"""Run Merl's model from the command line.

`merl <prompt>` or piped input generates text with the shipped weights.
`merl --train` uses fluksio to record training settings, losses, and weights,
then replaces the package's weights file. The editor uses `runtime/merl.c`.
"""

import argparse
import pathlib
import sys
import time
from typing import Any

from merl import tokenizer
from merl.model import WEIGHTS, generate, load
from merl.train import DEFAULTS


def main() -> None:
    parser = argparse.ArgumentParser(prog="merl", description=__doc__)
    parser.add_argument("prompt", nargs="*", help="prompt; read from stdin if omitted")
    parser.add_argument(
        "--train",
        nargs="?",
        const="",
        metavar="CORPUS",
        help="train on CORPUS (a FIM record stream; default: WikiText) and save",
    )
    parser.add_argument(
        "-n", "--tokens", type=int, default=64, help="bytes to generate"
    )
    parser.add_argument("-t", "--temperature", type=float, default=0.1)
    # `--dim`, `--steps`, `--lr`, ... -- one flag per knob, from the one place
    # its default lives. Every value is sent, so the run records the whole
    # configuration rather than only what was said out loud.
    for name, default in DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=type(default),
            default=default,
            help=f"training: {name} (default: %(default)s)",
        )
    args = parser.parse_args()

    if args.train is not None:
        train_run(args)
        return

    prompt = " ".join(args.prompt) if args.prompt else sys.stdin.read()
    if not prompt.strip():
        parser.error("empty prompt")
    if not WEIGHTS.exists():
        parser.error(f"no weights at {WEIGHTS}; run `merl --train` first")
    # Persist compiled programs across invocations: the generation loop
    # otherwise recompiles on every CLI start, dwarfing the generation.
    import jax

    jax.config.update(
        "jax_compilation_cache_dir",
        str(pathlib.Path.home() / ".cache" / "merl" / "jax"),
    )
    model = load(WEIGHTS)
    # Only the tail of an over-long prompt fits the context window.
    tokens = tokenizer.encode(prompt)[-model.cfg.context :]
    out = generate(model, tokens, args.tokens, args.temperature)
    sys.stdout.write(tokenizer.decode(out[len(tokens) :]))


def train_run(args: argparse.Namespace) -> None:
    """Submit training and save the resulting weights."""
    try:
        from fluksio.sdk import SyncError
        from fluksio.sdk.cli import discover
        from fluksio.sdk.client import ApiError, Client, repo_root, sync

        from merl import flow
    except ImportError:  # fluksio is a dev-group dependency; so is training.
        sys.exit("merl --train needs fluksio: `uv sync --group dev`")

    # Absolute: the worker reads the corpus from its own working directory.
    params: dict[str, Any] = {
        "corpus": str(pathlib.Path(args.train).resolve()) if args.train else ""
    }
    params |= {name: getattr(args, name) for name in DEFAULTS}

    try:
        client = Client()
        # Upload what this checkout declares before running it, the way
        # `fluksio run` does: the reason to train is usually the edit before it.
        sync(discover([flow.__name__]), client, repo=repo_root(__file__))
        # no_cache: `--train` is asked for to train, not to be handed the
        # result an identical earlier run already produced.
        handle = client.submit(flow.NAME, params, no_cache=True, cause="cli")
        log(f"{handle.id}  {flow.NAME}  {params}")
        follow(client, handle)
        if handle.status != "ok":
            for failed in handle.failures:
                log(f"{failed.get('node', '')}  {failed.get('error', '')}")
                log(str(failed.get("logs") or ""))
            sys.exit(f"run {handle.id} {handle.status}")
        # Artifacts and metrics are named for the flow that reported them.
        weights = handle.download(f"{flow.NAME}.weights")
    except (SyncError, ApiError) as exc:
        sys.exit(f"fluksio: {exc} (is `uv run fluksio serve` running?)")

    WEIGHTS.write_bytes(weights)
    result = handle.result
    log(
        f"{result['n_params']:,.0f} params, final loss "
        f"{result['final_loss']:.3f}; saved {WEIGHTS}"
    )


def follow(client: Any, handle: Any, poll: float = 2.0) -> None:
    """Poll until training finishes and report the latest loss each time.

    Read final metrics after completion so the last loss is included.
    """
    while True:
        done = handle.refresh().done
        points = client.metrics(handle.id)
        if points:
            log(f"step {len(points):4d}  loss {points[-1]['value']:.3f}")
        if done:
            return
        time.sleep(poll)


def log(message: str) -> None:
    """Write progress to stderr, leaving stdout for model output."""
    print(message, file=sys.stderr)
