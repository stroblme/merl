"""Declare the `merl --train` run as a fluksio flow.

The local engine records inputs, losses, and weights. Training settings,
including model shape, come from `merl.train.DEFAULTS` as sweepable inputs.
The node wraps `merl.train.train`, which remains usable without fluksio.
"""

import pathlib
import tempfile

import fluksio
from fluksio import Flow, Port, node

from merl.train import DEFAULTS

NAME = "merl_train"

#: One input per knob; the CLI spells the same names `--dim`, `--lr`, ...
PORTS = [
    Port(name, "float" if isinstance(default, float) else "int", initial=default)
    for name, default in DEFAULTS.items()
]


@node(
    requires=[Port("corpus", "str"), *(Port(p.name, p.dtype) for p in PORTS)],
    provides=[
        Port("loss", "float", stream=True),
        Port("final_loss", "float"),
        Port("n_params", "float"),
        Port("weights", "artifact"),
    ],
    # No `gpus` declaration: this is the shipped training command, so a machine
    # without a card should train on the CPU rather than refuse the run.
    # preallocate=false leaves room beside a study holding the same card.
    resources={"env": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}},
)
def fit(
    corpus: str,
    dim: int,
    n_layers: int,
    n_heads: int,
    context: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
) -> dict:
    """Train from scratch on a FIM record file, or WikiText if `corpus` is empty."""
    from flax import nnx

    from merl.dataset import base_corpus
    from merl.model import Config, Merl, save
    from merl.train import train

    data = pathlib.Path(corpus).read_bytes() if corpus else base_corpus()
    cfg = Config(dim=dim, n_layers=n_layers, n_heads=n_heads, context=context)
    model = Merl(cfg, rngs=nnx.Rngs(0))
    losses = train(
        model,
        data,
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        on_log=lambda step, loss: fluksio.emit(loss=loss),
    )

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.npz"
    save(model, path)
    return {
        "final_loss": losses[-1],
        "n_params": model.n_params(),
        "weights": fluksio.save_artifact(path.read_bytes(), name="weights.npz"),
    }


FLOW = Flow(
    NAME,
    title="merl --train",
    nodes=[fit],
    # The corpus travels as a path: megabytes of records are a file for the
    # worker to read, not a parameter to record on every run.
    inputs=[Port("corpus", "str", initial=""), *PORTS],
    outputs=["final_loss", "n_params"],
)
