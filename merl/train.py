"""Train next-byte prediction on byte windows.

`merl.dataset` builds fill-in-the-middle records; the loss still predicts
the next byte at each position. Sentinels require a `bytes` corpus.
"""

import inspect
from collections.abc import Callable
from dataclasses import fields

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from merl import tokenizer
from merl.model import Config, Merl


def muon(lr: float) -> optax.GradientTransformation:
    """Use Muon for attention matrices and its AdamW path for other parameters."""

    def hidden_matrices(params: dict) -> dict:
        return jax.tree_util.tree_map_with_path(
            lambda path, p: (
                optax.contrib.MuonDimensionNumbers()
                if p.ndim == 2 and "embed" not in jax.tree_util.keystr(path)
                else None
            ),
            params,
        )

    # ty: muon's annotation omits the callable form its source handles
    return optax.contrib.muon(
        lr,
        muon_weight_dimension_numbers=hidden_matrices,  # ty: ignore[invalid-argument-type]
    )


def sample_batch(
    data: jax.Array, batch_size: int, seq_len: int, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Sample `batch_size` windows, with each target one byte after its input."""
    starts = jax.random.randint(key, (batch_size,), 0, len(data) - seq_len - 1)
    idx = starts[:, None] + jnp.arange(seq_len + 1)[None, :]
    chunk = data[idx]
    return chunk[:, :-1], chunk[:, 1:]


@nnx.jit
def train_step(
    model: Merl, optimizer: nnx.Optimizer, x: jax.Array, y: jax.Array
) -> jax.Array:
    def loss_fn(model: Merl) -> jax.Array:
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def train(
    model: Merl,
    data: bytes,
    steps: int = 4000,
    batch_size: int = 32,
    seq_len: int = 384,
    lr: float = 3e-3,
    on_log: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Train `model` in place and return one loss per step.

    Call `on_log(step, loss)` after each step when provided.
    """
    stream = jnp.array(tokenizer.encode(data), dtype=jnp.int32)
    optimizer = nnx.Optimizer(model, muon(lr), wrt=nnx.Param)
    key = jax.random.PRNGKey(0)
    losses = []
    for step in range(steps):
        key, sub = jax.random.split(key)
        x, y = sample_batch(stream, batch_size, seq_len, sub)
        loss = float(train_step(model, optimizer, x, y))
        losses.append(loss)
        if on_log is not None:
            on_log(step, loss)
    return losses


#: Every knob `merl --train` exposes, and the one place its default lives:
#: the model's shape, then training's own arguments. Read off `Config` and
#: the signature above rather than listed again, so editing either is the
#: whole change -- the CLI turns each into a `--flag`, the flow into an input
#: (merl/flow.py), and the run records what it was given.
DEFAULTS: dict[str, int | float] = {
    field.name: field.default
    for field in fields(Config)
    # vocab_size is the byte tokenizer's business, not a knob.
    if field.name != "vocab_size" and isinstance(field.default, int | float)
} | {
    name: parameter.default
    for name, parameter in inspect.signature(train).parameters.items()
    if isinstance(parameter.default, int | float)
}
