"""Merl's attention-only transformer, based on wren's Simple Attention Network.

It replaces feed-forward layers with more attention layers and uses QK-norm
to keep deep stacks trainable (https://arxiv.org/abs/2607.18363). The vault
is supplied in the prompt, where attention-only models are strongest.
"""

import pathlib
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from merl.tokenizer import VOCAB_SIZE


@dataclass
class Config:
    vocab_size: int = VOCAB_SIZE
    dim: int = 256  # width of the residual stream
    n_layers: int = 8  # depth: the CPU latency axis -- keep shallow (README)
    n_heads: int = 4
    context: int = 384  # maximum sequence length; holds a pruned state block + calls


def rope(x: jax.Array, offset: jax.Array | int = 0) -> jax.Array:
    """Apply rotary positions to query or key feature pairs.

    Each pair uses a distinct frequency, making dot products depend on
    relative position. `offset` sets the starting position for decoding.
    """
    _, seq, _, head_dim = x.shape  # (batch, seq, heads, head_dim)
    half = head_dim // 2
    freqs = 1.0 / (10000.0 ** (jnp.arange(half) / half))
    angles = (jnp.arange(seq) + offset)[:, None] * freqs[None, :]  # (seq, half)
    cos = jnp.cos(angles)[None, :, None, :]
    sin = jnp.sin(angles)[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class AttentionBlock(nnx.Module):
    """Pre-norm causal self-attention with QK-norm and no feed-forward layer."""

    def __init__(self, cfg: Config, rngs: nnx.Rngs):
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.norm = nnx.RMSNorm(cfg.dim, rngs=rngs)
        self.wqkv = nnx.Linear(cfg.dim, 3 * cfg.dim, use_bias=False, rngs=rngs)
        self.wo = nnx.Linear(cfg.dim, cfg.dim, use_bias=False, rngs=rngs)
        # QK-norm: RMS-normalize queries and keys per head before their dot
        # product, bounding attention logits. Per the paper, this is the one
        # ingredient that keeps deep attention-only stacks trainable.
        self.q_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)
        self.k_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch, seq, dim = x.shape
        h = self.norm(x)
        q, k, v = jnp.split(self.wqkv(h), 3, axis=-1)
        q = q.reshape(batch, seq, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq, self.n_heads, self.head_dim)
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(self.head_dim)
        # Causal mask: token q may only attend to tokens k <= q, so the
        # model can't peek at the future it is being trained to predict.
        causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        scores = jnp.where(causal, scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)

        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(batch, seq, dim)
        return x + self.wo(out)

    def prefill(
        self, x: jax.Array, k_cache: jax.Array, v_cache: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Attend to the full prompt and cache keys and values in slots `0:seq`.

        This is the batched equivalent of decoding those positions one by one.
        """
        batch, seq, dim = x.shape
        h = self.norm(x)
        q, k, v = jnp.split(self.wqkv(h), 3, axis=-1)
        q = q.reshape(batch, seq, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq, self.n_heads, self.head_dim)
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(self.head_dim)
        causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        scores = jnp.where(causal, scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)

        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(batch, seq, dim)
        k_cache = jax.lax.dynamic_update_slice_in_dim(k_cache, k, 0, axis=1)
        v_cache = jax.lax.dynamic_update_slice_in_dim(v_cache, v, 0, axis=1)
        return x + self.wo(out), k_cache, v_cache

    def decode(
        self, x: jax.Array, k_cache: jax.Array, v_cache: jax.Array, pos: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Attend from one token at `pos` to cached keys and values.

        Update both caches at `pos`. Caching reduces each generation step
        from quadratic to linear work in sequence length.
        """
        batch, _, dim = x.shape
        h = self.norm(x)
        q, k, v = jnp.split(self.wqkv(h), 3, axis=-1)
        q = q.reshape(batch, 1, self.n_heads, self.head_dim)
        k = k.reshape(batch, 1, self.n_heads, self.head_dim)
        v = v.reshape(batch, 1, self.n_heads, self.head_dim)
        q = rope(self.q_norm(q), offset=pos)
        k = rope(self.k_norm(k), offset=pos)
        k_cache = jax.lax.dynamic_update_slice_in_dim(k_cache, k, pos, axis=1)
        v_cache = jax.lax.dynamic_update_slice_in_dim(v_cache, v, pos, axis=1)

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k_cache) / jnp.sqrt(self.head_dim)
        mask = jnp.arange(k_cache.shape[1]) <= pos  # causal: only filled slots
        scores = jnp.where(mask[None, None, None, :], scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)

        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v_cache).reshape(batch, 1, dim)
        return x + self.wo(out), k_cache, v_cache


class Merl(nnx.Module):
    """Decoder that maps byte tokens to next-byte logits."""

    def __init__(self, cfg: Config, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embed = nnx.Embed(cfg.vocab_size, cfg.dim, rngs=rngs)
        self.blocks = nnx.List(AttentionBlock(cfg, rngs) for _ in range(cfg.n_layers))
        self.norm = nnx.RMSNorm(cfg.dim, rngs=rngs)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """Map `(batch, seq)` byte IDs to `(batch, seq, vocab)` logits."""
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        # Tied output head: attend() reuses the embedding matrix transposed
        # as the unembedding -- standard for small models, saves vocab*dim.
        return self.embed.attend(self.norm(x))

    def n_params(self) -> int:
        return sum(p.size for p in jax.tree.leaves(nnx.state(self, nnx.Param)))

    def init_cache(self, batch: int) -> list[tuple[jax.Array, jax.Array]]:
        """Create empty per-layer key and value caches for `batch` rows."""
        shape = (
            batch,
            self.cfg.context,
            self.cfg.n_heads,
            self.cfg.dim // self.cfg.n_heads,
        )
        return [(jnp.zeros(shape), jnp.zeros(shape)) for _ in self.blocks]

    def decode(
        self,
        tokens: jax.Array,
        caches: list[tuple[jax.Array, jax.Array]],
        pos: jax.Array,
    ) -> tuple[jax.Array, list[tuple[jax.Array, jax.Array]]]:
        """Decode one token per row at `pos`, returning logits and updated caches."""
        x = self.embed(tokens)
        updated = []
        for block, (k, v) in zip(self.blocks, caches):
            x, k, v = block.decode(x, k, v, pos)
            updated.append((k, v))
        return self.embed.attend(self.norm(x))[:, 0], updated

    def prefill(
        self, tokens: jax.Array, caches: list[tuple[jax.Array, jax.Array]]
    ) -> list[tuple[jax.Array, jax.Array]]:
        """Fill prompt caches in one batched pass."""
        x = self.embed(tokens)
        updated = []
        for block, (k, v) in zip(self.blocks, caches):
            x, k, v = block.prefill(x, k, v)
            updated.append((k, v))
        return updated


def generate(
    model: Merl,
    prompt: list[int],
    n_new: int,
    temperature: float = 1.0,
    seed: int = 0,
) -> list[int]:
    """Return `prompt` followed by `n_new` generated bytes.

    Generation uses `generate_batch`; prompts longer than the available
    context are truncated at the front during decoding.
    """
    assert prompt, "empty prompt"
    outs, _ = generate_batch(model, [prompt], n_new, temperature=temperature, seed=seed)
    return list(prompt) + outs[0]


@nnx.jit
def _decode_step(
    model: Merl,
    tokens: jax.Array,
    caches: list[tuple[jax.Array, jax.Array]],
    pos: jax.Array,
) -> tuple[jax.Array, list[tuple[jax.Array, jax.Array]]]:
    return model.decode(tokens, caches, pos)


@nnx.jit
def _prefill(
    model: Merl, tokens: jax.Array, caches: list[tuple[jax.Array, jax.Array]]
) -> list[tuple[jax.Array, jax.Array]]:
    return model.prefill(tokens, caches)


@nnx.jit
def _decode_loop(
    model: Merl,
    buf: jax.Array,
    caches: list[tuple[jax.Array, jax.Array]],
    lens: jax.Array,
    ends: jax.Array,
    start: jax.Array,
    stop_arr: jax.Array,
    temperature: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Decode on device, greedily at temperature 0 or by sampling above 0.

    Return the buffer, row end positions, log-probability sums, and byte counts.
    """
    batch, width = buf.shape
    s = stop_arr.shape[0]

    def cond(carry):
        t, _buf, _caches, done, ends, _logprob, _written, _ring = carry
        return (t < width - 1) & jnp.any(~done & (t + 1 < ends))

    def body(carry):
        t, buf, caches, done, ends, logprob, written, ring = carry
        tokens = jax.lax.dynamic_slice_in_dim(buf, t, 1, axis=1)
        logits, caches = model.decode(tokens, caches, t)
        sampled = jax.random.categorical(
            jax.random.fold_in(key, t), logits / jnp.maximum(temperature, 1e-6)
        )
        greedy = jnp.argmax(logits, axis=-1)
        nxt = jnp.where(temperature > 0, sampled, greedy).astype(jnp.int32)
        write = ~done & (t + 1 < ends) & (t + 1 >= lens)
        cur = jax.lax.dynamic_slice_in_dim(buf, t + 1, 1, axis=1)[:, 0]
        new = jnp.where(write, nxt, cur)
        buf = jax.lax.dynamic_update_slice_in_dim(buf, new[:, None], t + 1, axis=1)
        picked = jnp.take_along_axis(jax.nn.log_softmax(logits), nxt[:, None], 1)[:, 0]
        logprob += jnp.where(write, picked, 0.0)
        written += write
        if s:  # static: traced once per stop length
            ring = jnp.concatenate([ring[:, 1:], new[:, None]], axis=1)
            # Only a window of purely generated bytes may match the stop.
            matched = write & (written >= s) & jnp.all(ring == stop_arr, axis=1)
            done |= matched
            ends = jnp.where(matched, t + 2, ends)
        return t + 1, buf, caches, done, ends, logprob, written, ring

    carry = (
        start,
        buf,
        caches,
        jnp.zeros(batch, dtype=bool),
        ends,
        jnp.zeros(batch),
        jnp.zeros(batch, dtype=jnp.int32),
        jnp.zeros((batch, max(s, 1)), dtype=jnp.int32),
    )
    _, buf, _, _, ends, logprob, written, _ = jax.lax.while_loop(cond, body, carry)
    return buf, ends, logprob, written


def generate_batch(
    model: Merl,
    prompts: list[list[int]],
    n_new: int,
    stop: tuple[int, ...] = (),
    temperature: float = 0.0,
    seed: int = 0,
) -> tuple[list[list[int]], list[float]]:
    """Generate continuations for a batch of prompts.

    Prefill caches together, then decode in one compiled loop. Truncate
    prompts from the front to leave room for `n_new` tokens and stop each
    continuation after `stop`. Return generated tokens and per-byte
    confidence, the geometric mean of generated token probabilities.
    Temperature 0 is greedy; higher values sample.
    """
    cfg = model.cfg
    prompts = [p[-(cfg.context - n_new) :] for p in prompts]
    batch = len(prompts)
    lens = np.array([len(p) for p in prompts])
    # Rounding the width up bounds recompilation to a handful of shapes.
    width = min(-(-(int(lens.max()) + n_new) // 64) * 64, cfg.context)
    buf = np.zeros((batch, width), dtype=np.int32)
    for i, p in enumerate(prompts):
        buf[i, : len(p)] = p
    caches = model.init_cache(batch)
    if width > 2:  # one batched pass instead of width sequential steps
        caches = _prefill(model, jnp.asarray(buf[:, : width - 1]), caches)
    buf, ends, logprob, written = _decode_loop(
        model,
        jnp.asarray(buf),
        caches,
        jnp.asarray(lens, dtype=jnp.int32),
        jnp.asarray(lens + n_new, dtype=jnp.int32),
        jnp.int32(max(int(lens.min()) - 1, 0)),
        jnp.array(stop, dtype=jnp.int32),
        jnp.float32(temperature),
        jax.random.PRNGKey(seed),
    )
    buf, ends = np.asarray(buf), np.asarray(ends)
    outs = [buf[i, lens[i] : ends[i]].tolist() for i in range(batch)]
    conf = np.exp(np.asarray(logprob) / np.maximum(np.asarray(written), 1))
    return outs, [float(c) for c in conf]


WEIGHTS = pathlib.Path(__file__).parent / "weights.npz"


def save(model: Merl, path: pathlib.Path = WEIGHTS) -> None:
    """Save model parameters and configuration to `path`."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))
    arrays = {jax.tree_util.keystr(k): np.asarray(v) for k, v in leaves}
    arrays |= {f"cfg:{k}": v for k, v in asdict(model.cfg).items()}
    np.savez(path, **arrays)


def load(path: pathlib.Path = WEIGHTS) -> Merl:
    """Load a model and its configuration from `path`."""
    npz = np.load(path)
    cfg = Config(**{k.removeprefix("cfg:"): int(npz[k]) for k in npz if "cfg:" in k})
    model = Merl(cfg, rngs=nnx.Rngs(0))
    leaves, treedef = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))
    restored = [jnp.asarray(npz[jax.tree_util.keystr(k)]) for k, _ in leaves]
    nnx.update(model, jax.tree_util.tree_unflatten(treedef, restored))
    return model
