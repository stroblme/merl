# Merl

Merl is a tiny ghost-text engine for markdown notes.
Based on other text in a vault, it proposes the next words of the line you're writing.
I'm developing it as an inline-completion backend for [accent](https://github.com/stroblme/accent), but because it speaks LSP 3.18, basically any editor with an inline-completion client could use it.

Merl uses an ∞-gram to extract text from other files in the vault within milliseconds of typing.
In the next stage, I'm planning to extend Merl with a fine-tuned attention model, [Wren](https://github.com/stroblme/wren), to give more precise suggestions.
No GPU needed btw.

## Usage

```sh
make install                                          # build/merl-rt, copied to ~/.local/bin
merl-rt --vault ~/notes                               # LSP server on stdio
merl-rt --vault ~/notes --suggest "the completion"    # one answer, then exit
```

Training and evaluation live in Python (`uv sync --group dev`):

```sh
uv run merl --train              # WikiText base corpus, on the fluksio engine
uv run merl --train records.bin  # any byte stream of fill-in-the-middle records
make test                        # includes the C runtime, pinned against merl.ngram
```

Datasets download into `data/` on demand (`merl.dataset`).

## Credits

- [Infini-gram](https://arxiv.org/abs/2401.17377) by Jiacheng Liu et al. for the ∞-gram idea.
- [libsais](https://github.com/IlyaGrebnov/libsais) by Ilya Grebnov for suffix arrays (Apache-2.0).
- [jsmn](https://github.com/zserge/jsmn) by Serge Zaitsev for JSON parsing (MIT).
- [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/google-deepmind/optax) for training, and fluksio for tracked runs.
- [WikiText](https://arxiv.org/abs/1609.07843) by Stephen Merity et al. for the base corpus.
