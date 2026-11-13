<p align="center">
  <img src="docs/merl-logo.svg" width="160" alt="Merl logo">
</p>

<p align="center">
  <a href="https://github.com/stroblme/merl/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/stroblme/merl/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

Merl is a tiny ghost-text engine for markdown notes.
Based on other text in a vault, it proposes the next words of the line you're writing.
I'm developing it as an inline-completion backend for [accent](https://github.com/stroblme/accent), but because it speaks LSP 3.18, basically any editor with an inline-completion client could use it.

Merl uses an ∞-gram (see [credits](#credits)) to extract text from other files in the vault within milliseconds of typing.
In the next stage, I'm planning to extend Merl with a fine-tuned attention model, [Wren](https://github.com/stroblme/wren), to give more precise suggestions.
No GPU needed btw.

The runtime indexes a 67 MB vault in **2.4 s** with **330 MB** of memory on a CPU, so no GPU is needed.

## Setup

You need a C compiler, make and Git.
Build the runtime and copy it to `~/.local/bin` by running:

```sh
git clone https://github.com/stroblme/merl.git
cd merl
make install
```

`PREFIX` changes the install location and `CFLAGS` the compiler flags (default `-O3`).
To run the runtime inside a Rust program, add the `merl-sys` crate from the repo:

```toml
[dependencies]
merl-sys = { git = "https://github.com/stroblme/merl" }
```

For training and development, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and, in the checkout, run:

```sh
uv sync --group dev
uv run fluksio serve
```

Keep the Fluksio engine running for training.
`make help` lists the build and test chores.

## Usage

Ask for one suggestion from a vault of notes:

```sh
merl-rt --vault ~/notes --suggest "the completion"
```

For an editor, run it as an LSP server on stdio:

```sh
merl-rt --vault ~/notes
```

Without `--vault`, the vault is the client's first workspace folder, else its `rootUri`.
In Rust, `merl_sys::run` does what `merl-rt`'s `main` does.
That is how accent ships ghost text inside `accent-cli`.

Training and evaluation live in Python.
From a checkout with the engine running, train in another terminal:

```sh
uv run merl --train              # WikiText base corpus
uv run merl --train records.bin  # any byte stream of fill-in-the-middle records
```

This records the training run and saves `merl/weights.npz`.
Datasets download into `data/` on demand (`merl.dataset`).
The tests include the C runtime, pinned against `merl.ngram`:

```sh
make test
```

## Credits

- [Infini-gram](https://arxiv.org/abs/2401.17377) by Jiacheng Liu et al. for the ∞-gram idea.
- [libsais](https://github.com/IlyaGrebnov/libsais) by Ilya Grebnov for suffix arrays (Apache-2.0).
- [jsmn](https://github.com/zserge/jsmn) by Serge Zaitsev for JSON parsing (MIT).
- [WikiText](https://arxiv.org/abs/1609.07843) by Stephen Merity et al. for the base corpus.
