# Merl

Merl is a tiny ghost-text engine for a markdown vault: at the caret it
proposes the rest of the line, from what the vault already says. It is the
inline-completion backend for [accent](https://github.com/stroblme/accent)
and speaks LSP 3.18 `textDocument/inlineCompletion`, so any editor with an
inline-completion client can use it.

Two things answer. A suffix array over every note (an ∞-gram,
`runtime/merl.c`) copies what the vault has literally written before, in
milliseconds and with no training. A byte-level attention-only model, wren's
architecture fine-tuned on the vault, will continue what it has not
(planned, not in the runtime yet). Everything runs on a CPU; the runtime is
one C file with no dependencies.

## Usage

```sh
make install                                          # build/merl-rt, copied to ~/.local/bin
merl-rt --vault ~/notes                               # LSP server on stdio
merl-rt --vault ~/notes --suggest "the completion"    # one answer, then exit
```

`make build` only compiles, `PREFIX=/usr/local` moves the install, and
`make help` lists the rest (`test`, `clean`).

The vault is walked for `*.md`. Under git that means what `git ls-files
--cached --others --exclude-standard` lists, so `.gitignore`, `.git/info/exclude`
and the global `core.excludesFile` all apply and a build tree stays out of the
suggestions. The index is built on a thread and swapped in when it is ready, so
neither start-up nor a save blocks an answer.

The note being typed in is searched too, and so are the other open documents:
the longest match across the three decides, so a line written a minute ago and
not yet saved is offered like any other. The constants are compiled in and can
be moved from the build line (`-DMIN_N=8`, `-DLONG_N=24`, ... in `CFLAGS`).

Training and evaluation live in Python (`uv sync --group dev`):

```sh
uv run merl --train              # WikiText base corpus, on the fluksio engine
uv run merl --train records.bin  # any byte stream of fill-in-the-middle records
make test                        # includes the C runtime, pinned against merl.ngram
```

Datasets download into `data/` on demand (`merl.dataset`).

## Layout

- `merl/` — the library: `tokenizer` (bytes plus three fill-in-the-middle
  sentinels), `model` (wren's transformer), `train`, `flow` (the fluksio
  run), `dataset` (vault walk, WikiText, FIM records), `ngram` (the ∞-gram
  reference the C binary is pinned to), `lsp` (a client for driving the
  binary), `cli`.
- `runtime/merl.c` — the server. `runtime/jsmn.h` (JSON, MIT) and
  `runtime/libsais.{c,h}` (suffix arrays, Apache-2.0) are vendored verbatim.

## Toolstack

jax, flax, optax for training; uv, ruff, ty, pytest; fluksio for
tracked runs; a C compiler for the runtime.
