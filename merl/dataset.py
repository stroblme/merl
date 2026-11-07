"""Build fill-in-the-middle records from vault notes or WikiText.

Each record is `SUF <suffix> PRE <prefix> MID <middle> \\n`. The middle
continues from a random split to the line end, teaching the model to stop
where a ghost-text suggestion stops.
"""

import os
import pathlib
import random
import re
import subprocess
import urllib.request
import zipfile

from merl.tokenizer import MID, PRE, SUF

DATA = pathlib.Path(__file__).parent.parent / "data"

#: The record budget, in bytes. runtime/merl.c assembles prompts under the
#: same numbers, and the three parts plus their sentinels fit
#: `Config.context` (384), which tests/test_dataset.py pins.
SUFFIX_MAX = 64
MIDDLE_MAX = 64
PREFIX_MAX = 253


def _git_paths(root: pathlib.Path) -> list[str]:
    """List Git's tracked and untracked, nonignored Markdown paths.

    Return an empty list if Git or the repository is unavailable.
    """
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [n for n in out.decode(errors="replace").split("\0") if n.endswith(".md")]


def vault(root: str | os.PathLike) -> list[tuple[str, bytes]]:
    """Return Markdown notes as sorted (relative path, raw bytes) pairs.

    Use Git's nonignored paths when available; otherwise walk the filesystem,
    skipping dot entries and symlinks. Ordering matches `runtime/merl.c`.
    """
    root = pathlib.Path(root)
    if names := _git_paths(root):
        return sorted(
            (n, (root / n).read_bytes()) for n in names if (root / n).is_file()
        )
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = pathlib.Path(dirpath)
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and not (here / d).is_symlink()
        ]
        for name in filenames:
            path = here / name
            if name.startswith(".") or not name.endswith(".md") or path.is_symlink():
                continue
            files.append((path.relative_to(root).as_posix(), path.read_bytes()))
    files.sort()
    return files


def corpus(files: list[tuple[str, bytes]]) -> bytes:
    """Join file contents, appending a newline to each."""
    return b"".join(text + b"\n" for _, text in files)


def record(data: bytes, split: int, rng: random.Random) -> bytes:
    """Build a FIM record at `split`, with the line's remainder as the middle.

    Each field has a byte limit. The prefix precedes the split; the suffix
    follows the line and is empty one time in ten to train on document endings.
    """
    end = data.find(b"\n", split)
    if end < 0:
        end = len(data)
    middle = data[split:end][:MIDDLE_MAX]
    suffix = b"" if rng.random() < 0.1 else data[end + 1 : end + 1 + SUFFIX_MAX]
    prefix = data[max(0, split - PREFIX_MAX) : split]
    return bytes([SUF]) + suffix + bytes([PRE]) + prefix + bytes([MID]) + middle + b"\n"


def fim(data: bytes, rng: random.Random, n: int) -> bytes:
    """Sample `n` byte positions for records, weighting longer lines more."""
    return b"".join(record(data, rng.randrange(len(data)), rng) for _ in range(n))


WIKITEXT = "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"


def wikitext(max_bytes: int = 8_000_000) -> str:
    """Download and detokenize the first `max_bytes` of WikiText-103 training data.

    Cache the archive in `data/wikitext/` and convert headings to Markdown.
    """
    folder = DATA / "wikitext"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / WIKITEXT.rsplit("/", 1)[-1]
    if not path.exists():
        urllib.request.urlretrieve(WIKITEXT, path)
    with zipfile.ZipFile(path) as zf, zf.open("wikitext-103-raw/wiki.train.raw") as f:
        return detokenize(f.read(max_bytes).decode("utf-8", errors="ignore"))


def _heading(m: re.Match) -> str:
    return "#" * m.group(1).count("=") + " " + m.group(2)


def detokenize(text: str) -> str:
    """Restore readable prose and Markdown headings from WikiText."""
    text = re.sub(r"^ ((?:= )+)(.+?)(?: =)+ $", _heading, text, flags=re.MULTILINE)
    text = re.sub(r" @(.)@ ", r"\1", text)  # 1 @,@ 000 -> 1,000; @-@ -> -
    text = re.sub(r" ([,.;:!?%)\]'])", r"\1", text)  # no space before closers
    text = re.sub(r"([(\[$]) ", r"\1", text)  # none after openers
    text = re.sub(r" (n't|'s|'m|'re|'ve|'ll|'d)\b", r"\1", text)
    text = re.sub(r'" (.*?) "', r'"\1"', text)
    return re.sub(r"^ | $", "", text, flags=re.MULTILINE)


def base_corpus(max_bytes: int = 8_000_000, n: int = 100_000, seed: int = 0) -> bytes:
    """Build the default FIM training stream from a WikiText slice."""
    return fim(wikitext(max_bytes).encode(), random.Random(seed), n)
