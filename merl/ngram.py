"""Reference ∞-gram for the vault's notes.

A suffix array finds the longest matching context, then extends it while
occurrences agree. `runtime/merl.c` is tested against this implementation.
Infini-gram (arXiv:2401.17377) applies the idea at trillion-token scale.
"""

import bisect

#: Bytes a suffix is sorted by: MAXN of prefix plus MAX_LEN of continuation,
#: so every narrowed range stays contiguous for the whole walk. `find` and the
#: walk use each range as a *set*, never in order, so any ordering that groups
#: suffixes by their first bytes gives the same answers -- which is why
#: runtime/merl.c can hand the job to libsais (a full lexicographic order) and
#: stay pinned to this file.
KEY = 128
MAXN = 64  # bytes of prefix consulted
MAX_LEN = 64  # bytes suggested at most

# ponytail: swept on a stand-in vault; runtime/merl.c has the same numbers.
MIN_N = 6  # shortest match worth continuing
MIN_COUNT = 2  # occurrences a match needs
LONG_N = 32  # bytes of context past which one occurrence is enough
SHARE = 0.5  # agreement the next byte needs


def _enough(count: int, n: int, min_count: int, long_n: int) -> bool:
    """Accept `min_count` occurrences, or one when `n >= long_n`."""
    return count >= min_count or (n >= long_n and count >= 1)


def _sep(b: int) -> bool:
    """Whether `b` is an ASCII non-alphanumeric separator."""
    return not (bytes([b]).isalnum() or b >= 0x80)


def _aligned(prefix: bytes, n: int) -> int:
    """Classify the last `n` bytes as separator-led (1), word-aligned (2),
    or mid-word (0). Word-aligned matches require each occurrence to start a
    word, allowing `compute_` to match `(compute_`. A whole prefix shorter
    than MAXN also counts as separator-led."""
    i = len(prefix) - n
    if _sep(prefix[i]) or (n == len(prefix) and n < MAXN):
        return 1
    return 2 if i > 0 and _sep(prefix[i - 1]) else 0


def _sentence_end(text: bytes) -> bool:
    """Whether whitespace follows sentence-ending punctuation."""
    return len(text) > 1 and text[-1:].isspace() and text[-2] in b".!?"


class Index:
    def __init__(self, corpus: bytes):
        self.data = corpus
        # ponytail: a KEY-byte key object per position -- fine for a test
        # vault; the C runtime is what indexes a real one.
        self.sa = sorted(range(len(corpus)), key=lambda i: corpus[i : i + KEY])

    def find(
        self, pattern: bytes, lo: int = 0, hi: int | None = None
    ) -> tuple[int, int]:
        """Return the range in `sa[lo:hi]` whose suffixes start with `pattern`."""
        data, m = self.data, len(pattern)
        if hi is None:
            hi = len(self.sa)

        def key(i: int) -> bytes:
            return data[i : i + m]

        return (
            bisect.bisect_left(self.sa, pattern, lo, hi, key=key),
            bisect.bisect_right(self.sa, pattern, lo, hi, key=key),
        )

    def at_word(self, i: int) -> bool:
        """Whether the suffix at `i` starts a word."""
        return i == 0 or _sep(self.data[i - 1])

    def suggest(
        self,
        prefix: bytes,
        after: bytes = b"",
        max_len: int = MAX_LEN,
        min_n: int = MIN_N,
        min_count: int = MIN_COUNT,
        long_n: int = LONG_N,
        share: float = SHARE,
    ) -> bytes:
        """Suggest a continuation through the sentence or line end.

        Return empty bytes if no match qualifies or the suggestion repeats
        `after`, the text already following the caret.
        """
        prefix = prefix[-MAXN:]
        lo = hi = 0
        for n in range(len(prefix), min_n - 1, -1):
            kind = _aligned(prefix, n)
            if not kind:
                continue
            lo, hi = self.find(prefix[-n:])
            count = hi - lo if kind == 1 else sum(map(self.at_word, self.sa[lo:hi]))
            if _enough(count, n, min_count, long_n):
                break
        else:
            return b""
        pattern = prefix[-n:]
        data, sa = self.data, self.sa
        out = bytearray()
        whole = False  # a complete answer, rather than one cut at the budget
        while len(out) < max_len:
            counts = [0] * 256
            total = 0
            for k in range(lo, hi):
                if kind == 2 and not self.at_word(sa[k]):
                    continue
                total += 1
                j = sa[k] + n + len(out)
                if j < len(data):
                    counts[data[j]] += 1
            byte = max(range(256), key=lambda b: (counts[b], -b))
            c = counts[byte]
            if not _enough(c, n + len(out), min_count, long_n):
                break
            if c <= share * total:
                break  # a tie is not a majority
            if byte == 10:
                whole = True  # the line ends here, and so does the suggestion
                break
            out.append(byte)
            if _sentence_end(pattern + out):
                whole = True  # so does the sentence the caret was inside
                break
            lo, hi = self.find(pattern + bytes(out), lo, hi)
        text = trim_utf8(bytes(out))
        if not whole:
            # Stopped in the middle of what it was copying: a half-written word
            # is worse than a shorter suggestion.
            while text and not _sep(text[-1]):
                text = text[:-1]
        if not text.strip() or after.split(b"\n", 1)[0].startswith(text):
            return b""
        return text


def trim_utf8(b: bytes) -> bytes:
    """Remove a trailing incomplete UTF-8 sequence."""
    p = len(b)
    while p > 0 and len(b) - p < 4 and b[p - 1] & 0xC0 == 0x80:
        p -= 1
    if p == 0:
        return b
    lead = b[p - 1]
    p -= 1
    need = (
        1
        if lead < 0xC0
        else 2
        if lead < 0xE0
        else 3
        if lead < 0xF0
        else 4
        if lead < 0xF8
        else 1
    )
    return b[:p] if len(b) - p < need else b
