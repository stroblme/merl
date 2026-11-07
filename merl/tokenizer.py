"""Tokenize text as UTF-8 bytes with a 256-token vocabulary.

Text round-trips without a learned vocabulary. The FIM sentinels use bytes
0xF5-0xF7, which valid UTF-8 never contains. Keep records containing them
as `bytes`; decoding a sentinel produces U+FFFD.
"""

VOCAB_SIZE = 256

#: Fill-in-the-middle sentinels, in the order a record is written (SPM):
#: `SUF <suffix> PRE <prefix> MID <middle> \n`. The prefix is what changes
#: while the user types, so it comes last -- a KV cache can be extended.
SUF, PRE, MID = 0xF5, 0xF6, 0xF7


def encode(text: str | bytes) -> list[int]:
    return list(text.encode("utf-8") if isinstance(text, str) else text)


def decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")
