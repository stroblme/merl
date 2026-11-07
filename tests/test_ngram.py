from merl.ngram import Index, trim_utf8

CORPUS = b"""# Notes
Every text tab asks one Vault API about its own content.
Every text tab asks one Vault API about its own content.
Every text tab asks one Vault API about nothing at all.
naive cafe au lait
"""


def test_longest_match_then_majority_walk():
    index = Index(CORPUS)
    # Three matches agree on " Vault API about ", then two of three say "its".
    assert index.suggest(b"tab asks one") == b" Vault API about its own content."


def test_stops_where_agreement_drops():
    index = Index(CORPUS)
    assert index.suggest(b"tab asks one", share=0.8) == b" Vault API about "


def test_min_count_and_short_prefixes():
    index = Index(CORPUS)
    assert index.suggest(b"naive caf") == b""  # seen once, and short: no
    assert index.suggest(b"naive caf", min_count=1) == b"e au lait"
    assert index.suggest(b"asks") == b""  # shorter than min_n


def test_a_long_match_is_followed_on_a_single_occurrence():
    """One occurrence suffices once the match reaches `long_n` bytes."""
    index = Index(CORPUS)
    assert index.suggest(b"tab asks one Vault API about nothing") == b" at all."
    assert index.suggest(b"naive caf") == b""  # nine bytes: still needs two
    assert index.suggest(b"naive caf", long_n=9) == b"e au lait"


def test_a_suggestion_does_not_end_mid_word():
    """A byte-budget cutoff drops the unfinished word."""
    index = Index(CORPUS)
    assert index.suggest(b"tab asks one", max_len=20) == b" Vault API about "


def test_a_match_may_not_start_mid_word():
    """A mid-word match cannot complete `he walk` from `she walked`."""
    index = Index(b"then she walked away\n" * 2)
    assert index.suggest(b"and then he walk") == b""
    assert index.suggest(b"and then she walk") == b"ed away"


def test_a_match_may_start_at_a_word_through_any_separator():
    """Word-aligned matches cross separators but exclude `recompute_` hits."""
    index = Index(
        b"call(compute_total(price, qty))\nlist[compute_total(price, qty)]\n"
        + b"recompute_total(nothing)\n" * 3
    )
    assert index.suggest(b"y = compute_") == b"total(price, qty)"
    assert index.suggest(b"y = recompute_") == b"total(nothing)"


def test_a_tie_is_not_a_majority():
    """A tied next byte cannot extend a suggestion."""
    index = Index(b"the kettle was already boiling\nthe kettle was already cold\n")
    assert index.suggest(b"the kettle was already ") == b""


def test_a_suggestion_stops_at_the_end_of_a_sentence():
    """A suggestion stops after sentence-ending whitespace."""
    index = Index(b"the kettle boiled. then it cooled again\n" * 2)
    assert index.suggest(b"the kettle boil") == b"ed. "


def test_does_not_echo_the_text_after_the_caret():
    index = Index(CORPUS)
    after = b" Vault API about its own content.\nnext line"
    assert index.suggest(b"tab asks one", after=after) == b""
    assert index.suggest(b"tab asks one", after=b" Vault API") != b""


def test_trim_utf8():
    assert trim_utf8("café".encode()[:-1]) == b"caf"
    assert trim_utf8("café".encode()) == "café".encode()
    assert trim_utf8(b"") == b""
