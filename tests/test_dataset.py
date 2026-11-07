import random

from merl import dataset
from merl.model import Config
from merl.tokenizer import MID, PRE, SUF


def test_record_shape_and_budget():
    data = b"a" * 500 + b"\n" + b"middle rest of line" + b"\n" + b"b" * 500
    rng = random.Random(0)
    rec = dataset.record(data, 501, rng)  # split at the start of line two
    assert rec[0] == SUF and rec[-1] == 10
    suffix, rest = rec[1:].split(bytes([PRE]), 1)
    prefix, middle = rest.split(bytes([MID]), 1)
    assert suffix == b"b" * dataset.SUFFIX_MAX
    assert prefix == b"a" * (dataset.PREFIX_MAX - 1) + b"\n"
    assert middle == b"middle rest of line\n"
    budget = dataset.SUFFIX_MAX + dataset.PREFIX_MAX + dataset.MIDDLE_MAX + 3
    assert budget <= Config().context


def test_fim_records_end_at_newlines():
    stream = dataset.fim(b"one line\nanother line\n", random.Random(1), 20)
    assert stream.count(bytes([SUF])) == 20
    assert stream.count(bytes([MID])) == 20
    assert stream.endswith(b"\n")


def test_detokenize():
    text = ' = = Gameplay = = \n The game \'s 1 @,@ 000 units ( see below ) , said " no " . \n'
    assert dataset.detokenize(text) == (
        '## Gameplay\nThe game\'s 1,000 units (see below), said "no".\n'
    )


def test_vault_walk(tmp_path):
    (tmp_path / "b.md").write_bytes(b"bee")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.md").write_bytes(b"ay")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "x.md").write_bytes(b"no")
    (tmp_path / "notes.txt").write_bytes(b"no")
    (tmp_path / "link.md").symlink_to(tmp_path / "b.md")
    files = dataset.vault(tmp_path)
    assert files == [("b.md", b"bee"), ("sub/a.md", b"ay")]
    assert dataset.corpus(files) == b"bee\nay\n"
