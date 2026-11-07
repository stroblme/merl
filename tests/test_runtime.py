"""Check C runtime parity with `merl.ngram` over the LSP wire."""

import itertools
import pathlib
import shutil
import subprocess
import time

import pytest

from merl import dataset
from merl.lsp import Server
from merl.ngram import Index

RUNTIME = pathlib.Path(__file__).parent.parent / "runtime"

NOTES = {
    "a.md": "# Notes\nEvery text tab asks one Vault API about its own content.\n"
    "Every text tab asks one Vault API about its own content.\n",
    "sub/b.md": "Every text tab asks one Vault API about nothing at all.\n"
    "Ünïcode line: naïve café au lait\nÜnïcode line: naïve café au lait\n"
    "call(compute_total(price, qty))\nlist[compute_total(price, qty)]\n"
    "recompute_total(nothing)\nrecompute_total(nothing)\nrecompute_total(nothing)\n",
}

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runtime")
    exe = tmp / "merl-rt"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-pthread",
            "-o",
            str(exe),
            str(RUNTIME / "merl.c"),
            str(RUNTIME / "libsais.c"),
        ],
        check=True,
    )
    vault = tmp / "vault"
    for rel, text in NOTES.items():
        path = vault / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return exe, vault, Index(dataset.corpus(dataset.vault(vault)))


PROMPTS = [
    ("Every text tab asks one", " Vault API about its own content."),
    ("Ünïcode line: naïve caf", "é au lait"),
    ("nothing in the vault says this", ""),
    # Seen once, but 50 bytes of agreed context is past LONG_N.
    ("Every text tab asks one Vault API about nothing at", " all."),
    # Through any separator, and only the occurrences that start a word.
    ("y = compute_", "total(price, qty)"),
]


def indexed_server(exe, vault, **kwargs):
    """Wait for the first index before checking suggestion content."""
    server = Server([str(exe), "--vault", str(vault)], progress=True, **kwargs)
    server.drain(
        "$/progress",
        until=lambda messages: any(
            message["params"]["value"]["kind"] == "end" for message in messages
        ),
    )
    return server


def test_utf8_positions_match_reference(runtime):
    exe, vault, index = runtime
    server = indexed_server(exe, vault)
    assert server.initialize["capabilities"]["positionEncoding"] == "utf-8"
    for prefix, want in PROMPTS:
        got = server.suggest(prefix.encode())
        assert got == index.suggest(prefix.encode()) == want.encode(), prefix
    assert server.close() == 0


def test_utf16_positions_and_clamping(runtime):
    exe, vault, index = runtime
    server = indexed_server(exe, vault, encodings=("utf-16",))
    assert server.initialize["capabilities"]["positionEncoding"] == "utf-16"
    uri = "file:///t.md"
    line = "Ünïcode line: naïve caf"
    server.open(uri, "first\n" + line + "\nlast")
    assert server.inline(uri, 1, len(line)) == "é au lait"
    # Past the end of the line: the caret clamps to the line end.
    assert server.inline(uri, 1, 999) == "é au lait"
    # Past the end of the document: nothing to complete from.
    assert (
        server.inline(uri, 99, 0)
        == index.suggest(b"first\n" + line.encode() + b"\nlast").decode()
    )
    assert server.close() == 0


def test_does_not_echo_existing_line(runtime):
    exe, vault, _ = runtime
    server = indexed_server(exe, vault)
    assert (
        server.suggest(
            b"Every text tab asks one", b" Vault API about its own content.\n"
        )
        == b""
    )
    assert server.suggest(b"Every text tab asks one", b" Vault API\n") != b""
    assert server.close() == 0


def test_an_open_document_answers_for_itself(runtime):
    """An unsaved open document can suggest its own earlier text."""
    exe, vault, _ = runtime
    server = indexed_server(exe, vault)
    line = "The kettle in the kitchen was already"
    uri = "file:///draft.md"
    server.open(uri, line + " boiling.\n" + line)
    assert server.inline(uri, 1, len(line)) == " boiling."
    assert server.close() == 0


def test_the_caret_never_matches_itself(runtime):
    """The prefix at the caret cannot count as its own match."""
    exe, vault, _ = runtime
    server = Server([str(exe), "--vault", str(vault)])
    line = "The kettle was already"
    uri = "file:///alone.md"
    server.open(uri, line)
    assert server.inline(uri, 0, len(line)) == ""
    assert server.close() == 0


def test_first_build_does_not_hold_shutdown(runtime, tmp_path):
    """An inline request during the first index must not hold the LSP loop."""
    exe, _, _ = runtime
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "large.md").write_bytes(
        b"Every text tab asks one Vault API about its own content.\n" * 750_000
    )
    server = Server([str(exe), "--vault", str(vault)], progress=True)
    server.drain(
        "$/progress",
        until=lambda messages: any(
            message["params"]["value"].get("message", "").endswith("sorting")
            for message in messages
        ),
    )
    start = time.monotonic()
    assert server.inline("file:///draft.md", 0, 0) == ""
    elapsed = time.monotonic() - start
    assert server.close() == 0
    assert elapsed < 0.5, f"first-build inline request held LSP loop for {elapsed:.2f}s"


def test_the_longest_match_wins_over_the_vault(runtime):
    """A longer open-document match beats the vault's continuation."""
    exe, vault, _ = runtime
    server = indexed_server(exe, vault)
    tail = "Note: Every text tab asks one Vault API about"
    assert server.suggest(tail[len("Note: ") :].encode()) == b" its own content."
    uri = "file:///draft.md"
    server.open(uri, tail + " its own quiet corner.\n" + tail)
    assert server.inline(uri, 1, len(tail)) == " its own quiet corner."
    assert server.close() == 0


def test_protocol_edges(runtime):
    exe, vault, _ = runtime
    proc = subprocess.Popen(
        [str(exe), "--vault", str(vault)], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    server = Server.__new__(Server)
    server.proc, server.next_id, server.opened = proc, 0, False
    err = server.request("textDocument/inlineCompletion", {})["error"]
    assert err["code"] == -32002
    assert server.request("initialize", {"rootUri": None, "capabilities": {}})["result"]
    assert server.request("no/such", {})["error"]["code"] == -32601
    server.notify("exit", None)  # without shutdown: exit code 1
    assert proc.wait(timeout=10) == 1


def test_a_save_reindexes_without_stalling(runtime, tmp_path):
    """A save rebuilds the index while the old index keeps answering."""
    exe, _, _ = runtime
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text(NOTES["a.md"])
    server = indexed_server(exe, vault)

    phrase = b"The kettle was already"
    assert server.suggest(phrase) == b"", "not in the vault yet"
    # The old index still answers while the new one is being built.
    assert server.suggest(b"Every text tab asks one") != b""

    (vault / "b.md").write_text(
        "The kettle was already boiling.\nThe kettle was already boiling.\n"
    )
    server.notify(
        "textDocument/didSave",
        {"textDocument": {"uri": (vault / "b.md").as_uri()}},
    )
    for _ in range(100):
        if server.suggest(phrase) == b" boiling.":
            break
        time.sleep(0.05)
    else:
        pytest.fail("the save never reached the index")
    assert server.close() == 0


def test_git_decides_what_is_indexed(runtime, tmp_path):
    """Git excludes ignored notes; a plain vault includes their Markdown."""
    exe, _, _ = runtime
    vault = tmp_path / "repo"
    (vault / "build").mkdir(parents=True)
    (vault / "a.md").write_text(NOTES["a.md"])
    (vault / "build" / "vendored.md").write_text(
        "Nobody here wrote this line.\nNobody here wrote this line.\n"
    )
    (vault / ".gitignore").write_text("build/\n")
    subprocess.run(["git", "init", "-q", str(vault)], check=True)
    subprocess.run(["git", "-C", str(vault), "add", "-A"], check=True)

    assert dataset.vault(vault) == [("a.md", NOTES["a.md"].encode())], (
        "the reference walk agrees"
    )

    server = indexed_server(exe, vault)
    assert server.suggest(b"Every text tab asks one") != b"", "a tracked note answers"
    assert server.suggest(b"Nobody here wrote") == b"", "an ignored one does not"
    assert server.close() == 0

    shutil.rmtree(vault / ".git")
    plain = indexed_server(exe, vault)
    assert plain.suggest(b"Nobody here wrote") == b" this line.", "no repo, no rule"
    assert plain.close() == 0


def test_a_rebuild_is_announced_to_a_client_that_asked(runtime, tmp_path):
    """Subscribed clients receive rebuild progress; others stay silent.

    Progress begins, reports files and sorting, then ends.
    """
    exe, _, _ = runtime
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text(NOTES["a.md"])

    server = Server([str(exe), "--vault", str(vault)], progress=True)

    def ends(n):
        return lambda ms: [m["params"]["value"]["kind"] for m in ms].count("end") == n

    def progress():
        """Collect each build's progress kinds and final report message."""
        values = [m["params"]["value"] for m in server.drain("$/progress")]
        kinds = [k for k, _ in itertools.groupby(v["kind"] for v in values)]
        last = [
            values[i - 1].get("message")
            for i, v in enumerate(values)
            if v["kind"] == "end"
        ]
        return kinds, last

    server.drain("$/progress", until=ends(1))
    assert progress() == (["begin", "report", "end"], ["1 file, sorting"]), (
        "the first build"
    )
    assert any(m["method"] == "window/workDoneProgress/create" for m in server.notes), (
        "the token is registered before it is used"
    )

    # A save starts another run, and the old index answers through it.
    (vault / "b.md").write_text("Fresh lines appear here.\nFresh lines appear here.\n")
    server.notify(
        "textDocument/didSave", {"textDocument": {"uri": (vault / "b.md").as_uri()}}
    )
    assert server.suggest(b"Every text tab asks one") != b""
    server.drain("$/progress", until=ends(2))
    assert progress() == (
        ["begin", "report", "end"] * 2,
        ["1 file, sorting", "2 files, sorting"],
    ), "and the rebuild"
    assert server.close() == 0

    quiet = Server([str(exe), "--vault", str(vault)])
    for _ in range(100):
        if quiet.suggest(b"Every text tab asks one"):
            break
        time.sleep(0.01)
    else:
        pytest.fail("the quiet client's first index never completed")
    assert quiet.notes == [], "nothing for a client that did not ask"
    assert quiet.close() == 0
