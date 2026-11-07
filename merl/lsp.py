"""Minimal Python LSP client for testing and measuring `merl-rt`."""

import json
import subprocess
import time
from typing import Any

URI = "file:///merl/eval.md"


class Server:
    def __init__(
        self,
        argv: list[str],
        encodings: tuple[str, ...] = ("utf-8",),
        progress: bool = False,
    ):
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        self.next_id = 0
        self.notes: list[dict] = []
        caps: dict = {"general": {"positionEncodings": list(encodings)}}
        if progress:
            caps["window"] = {"workDoneProgress": True}
        self.initialize = self.request("initialize", {"capabilities": caps})["result"]
        self.notify("initialized", {})
        self.opened = False

    def _send(self, msg: dict) -> None:
        body = json.dumps(msg).encode()
        assert self.proc.stdin is not None
        self.proc.stdin.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
        self.proc.stdin.flush()

    def _recv(self) -> dict:
        assert self.proc.stdout is not None
        length = -1
        while True:
            line = self.proc.stdout.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode().partition(":")
            if name.strip().lower() == "content-length":
                length = int(value)
        assert length >= 0, "server closed the pipe"
        return json.loads(self.proc.stdout.read(length))

    def notify(self, method: str, params: Any) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: Any) -> dict:
        self.next_id += 1
        self._send(
            {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params}
        )
        while True:
            msg = self._recv()
            if msg.get("id") == self.next_id:
                return msg
            self._aside(msg)

    def _aside(self, msg: dict) -> None:
        """Store server messages and acknowledge server requests."""
        if "method" not in msg:
            return
        self.notes.append(msg)
        if "id" in msg:
            self._send({"jsonrpc": "2.0", "id": msg["id"], "result": None})

    def drain(self, method: str, until: Any = None, timeout: float = 10.0) -> list:
        """Return messages of `method`, reading until `until` or the deadline."""
        deadline = time.monotonic() + timeout
        while until is not None and not until(
            [m for m in self.notes if m["method"] == method]
        ):
            if time.monotonic() > deadline:
                raise AssertionError(f"no {method} in {timeout}s: {self.notes}")
            self._aside(self._recv())
        return [m for m in self.notes if m["method"] == method]

    def open(self, uri: str, text: str) -> None:
        self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "markdown",
                    "version": 1,
                    "text": text,
                }
            },
        )

    def change(self, uri: str, text: str) -> None:
        self.notify(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": 2},
                "contentChanges": [{"text": text}],
            },
        )

    def inline(self, uri: str, line: int, character: int) -> str:
        result = self.request(
            "textDocument/inlineCompletion",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
                "context": {"triggerKind": 2},
            },
        )["result"]
        return "".join(item["insertText"] for item in result["items"])

    def suggest(self, prefix: bytes, after: bytes = b"") -> bytes:
        """Suggest at the byte-positioned caret between `prefix` and `after`."""
        text = (prefix + after).decode("utf-8", errors="replace")
        (self.change if self.opened else self.open)(URI, text)
        self.opened = True
        head = prefix.rsplit(b"\n", 1)[-1]
        return self.inline(URI, prefix.count(b"\n"), len(head)).encode()

    def close(self) -> int:
        self.request("shutdown", None)
        self.notify("exit", None)
        return self.proc.wait(timeout=10)
