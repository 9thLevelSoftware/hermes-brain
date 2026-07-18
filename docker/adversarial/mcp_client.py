#!/usr/bin/env python3
"""A tiny stdlib newline-delimited JSON-RPC client for the brain's MCP server.

The brain's ``mcp_server.serve()`` speaks MCP stdio framing — one JSON object
per line on stdin, one response line on stdout (mcp_server.py:176-225). This
client spawns that server as a subprocess and round-trips frames, INCLUDING the
malformed ones the adversarial suite needs (a bare int, an empty batch, a
non-object ``arguments``) to prove a single bad line yields a ``-32600`` /
``-32700`` error and never crashes the serve loop.

Hand-rolled (no ``mcp`` SDK) so it runs in the floor image and standalone
pytest with zero extra deps. Two entry points:

  * ``MCPClient`` — programmatic use (pytest / the live phase driver).
  * ``python mcp_client.py --self-check`` — a runnable assertion sweep used as
    a Docker build/run proof.

The server command defaults to ``hermes brain mcp`` (real install); pass
``--python`` to instead launch ``python -c "from brain.mcp_server import serve;
serve(HERMES_HOME)"`` which needs only the brain importable (no hermes-agent),
so the same client drives both the standalone and live phases.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time


class MCPClient:
    def __init__(self, cmd: list[str], env: dict | None = None):
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
            env={**os.environ, **(env or {})},
        )

    # -- raw line I/O (lets us send deliberately malformed frames) ----------
    def send_raw(self, line: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(line if line.endswith("\n") else line + "\n")
        self._proc.stdin.flush()

    def read_response(self, timeout: float = 10.0) -> dict | None:
        """Read one response line as a dict. Returns None on EOF/timeout.
        (The server writes exactly one line per non-notification request.)"""
        assert self._proc.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if line == "":
                return None  # EOF — server exited
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
        return None

    # -- structured requests ------------------------------------------------
    def request(self, method: str, params: dict | None = None, msg_id=1) -> dict | None:
        req = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            req["params"] = params
        self.send_raw(json.dumps(req))
        return self.read_response()

    def notify(self, method: str, params: dict | None = None) -> None:
        req = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            req["params"] = params
        self.send_raw(json.dumps(req))  # no response expected

    def initialize(self) -> dict | None:
        return self.request("initialize", {"protocolVersion": "2024-11-05",
                                            "capabilities": {},
                                            "clientInfo": {"name": "adv-client",
                                                           "version": "0"}})

    def tools_list(self, msg_id=2) -> dict | None:
        return self.request("tools/list", msg_id=msg_id)

    def call_tool(self, name: str, arguments: dict, msg_id=3) -> dict | None:
        return self.request("tools/call", {"name": name, "arguments": arguments},
                            msg_id=msg_id)

    # -- teardown -----------------------------------------------------------
    def close(self) -> tuple[str, int]:
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)
        stderr = ""
        if self._proc.stderr:
            stderr = self._proc.stderr.read()
        return stderr, self._proc.returncode

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def server_cmd(hermes_home: str, *, use_python: bool = False) -> list[str]:
    if use_python:
        return [sys.executable, "-c",
                "import sys; from brain.mcp_server import serve; "
                f"sys.exit(serve({hermes_home!r}))"]
    return ["hermes", "brain", "mcp"]


def self_check(hermes_home: str, use_python: bool) -> int:
    """Adversarial sweep proving the serve loop survives malformed frames and
    serves reads at 'tool' trust. Exit 0 == all invariants held."""
    cmd = server_cmd(hermes_home, use_python=use_python)
    fails: list[str] = []
    with MCPClient(cmd, env={"HERMES_HOME": hermes_home}) as c:
        init = c.initialize()
        if not (init and init.get("result", {}).get("serverInfo", {}).get("name") == "hermes-brain"):
            fails.append(f"initialize did not return the brain serverInfo: {init!r}")

        # 1. a bare int is valid JSON but not a JSON-RPC object -> -32600
        c.send_raw("12345")
        r = c.read_response()
        if not (r and r.get("error", {}).get("code") == -32600):
            fails.append(f"bare int should be -32600, got {r!r}")

        # 2. an empty batch -> -32600, loop survives
        c.send_raw("[]")
        r = c.read_response()
        if not (r and r.get("error", {}).get("code") == -32600):
            fails.append(f"empty batch should be -32600, got {r!r}")

        # 3. unparseable line -> -32700, loop survives
        c.send_raw("{not json")
        r = c.read_response()
        if not (r and r.get("error", {}).get("code") == -32700):
            fails.append(f"garbage line should be -32700, got {r!r}")

        # 4. the loop is STILL alive: tools/list returns the four brain tools
        tl = c.tools_list(msg_id=99)
        names = [t.get("name") for t in (tl or {}).get("result", {}).get("tools", [])]
        if "brain_recall" not in names:
            fails.append(f"tools/list did not survive malformed frames: {tl!r}")

        # 5. a read works (empty query is fine — just proves the path)
        rc = c.call_tool("brain_recall", {"query": "staging database"}, msg_id=100)
        if not (rc and "result" in rc):
            fails.append(f"brain_recall did not return a result: {rc!r}")

    if fails:
        for f in fails:
            print("MCP SELF-CHECK FAIL:", f, file=sys.stderr)
        return 1
    print("MCP SELF-CHECK OK: serve loop survived malformed frames; reads work at tool trust")
    return 0


def main(argv: list[str]) -> int:
    use_python = "--python" in argv
    home = os.environ.get("HERMES_HOME") or "/hermes-home"
    for i, a in enumerate(argv):
        if a == "--home" and i + 1 < len(argv):
            home = argv[i + 1]
    if "--self-check" in argv:
        return self_check(home, use_python)
    print("usage: mcp_client.py --self-check [--python] [--home DIR]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
