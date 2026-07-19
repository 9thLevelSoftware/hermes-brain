"""stdio MCP server — the brain's surface for external agents (Claude Code).

This is the "cross-platform money shot" (plan P5): a memory the owner wrote
from Telegram is recalled here by a different agent entirely. Invariants
(critique item 35, pinned here as design, not accident):

  * stdio ONLY — no network listener, ever. A future SSE transport would need
    an auth design first.
  * Same capability probe as everything else (store/db.connect) — degrades,
    never crashes, on a Python without FTS5/sqlite-vec.
  * NEVER runs consolidation or any dream strategy. Sleep-time work belongs to
    the dream process alone; the MCP server only reads and does capped writes.
  * External trust: the connecting agent speaks at the 'tool' tier. Reads see
    the owner's global + owner-scoped memories (that IS the money shot — the
    server runs locally, owner-launched, on the owner's behalf); writes are
    tool-tier — capped, scoped, instruction-shaped content quarantined, and
    never lane-1 eligible.

Protocol: newline-delimited JSON-RPC 2.0 over stdin/stdout (the MCP stdio
framing). Hand-rolled, stdlib-only, mirroring hermes-agent's own mcp_serve.py
(which imports no MCP SDK either). Module level stays import-light — the
plugin loader eagerly imports every root *.py, so brain siblings load lazily
inside the handlers.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "hermes-brain"


# ---------------------------------------------------------------------------
# tool schema conversion (OpenAI function-calling -> MCP inputSchema)
# ---------------------------------------------------------------------------

def _mcp_tools() -> list[dict]:
    from . import tools

    # The MCP surface is at 'tool' trust and is a primary ask surface — expose
    # brain_ask here (read-only, cited) alongside the four core tools.
    schemas = [*tools.get_schemas(), tools.ask_schema(), tools.context_schema()]
    out = []
    for schema in schemas:
        fn = schema["function"]
        out.append({
            "name": fn["name"],
            "description": fn["description"],
            "inputSchema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

class BrainMCPServer:
    """One brain.db connection, one stdio loop. Single-threaded by design:
    MCP stdio is a serial request/response channel, so a single connection
    with short transactions is correct and simplest."""

    def __init__(self, hermes_home: str) -> None:
        self._hermes_home = hermes_home
        self._conn = None
        self._embedder = None
        self._initialized = False

    # -- lifecycle ----------------------------------------------------------

    def _open(self) -> None:
        from . import config as brain_config
        from .store import db, sysinfo

        self._conn = db.connect(self._hermes_home)
        cfg = brain_config.load_config(self._hermes_home)
        self._config = cfg
        # Embedder is best-effort and never downloads (that is setup's job).
        try:
            from .recall.embed import get_embedder

            mode = sysinfo.resolve_mode(str(cfg.get("mode", "auto")))
            self._embedder = get_embedder(cfg, mode, allow_download=False)
        except Exception:
            logger.warning("MCP: embedder unavailable; FTS-only", exc_info=True)

    def _tool_context(self):
        from .tools import ToolContext

        # trust='tool' + principal='owner': reads see the owner's global +
        # owner-scoped memories (the money shot); writes are capped to 'tool',
        # scoped to the owner, and instruction-shaped content is quarantined.
        return ToolContext(
            session_id="mcp", principal_id="owner", trust_tier="tool",
            source_author="mcp", platform="mcp", embedder=self._embedder,
            config=self._config, hermes_home=self._hermes_home)

    # -- JSON-RPC dispatch --------------------------------------------------

    def handle(self, msg: dict) -> dict | None:
        """Return a JSON-RPC response dict, or None for a notification.

        Hardened against the untrusted external channel: valid JSON that is
        not a JSON-RPC object (a bare int/string/bool/null, or a list member
        that isn't an object) yields a -32600 error, never an exception —
        the serve loop must not be crashable by a single malformed line."""
        if not isinstance(msg, dict):
            return _error(None, -32600, "invalid request: not a JSON-RPC object")
        method = msg.get("method")
        msg_id = msg.get("id")
        is_notification = "id" not in msg

        try:
            if method == "initialize":
                result = self._on_initialize(msg.get("params") or {})
            elif method == "notifications/initialized":
                return None
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": _mcp_tools()}
            elif method == "tools/call":
                result = self._on_tools_call(msg.get("params") or {})
            elif is_notification:
                return None
            else:
                return _error(msg_id, -32601, f"method not found: {method}")
        except Exception as e:
            logger.warning("MCP: handler for %s failed: %s", method, e, exc_info=True)
            if is_notification:
                return None
            return _error(msg_id, -32603, f"internal error: {e}")

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _on_initialize(self, params: dict) -> dict:
        if self._conn is None:
            self._open()
        self._initialized = True
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": SERVER_NAME, "version": _version()},
            "capabilities": {"tools": {}},
            "instructions": (
                "hermes-brain: the owner's global long-term memory. "
                "brain_recall to search past memories and conversations across "
                "every platform; brain_remember to save a durable fact "
                "(external writes are held for review). This memory is shared "
                "with the owner's other agents."),
        }

    def _on_tools_call(self, params: dict) -> dict:
        from . import tools

        name = params.get("name")
        args = params.get("arguments") or {}
        if self._conn is None:
            self._open()
        payload = tools.dispatch(self._conn, name, args, ctx=self._tool_context())
        # MCP tool results are a content array. The brain returns a JSON
        # string; surface it as text and flag errors via isError.
        is_error = False
        try:
            is_error = "error" in json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            pass
        return {"content": [{"type": "text", "text": payload}], "isError": is_error}

    # -- the loop -----------------------------------------------------------

    def serve(self, stdin=None, stdout=None) -> int:
        """Read newline-delimited JSON-RPC from stdin, reply on stdout.
        Returns an exit code. Blank lines and parse errors are tolerated."""
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        logger.info("hermes-brain MCP server: ready on stdio")
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                _write(stdout, _error(None, -32700, "parse error"))
                continue
            try:
                if isinstance(msg, list):          # batch
                    if not msg:
                        _write(stdout, _error(None, -32600, "empty batch"))
                        continue
                    for sub in msg:
                        resp = self.handle(sub)     # handle() tolerates non-dicts
                        if resp is not None:
                            _write(stdout, resp)
                    continue
                resp = self.handle(msg)
                if resp is not None:
                    _write(stdout, resp)
            except Exception as e:  # last-resort backstop: never let one line kill the server
                logger.warning("MCP: dispatch crashed: %s", e, exc_info=True)
                _write(stdout, _error(None, -32603, f"internal error: {e}"))
        self.close()
        return 0

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write(stdout, obj: dict) -> None:
    stdout.write(json.dumps(obj) + "\n")
    stdout.flush()


def _error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:
        return "0"


def serve(hermes_home: str) -> int:
    """Entry point for `hermes brain mcp` and `python -m brain.mcp_server`."""
    return BrainMCPServer(hermes_home).serve()


if __name__ == "__main__":  # pragma: no cover
    import os

    logging.basicConfig(level=logging.WARNING)
    home = os.environ.get("HERMES_HOME") or str(
        __import__("pathlib").Path.home() / ".hermes")
    sys.exit(serve(home))
