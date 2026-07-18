#!/usr/bin/env python3
"""A dependency-free mock OpenAI-compatible chat-completions server.

Phase-2 live-integration harness for Hermes-Brain: this stands in for a real
LLM so a REAL hermes-agent + brain plugin can round-trip an agent turn and a
dream cycle fully offline, with no API key and no network.

It serves ``POST /v1/chat/completions`` and returns a *valid* OpenAI
chat-completion object (parseable by the ``openai`` Python SDK that hermes's
auxiliary client uses). It distinguishes three request shapes by inspecting the
system prompt:

  1. brain EXTRACTION calls (``brain.capture.extract._EXTRACT_SYSTEM`` —
     "You distill a conversation digest into durable memories ...") -> return a
     JSON array of extraction items shaped EXACTLY per that prompt's schema
     (content/kind/about_user/time_sensitive/instruction_shaped/source_uids/
     search_aids), so the brain actually writes memories. The item content is
     derived from the digest (staging-db fact) so recall can find it.

  2. any other JSON-expecting brain task (consolidate / distill / cases —
     the prompt asks for JSON) -> return an empty JSON array ``[]`` (a safe
     no-op: the strategy simply distills nothing this run).

  3. a normal chat turn -> a short deterministic assistant reply. The main
     agent turn requests ``stream:true``, so those are served as an SSE stream
     of ``chat.completion.chunk`` objects ending in ``data: [DONE]``; the
     brain's auxiliary calls are non-streaming JSON. (stream flag = the clean
     discriminator: aux calls never stream, the agent turn always does.)

Every response carries a realistic ``usage`` object (prompt/completion tokens
derived deterministically from the payload) so hermes's cost accounting (B2)
and the brain's llm_ledger metering exercise too. Responses are deterministic:
identical input -> identical output.

Stdlib only (http.server). Start it, poll ``GET /health`` until 200, then drive
the agent.
"""

from __future__ import annotations

import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The distinctive opening of brain.capture.extract._EXTRACT_SYSTEM. If the brain
# rewords its extraction prompt, widen this set — any one match flags extraction.
_EXTRACT_MARKERS = (
    "distill a conversation digest into durable memories",
    "search_aids",
    "instruction_shaped",
)

# Pull the source uid out of a digest line like: "[a1b2c3d4|owner|0.45] U: ..."
_DIGEST_UID_RE = re.compile(r"\[([0-9A-Za-z]{6,26})\|")


def _approx_tokens(text: str) -> int:
    # Mirror the brain's char/4 proxy so token counts look plausible.
    return max(1, len(text or "") // 4)


def _system_text(messages: list) -> str:
    return "\n".join(
        m.get("content", "") for m in (messages or [])
        if isinstance(m, dict) and m.get("role") == "system"
        and isinstance(m.get("content"), str)
    )


def _user_text(messages: list) -> str:
    return "\n".join(
        m.get("content", "") for m in (messages or [])
        if isinstance(m, dict) and m.get("role") == "user"
        and isinstance(m.get("content"), str)
    )


def _is_extraction(messages: list) -> bool:
    sys_txt = _system_text(messages).lower()
    return any(marker in sys_txt for marker in _EXTRACT_MARKERS)


def _wants_json(messages: list) -> bool:
    blob = (_system_text(messages) + "\n" + _user_text(messages)).lower()
    return "json" in blob


def _extraction_items(messages: list) -> list:
    """Build the extraction JSON the brain expects for the digest at hand.

    Deterministic + digest-aware: when the digest is the staging-database turn
    we return the crisp fact the test asserts on; otherwise we still emit ONE
    well-formed item echoing the salient user text, so the round-trip writes a
    memory for any driver prompt (graceful, never empty on a real turn).
    """
    digest = _user_text(messages)
    low = digest.lower()
    m = _DIGEST_UID_RE.search(digest)
    source_uid = m.group(1) if m else "session"

    if "staging" in low and ("postgres" in low or "fly.io" in low):
        return [{
            "content": "The user's staging database is PostgreSQL 14 running on Fly.io.",
            "kind": "fact",
            "about_user": True,
            "time_sensitive": False,
            "instruction_shaped": False,
            "source_uids": [source_uid],
            "search_aids": ["staging db", "staging postgres version",
                            "where is staging hosted"],
        }]

    # Generic fallback: distill the first user utterance from the digest into a
    # single fact so any prompt still produces a writable memory.
    snippet = ""
    for line in digest.splitlines():
        u = line.split("U:", 1)
        if len(u) == 2:
            snippet = u[1].split(" A:", 1)[0].strip()
            break
    snippet = (snippet or digest.strip())[:280]
    if len(snippet) < 10:
        return []  # nothing worth remembering (valid empty answer)
    return [{
        "content": f"User noted: {snippet}"[:400],
        "kind": "fact",
        "about_user": True,
        "time_sensitive": False,
        "instruction_shaped": False,
        "source_uids": [source_uid],
        "search_aids": ["what did the user say", "user note"],
    }]


def _chat_reply(messages: list) -> str:
    """Plain-text assistant reply for a normal (streaming) chat turn. Never a
    tool call — a clean text answer makes the agent turn finalize cleanly and
    fire the memory ``sync_turn`` hook. Acknowledges a 'remember ...' request so
    the downstream extraction has salient content to distill."""
    user = _user_text(messages).strip().lower()
    if "staging" in user:
        return ("Noted — your staging database is PostgreSQL 14 on Fly.io. "
                "I'll remember that.")
    return "Understood. I've noted that."


def _reply_content(messages: list) -> str:
    """Assistant text for a NON-streaming request. Non-streaming requests in
    this harness are the brain's auxiliary calls (extraction / consolidate),
    which expect JSON; the main agent turn always streams (handled separately).
    """
    if _is_extraction(messages):
        return json.dumps(_extraction_items(messages))
    if _wants_json(messages):
        # consolidate / distill / cases and friends: a safe, valid empty result.
        return "[]"
    return _chat_reply(messages)


def _completion(model: str, messages: list) -> dict:
    content = _reply_content(messages)
    prompt_tokens = sum(_approx_tokens(m.get("content", "")) for m in messages
                        if isinstance(m, dict) and isinstance(m.get("content"), str))
    prompt_tokens = max(prompt_tokens, 1)
    completion_tokens = _approx_tokens(content)
    return {
        "id": "chatcmpl-mock-0000000000",
        "object": "chat.completion",
        "created": 1700000000,  # fixed for determinism
        "model": model or "mock-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") in ("/health", "/healthz"):
            self._send(200, {"status": "ok"})
            return
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": "mock-main", "object": "model", "owned_by": "mock"},
                {"id": "mock-extract", "object": "model", "owned_by": "mock"},
                {"id": "mock-consolidate", "object": "model", "owned_by": "mock"},
            ]})
            return
        self._send(404, {"error": {"message": f"no route {self.path}"}})

    def _send_stream(self, model: str, messages: list) -> None:
        """SSE stream of OpenAI chat.completion.chunk objects ending in
        `data: [DONE]` — the shape the main agent turn consumes (it calls
        chat.completions.create(stream=True, stream_options={include_usage})).
        """
        content = _chat_reply(messages)
        prompt_tokens = sum(_approx_tokens(m.get("content", "")) for m in messages
                            if isinstance(m, dict) and isinstance(m.get("content"), str))
        prompt_tokens = max(prompt_tokens, 1)
        completion_tokens = _approx_tokens(content)
        base = {"id": "chatcmpl-mock-0000000000", "object": "chat.completion.chunk",
                "created": 1700000000, "model": model or "mock-model"}

        def chunk(delta, finish=None, choices=None, usage=None):
            obj = dict(base)
            if choices is not None:
                obj["choices"] = choices
            else:
                obj["choices"] = [{"index": 0, "delta": delta,
                                   "finish_reason": finish}]
            if usage is not None:
                obj["usage"] = usage
            return "data: " + json.dumps(obj) + "\n\n"

        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        parts = [
            chunk({"role": "assistant", "content": ""}),
            chunk({"content": content}),
            chunk({}, finish="stop"),
            # final usage-only chunk (stream_options.include_usage=True)
            chunk(None, choices=[], usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens}),
            "data: [DONE]\n\n",
        ]
        for p in parts:
            self.wfile.write(p.encode("utf-8"))
            self.wfile.flush()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": {"message": "invalid JSON body"}})
            return
        path = self.path.rstrip("/")
        if path.endswith("/chat/completions"):
            model = str(req.get("model") or "mock-model")
            messages = req.get("messages") or []
            if req.get("stream"):
                self._send_stream(model, messages)
            else:
                self._send(200, _completion(model, messages))
            return
        self._send(404, {"error": {"message": f"no route {self.path}"}})

    def log_message(self, fmt, *args):  # keep the container log readable
        sys.stderr.write("[mock_llm] " + (fmt % args) + "\n")


def main(argv: list[str]) -> int:
    host = "127.0.0.1"
    port = 8080
    for i, a in enumerate(argv):
        if a in ("--port", "-p") and i + 1 < len(argv):
            port = int(argv[i + 1])
        elif a in ("--host", "-h") and i + 1 < len(argv):
            host = argv[i + 1]
    server = ThreadingHTTPServer((host, port), Handler)
    sys.stderr.write(f"[mock_llm] serving OpenAI-compatible API on "
                     f"http://{host}:{port}/v1 (started {time.time():.0f})\n")
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
