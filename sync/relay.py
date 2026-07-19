"""Zero-knowledge sync relay + client (Phase G, multi-device encrypted sync).

The relay is a DUMB, ZERO-KNOWLEDGE store. It only ever sees OPAQUE ENCRYPTED
BLOBS — the ciphertext produced by ``sync/crypto.py`` on the client side. This
module NEVER decrypts, inspects, or even base64-decodes a blob; it treats each
one as an opaque string, stores it verbatim keyed by ``namespace`` + a monotonic
server sequence, and hands it back byte-for-byte on pull. It has NO dependency on
``cryptography`` — pure stdlib only (``http.server``, ``json``, ``threading``,
``base64`` is not even needed because we never decode).

Wire protocol (JSON over HTTP):

- ``POST /push``  body ``{"namespace": str, "blobs": [<opaque str>, ...]}``
  -> ``{"cursor": <int>}``  (new high-water server sequence for the namespace)
- ``GET  /pull?namespace=...&since=<int>&limit=<int>``
  -> ``{"blobs": [...], "cursor": <int>}``  (blobs strictly after ``since``)
- ``GET  /status?namespace=...``
  -> ``{"namespace": str, "high_water": <int>}``

Namespaces isolate devices/accounts: a pull on one namespace can NEVER surface
another namespace's blobs. Malformed / oversized / unknown requests return a JSON
error body with an appropriate HTTP status; the server never crashes on bad input.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

# Guardrails so a single hostile request can neither exhaust memory nor smuggle
# a non-opaque payload past the relay. These are size limits, not inspection.
MAX_BODY_BYTES = 16 * 1024 * 1024  # 16 MiB per push request
MAX_BLOBS_PER_PUSH = 10_000
MAX_BLOB_CHARS = 4 * 1024 * 1024  # 4 MiB per individual opaque blob
DEFAULT_LIMIT = 500
MAX_LIMIT = 5_000


class RelayStore:
    """In-process ciphertext store (the server's backing).

    Append-only log of opaque blobs per namespace. Each appended blob gets a
    monotonic, 1-based server sequence used as the paging cursor. Blobs are
    stored VERBATIM — the store never interprets, decodes, or validates their
    content beyond a size cap. Thread-safe via a single lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # namespace -> list of blob strings; index i holds sequence (i + 1).
        self._logs: dict[str, list[str]] = {}

    def append(self, namespace: str, blobs: list) -> int:
        """Append ``blobs`` to ``namespace``; return the new high-water seq."""
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace must be a non-empty string")
        if not isinstance(blobs, list):
            raise ValueError("blobs must be a list")
        if len(blobs) > MAX_BLOBS_PER_PUSH:
            raise ValueError("too many blobs in one push")
        for blob in blobs:
            # We accept only strings: the opaque ciphertext envelope. We do NOT
            # decode or parse it — we only bound its length.
            if not isinstance(blob, str):
                raise ValueError("each blob must be an opaque string")
            if len(blob) > MAX_BLOB_CHARS:
                raise ValueError("blob exceeds maximum size")
        with self._lock:
            log = self._logs.setdefault(namespace, [])
            log.extend(blobs)
            return len(log)

    def since(self, namespace: str, cursor: int, limit: int = DEFAULT_LIMIT) -> tuple[list, int]:
        """Return ``(blobs, next_cursor)`` for blobs with seq > ``cursor``.

        ``cursor`` is the last sequence the caller already has (0 = from start).
        ``next_cursor`` is the sequence of the last returned blob (== ``cursor``
        when nothing new). Never returns overlapping or skipped blobs.
        """
        if cursor < 0:
            cursor = 0
        if limit <= 0:
            limit = DEFAULT_LIMIT
        limit = min(limit, MAX_LIMIT)
        with self._lock:
            log = self._logs.get(namespace, [])
            high_water = len(log)
            if cursor >= high_water:
                return [], high_water
            end = min(cursor + limit, high_water)
            # log index (cursor) holds sequence (cursor + 1), so slice [cursor:end].
            blobs = list(log[cursor:end])
            return blobs, end

    def high_water(self, namespace: str) -> int:
        """Current high-water sequence for ``namespace`` (0 if unknown)."""
        with self._lock:
            return len(self._logs.get(namespace, []))


class _RelayError(Exception):
    """Internal: raised to produce a clean JSON HTTP error."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def make_handler(store: RelayStore):
    """Build a ``BaseHTTPRequestHandler`` subclass bound to ``store``."""

    class _RelayHandler(BaseHTTPRequestHandler):
        server_version = "HermesRelay/1.0"
        protocol_version = "HTTP/1.1"

        # -- helpers --------------------------------------------------------
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._send_json(status, {"error": message})

        def _query(self) -> dict:
            parsed = urllib_parse.urlsplit(self.path)
            raw = urllib_parse.parse_qs(parsed.query, keep_blank_values=True)
            return {k: v[0] for k, v in raw.items()}

        def _path(self) -> str:
            return urllib_parse.urlsplit(self.path).path.rstrip("/") or "/"

        # -- routing --------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            try:
                path = self._path()
                if path == "/pull":
                    self._handle_pull()
                elif path == "/status":
                    self._handle_status()
                else:
                    self._error(404, "unknown endpoint")
            except _RelayError as exc:
                self._error(exc.status, exc.message)
            except Exception as exc:  # never crash the server on bad input
                self._error(500, f"internal error: {exc.__class__.__name__}")

        def do_POST(self) -> None:  # noqa: N802
            try:
                path = self._path()
                if path == "/push":
                    self._handle_push()
                else:
                    self._error(404, "unknown endpoint")
            except _RelayError as exc:
                self._error(exc.status, exc.message)
            except Exception as exc:
                self._error(500, f"internal error: {exc.__class__.__name__}")

        # -- endpoints ------------------------------------------------------
        def _read_json_body(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                raise _RelayError(411, "missing or invalid Content-Length") from None
            if length < 0:
                raise _RelayError(400, "invalid Content-Length")
            if length > MAX_BODY_BYTES:
                raise _RelayError(413, "request body too large")
            raw = self.rfile.read(length) if length else b""
            try:
                data = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise _RelayError(400, "malformed JSON body") from None
            if not isinstance(data, dict):
                raise _RelayError(400, "body must be a JSON object")
            return data

        def _handle_push(self) -> None:
            data = self._read_json_body()
            namespace = data.get("namespace")
            blobs = data.get("blobs", [])
            if not isinstance(namespace, str) or not namespace:
                raise _RelayError(400, "namespace must be a non-empty string")
            if not isinstance(blobs, list):
                raise _RelayError(400, "blobs must be a list")
            try:
                cursor = store.append(namespace, blobs)
            except ValueError as exc:
                raise _RelayError(400, str(exc)) from None
            self._send_json(200, {"cursor": cursor})

        def _handle_pull(self) -> None:
            q = self._query()
            namespace = q.get("namespace")
            if not namespace:
                raise _RelayError(400, "namespace query param required")
            since = _parse_int(q.get("since"), default=0, field="since")
            limit = _parse_int(q.get("limit"), default=DEFAULT_LIMIT, field="limit")
            blobs, cursor = store.since(namespace, since, limit)
            self._send_json(200, {"blobs": blobs, "cursor": cursor})

        def _handle_status(self) -> None:
            q = self._query()
            namespace = q.get("namespace")
            if not namespace:
                raise _RelayError(400, "namespace query param required")
            self._send_json(200, {
                "namespace": namespace,
                "high_water": store.high_water(namespace),
            })

        # Silence the default stderr request logging (keeps test output clean).
        def log_message(self, fmt: str, *args) -> None:  # noqa: A002
            return

    return _RelayHandler


def _parse_int(value, *, default: int, field: str) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise _RelayError(400, f"{field} must be an integer") from None


def serve(host: str, port: int, *, store: RelayStore | None = None):
    """Run a blocking threaded relay server on ``host:port``.

    Returns the ``ThreadingHTTPServer`` (after ``serve_forever`` — which blocks;
    callers wanting a non-blocking server construct it via ``make_relay_server``).
    """
    httpd = make_relay_server(host, port, store=store)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
    return httpd


def make_relay_server(host: str, port: int, *, store: RelayStore | None = None) -> ThreadingHTTPServer:
    """Construct (but do not start) a ``ThreadingHTTPServer`` for the relay.

    Handy for tests: bind ``port=0`` for an ephemeral port, read
    ``server.server_address[1]``, then drive ``serve_forever`` on a thread.
    """
    if store is None:
        store = RelayStore()
    handler = make_handler(store)
    httpd = ThreadingHTTPServer((host, port), handler)
    # Stash the store so tests / callers can introspect the backing directly.
    httpd.relay_store = store  # type: ignore[attr-defined]
    return httpd


class RelayClient:
    """Client the sync engine uses. Talks HTTP to a relay base URL via stdlib
    ``urllib`` only (no third-party HTTP). All payloads are opaque strings;
    the client neither encrypts nor decrypts — that is ``sync/crypto.py``."""

    def __init__(self, base_url: str, *, namespace: str, timeout: float = 10.0) -> None:
        if not namespace:
            raise ValueError("namespace is required")
        self.base_url = base_url.rstrip("/")
        self.namespace = namespace
        self.timeout = timeout

    # -- HTTP plumbing -----------------------------------------------------
    def _request(self, method: str, path: str, *, params: dict | None = None,
                 body: dict | None = None) -> dict:
        url = self.base_url + path
        if params:
            url = url + "?" + urllib_parse.urlencode(params)
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib_request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib_request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib_error.HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {"error": f"HTTP {exc.code}"}
            raise RelayError(exc.code, payload.get("error", f"HTTP {exc.code}")) from None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RelayError(502, "relay returned malformed JSON") from None

    # -- API ---------------------------------------------------------------
    def push(self, blobs: list) -> int:
        """POST opaque ``blobs`` to the relay; return the new server cursor."""
        resp = self._request(
            "POST", "/push",
            body={"namespace": self.namespace, "blobs": list(blobs)},
        )
        return int(resp["cursor"])

    def pull(self, cursor: int, *, limit: int = DEFAULT_LIMIT) -> tuple[list, int]:
        """GET blobs with seq > ``cursor``; return ``(blobs, next_cursor)``."""
        resp = self._request(
            "GET", "/pull",
            params={"namespace": self.namespace, "since": cursor, "limit": limit},
        )
        return resp["blobs"], int(resp["cursor"])

    def status(self) -> dict:
        """GET the namespace status (``{"namespace", "high_water"}``)."""
        return self._request("GET", "/status", params={"namespace": self.namespace})


class RelayError(Exception):
    """Raised by ``RelayClient`` when the relay returns an HTTP error."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"[{status}] {message}")
        self.status = status
        self.message = message


if __name__ == "__main__":  # pragma: no cover - manual smoke entrypoint
    import argparse

    parser = argparse.ArgumentParser(description="Hermes zero-knowledge sync relay")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    print(f"relay listening on {args.host}:{args.port}")  # noqa: T201
    serve(args.host, args.port)
