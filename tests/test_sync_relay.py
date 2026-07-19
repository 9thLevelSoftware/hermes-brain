"""Tests for the zero-knowledge sync relay (``sync/relay.py``).

Spins up a REAL relay on an ephemeral port (127.0.0.1:0) in a background thread
and drives it through the stdlib ``RelayClient``. The load-bearing property under
test is that the relay is a dumb, zero-knowledge store: it hands opaque blobs back
byte-for-byte and never interprets them, isolates namespaces, pages by cursor
without overlap/gap, and survives malformed input.
"""

from __future__ import annotations

import threading

import pytest
from brain.sync.relay import (
    RelayClient,
    RelayError,
    RelayStore,
    make_handler,
    make_relay_server,
    serve,
)


@pytest.fixture
def relay():
    """A running relay on an ephemeral port; yields its base URL + store."""
    server = make_relay_server("127.0.0.1", 0)
    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield base_url, server.relay_store
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_push_pull_roundtrip(relay):
    base_url, _ = relay
    client = RelayClient(base_url, namespace="dev-A")
    blobs = [" b64blob-one==", "b64blob-two==", "b64blob-three=="]
    cursor = client.push(blobs)
    assert cursor == 3
    got, next_cursor = client.pull(0)
    assert got == blobs
    assert next_cursor == 3


def test_cursor_paging_no_overlap_or_gap(relay):
    base_url, _ = relay
    client = RelayClient(base_url, namespace="dev-A")
    first = ["a1", "a2"]
    second = ["a3", "a4", "a5"]
    c1 = client.push(first)
    assert c1 == 2
    c2 = client.push(second)
    assert c2 == 5

    # Pull from the first cursor: only the NEWER batch, no overlap.
    newer, cursor = client.pull(c1)
    assert newer == second
    assert cursor == 5

    # Pull from the tip: nothing new, cursor unchanged.
    tail, tail_cursor = client.pull(cursor)
    assert tail == []
    assert tail_cursor == 5

    # Pull from the very start: the full ordered log, exactly once each.
    everything, all_cursor = client.pull(0)
    assert everything == first + second
    assert all_cursor == 5


def test_paging_limit(relay):
    base_url, _ = relay
    client = RelayClient(base_url, namespace="dev-A")
    client.push([f"blob-{i}" for i in range(10)])
    page1, c1 = client.pull(0, limit=4)
    assert page1 == ["blob-0", "blob-1", "blob-2", "blob-3"]
    assert c1 == 4
    page2, c2 = client.pull(c1, limit=4)
    assert page2 == ["blob-4", "blob-5", "blob-6", "blob-7"]
    assert c2 == 8
    page3, c3 = client.pull(c2, limit=4)
    assert page3 == ["blob-8", "blob-9"]
    assert c3 == 10


def test_namespace_isolation(relay):
    base_url, _ = relay
    client_a = RelayClient(base_url, namespace="account-A")
    client_b = RelayClient(base_url, namespace="account-B")
    client_a.push(["secret-A-1", "secret-A-2"])
    client_b.push(["secret-B-1"])

    got_b, _ = client_b.pull(0)
    assert got_b == ["secret-B-1"]
    assert "secret-A-1" not in got_b
    assert "secret-A-2" not in got_b

    got_a, _ = client_a.pull(0)
    assert got_a == ["secret-A-1", "secret-A-2"]

    assert client_a.status() == {"namespace": "account-A", "high_water": 2}
    assert client_b.status() == {"namespace": "account-B", "high_water": 1}


def test_verbatim_opaque_storage(relay):
    """The relay stores and returns blobs byte-for-byte, proving no interpretation.

    We push a payload that is deliberately NOT valid base64 and NOT valid JSON —
    if the relay tried to decode or parse it, this would fail or mangle. It must
    come back exactly.
    """
    base_url, store = relay
    client = RelayClient(base_url, namespace="dev-A")
    opaque = "!!!not-base64-not-json::\x01\x02 ☠ padding====??"
    client.push([opaque])
    got, _ = client.pull(0)
    assert got == [opaque]
    assert got[0] == opaque
    # The backing store holds the identical string, untouched.
    stored, _ = store.since("dev-A", 0)
    assert stored == [opaque]


def test_malformed_request_returns_json_error_and_server_stays_up(relay):
    base_url, _ = relay
    import json
    import urllib.error
    import urllib.request

    # Malformed JSON body to /push.
    req = urllib.request.Request(
        base_url + "/push",
        data=b"{not valid json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 400
    payload = json.loads(exc_info.value.read().decode("utf-8"))
    assert "error" in payload

    # Missing namespace on /pull -> 400 at the HTTP layer.
    req2 = urllib.request.Request(base_url + "/pull?since=0", method="GET")
    with pytest.raises(urllib.error.HTTPError) as exc2:
        urllib.request.urlopen(req2, timeout=5)
    assert exc2.value.code == 400

    # Unknown endpoint -> 404 JSON error.
    req3 = urllib.request.Request(base_url + "/nope", method="GET")
    with pytest.raises(urllib.error.HTTPError) as exc3:
        urllib.request.urlopen(req3, timeout=5)
    assert exc3.value.code == 404

    # Server is still alive: a normal push/pull round-trips fine.
    client = RelayClient(base_url, namespace="after-bad")
    client.push(["still-alive"])
    got, _ = client.pull(0)
    assert got == ["still-alive"]


def test_client_surfaces_relay_error(relay):
    base_url, _ = relay
    client = RelayClient(base_url, namespace="dev-A")
    # A non-string blob is rejected server-side (opaque strings only) -> 400,
    # surfaced to the client as a RelayError.
    with pytest.raises(RelayError) as exc_info:
        client.push([{"not": "a string"}])
    assert exc_info.value.status == 400


def test_relaystore_direct_semantics():
    store = RelayStore()
    assert store.high_water("ns") == 0
    assert store.append("ns", ["x", "y"]) == 2
    assert store.append("ns", ["z"]) == 3
    blobs, cursor = store.since("ns", 0)
    assert blobs == ["x", "y", "z"]
    assert cursor == 3
    blobs2, cursor2 = store.since("ns", 2)
    assert blobs2 == ["z"]
    assert cursor2 == 3
    # Non-string blob is rejected (opaque strings only).
    with pytest.raises(ValueError):
        store.append("ns", [123])
    # Empty / negative cursor handling.
    assert store.since("ns", -5)[0] == ["x", "y", "z"]
    assert store.since("missing", 0) == ([], 0)


def test_make_handler_and_serve_exist():
    # make_handler returns a handler class bound to a store; serve is callable.
    store = RelayStore()
    handler_cls = make_handler(store)
    assert isinstance(handler_cls, type)
    assert callable(serve)
