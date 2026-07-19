"""Tests for store/events.py — the append-only ``memory_events`` writer and
its ``(ts, event_id)`` cursor pager (Phase B, best-of-three).

Covers: the off-by-default gate, a real insert (valid ULID + JSON payload),
cursor ordering that is stable and resumable across two half-pages, CHECK
rejection of a bad op, cursor round-trip + malformed-input tolerance, and the
capture-path guarantee that a storage error is swallowed, never raised.
"""

from __future__ import annotations

import sqlite3

import pytest
from brain.store import events

_ULID_ALPHABET = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


def _row_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]


# ---------------------------------------------------------------------------
# ULID
# ---------------------------------------------------------------------------

def test_new_ulid_is_26_crockford_chars_and_monotonic():
    ids = [events._new_ulid() for _ in range(1000)]
    assert all(len(u) == 26 for u in ids)
    assert all(set(u) <= _ULID_ALPHABET for u in ids)
    assert len(set(ids)) == len(ids)          # unique
    assert ids == sorted(ids)                  # lexicographically monotonic


# ---------------------------------------------------------------------------
# record_event
# ---------------------------------------------------------------------------

def test_disabled_is_noop_returning_none(conn):
    result = events.record_event(conn, "create", "uid-1", enabled=False)
    assert result is None
    assert _row_count(conn) == 0


def test_enabled_inserts_row_with_ulid_and_json_payload(conn):
    eid = events.record_event(
        conn, "create", "uid-42",
        payload={"content": "hi", "n": 3}, origin="dev-a", enabled=True,
    )
    assert isinstance(eid, str) and len(eid) == 26
    assert set(eid) <= _ULID_ALPHABET

    row = conn.execute(
        "SELECT event_id, op, memory_uid, payload, origin, synced_at"
        " FROM memory_events WHERE event_id=?",
        (eid,),
    ).fetchone()
    assert row["event_id"] == eid
    assert row["op"] == "create"
    assert row["memory_uid"] == "uid-42"
    assert row["origin"] == "dev-a"
    assert row["synced_at"] is None            # unsynced by default
    import json
    assert json.loads(row["payload"]) == {"content": "hi", "n": 3}


def test_enabled_default_and_null_payload(conn):
    # enabled defaults to True; payload=None stores SQL NULL.
    eid = events.record_event(conn, "tombstone", "uid-7")
    assert eid is not None
    row = conn.execute(
        "SELECT payload FROM memory_events WHERE event_id=?", (eid,)
    ).fetchone()
    assert row["payload"] is None


def test_invalid_op_rejected_by_check_constraint(conn):
    # The CHECK guards direct writers; a bad op is a hard integrity error.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO memory_events (event_id, ts, op, memory_uid)"
            " VALUES (?,?,?,?)",
            (events._new_ulid(), "2026-07-18T00:00:00.000Z", "frobnicate", "uid-1"),
        )


def test_record_event_swallows_storage_error(conn):
    conn.close()  # any use now raises ProgrammingError (a storage error)
    # Must not raise into the capture-path caller — returns None instead.
    assert events.record_event(conn, "create", "uid-x", enabled=True) is None


# ---------------------------------------------------------------------------
# cursor encode/decode
# ---------------------------------------------------------------------------

def test_cursor_round_trip():
    ts, eid = "2026-07-18T12:34:56.789Z", "01J000000000000000000ABCDE"
    assert events.decode_cursor(events.encode_cursor(ts, eid)) == (ts, eid)


def test_decode_cursor_tolerates_bad_input():
    assert events.decode_cursor("") == ("", "")
    assert events.decode_cursor("!!!not base64!!!") == ("", "")
    # valid base64 but no '|' separator -> start-from-beginning sentinel
    import base64
    no_sep = base64.b64encode(b"nopipehere").decode()
    assert events.decode_cursor(no_sep) == ("", "")


# ---------------------------------------------------------------------------
# events_since paging
# ---------------------------------------------------------------------------

def test_events_since_from_beginning(conn):
    for i in range(5):
        events.record_event(conn, "create", f"uid-{i}")
    got, nxt = events.events_since(conn)
    assert [e.memory_uid for e in got] == [f"uid-{i}" for i in range(5)]
    assert nxt == events.encode_cursor(got[-1].ts, got[-1].event_id)
    # payload decoded back to a dict/None; Event shape intact
    assert got[0].payload is None
    assert got[0].op == "create"
    assert got[0].synced_at is None


def test_events_since_is_stable_and_resumable_across_two_halves(conn):
    uids = [f"uid-{i:02d}" for i in range(10)]
    for u in uids:
        events.record_event(conn, "create", u, payload={"u": u})

    first, cur = events.events_since(conn, limit=4)
    assert len(first) == 4
    assert cur is not None

    second, cur2 = events.events_since(conn, cur, limit=4)
    assert len(second) == 4

    third, cur3 = events.events_since(conn, cur2, limit=4)
    assert len(third) == 2                      # remainder

    # No overlap, no gap: the three pages reconstruct the full ordered log.
    paged = [e.event_id for e in first + second + third]
    assert paged == sorted(paged)               # globally ordered
    assert len(set(paged)) == 10                # every event exactly once
    assert [e.memory_uid for e in first + second + third] == uids

    # Draining to the end yields no rows and a None cursor.
    tail, cur4 = events.events_since(conn, cur3, limit=4)
    assert tail == []
    assert cur4 is None


def test_empty_log_returns_no_rows_and_none_cursor(conn):
    got, nxt = events.events_since(conn)
    assert got == []
    assert nxt is None
