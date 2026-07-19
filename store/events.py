"""Append-only lifecycle event log over ``memory_events`` — the seam every
later phase writes from day one (Phase G sync drains it).

This module is a *writer* and a *pager*, not a sync engine. It appends one row
per memory lifecycle op (``create``/``supersede``/``tombstone``/``purge``) and
pages the log strictly-after an opaque ``(ts, event_id)`` cursor. Building the
outbox now — even though sync ships later — means the log is populated from the
first write, so a device that turns sync on later has real history to push.

Off by default (the ``sync_events`` config gate): ``record_event`` is a no-op
returning ``None`` when ``enabled`` is False, so the floor tier never pays the
write cost until sync is actually turned on (and the cost stays *visible* here,
not hidden behind a DB trigger).

Capture-path discipline (CLAUDE.md invariant #3): ``record_event`` is called
from write chokepoints that run in the turn/worker path — it must NEVER raise
into a caller. On any storage error it logs at warning and returns ``None``.

Cursor/event-log pattern adapted from mnemosyne-oss/mnemosyne
(``mnemosyne/core/sync.py``, MIT, (c) 2026 Abdias J). Only the delta-paging
cursor idea is reused; the encoding here is this project's ``base64("ts|id")``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass

from . import db

logger = logging.getLogger(__name__)

_VALID_OPS = frozenset({"create", "supersede", "tombstone", "purge"})


# ---------------------------------------------------------------------------
# ULID — stdlib-only, monotonic, lexicographically sortable
# ---------------------------------------------------------------------------
# A ULID is a 128-bit value: a 48-bit big-endian millisecond timestamp followed
# by 80 bits of randomness, rendered as 26 Crockford-base32 chars. Because the
# timestamp is the high bits and base32 preserves byte order, ULIDs sort
# lexicographically in (roughly) creation order — which is exactly what the
# ``(ts, event_id)`` cursor relies on to tie-break events sharing a millisecond
# ISO timestamp. Within a single millisecond we increment the random field
# instead of redrawing it, so ids minted back-to-back stay strictly increasing
# (monotonic) and never collide against the UNIQUE constraint.
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32 (no I,L,O,U)
_ulid_lock = threading.Lock()
_ulid_last_ms = 0
_ulid_last_rand = 0


def _new_ulid() -> str:
    """Return a fresh 26-char Crockford-base32 ULID (monotonic within a ms)."""
    global _ulid_last_ms, _ulid_last_rand
    with _ulid_lock:
        ms = int(time.time() * 1000)
        if ms == _ulid_last_ms:
            # Same millisecond: bump the previous randomness to stay strictly
            # increasing (and unique) rather than risk an equal/lower draw.
            rand = (_ulid_last_rand + 1) & ((1 << 80) - 1)
        else:
            rand = int.from_bytes(os.urandom(10), "big")
        _ulid_last_ms = ms
        _ulid_last_rand = rand

    value = ((ms & ((1 << 48) - 1)) << 80) | rand
    chars = []
    for _ in range(26):
        chars.append(_ULID_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


# ---------------------------------------------------------------------------
# Event record
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """One row of the ``memory_events`` append-only log."""

    id: int
    event_id: str
    ts: str
    op: str
    memory_uid: str
    payload: dict | None
    origin: str | None
    synced_at: str | None

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> Event:
        raw = row["payload"]
        payload: dict | None = None
        if raw is not None:
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError):
                payload = None
        return cls(
            id=row["id"],
            event_id=row["event_id"],
            ts=row["ts"],
            op=row["op"],
            memory_uid=row["memory_uid"],
            payload=payload,
            origin=row["origin"],
            synced_at=row["synced_at"],
        )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def recording_enabled(config: dict | None) -> bool:
    """Whether lifecycle events should be recorded. True when EITHER sync_events
    (record without pushing — pre-populate the log) OR sync_enabled (the master
    push/pull switch) is set. Coupling them removes the footgun where a user
    follows the CLI hint to set sync_enabled but leaves the outbox empty because
    sync_events was still off (PR #5 review)."""
    config = config or {}
    return bool(config.get("sync_events") or config.get("sync_enabled"))


def record_event(
    conn: sqlite3.Connection,
    op: str,
    memory_uid: str,
    *,
    payload: dict | None = None,
    origin: str | None = None,
    enabled: bool = True,
) -> str | None:
    """Append one lifecycle event; return its ULID ``event_id``.

    When ``enabled`` is False this is a NO-OP returning ``None`` (the
    ``sync_events`` config gate, off by default). ``op`` must be one of
    ``create``/``supersede``/``tombstone``/``purge``. ``payload`` is
    JSON-serialized.

    Never raises into a caller — on any storage error it logs at warning and
    returns ``None`` (called from write chokepoints on the capture path).
    """
    if not enabled:
        return None
    try:
        event_id = _new_ulid()
        payload_json = json.dumps(payload) if payload is not None else None
        conn.execute(
            "INSERT INTO memory_events (event_id, ts, op, memory_uid, payload, origin)"
            " VALUES (?,?,?,?,?,?)",
            (event_id, db.iso_now(), op, memory_uid, payload_json, origin),
        )
        # No conn.commit() / rollback here — the CALLER owns the transaction
        # (this runs inside _write_item / forget, which have audit + promotion
        # + batch work still pending). Committing would persist a half-built
        # unit; rolling back would DESTROY the caller's pending writes. On a
        # failed insert we simply skip the event and leave the txn untouched
        # (PR #5 review).
        return event_id
    except Exception as exc:  # never raise into a turn (invariant #3)
        logger.warning("record_event(%s, %s) skipped: %s", op, memory_uid, exc)
        return None


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------

def encode_cursor(ts: str, event_id: str) -> str:
    """Opaque, resumable cursor: base64 of ``"ts|event_id"``.

    The ``ts`` and ``event_id`` are joined with a ``|`` (event_ids are
    Crockford base32 and never contain one) and base64-encoded so callers treat
    the value as opaque.
    """
    raw = f"{ts}|{event_id}".encode()
    return base64.b64encode(raw).decode("ascii")


def decode_cursor(cursor: str) -> tuple[str, str]:
    """Inverse of :func:`encode_cursor`; tolerant of malformed input.

    Returns ``(ts, event_id)``. Any input that is empty, not valid base64, or
    missing the ``|`` separator decodes to ``("", "")`` — a cursor that sorts
    before every real row, i.e. "start from the beginning".
    """
    if not cursor:
        return "", ""
    try:
        raw = base64.b64decode(cursor, validate=True).decode("utf-8")
    except (ValueError, TypeError):
        return "", ""
    ts, sep, event_id = raw.partition("|")
    if not sep:
        return "", ""
    return ts, event_id


# ---------------------------------------------------------------------------
# Pager
# ---------------------------------------------------------------------------

def events_since(
    conn: sqlite3.Connection,
    cursor: str | None = None,
    *,
    limit: int = 500,
) -> tuple[list[Event], str | None]:
    """Page events strictly after ``cursor`` in ``(ts, event_id)`` order.

    ``cursor=None`` starts from the beginning. Returns ``(events, next_cursor)``
    where ``next_cursor`` is the last returned row's cursor (feed it back to get
    the next page), or ``None`` when no rows were returned — the end of the log.
    Ordering is deterministic and resumable: ``(ts, event_id)`` is a total order
    (event_ids are UNIQUE), matching ``idx_events_cursor``.
    """
    if cursor:
        ts, event_id = decode_cursor(cursor)
    else:
        ts, event_id = "", ""

    rows = conn.execute(
        "SELECT id, event_id, ts, op, memory_uid, payload, origin, synced_at"
        " FROM memory_events"
        " WHERE ts > ? OR (ts = ? AND event_id > ?)"
        " ORDER BY ts, event_id"
        " LIMIT ?",
        (ts, ts, event_id, int(limit)),
    ).fetchall()

    events = [Event._from_row(r) for r in rows]
    if not events:
        return events, None
    last = events[-1]
    return events, encode_cursor(last.ts, last.event_id)
