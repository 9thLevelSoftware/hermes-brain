"""brain_lease: the single mutual-exclusion primitive for dream/sweep
processes (docs/design/critique.md item 2 — one mechanism, no lockfiles).

A lease is acquired by an atomic UPDATE that only succeeds when the row is
free or its TTL has lapsed, so two dream processes sharing one brain.db (a
CLI `--if-due` spawn racing a cron run) can never both hold it: WAL
serializes the writes and the loser's UPDATE matches zero rows. Timestamps
are ISO strings (lexically comparable), consistent with the rest of the
brain and portable to native Windows.
"""

from __future__ import annotations

import logging
import sqlite3
import time

from ..store import db

logger = logging.getLogger(__name__)

TTL_SECONDS = 120
RENEW_SECONDS = 30


def _future_iso(seconds: float) -> str:
    t = time.time() + seconds
    ms = int((t % 1) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{ms:03d}Z"


def acquire(conn: sqlite3.Connection, name: str, holder: str,
            ttl_seconds: float = TTL_SECONDS) -> bool:
    """Try to take the named lease. Atomic: only succeeds when the row is
    free (holder NULL) or expired. Returns True iff this holder now owns it."""
    now = db.iso_now()
    cur = conn.execute(
        "UPDATE brain_lease SET holder=?, acquired_at=?, expires_at=? "
        "WHERE name=? AND (holder IS NULL OR expires_at < ?)",
        (holder, now, _future_iso(ttl_seconds), name, now),
    )
    conn.commit()
    if cur.rowcount == 1:
        return True
    # Also succeed if we already hold it (idempotent re-acquire).
    row = conn.execute(
        "SELECT holder FROM brain_lease WHERE name=?", (name,)).fetchone()
    return bool(row and row["holder"] == holder)


def renew(conn: sqlite3.Connection, name: str, holder: str,
          ttl_seconds: float = TTL_SECONDS) -> bool:
    """Extend the TTL — only if we still hold it (a preempted holder must
    not clobber a new owner). Returns False if the lease was lost."""
    cur = conn.execute(
        "UPDATE brain_lease SET expires_at=? WHERE name=? AND holder=?",
        (_future_iso(ttl_seconds), name, holder),
    )
    conn.commit()
    return cur.rowcount == 1


def release(conn: sqlite3.Connection, name: str, holder: str) -> None:
    conn.execute(
        "UPDATE brain_lease SET holder=NULL, acquired_at=NULL, expires_at=NULL "
        "WHERE name=? AND holder=?",
        (name, holder),
    )
    conn.commit()


def held_by(conn: sqlite3.Connection, name: str) -> str | None:
    """Current live holder (None if free/expired) — for `doctor`/status."""
    row = conn.execute(
        "SELECT holder, expires_at FROM brain_lease WHERE name=?", (name,)
    ).fetchone()
    if not row or row["holder"] is None:
        return None
    if (row["expires_at"] or "") < db.iso_now():
        return None
    return row["holder"]
