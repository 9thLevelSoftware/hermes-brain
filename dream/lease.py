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

import calendar
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
    # `expires_at IS NULL` is load-bearing: in SQL `NULL < '2026-...'` is NULL,
    # not true, so a row left with a holder but no expiry could never be
    # reclaimed and the dream would be wedged forever with no recovery path.
    # held_by() already treats that state as free — this mirrors it.
    cur = conn.execute(
        "UPDATE brain_lease SET holder=?, acquired_at=?, expires_at=? "
        "WHERE name=? AND (holder IS NULL OR expires_at IS NULL OR expires_at < ?)",
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


# ---------------------------------------------------------------------------
# "is a dream due?" — shared by every trigger
# ---------------------------------------------------------------------------

DEFAULT_MIN_INTERVAL_HOURS = 6


def last_dream_finished(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT finished_at FROM shift_runs WHERE finished_at IS NOT NULL "
        "ORDER BY finished_at DESC LIMIT 1").fetchone()
    return row["finished_at"] if row else None


def _iso_add_hours(iso: str, hours: float) -> str:
    """ISO string + hours, staying in the lexically comparable text domain.

    Uses ``calendar.timegm``, not ``time.mktime``: every timestamp the brain
    writes is UTC (``db.iso_now``), and mktime would reinterpret it as LOCAL
    time — skewing the due-check by the machine's UTC offset, so a 6h interval
    fired hours early or late depending on the timezone.
    """
    try:
        base = time.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError):
        return iso
    return time.strftime("%Y-%m-%dT%H:%M:%S",
                         time.gmtime(calendar.timegm(base) + hours * 3600))


def is_due(conn: sqlite3.Connection, config: dict) -> bool:
    """True when a dream shift may start now.

    ONE implementation for every trigger — `hermes brain dream --if-due` (cron)
    and the provider's opt-in on-idle path — so the two can never disagree
    about what "due" means. A held lease reads as not-due, which is also what
    makes triple-triggering harmless: the loser simply no-ops.
    """
    if held_by(conn, "dream"):
        return False
    last = last_dream_finished(conn)
    if not last:
        return True
    hours = float(config.get("dream_min_interval_hours", DEFAULT_MIN_INTERVAL_HOURS))
    return _iso_add_hours(last, hours) < db.iso_now()
