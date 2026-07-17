"""Strategy 'forget': multi-factor value scoring + tiered demotion
(docs/design/learning-system.md §1.2g — distill, don't delete).

Of the four forgetting levers, three live elsewhere (importance gating and
dup-merge at write time; decay as read-time modulation; hard eviction only
on the compliance path). This pass's actual job is the fourth: compute a
consolidation-time value score for every current-truth memory, annotate it
to memories.importance, and DEMOTE the provably worthless tail one tier at
a time — active -> summarized -> tombstone -> (grace) content stub. The
row, its uid, and its provenance survive every tier; the raw text lives on
in the episodic archive. Nothing is ever hard-deleted here.

value = weighted sum of normalized 0..1 terms: reliability (log-scaled
verification_count), user_relevance (scope_user set & recalled),
task_utility (sigmoid of helpful-harmful, zero without signal), usage
(log-scaled recall_count + recency of last recall), recency (age vs
half-life), surprise (stored kNN distance at write). Pinned rows score
1.0 and are immune to every lever.

No LLM. Preemption-aware (shift.tick()); all mutations are staged during
the read pass and applied at the end, so a mid-loop lease renewal can
never commit a half-applied batch. Modes: active mutates + audits
'forget_demote'/'forget_tombstone'/'forget_purge'; dry_run/shadow compute
the same counts and audit 'would_*' rows, mutating nothing.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from calendar import timegm

from ..store import db
from .shift import Shift

logger = logging.getLogger(__name__)

_MAX_SCORED = 5000       # value-scoring pass cap per run
_MAX_DEMOTE = 200        # tier moves per run (each tier)
_MAX_PURGE = 100         # tombstone content purges per run
_DEFAULT_DEMOTE_BELOW = 0.15
_DEFAULT_GRACE_DAYS = 30.0
_NO_HALF_LIFE_AGE_DAYS = 180.0   # stale bar when the row has no half-life
_USAGE_RECENCY_HALF_LIFE_DAYS = 90.0
_NO_DECAY_REFERENCE_DAYS = 365.0
_STUB_CHARS = 200

# Weights sum to 1.0 so the score stays in 0..1.
_W_RELIABILITY = 0.15
_W_USER_RELEVANCE = 0.15
_W_TASK_UTILITY = 0.25
_W_USAGE = 0.20
_W_RECENCY = 0.15
_W_SURPRISE = 0.10


def run(shift: Shift) -> dict:
    """Never raises — a forgetting failure must not sink the pipeline."""
    try:
        return _run(shift)
    except Exception as e:  # noqa: BLE001 — strategy contract
        logger.warning("forget: failed: %s", e, exc_info=True)
        try:
            shift.conn.rollback()
        except sqlite3.Error:
            pass
        return {"error": str(e)}


def _run(shift: Shift) -> dict:
    mode = shift.config.get("_forced_mode") or shift.mode("forget")
    active = mode == "active"
    demote_below = float(shift.config.get("forget_demote_below", _DEFAULT_DEMOTE_BELOW))
    grace_secs = float(shift.config.get("forget_grace_days", _DEFAULT_GRACE_DAYS)) * 86400.0
    now_epoch = time.time()

    counts = {"scored": 0, "demoted": 0, "tombstoned": 0, "purged": 0}
    importance_updates: list[tuple[float, int]] = []
    demote: list[tuple[sqlite3.Row, float]] = []      # active -> summarized
    tombstone: list[tuple[sqlite3.Row, float]] = []   # summarized -> tombstone

    # -- value scoring over current-truth rows (active + summarized: the
    #    summarized tier must keep proving it deserves to stay). A WRAPPING
    #    id cursor (review finding #10) rotates the bounded window across the
    #    whole set over successive runs, instead of re-reading the same
    #    lowest-id prefix forever (where pinned/outcome/recalled rows are
    #    immune and permanently squat the low ids). ---------------------------
    cursor = _get_cursor(shift.conn)
    rows = shift.conn.execute(
        "SELECT id, uid, status, pinned, outcome, scope_user, surprise,"
        " helpful_count, harmful_count, recall_count, last_recalled_at,"
        " verification_count, half_life_days, valid_from, recorded_at"
        " FROM memories WHERE valid_to IS NULL AND live=1"
        " AND status IN ('active','summarized') AND id > ? ORDER BY id LIMIT ?",
        (cursor, _MAX_SCORED),
    ).fetchall()
    # Wrap to the start once the tail is reached (short batch = end of set).
    next_cursor = rows[-1]["id"] if len(rows) == _MAX_SCORED else 0

    for row in rows:
        if not shift.tick():
            break
        score = _value_score(row, now_epoch)
        counts["scored"] += 1
        importance_updates.append((round(score, 6), row["id"]))
        if not _qualifies_demotion(row, score, demote_below, now_epoch):
            continue
        if row["status"] == "active" and len(demote) < _MAX_DEMOTE:
            demote.append((row, score))
        elif row["status"] == "summarized" and len(tombstone) < _MAX_DEMOTE:
            start = _grace_start(shift.conn, row, ("forget_demote",))
            if start is not None and now_epoch - start > grace_secs:
                tombstone.append((row, score))

    # -- grace purge: tombstones past grace lose their content (stub stays) -
    purge: list[sqlite3.Row] = []
    t_rows = shift.conn.execute(
        "SELECT id, uid, content, summary, recorded_at FROM memories"
        " WHERE status='tombstone' AND content IS NOT NULL ORDER BY id LIMIT ?",
        (_MAX_PURGE,),
    ).fetchall()
    for row in t_rows:
        if not shift.tick():
            break
        start = _grace_start(shift.conn, row, ("forget_tombstone", "tombstone"))
        if start is not None and now_epoch - start > grace_secs:
            purge.append(row)

    counts["demoted"] = len(demote)
    counts["tombstoned"] = len(tombstone)
    counts["purged"] = len(purge)

    # -- apply (active) or record intent (dry_run/shadow) -------------------
    if active:
        shift.conn.executemany(
            "UPDATE memories SET importance=? WHERE id=?", importance_updates)
        for row, score in demote:
            shift.conn.execute(
                "UPDATE memories SET status='summarized' WHERE id=?", (row["id"],))
            shift.audit("forget_demote", row["uid"],
                        {"score": round(score, 4), "from": "active",
                         "to": "summarized"})
        for row, score in tombstone:
            shift.conn.execute(
                "UPDATE memories SET status='tombstone' WHERE id=?", (row["id"],))
            shift.audit("forget_tombstone", row["uid"],
                        {"score": round(score, 4), "from": "summarized",
                         "to": "tombstone"})
        for row in purge:
            stub = (row["summary"] or (row["content"] or "")[:_STUB_CHARS]).strip()
            shift.conn.execute(
                "UPDATE memories SET content=NULL, summary=? WHERE id=?",
                (stub or None, row["id"]))
            shift.audit("forget_purge", row["uid"],
                        {"stub_kept": bool(stub),
                         "note": "content distilled; raw text in episodic archive"})
        if demote or tombstone or purge:
            db.bump_generation(shift.conn)
    elif mode == "dry_run":                            # shadow is audit-silent (#8)
        for row, score in demote:
            shift.audit("would_demote", row["uid"],
                        {"score": round(score, 4), "from": "active",
                         "to": "summarized", "mode": mode})
        for row, score in tombstone:
            shift.audit("would_tombstone", row["uid"],
                        {"score": round(score, 4), "from": "summarized",
                         "to": "tombstone", "mode": mode})
        for row in purge:
            shift.audit("would_purge", row["uid"], {"mode": mode})
    _set_cursor(shift.conn, next_cursor)               # rotate window (any mode)
    shift.conn.commit()
    return counts


def _get_cursor(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT watermark FROM sweep_state WHERE key='forget:cursor'").fetchone()
    try:
        return int(row["watermark"]) if row else 0
    except (TypeError, ValueError):
        return 0


def _set_cursor(conn: sqlite3.Connection, value: int) -> None:
    conn.execute(
        "INSERT INTO sweep_state (key, watermark, updated_at) VALUES "
        "('forget:cursor', ?, ?) ON CONFLICT(key) DO UPDATE SET "
        "watermark=excluded.watermark, updated_at=excluded.updated_at",
        (str(int(value)), db.iso_now()))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _value_score(row: sqlite3.Row, now_epoch: float) -> float:
    """Weighted multi-factor value, 0..1 (pinned rows peg to 1.0)."""
    if row["pinned"]:
        return 1.0

    reliability = min(1.0, math.log1p(max(0, row["verification_count"] or 0))
                      / math.log1p(10))

    recalls = row["recall_count"] or 0
    user_relevance = (1.0 if recalls > 0 else 0.4) if row["scope_user"] else 0.0

    helpful = float(row["helpful_count"] or 0)
    harmful = float(row["harmful_count"] or 0)
    # No outcome signal at all -> no utility credit (not the sigmoid's 0.5).
    task_utility = (1.0 / (1.0 + math.exp(-(helpful - harmful)))
                    if helpful or harmful else 0.0)

    usage_volume = min(1.0, math.log1p(recalls) / math.log1p(20))
    last = _iso_to_epoch(row["last_recalled_at"] or "")
    usage_recency = 0.0
    if last is not None:
        days_since = max(0.0, now_epoch - last) / 86400.0
        usage_recency = 0.5 ** (days_since / _USAGE_RECENCY_HALF_LIFE_DAYS)
    usage = 0.5 * usage_volume + 0.5 * usage_recency

    age_days = _age_days(row, now_epoch)
    half_life = row["half_life_days"]
    if half_life:
        recency = 0.5 ** (age_days / float(half_life))
    else:
        recency = math.exp(-age_days / _NO_DECAY_REFERENCE_DAYS)

    surprise = min(1.0, max(0.0, float(row["surprise"] or 0.0)))

    value = (_W_RELIABILITY * reliability
             + _W_USER_RELEVANCE * user_relevance
             + _W_TASK_UTILITY * task_utility
             + _W_USAGE * usage
             + _W_RECENCY * recency
             + _W_SURPRISE * surprise)
    return max(0.0, min(1.0, value))


def _qualifies_demotion(row: sqlite3.Row, score: float, demote_below: float,
                        now_epoch: float) -> bool:
    """Low-value AND stale AND unproven AND untouched — every guard must
    agree before a row moves down a tier."""
    if row["pinned"]:
        return False
    if row["outcome"]:
        return False
    if (row["recall_count"] or 0) > 0:
        return False
    if score >= demote_below:
        return False
    age_days = _age_days(row, now_epoch)
    half_life = row["half_life_days"]
    if half_life:
        return age_days > 2.0 * float(half_life)
    return age_days > _NO_HALF_LIFE_AGE_DAYS


def _age_days(row: sqlite3.Row, now_epoch: float) -> float:
    born = _iso_to_epoch(row["valid_from"] or "") \
        or _iso_to_epoch(row["recorded_at"] or "") or now_epoch
    return max(0.0, (now_epoch - born) / 86400.0)


def _grace_start(conn: sqlite3.Connection, row: sqlite3.Row,
                 actions: tuple) -> float | None:
    """When did this row enter its current tier? Latest matching audit row
    ('forgotten_at' tracking); falls back to the row's recorded_at so
    manually-tiered rows still age out."""
    placeholders = ",".join("?" * len(actions))
    hit = conn.execute(
        f"SELECT MAX(ts) AS t FROM audit_log WHERE target=?"
        f" AND action IN ({placeholders})",
        (row["uid"], *actions),
    ).fetchone()
    ts = hit["t"] if hit else None
    if ts:
        return _iso_to_epoch(ts)
    return _iso_to_epoch(row["recorded_at"] or "")


def _iso_to_epoch(ts: str) -> float | None:
    """ISO-8601 UTC ('2026-07-16T21:04:05.123Z') -> epoch seconds."""
    try:
        base = timegm(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return None
    ms = 0.0
    if len(ts) > 20 and ts[19] == ".":
        try:
            ms = float("0." + ts[20:23])
        except ValueError:
            ms = 0.0
    return base + ms
