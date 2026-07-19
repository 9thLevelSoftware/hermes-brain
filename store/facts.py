"""Temporal fact index over ``memories`` — single-current-truth triples.

Adapted from mnemosyne-oss/mnemosyne (``mnemosyne/core/triples.py``, MIT,
(c) 2026 Abdias J). Table semantics taken wholesale: close-then-insert
supersession (exactly one current row per ``(subject, predicate)``) and the
as-of predicate ``valid_from <= as_of AND (valid_until IS NULL OR
valid_until > as_of)``. The donor's connection/data-dir machinery is
discarded — this module operates on hermes-brain's own ``sqlite3.Connection``
(house pragmas + schema owned by ``store/db.py``).

``facts`` is an INDEX OVER ``memories``, not a parallel store (critique item
9): a triple carries the ``memory_id`` of the natural-language row it indexes.
Because a fact and its memory are one truth in two shapes, superseding a fact
that points at a *different* memory retires the OLD linked memory row in
lockstep via the memories version-chain (plan finding 5).

stdlib-only: sqlite3 + json + dataclasses. Parameterized SQL only.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from . import db

# Column list shared by every SELECT; order MUST match the ``Fact`` fields so
# ``Fact(*row)`` maps positionally regardless of the connection row_factory.
_COLS = (
    "id, subject, predicate, object, memory_id, entity_id, confidence, "
    "source, valid_from, valid_until, recorded_at, superseded_by"
)


@dataclass
class Fact:
    """One temporal triple. Fields mirror the ``facts`` table columns."""

    id: int
    subject: str
    predicate: str
    object: str
    memory_id: int | None
    entity_id: int | None
    confidence: float
    source: str | None
    valid_from: str
    valid_until: str | None
    recorded_at: str
    superseded_by: int | None


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(*tuple(row))


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

def add_fact(
    conn: sqlite3.Connection,
    subject: str,
    predicate: str,
    object: str,
    *,
    memory_id: int | None = None,
    entity_id: int | None = None,
    confidence: float = 1.0,
    source: str | None = None,
    supersede: bool = True,
    valid_from: str | None = None,
) -> int:
    """Insert a triple; return its new fact id.

    When ``supersede`` is True (the default), first CLOSE the current
    ``(subject, predicate)`` row(s): set ``valid_until = now`` and
    ``superseded_by = <new fact id>``. If a closed fact carried a
    ``memory_id`` and this call carries a *different* ``memory_id``, the OLD
    linked memory row is retired in lockstep via the version-chain
    (``UPDATE memories SET valid_to=now, superseded_by=<new memory_id>``) —
    facts are an index over memories, so the NL memory must retire with the
    triple (plan finding 5).

    ``valid_from`` defaults to now. With ``supersede`` there is exactly one
    current-truth row per ``(subject, predicate)`` after the call.
    """
    now = db.iso_now()
    vfrom = valid_from or now

    # Snapshot the current truth BEFORE inserting so we still see the row(s)
    # this write is about to close (and their linked memory ids).
    closing: list[sqlite3.Row] = []
    if supersede:
        closing = conn.execute(
            "SELECT id, memory_id FROM facts "
            "WHERE subject=? AND predicate=? AND valid_until IS NULL",
            (subject, predicate),
        ).fetchall()

    cur = conn.execute(
        "INSERT INTO facts (subject, predicate, object, memory_id, entity_id, "
        "confidence, source, valid_from, valid_until, recorded_at, superseded_by) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (subject, predicate, object, memory_id, entity_id, confidence, source,
         vfrom, None, now, None),
    )
    new_id = int(cur.lastrowid)

    for old in closing:
        old_id = old["id"]
        old_mem = old["memory_id"]
        conn.execute(
            "UPDATE facts SET valid_until=?, superseded_by=? WHERE id=?",
            (now, new_id, old_id),
        )
        # The fact index moved to a new memory -> retire the old NL row in
        # lockstep — but ONLY when that memory no longer backs any OTHER
        # current-truth fact. One memory can carry MANY triples (extraction
        # emits several per item), so retiring it because a single triple
        # changed would orphan the siblings: they'd point at a non-current row
        # and drop out of current-truth recall (PR #5 review).
        if old_mem is not None and memory_id is not None and old_mem != memory_id:
            still_used = conn.execute(
                "SELECT 1 FROM facts WHERE memory_id=? AND valid_until IS NULL LIMIT 1",
                (old_mem,),
            ).fetchone()
            if still_used is None:
                conn.execute(
                    "UPDATE memories SET valid_to=?, superseded_by=? WHERE id=?",
                    (now, memory_id, old_mem),
                )

    # NOTE: no conn.commit() here — the CALLER owns the transaction. add_fact
    # runs inside _write_item's extraction transaction (audit + promotion still
    # pending) and inside dream strategies that batch multiple writes; a commit
    # here would prematurely persist a half-built unit and defeat the caller's
    # rollback (PR #5 review).
    return new_id


def end_fact(
    conn: sqlite3.Connection,
    subject: str,
    predicate: str,
    *,
    valid_until: str | None = None,
) -> int:
    """Close the current ``(subject, predicate)`` fact with no replacement.

    Returns the number of rows closed (0 or 1 under single-current-truth).
    ``superseded_by`` stays NULL — nothing replaced it.
    """
    until = valid_until or db.iso_now()
    cur = conn.execute(
        "UPDATE facts SET valid_until=? "
        "WHERE subject=? AND predicate=? AND valid_until IS NULL",
        (until, subject, predicate),
    )
    conn.commit()
    return cur.rowcount


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------

def query_facts(
    conn: sqlite3.Connection,
    *,
    subject: str | None = None,
    predicate: str | None = None,
    object: str | None = None,
    as_of: str | None = None,
) -> list[Fact]:
    """Point-in-time query.

    ``as_of=None`` returns current truth (``valid_until IS NULL``). With
    ``as_of`` set, returns the triples valid at that instant:
    ``valid_from <= as_of AND (valid_until IS NULL OR valid_until > as_of)``.
    Results are ordered newest-first by ``valid_from``.
    """
    conditions: list[str] = []
    params: list[object] = []
    if subject is not None:
        conditions.append("subject=?")
        params.append(subject)
    if predicate is not None:
        conditions.append("predicate=?")
        params.append(predicate)
    if object is not None:
        conditions.append("object=?")
        params.append(object)

    if as_of is None:
        conditions.append("valid_until IS NULL")
    else:
        conditions.append("valid_from <= ?")
        params.append(as_of)
        conditions.append("(valid_until IS NULL OR valid_until > ?)")
        params.append(as_of)

    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    rows = conn.execute(
        f"SELECT {_COLS} FROM facts{where} ORDER BY valid_from DESC, id DESC",
        params,
    ).fetchall()
    return [_row_to_fact(r) for r in rows]


def current_facts_for(conn: sqlite3.Connection, subject: str) -> list[Fact]:
    """All current-truth facts with this subject."""
    return query_facts(conn, subject=subject)


def reasoning_chain(conn: sqlite3.Connection, memory_id: int) -> list[dict]:
    """Provenance walk for a memory row.

    Follows ``memories.supersedes_id`` back through the version chain, reading
    each row's ``source_refs`` JSON, and returns the ordered chain
    newest->oldest as plain dicts. Honcho's ``get_reasoning_chain``, expressed
    over hermes-brain's existing provenance columns. Cycle-guarded.
    """
    chain: list[dict] = []
    seen: set[int] = set()
    current: int | None = memory_id
    while current is not None and current not in seen:
        seen.add(current)
        row = conn.execute(
            "SELECT id, uid, kind, content, summary, source_refs, supersedes_id, "
            "created_by, valid_from, valid_to, recorded_at "
            "FROM memories WHERE id=?",
            (current,),
        ).fetchone()
        if row is None:
            break
        try:
            refs = json.loads(row["source_refs"] or "[]")
        except (json.JSONDecodeError, TypeError):
            refs = []
        chain.append({
            "id": row["id"],
            "uid": row["uid"],
            "kind": row["kind"],
            "content": row["content"],
            "summary": row["summary"],
            "source_refs": refs,
            "supersedes_id": row["supersedes_id"],
            "created_by": row["created_by"],
            "valid_from": row["valid_from"],
            "valid_to": row["valid_to"],
            "recorded_at": row["recorded_at"],
        })
        current = row["supersedes_id"]
    return chain
