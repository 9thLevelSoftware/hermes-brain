"""Entity population + lookup — the PPR substrate (schema.sql `entities` /
`entity_mentions`).

These tables shipped with no writers; this module is the writer. The
consolidate strategy already extracts one concrete typed entity per distilled
lesson — linking that entity to the pattern AND its member memories builds the
co-mention structure recall/graph.py runs Personalized PageRank over, and it
finally gives consolidate's own specificity gate real data to check against.

Global entity identity is the normalized `canonical` name (UNIQUE); a mention
is one (entity, memory) pair. Stdlib only — safe in the store subpackage.
"""

from __future__ import annotations

import logging
import re

from . import db

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")
_MAX_CANONICAL = 120


def normalize(name: str) -> str:
    """Global identity key: casefold, trim, collapse internal whitespace."""
    return _WS_RE.sub(" ", (name or "").strip()).casefold()[:_MAX_CANONICAL]


def link(conn, name: str, memory_id: int | None, *, entity_type: str = "concept",
         display: str | None = None, scope_project: str | None = None,
         principal_id: str | None = None, ts: str | None = None) -> int | None:
    """Upsert the entity and record a (entity, memory) mention. Bumps
    mention_count only on a genuinely new mention. Returns the entity id, or
    None (bad input). Never raises — capture/dream path."""
    try:
        canonical = normalize(name)
        if not canonical or memory_id is None:
            return None
        ts = ts or db.iso_now()
        display = (display or name or "").strip() or canonical
        conn.execute(
            "INSERT INTO entities (canonical, display_name, entity_type,"
            " principal_id, mention_count, created_at) VALUES (?,?,?,?,0,?)"
            " ON CONFLICT(canonical) DO UPDATE SET"
            " principal_id=COALESCE(entities.principal_id, excluded.principal_id)",
            (canonical, display, entity_type, principal_id, ts))
        ent = conn.execute("SELECT id FROM entities WHERE canonical=?",
                           (canonical,)).fetchone()
        entity_id = ent["id"]
        cur = conn.execute(
            "INSERT OR IGNORE INTO entity_mentions (entity_id, memory_id,"
            " scope_project, ts) VALUES (?,?,?,?)",
            (entity_id, memory_id, scope_project, ts))
        if cur.rowcount:
            conn.execute("UPDATE entities SET mention_count=mention_count+1"
                         " WHERE id=?", (entity_id,))
        return entity_id
    except Exception as e:  # never break a turn/dream over graph bookkeeping
        logger.warning("entities.link failed for %r: %s", name, e)
        return None


def entities_of(conn, memory_ids) -> set[int]:
    """Entity ids mentioned by any of these memories."""
    ids = [int(m) for m in memory_ids if m is not None]
    if not ids:
        return set()
    rows = conn.execute(
        f"SELECT DISTINCT entity_id FROM entity_mentions"
        f" WHERE memory_id IN ({','.join('?' * len(ids))})", ids).fetchall()
    return {r["entity_id"] for r in rows}


def memories_of(conn, entity_ids, *, limit: int = 400) -> list[int]:
    """Memory ids mentioning any of these entities (bounded)."""
    ids = [int(e) for e in entity_ids if e is not None]
    if not ids:
        return []
    rows = conn.execute(
        f"SELECT DISTINCT memory_id FROM entity_mentions"
        f" WHERE entity_id IN ({','.join('?' * len(ids))}) LIMIT ?",
        [*ids, limit]).fetchall()
    return [r["memory_id"] for r in rows]


def co_mentioned(conn, memory_ids, *, limit: int = 24) -> list[int]:
    """Memories sharing at least one entity with the seeds (1 hop), seeds
    excluded. Cheap neighbor lookup for brain_recall(depth=deep)."""
    seeds = {int(m) for m in memory_ids if m is not None}
    if not seeds:
        return []
    ent = entities_of(conn, seeds)
    out: list[int] = []
    for mid in memories_of(conn, ent, limit=limit * 4):
        if mid not in seeds:
            out.append(mid)
        if len(out) >= limit:
            break
    return out
