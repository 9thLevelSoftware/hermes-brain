"""Authoritative current-truth and reversible TTL expiry operations."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from typing import Any

from . import db
from . import vec as vec_store

logger = logging.getLogger(__name__)

_SQL_NOW = "strftime('%Y-%m-%dT%H:%M:%fZ','now')"


def current_memory_predicate(alias: str = "") -> str:
    """SQL for a memory that is valid as current truth *right now*.

    Keep this expression centralized: background expiry is maintenance, while
    this query-time guard is the correctness boundary when dream has not run.
    """
    prefix = f"{alias.rstrip('.')}." if alias else ""
    return (
        f"{prefix}valid_to IS NULL AND {prefix}status = 'active' "
        f"AND {prefix}live = 1 AND "
        f"({prefix}ttl_at IS NULL OR {prefix}ttl_at > {_SQL_NOW})"
    )


def metadata(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def lifecycle_fields(row) -> dict[str, Any]:
    try:
        raw_meta = row["meta"]
    except (IndexError, KeyError):
        raw_meta = None
    meta = metadata(raw_meta)
    return {
        "retention": meta.get("retention") or (
            "temporary" if row["ttl_at"] else "durable"
        ),
        "ttl_at": row["ttl_at"],
        "expiry_source": meta.get("expiry_source"),
    }


def apply_legacy_retention(
    conn: sqlite3.Connection,
    *,
    now: datetime | None = None,
    actor: str = "lifecycle:legacy",
    limit: int = 500,
) -> dict[str, Any]:
    """Apply only high-precision deterministic TTLs to legacy rows.

    Ambiguous/model-only guesses are deliberately left untouched.  A later
    dream may propose them in shadow, but this pass never guesses.
    """
    from dataclasses import replace

    from ..capture.retention import classify_retention, lifecycle_meta

    clock = now or datetime.now(UTC)
    rows = conn.execute(
        f"SELECT * FROM memories WHERE {current_memory_predicate()} "
        "AND ttl_at IS NULL ORDER BY id LIMIT ?",
        (max(1, min(int(limit), 5000)),),
    ).fetchall()
    updated: list[int] = []
    stamped_at = db.iso_now()
    for row in rows:
        policy = classify_retention(row["content"] or "", row["kind"], now=clock)
        if policy.retention == "episode_only":
            # The episode already exists, but deleting a historical durable
            # row would be destructive.  Give it the same short operational
            # horizon instead and let normal expiry preserve it as history.
            policy = classify_retention(
                "deployment operational state", "fact", now=clock,
            )
            policy = replace(policy, expiry_source="legacy_process_narration")
        if policy.retention != "temporary":
            continue
        if policy.expiry_source == "operational_default":
            policy = replace(policy, expiry_source="legacy_operational_pattern")
        meta = lifecycle_meta(row["meta"], policy)
        conn.execute(
            "UPDATE memories SET ttl_at=?, meta=? WHERE id=? AND ttl_at IS NULL",
            (policy.ttl_at, meta, row["id"]),
        )
        conn.execute(
            "INSERT INTO audit_log (actor,action,target,detail,ts) "
            "VALUES (?,?,?,?,?)",
            (actor, "legacy_retention_applied", row["uid"], json.dumps({
                "retention": policy.retention,
                "ttl_at": policy.ttl_at,
                "expiry_source": policy.expiry_source,
            }, sort_keys=True), stamped_at),
        )
        updated.append(int(row["id"]))
    if updated:
        db.bump_generation(conn, "mem")
        conn.commit()
    return {"legacy_ttl_applied": len(updated), "legacy_ids": updated}


def expire_due(
    conn: sqlite3.Connection,
    *,
    now: str | None = None,
    actor: str = "lifecycle",
) -> dict[str, Any]:
    """Expire due current rows without deleting content or provenance.

    Facts and graph edges are temporal indexes over the memory, so they close
    in the same transaction.  Vectors and lane-1 rows are derived active
    indexes and are removed.  Re-running is idempotent.
    """
    closed_at = now or db.iso_now()
    rows = conn.execute(
        "SELECT * FROM memories WHERE valid_to IS NULL AND status='active' "
        "AND live=1 AND ttl_at IS NOT NULL AND ttl_at <= ? ORDER BY id",
        (closed_at,),
    ).fetchall()
    if not rows:
        return {"expired": 0, "ids": []}

    ids: list[int] = []
    try:
        for row in rows:
            mid = int(row["id"])
            ids.append(mid)
            conn.execute(
                "UPDATE memories SET status='expired', valid_to=? "
                "WHERE id=? AND valid_to IS NULL AND status='active'",
                (closed_at, mid),
            )
            conn.execute(
                "UPDATE facts SET valid_until=? "
                "WHERE memory_id=? AND valid_until IS NULL",
                (closed_at, mid),
            )
            conn.execute(
                "UPDATE edges SET valid_to=? WHERE valid_to IS NULL "
                "AND (src_id=? OR dst_id=?)",
                (closed_at, mid, mid),
            )
            conn.execute("DELETE FROM lane1_snapshot WHERE memory_id=?", (mid,))
            try:
                if vec_store.vec_available(conn):
                    vec_store.delete(conn, "mem_vec", mid)
            except Exception as exc:  # derived data cleanup is best effort
                logger.warning("expiry: vector cleanup failed for %s: %s", mid, exc)

            fields = lifecycle_fields(row)
            conn.execute(
                "INSERT INTO audit_log (actor,action,target,detail,ts) "
                "VALUES (?,?,?,?,?)",
                (
                    actor,
                    "memory_expired",
                    row["uid"],
                    json.dumps({
                        **fields,
                        "closed_at": closed_at,
                        "source_refs": _json_value(row["source_refs"], []),
                    }, sort_keys=True),
                    closed_at,
                ),
            )
        db.bump_generation(conn, "mem")
        db.bump_generation(conn, "graph")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"expired": len(ids), "ids": ids}


def _json_value(raw, default):
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return default
