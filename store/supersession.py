"""One reversible version-chain service for every memory correction path."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from ..capture.symbols import symbols_field
from . import db
from . import vec as vec_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupersessionResult:
    old_id: int
    old_uid: str
    new_id: int
    new_uid: str
    version: int
    selected_uid: str
    root_uid: str

    @property
    def restore_call(self) -> str:
        return (
            f'brain_manage(action="restore", id="{self.selected_uid}", '
            'reason="undo correction")'
        )


def _root(conn: sqlite3.Connection, row: sqlite3.Row) -> sqlite3.Row:
    seen: set[int] = set()
    current = row
    while current["supersedes_id"] is not None and current["id"] not in seen:
        seen.add(int(current["id"]))
        parent = conn.execute(
            "SELECT * FROM memories WHERE id=?", (current["supersedes_id"],),
        ).fetchone()
        if parent is None:
            break
        current = parent
    return current


def chain_head(conn: sqlite3.Connection, selected: sqlite3.Row) -> tuple[sqlite3.Row, sqlite3.Row]:
    """Return ``(root, latest descendant)`` for any historical version."""
    root = _root(conn, selected)
    head = conn.execute(
        "WITH RECURSIVE chain(id) AS ("
        " SELECT ? UNION ALL"
        " SELECT m.id FROM memories m JOIN chain c ON m.supersedes_id=c.id"
        ") SELECT m.* FROM memories m JOIN chain c ON c.id=m.id"
        " ORDER BY m.version DESC, m.id DESC LIMIT 1",
        (root["id"],),
    ).fetchone()
    return root, head or selected


def _meta(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        value = {}
    return value if isinstance(value, dict) else {}


def _drop_vector(conn: sqlite3.Connection, row_id: int) -> None:
    try:
        if vec_store.vec_available(conn):
            vec_store.delete(conn, "mem_vec", row_id)
    except Exception as exc:
        logger.warning("supersession: vector cleanup failed for %s: %s", row_id, exc)


def _embed(conn: sqlite3.Connection, embedder, row_id: int, content: str) -> None:
    if embedder is None:
        return
    try:
        if vec_store.vec_available(conn):
            vector = embedder.encode_documents([content[:8000]])[0]
            vec_store.upsert(conn, "mem_vec", row_id, vector)
            conn.execute(
                "UPDATE memories SET embedded_with=? WHERE id=?",
                (embedder.name, row_id),
            )
    except Exception as exc:
        logger.warning("supersession: embedding failed for %s: %s", row_id, exc)


def _retire_indexes(
    conn: sqlite3.Connection,
    old: sqlite3.Row,
    new_id: int,
    now: str,
    *,
    retired_status: str | None = None,
) -> None:
    status_sql = ", status=?" if retired_status else ""
    params: tuple = ((now, new_id, retired_status, old["id"])
                     if retired_status else (now, new_id, old["id"]))
    conn.execute(
        "UPDATE memories SET valid_to=COALESCE(valid_to,?), superseded_by=?"
        + status_sql + " WHERE id=?",
        params,
    )
    conn.execute(
        "UPDATE facts SET valid_until=? WHERE memory_id=? AND valid_until IS NULL",
        (now, old["id"]),
    )
    conn.execute(
        "UPDATE edges SET valid_to=? WHERE valid_to IS NULL "
        "AND (src_id=? OR dst_id=?)",
        (now, old["id"], old["id"]),
    )
    conn.execute("DELETE FROM lane1_snapshot WHERE memory_id=?", (old["id"],))
    _drop_vector(conn, int(old["id"]))


def _supersedes_edge(
    conn: sqlite3.Connection,
    new_id: int,
    old_id: int,
    now: str,
    actor: str,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO edges "
        "(src_id,dst_id,edge_type,confidence,created_by,valid_from,recorded_at) "
        "VALUES (?,?, 'supersedes',1.0,?,?,?)",
        (new_id, old_id, actor[:80], now, now),
    )


def create_successor(
    conn: sqlite3.Connection,
    selected: sqlite3.Row,
    content: str,
    *,
    actor: str,
    reason: str | None = None,
    mode: str = "correct",
    evidence: str = "explicit_manage",
    source_episodes: list[str] | None = None,
    adjudication: dict[str, Any] | None = None,
    embedder=None,
    retired_status: str | None = None,
    commit: bool = True,
) -> SupersessionResult:
    """Append a current version; never mutate or reopen historical content."""
    text = " ".join((content or "").split()).strip()
    if not text:
        raise ValueError("successor content must not be empty")
    root, head = chain_head(conn, selected)
    now = db.iso_now()
    uid = db.new_ulid()
    version = max(int(head["version"] or 1), int(selected["version"] or 1)) + 1

    meta = _meta(head["meta"])
    meta.update({
        "restore_lineage": {
            "selected_uid": selected["uid"],
            "previous_head_uid": head["uid"],
            "root_uid": root["uid"],
        },
    })
    ttl_at = head["ttl_at"]
    if mode == "restore":
        meta["restored_from_uid"] = selected["uid"]
        meta["retention"] = "durable"
        meta.pop("expiry_source", None)
        ttl_at = None
    elif reason:
        meta["correction_reason"] = reason

    # Copy the envelope generically so new schema metadata is inherited by
    # default.  Only identity/content/temporal/vector fields are regenerated.
    generated = {
        "id", "uid", "content", "summary", "content_hash", "symbols",
        "token_len", "status", "live", "version", "supersedes_id",
        "superseded_by", "valid_from", "valid_to", "recorded_at",
        "invalidated_by", "embedded_with", "prompt_version", "meta", "ttl_at",
    }
    inherited = [
        r["name"] for r in conn.execute("PRAGMA table_info(memories)").fetchall()
        if r["name"] not in generated
    ]
    columns = inherited + [
        "uid", "content", "summary", "content_hash", "symbols", "token_len",
        "status", "live", "version", "supersedes_id", "valid_from",
        "recorded_at", "prompt_version", "meta", "ttl_at",
    ]
    values = [head[name] for name in inherited] + [
        uid, text, None, db.content_hash(text), symbols_field(text),
        db.approx_tokens(text), "active", 1, version, head["id"], now, now,
        f"{mode}-v1", json.dumps(meta, sort_keys=True), ttl_at,
    ]
    cur = conn.execute(
        f"INSERT INTO memories ({','.join(columns)}) "
        f"VALUES ({','.join('?' * len(columns))})",
        values,
    )
    new_id = int(cur.lastrowid)
    _retire_indexes(
        conn, head, new_id, now, retired_status=retired_status,
    )
    _supersedes_edge(conn, new_id, int(head["id"]), now, actor)
    _embed(conn, embedder, new_id, text)

    action = "memory_restored" if mode == "restore" else "memory_corrected"
    detail = {
        "reason": reason,
        "evidence": evidence,
        "source_episodes": source_episodes or [],
        "adjudication": adjudication,
        "supersedes_uid": head["uid"],
        "restore_lineage": meta["restore_lineage"],
    }
    conn.execute(
        "INSERT INTO audit_log (actor,action,target,detail,ts) VALUES (?,?,?,?,?)",
        (actor, action, uid, json.dumps(detail, sort_keys=True), now),
    )
    db.bump_generation(conn, "mem")
    db.bump_generation(conn, "graph")
    if commit:
        conn.commit()
    return SupersessionResult(
        old_id=int(head["id"]), old_uid=head["uid"], new_id=new_id,
        new_uid=uid, version=version, selected_uid=selected["uid"],
        root_uid=root["uid"],
    )


def link_existing_successor(
    conn: sqlite3.Connection,
    old_id: int,
    new_id: int,
    *,
    actor: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
) -> bool:
    """Make an already-inserted memory the successor using the same cleanup."""
    old = conn.execute("SELECT * FROM memories WHERE id=?", (old_id,)).fetchone()
    new = conn.execute("SELECT * FROM memories WHERE id=?", (new_id,)).fetchone()
    if old is None or new is None or old_id == new_id:
        return False
    if old["valid_to"] is not None and old["superseded_by"] not in (None, new_id):
        return False
    if old["superseded_by"] == new_id and new["supersedes_id"] == old_id:
        return True
    now = db.iso_now()
    version = int(old["version"] or 1) + 1
    conn.execute(
        "UPDATE memories SET version=?, supersedes_id=? WHERE id=?",
        (version, old_id, new_id),
    )
    _retire_indexes(conn, old, new_id, now)
    _supersedes_edge(conn, new_id, old_id, now, actor)
    detail = {
        "reason": reason,
        "evidence": evidence or {},
        "supersedes_uid": old["uid"],
        "restore_lineage": {"selected_uid": old["uid"]},
    }
    conn.execute(
        "INSERT INTO audit_log (actor,action,target,detail,ts) VALUES (?,?,?,?,?)",
        (actor, "memory_superseded", new["uid"],
         json.dumps(detail, sort_keys=True), now),
    )
    db.bump_generation(conn, "mem")
    db.bump_generation(conn, "graph")
    return True
