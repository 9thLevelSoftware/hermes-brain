"""Daem0n-MCP migration: import a per-project ``.daem0nmcp/memory.db``.

Optional, CLI-triggered (``hermes brain bootstrap --daemon <path>``), one call
per project. The source is opened READ-ONLY (same percent-encoded URI pattern
as store/db.connect) and only its ``memories`` table is read — column set per
Daem0n-MCP daem0nmcp/models.py (category, content, rationale, tags, outcome,
worked, pinned, archived, created_at).

Mapping (docs/design/integration.md §5.4, refined against models.py):
  category  decision->'decision'  pattern/learning->'insight'
            warning->'warning'    anything else->'fact'
  content   content (+ '\\nRationale: <rationale>' when present — Daem0n kept
            the why in a separate column; the brain keeps one content field)
  worked    True->'worked'  False->'failed'  (its free-text outcome column
            lands in outcome_note so nothing is lost)
  type      pattern/warning are project facts -> 'semantic';
            decision/learning decay -> 'episodic' with half_life_days=90
  scope_project = name of the directory containing .daem0nmcp
  valid_from = its created_at (bi-temporal: when it became true, not now)
  trust 'owner', created_by='migration', tags = its tags + ['daem0n-import']

Archived rows are SKIPPED (Daem0n already retired them); dedup is
content_hash against current rows, so the import is re-runnable.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..capture.symbols import symbols_field
from ..store import db

logger = logging.getLogger(__name__)

_KIND_MAP = {
    "decision": "decision",
    "pattern": "insight",
    "warning": "warning",
    "learning": "insight",
}
_SEMANTIC_CATEGORIES = {"pattern", "warning"}
_EPISODIC_HALF_LIFE_DAYS = 90.0

_PATH_REMEDY = (
    "expected <project>/.daem0nmcp/memory.db — point --daemon at the memory.db "
    "inside the project's .daem0nmcp directory"
)


def _iso_from_daemon(value: Any) -> str | None:
    """Daem0n (SQLAlchemy DateTime) stores 'YYYY-MM-DD HH:MM:SS[.ffffff][+00:00]'."""
    if not value:
        return None
    text = str(value).strip().replace(" ", "T", 1)
    if text.endswith("+00:00"):
        text = text[:-6] + "Z"
    elif not text.endswith("Z") and "+" not in text[10:]:
        text += "Z"
    return text


def _tags_from_daemon(value: Any) -> list:
    try:
        tags = json.loads(value) if isinstance(value, str) else (value or [])
        return [str(t) for t in tags] if isinstance(tags, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _scope_project(db_path: Path) -> str:
    parent = db_path.resolve().parent
    return parent.parent.name if parent.name == ".daem0nmcp" else parent.name


def import_daemon_db(conn: sqlite3.Connection, daemon_db_path: str | Path) -> dict[str, Any]:
    """Import one Daem0n memory.db. Returns {'imported', 'skipped'} (+ 'error')."""
    counts: dict[str, Any] = {"imported": 0, "skipped": 0}
    path = Path(daemon_db_path)
    if not path.is_file():
        counts["error"] = f"daemon db not found at {path}: {_PATH_REMEDY}"
        return counts

    uri = "file:" + quote(str(path).replace("\\", "/")) + "?mode=ro"
    try:
        src = sqlite3.connect(uri, uri=True, timeout=5.0)
    except sqlite3.Error as e:
        counts["error"] = f"cannot open {path} ({e}): {_PATH_REMEDY}"
        return counts
    src.row_factory = sqlite3.Row

    try:
        try:
            rows = [dict(r) for r in src.execute("SELECT * FROM memories").fetchall()]
        except sqlite3.Error as e:
            counts["error"] = f"{path} has no readable memories table ({e}): {_PATH_REMEDY}"
            return counts
    finally:
        src.close()

    project = _scope_project(path)
    now = db.iso_now()
    added = 0
    for row in rows:
        if row.get("archived"):
            counts["skipped"] += 1
            continue
        content = str(row.get("content") or "").strip()
        if not content:
            counts["skipped"] += 1
            continue
        rationale = str(row.get("rationale") or "").strip()
        if rationale:
            content = f"{content}\nRationale: {rationale}"

        chash = db.content_hash(content)
        if conn.execute(
            "SELECT 1 FROM memories WHERE content_hash=? AND valid_to IS NULL LIMIT 1",
            (chash,),
        ).fetchone():
            counts["skipped"] += 1
            continue

        category = str(row.get("category") or "")
        semantic = category in _SEMANTIC_CATEGORIES
        worked = row.get("worked")
        conn.execute(
            "INSERT INTO memories (uid, epistemic, memory_type, kind, status, live,"
            " content, content_hash, symbols, tags, token_len, trust_tier, created_by,"
            " scope_project, valid_from, recorded_at, pinned, half_life_days,"
            " outcome, outcome_note)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                db.new_ulid(),
                "observation",
                "semantic" if semantic else "episodic",
                _KIND_MAP.get(category, "fact"),
                "active",
                1,
                content,
                chash,
                symbols_field(content),
                json.dumps(_tags_from_daemon(row.get("tags")) + ["daem0n-import"]),
                db.approx_tokens(content),
                "owner",
                "migration",
                project,
                _iso_from_daemon(row.get("created_at")) or now,
                now,
                1 if row.get("pinned") else 0,
                None if semantic else _EPISODIC_HALF_LIFE_DAYS,
                None if worked is None else ("worked" if worked else "failed"),
                str(row["outcome"]) if row.get("outcome") else None,
            ),
        )
        counts["imported"] += 1
        added += 1

    if added:
        db.bump_generation(conn, "mem")
    conn.commit()
    logger.info(
        "bootstrap: daemon import from %s — %d imported, %d skipped (project=%s)",
        path, counts["imported"], counts["skipped"], project,
    )
    return counts
