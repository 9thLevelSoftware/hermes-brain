"""MEMORY.md / USER.md import: seed the brain from Hermes' built-in memory.

The built-in memory tool (hermes-agent tools/memory_tool.py) persists two
§-delimited files under ``<hermes_home>/memories/``. This module parses them
with the SAME split the tool uses (``"\\n§\\n"``, strip, drop empties, dedup
preserving first occurrence) so an entry round-trips byte-identically, and
writes each entry as a current ``memories`` row:

    epistemic='observation'  memory_type='semantic'
    kind='profile' (USER.md) | 'fact' (MEMORY.md)
    trust_tier='owner'  created_by='bootstrap'  tags=['builtin-import']
    scope_user='owner' for USER.md rows (profile facts are about the owner)

Idempotency is content_hash dedup against current rows (valid_to IS NULL) —
re-running adds nothing; the source files are only ever READ.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from ..capture.symbols import symbols_field
from ..store import db

logger = logging.getLogger(__name__)

# Must match hermes-agent tools/memory_tool.py ENTRY_DELIMITER exactly.
ENTRY_DELIMITER = "\n§\n"

# file name -> (counts key, kind, scope_user)
_FILES = {
    "MEMORY.md": ("memory", "fact", None),
    "USER.md": ("user", "profile", "owner"),
}


def _parse_entries(path: Path) -> list[str]:
    """Mirror MemoryStore._read_file: split, strip, drop empties, dedup."""
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("bootstrap: cannot read %s (%s); skipping", path, e)
        return []
    if not raw.strip():
        return []
    entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
    return list(dict.fromkeys(e for e in entries if e))


def _hash_exists(conn: sqlite3.Connection, chash: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM memories WHERE content_hash=? AND valid_to IS NULL LIMIT 1",
        (chash,),
    ).fetchone() is not None


def import_memory_files(conn: sqlite3.Connection, hermes_home: str | Path) -> dict[str, int]:
    """Import MEMORY.md + USER.md entries. Returns {'memory', 'user', 'skipped'}.

    Missing files are simply zero counts — a fresh profile has none yet.
    """
    counts = {"memory": 0, "user": 0, "skipped": 0}
    mem_dir = Path(hermes_home) / "memories"
    now = db.iso_now()
    added = 0

    for filename, (key, kind, scope_user) in _FILES.items():
        for entry in _parse_entries(mem_dir / filename):
            chash = db.content_hash(entry)
            if _hash_exists(conn, chash):
                counts["skipped"] += 1
                continue
            conn.execute(
                "INSERT INTO memories (uid, epistemic, memory_type, kind, status, live,"
                " content, content_hash, symbols, tags, token_len, trust_tier,"
                " created_by, scope_user, valid_from, recorded_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    db.new_ulid(),
                    "observation",
                    "semantic",
                    kind,
                    "active",
                    1,
                    entry,
                    chash,
                    symbols_field(entry),
                    json.dumps(["builtin-import"]),
                    db.approx_tokens(entry),
                    "owner",
                    "bootstrap",
                    scope_user,
                    now,
                    now,
                ),
            )
            counts[key] += 1
            added += 1

    if added:
        db.bump_generation(conn, "mem")
    conn.commit()
    logger.info(
        "bootstrap: memory files imported %d MEMORY.md + %d USER.md entries (%d dup-skipped)",
        counts["memory"], counts["user"], counts["skipped"],
    )
    return counts
