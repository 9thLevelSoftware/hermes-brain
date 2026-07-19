"""The migration invariant: a fresh database and a migrated one are the same.

store/schema.sql (fresh create) and store/migrations/*.sql (upgrade path)
are two descriptions of one schema. Nothing but a test stops them drifting,
so this is that test — it fails loudly the day someone adds a table to
schema.sql and forgets the delta (or vice versa).

Applies to every version: `_v1_schema()` reconstructs a v1 database by
creating a fresh one and dropping what the v2 delta added, which keeps the
fixture honest without pinning a stale copy of the old schema in the repo.
"""

from __future__ import annotations

import sqlite3

import pytest
from brain.store import db


def _structure(conn: sqlite3.Connection) -> list[str]:
    """Every named object's DDL, ordered — the comparable shape of a schema.

    Auto-created objects (sqlite_autoindex_*) have sql IS NULL and are
    excluded: they are implied by the DDL we do compare.
    """
    return sorted(
        r["sql"] for r in conn.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%'"
        )
    )


def _migration_objects() -> set[str]:
    """Names every delta introduces (parsed from the migration files).

    Collected across ALL migrations (not just v2), so reconstructing a v1
    database drops what v2 AND v3+ added — the migrate-forward path then
    genuinely re-applies each delta and drift in any of them fails here,
    rather than a later migration's CREATE ... IF NOT EXISTS silently
    no-op'ing over a table the fresh schema already made."""
    import re

    names: set[str] = set()
    for sql_file in sorted(db._MIGRATIONS_DIR.glob("*.sql")):
        sql = sql_file.read_text(encoding="utf-8")
        names |= set(re.findall(
            r"CREATE (?:TABLE|INDEX) IF NOT EXISTS (\w+)", sql))
    return names


def test_migration_files_exist_for_every_version():
    """Every version between 2 and SCHEMA_VERSION must have a step."""
    for version in range(2, db.SCHEMA_VERSION + 1):
        assert version in db.MIGRATIONS, f"no migration registered for v{version}"


def test_fresh_and_migrated_schemas_are_identical(tmp_path):
    fresh_home = tmp_path / "fresh"
    conn = db.connect(fresh_home)
    fresh = _structure(conn)
    assert db.get_meta(conn, "schema_version") == str(db.SCHEMA_VERSION)
    conn.close()

    # Build a v1 database: fresh v2 minus everything the v2 delta added.
    old_home = tmp_path / "old"
    conn = db.connect(old_home)
    for name in _migration_objects():
        kind = "INDEX" if name.startswith("idx_") else "TABLE"
        conn.execute(f"DROP {kind} IF EXISTS {name}")
    db.set_meta(conn, "schema_version", "1")
    conn.commit()
    assert not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='proposals'").fetchone()
    conn.close()

    # Reopening must migrate it forward to an identical structure.
    conn = db.connect(old_home)
    assert db.get_meta(conn, "schema_version") == str(db.SCHEMA_VERSION)
    assert _structure(conn) == fresh, (
        "schema.sql and store/migrations/*.sql have drifted: a fresh database "
        "and a migrated one no longer match"
    )
    conn.close()


def test_migration_takes_a_backup_first(tmp_path):
    """critique item 34: VACUUM INTO backup before migrating."""
    home = tmp_path / "h"
    conn = db.connect(home)
    for name in _migration_objects():
        kind = "INDEX" if name.startswith("idx_") else "TABLE"
        conn.execute(f"DROP {kind} IF EXISTS {name}")
    db.set_meta(conn, "schema_version", "1")
    conn.commit()
    conn.close()

    conn = db.connect(home)
    conn.close()
    backups = list(db.brain_dir(home).glob("brain.pre-v*.db"))
    assert backups, "no pre-migration backup was written"


def test_migration_is_idempotent(tmp_path):
    """A re-run (crash between executescript and the version stamp) is a no-op."""
    conn = db.connect(tmp_path / "h")
    before = _structure(conn)
    for version in range(2, db.SCHEMA_VERSION + 1):
        db.MIGRATIONS[version](conn)
    conn.commit()
    assert _structure(conn) == before
    conn.close()


def test_refuses_to_open_a_future_database(tmp_path):
    """An older plugin must never touch a newer brain.db (critique item 34)."""
    home = tmp_path / "h"
    conn = db.connect(home)
    db.set_meta(conn, "schema_version", str(db.SCHEMA_VERSION + 1))
    conn.commit()
    conn.close()
    with pytest.raises(db.FutureSchemaError):
        db.connect(home)
