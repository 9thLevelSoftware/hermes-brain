"""Store-layer tests: schema creation, id/hash helpers, forward-version gate.

Column and table names come straight from store/schema.sql (the law);
helper behavior from store/db.py docstrings and critique items 19/34.
"""

from __future__ import annotations

import time

import pytest
from brain.store import db
from brain.store.db import FutureSchemaError

# Every non-virtual table schema.sql v1 creates.
EXPECTED_TABLES = {
    "meta", "identities", "episodes", "memories", "edges", "entities",
    "entity_mentions", "ingest_buffer", "retrieval_log", "lane1_snapshot",
    "brain_lease", "activity", "sweep_state", "shift_runs", "strategy_state",
    "llm_ledger", "work_queue", "audit_log",
}


def _table_names(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {row["name"] for row in rows}


def test_schema_creates_all_tables(conn):
    names = _table_names(conn)
    missing = EXPECTED_TABLES - names
    assert not missing, f"schema.sql did not create: {sorted(missing)}"
    if db.capabilities(conn).get("fts5"):
        assert {"episode_fts", "memory_fts"} <= names


def test_brain_lease_seed_rows(conn):
    rows = conn.execute("SELECT name, holder FROM brain_lease ORDER BY name").fetchall()
    assert [row["name"] for row in rows] == ["dream", "sweep"]
    assert all(row["holder"] is None for row in rows)


def test_ulids_unique_26_chars():
    ids = [db.new_ulid() for _ in range(500)]
    assert len(set(ids)) == 500
    assert all(len(uid) == 26 for uid in ids)
    alphabet = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")
    assert all(set(uid) <= alphabet for uid in ids)


def test_ulids_lexically_increase_across_ms_boundary():
    first = db.new_ulid()
    time.sleep(0.003)  # guarantee a new millisecond
    second = db.new_ulid()
    assert second > first


def test_content_hash_normalization():
    assert db.content_hash("Hello   World") == db.content_hash("hello world")
    assert db.content_hash("  hello\tWORLD \n") == db.content_hash("hello world")
    assert db.content_hash("hello world") != db.content_hash("hello worlds")


def test_future_schema_refused(tmp_home):
    conn = db.connect(tmp_home)
    conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(FutureSchemaError):
        db.connect(tmp_home)


def test_capabilities_probe_records_fts5(conn):
    caps = db.capabilities(conn)
    assert "fts5" in caps
    assert isinstance(caps["fts5"], bool)


def test_reconnect_idempotent(tmp_home):
    first = db.connect(tmp_home)
    created_at = db.get_meta(first, "created_at")
    tables_before = _table_names(first)
    first.close()

    second = db.connect(tmp_home)  # must not raise duplicate-table errors
    assert db.get_meta(second, "created_at") == created_at
    assert _table_names(second) == tables_before
    second.close()
