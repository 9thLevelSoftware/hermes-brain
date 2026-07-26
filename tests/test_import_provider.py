"""Importing memory from the other Hermes memory providers.

The provider slot is exclusive, so adopting the brain means leaving whatever
you were on — and there was no way to bring memory across
(docs/design/alignment-audit.md §F5).
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from brain.bootstrap.providers import PROVIDERS, UNSUPPORTED, import_provider
from conftest import seed_memory


def build_holographic(home, facts):
    """facts: (content, category, tags, trust_score, created_at)"""
    path = home / "memory_store.db"
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE facts (fact_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " content TEXT NOT NULL UNIQUE, category TEXT DEFAULT 'general',"
        " tags TEXT DEFAULT '', trust_score REAL DEFAULT 0.5,"
        " retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0,"
        " created_at TIMESTAMP, updated_at TIMESTAMP, hrr_vector BLOB)")
    for content, category, tags, trust, created in facts:
        con.execute("INSERT INTO facts (content, category, tags, trust_score,"
                    " created_at) VALUES (?,?,?,?,?)",
                    (content, category, tags, trust, created))
    con.commit()
    con.close()
    return path


def _count(conn, where="1=1"):
    return conn.execute(f"SELECT COUNT(*) AS n FROM memories WHERE {where}").fetchone()["n"]


# ---------------------------------------------------------------------------
# holographic
# ---------------------------------------------------------------------------

def test_holographic_import_maps_categories_and_tags(conn, tmp_home):
    build_holographic(tmp_home, [
        ("The staging box is 10.0.0.7", "general", "infra,network", 0.5,
         "2026-01-02 03:04:05"),
        ("User prefers terse answers", "preference", '["style"]', 0.95, None),
        ("Never run pip inside Termux", "warning", "", 0.6, "2026-02-01"),
    ])
    counts = import_provider(conn, "holographic", hermes_home=tmp_home, apply=True)
    assert counts["imported"] == 3 and "error" not in counts

    pref = conn.execute(
        "SELECT * FROM memories WHERE content LIKE 'User prefers%'").fetchone()
    assert pref["kind"] == "preference"
    assert "style" in pref["tags"] and "holographic-import" in pref["tags"]
    # trust_score is CONFIDENCE, not provenance: a very high one pins, it does
    # not promote the row to owner trust.
    assert pref["pinned"] == 1
    assert pref["trust_tier"] == "agent"

    warn = conn.execute(
        "SELECT kind, pinned FROM memories WHERE content LIKE 'Never run pip%'").fetchone()
    assert warn["kind"] == "warning" and warn["pinned"] == 0

    infra = conn.execute(
        "SELECT valid_from FROM memories WHERE content LIKE 'The staging box%'").fetchone()
    assert infra["valid_from"].startswith("2026-01-02T03:04:05")


def test_import_never_lands_at_owner_trust(conn, tmp_home):
    """It did not come from the owner speaking to THIS brain; importing a
    foreign store at owner trust would put it straight into lane 1."""
    build_holographic(tmp_home, [("a fact from elsewhere", "general", "", 1.0, None)])
    import_provider(conn, "holographic", hermes_home=tmp_home, apply=True)
    assert _count(conn, "trust_tier='owner'") == 0
    assert _count(conn, "trust_tier='agent'") == 1


def test_import_is_dry_run_by_default(conn, tmp_home):
    build_holographic(tmp_home, [("a fact", "general", "", 0.5, None)])
    counts = import_provider(conn, "holographic", hermes_home=tmp_home)
    assert counts["imported"] == 1 and counts["applied"] is False
    assert _count(conn) == 0, "a dry run must write nothing"


def test_import_is_idempotent(conn, tmp_home):
    build_holographic(tmp_home, [("a fact worth keeping", "general", "", 0.5, None)])
    assert import_provider(conn, "holographic", hermes_home=tmp_home,
                           apply=True)["imported"] == 1
    second = import_provider(conn, "holographic", hermes_home=tmp_home, apply=True)
    assert second["imported"] == 0 and second["skipped"] == 1
    assert _count(conn) == 1


def test_import_dedups_against_existing_brain_content(conn, tmp_home):
    seed_memory(conn, "already known to the brain")
    build_holographic(tmp_home, [("already known to the brain", "general", "", 0.5, None)])
    counts = import_provider(conn, "holographic", hermes_home=tmp_home, apply=True)
    assert counts["imported"] == 0 and counts["skipped"] == 1


def test_holographic_missing_file_teaches(conn, tmp_home):
    counts = import_provider(conn, "holographic", hermes_home=tmp_home)
    assert "no such file" in counts["error"]


def test_wrong_schema_teaches(conn, tmp_home):
    path = tmp_home / "memory_store.db"
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE unrelated (x INTEGER)")
    con.commit()
    con.close()
    counts = import_provider(conn, "holographic", hermes_home=tmp_home)
    assert "no readable holographic" in counts["error"]


# ---------------------------------------------------------------------------
# jsonl — the universal path
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_home, rows):
    path = tmp_home / "export.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_jsonl_import_accepts_content_aliases(conn, tmp_home):
    path = _write_jsonl(tmp_home, [
        {"content": "first via content"},
        {"text": "second via text"},
        {"memory": "third via memory"},
    ])
    counts = import_provider(conn, "jsonl", hermes_home=tmp_home, path=path, apply=True)
    assert counts["imported"] == 3


def test_jsonl_import_honours_kind_tags_and_pinned(conn, tmp_home):
    path = _write_jsonl(tmp_home, [
        {"content": "a warning worth pinning", "kind": "warning",
         "tags": ["ops"], "pinned": True, "created_at": "2026-03-04"},
    ])
    import_provider(conn, "jsonl", hermes_home=tmp_home, path=path, apply=True)
    row = conn.execute("SELECT * FROM memories").fetchone()
    assert row["kind"] == "warning" and row["pinned"] == 1
    assert "ops" in row["tags"] and "jsonl-import" in row["tags"]
    assert row["valid_from"].startswith("2026-03-04")


def test_jsonl_unknown_kind_falls_back_to_fact(conn, tmp_home):
    path = _write_jsonl(tmp_home, [{"content": "x y z", "kind": "nonsense"}])
    import_provider(conn, "jsonl", hermes_home=tmp_home, path=path, apply=True)
    assert conn.execute("SELECT kind FROM memories").fetchone()["kind"] == "fact"


def test_jsonl_counts_malformed_lines_without_aborting(conn, tmp_home):
    path = tmp_home / "export.jsonl"
    path.write_text('{"content": "good one"}\nnot json at all\n[]\n'
                    '{"content": "another good one"}\n', encoding="utf-8")
    counts = import_provider(conn, "jsonl", hermes_home=tmp_home, path=path, apply=True)
    assert counts["imported"] == 2
    assert counts["malformed"] == 2


def test_jsonl_requires_a_path(conn, tmp_home):
    counts = import_provider(conn, "jsonl", hermes_home=tmp_home)
    assert "--path is required" in counts["error"]


# ---------------------------------------------------------------------------
# the honest refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_unsupported_providers_explain_the_route_in(conn, tmp_home, name):
    """A half-working importer that silently drops rows is worse than a
    documented export step — so these refuse AND teach."""
    counts = import_provider(conn, name, hermes_home=tmp_home)
    assert "error" in counts
    assert "jsonl" in counts["error"], "every refusal must name the way in"


def test_unknown_provider_lists_the_supported_ones(conn, tmp_home):
    counts = import_provider(conn, "totally-made-up", hermes_home=tmp_home)
    assert "unknown provider" in counts["error"]
    for name in PROVIDERS:
        assert name in counts["error"]


def test_import_never_raises_on_a_corrupt_source(conn, tmp_home):
    path = tmp_home / "memory_store.db"
    path.write_bytes(b"this is not a sqlite file at all")
    counts = import_provider(conn, "holographic", hermes_home=tmp_home)
    assert "error" in counts
