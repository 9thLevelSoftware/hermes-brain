"""Regression contract for TTL enforcement and extraction retention policy."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from brain import tools
from brain.capture.retention import RetentionError, classify_retention
from brain.config import DEFAULTS
from brain.memfs import _rows_for
from brain.recall import lane1
from brain.recall.blend import blend
from brain.recall.facts_leg import facts_leg
from brain.recall.search import search
from brain.store import db, facts
from brain.store.lifecycle import (
    apply_legacy_retention,
    current_memory_predicate,
    expire_due,
)
from conftest import seed_memory

PAST = "2000-01-01T00:00:00.000Z"
FUTURE = "2999-01-01T00:00:00.000Z"


def _ctx(tmp_home):
    return tools.ToolContext(
        session_id="lifecycle-test",
        principal_id="owner",
        trust_tier="owner",
        platform="cli",
        config=dict(DEFAULTS),
        hermes_home=str(tmp_home),
    )


def test_current_predicate_has_one_authoritative_ttl_rule():
    predicate = current_memory_predicate("m")
    assert "m.valid_to IS NULL" in predicate
    assert "m.status = 'active'" in predicate
    assert "m.live = 1" in predicate
    assert "m.ttl_at IS NULL OR m.ttl_at >" in predicate


def test_due_memory_is_hidden_from_every_current_truth_surface(conn, tmp_home):
    expired = seed_memory(
        conn, "walnut deployment is currently blocked", kind="fact", pinned=1,
    )
    conn.execute("UPDATE memories SET ttl_at=? WHERE id=?", (PAST, expired))
    facts.add_fact(
        conn, "walnut deployment", "status", "blocked", memory_id=expired,
    )
    conn.commit()

    assert not [h for h in search(
        conn, "walnut deployment", include_episodes=False, graph=False, facts=False,
    ) if h.id == expired]
    assert expired not in [h.id for h in blend(
        conn, "walnut deployment", include_episodes=False,
    )]
    assert expired not in facts_leg(conn, "walnut")
    assert facts.query_facts(conn, subject="walnut deployment") == []

    rows = _rows_for(conn, "profile", "", _ctx(tmp_home))
    assert expired not in [row["id"] for row in rows]

    lane1.materialize(conn, {})
    assert "walnut deployment" not in lane1.render(conn, 1000)

    # Query-time filtering is authoritative even before expiry maintenance:
    # an identical remember creates a fresh current row instead of reinforcing
    # the due one.
    out = json.loads(tools.dispatch(
        conn,
        "brain_remember",
        {"content": "walnut deployment is currently blocked", "kind": "fact"},
        ctx=_ctx(tmp_home),
    ))
    assert out["deduped_against"] is None
    assert conn.execute(
        "SELECT count(*) FROM memories WHERE content_hash=?",
        (db.content_hash("walnut deployment is currently blocked"),),
    ).fetchone()[0] == 2


def test_expiry_closes_indexes_and_preserves_history(conn, monkeypatch):
    target = seed_memory(conn, "temporary release train", kind="warning")
    peer = seed_memory(conn, "release train owner", kind="profile", pinned=1)
    conn.execute(
        "UPDATE memories SET ttl_at=?, meta=? WHERE id=?",
        (FUTURE, json.dumps({
            "retention": "temporary", "expiry_source": "operational_default",
        }), target),
    )
    facts.add_fact(conn, "release train", "state", "temporary", memory_id=target)
    now = db.iso_now()
    conn.execute(
        "INSERT INTO edges (src_id,dst_id,edge_type,created_by,valid_from,recorded_at)"
        " VALUES (?,?,?,?,?,?)",
        (target, peer, "related_to", "test", now, now),
    )
    conn.commit()
    lane1.materialize(conn, {})
    conn.execute("UPDATE memories SET ttl_at=? WHERE id=?", (PAST, target))
    conn.commit()

    removed_vectors: list[tuple[str, int]] = []
    monkeypatch.setattr("brain.store.vec.vec_available", lambda _conn: True)
    monkeypatch.setattr(
        "brain.store.vec.delete",
        lambda _conn, table, row_id: removed_vectors.append((table, row_id)),
    )
    mem_gen = int(db.get_meta(conn, "mem_generation"))
    graph_gen = int(db.get_meta(conn, "graph_generation"))
    closed_at = "2026-08-01T12:00:00.000Z"

    result = expire_due(conn, now=closed_at, actor="test:lifecycle")

    assert result == {"expired": 1, "ids": [target]}
    row = conn.execute("SELECT * FROM memories WHERE id=?", (target,)).fetchone()
    assert row["status"] == "expired" and row["valid_to"] == closed_at
    assert row["content"] == "temporary release train"
    assert conn.execute(
        "SELECT valid_until FROM facts WHERE memory_id=?", (target,),
    ).fetchone()[0] == closed_at
    assert conn.execute(
        "SELECT valid_to FROM edges WHERE src_id=?", (target,),
    ).fetchone()[0] == closed_at
    assert conn.execute(
        "SELECT count(*) FROM lane1_snapshot WHERE memory_id=?", (target,),
    ).fetchone()[0] == 0
    assert removed_vectors == [("mem_vec", target)]
    assert int(db.get_meta(conn, "mem_generation")) == mem_gen + 1
    assert int(db.get_meta(conn, "graph_generation")) == graph_gen + 1

    audit = conn.execute(
        "SELECT detail FROM audit_log WHERE action='memory_expired' AND target=?",
        (row["uid"],),
    ).fetchone()
    detail = json.loads(audit["detail"])
    assert detail["retention"] == "temporary"
    assert detail["expiry_source"] == "operational_default"
    assert detail["ttl_at"] == PAST
    assert expire_due(conn, now=closed_at) == {"expired": 0, "ids": []}


def test_deterministic_retention_policy():
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)

    assert classify_retention(
        "Created commit abc123 and pushed it", "fact", now=now,
    ).retention == "episode_only"

    operational = classify_retention(
        "PR #42 is currently waiting for CI", "fact", now=now,
    )
    assert operational.retention == "temporary"
    assert operational.ttl_days == 7
    assert operational.expiry_source == "operational_default"

    deploys = classify_retention(
        "Staging deploys to prod-box-9", "fact", now=now,
    )
    assert deploys.retention == "temporary"
    assert deploys.ttl_days == 7

    dated = classify_retention(
        "The deployment freeze runs through 2026-08-12", "fact", now=now,
    )
    assert dated.retention == "temporary"
    assert dated.ttl_at == "2026-08-13T00:00:00.000Z"
    assert dated.expiry_source == "explicit_end_date"

    dated_warning = classify_retention(
        "Rollback warning applies through 2026-08-12", "warning", now=now,
    )
    assert dated_warning.retention == "temporary"
    assert dated_warning.ttl_at == "2026-08-13T00:00:00.000Z"

    for kind in ("preference", "warning", "profile", "decision"):
        durable = classify_retention(
            "Currently use concise answers", kind, now=now,
        )
        assert durable.retention == "durable" and durable.ttl_at is None

    explicit = classify_retention(
        "Use verbose answers during the launch", "preference",
        requested="temporary", ttl_days=5, now=now,
    )
    assert explicit.retention == "temporary" and explicit.ttl_days == 5
    assert explicit.expiry_source == "extractor_ttl"

    for invalid in (0, -1, 366, 1.5, "7", True):
        with pytest.raises(RetentionError):
            classify_retention(
                "temporary state", "fact", requested="temporary",
                ttl_days=invalid, now=now,
            )


def test_legacy_pass_only_stamps_high_precision_operational_rows(conn):
    operational = seed_memory(conn, "PR #91 is waiting for CI", kind="fact")
    durable = seed_memory(conn, "User prefers terse answers", kind="preference")

    result = apply_legacy_retention(
        conn, now=datetime(2026, 7, 31, 12, tzinfo=UTC), actor="test:legacy",
    )

    assert result["legacy_ids"] == [operational]
    op = conn.execute(
        "SELECT ttl_at, meta FROM memories WHERE id=?", (operational,),
    ).fetchone()
    assert op["ttl_at"] == "2026-08-07T12:00:00.000Z"
    assert json.loads(op["meta"])["expiry_source"] == "legacy_operational_pattern"
    stable = conn.execute(
        "SELECT ttl_at, meta FROM memories WHERE id=?", (durable,),
    ).fetchone()
    assert stable["ttl_at"] is None and stable["meta"] is None
