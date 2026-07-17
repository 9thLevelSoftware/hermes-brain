"""Regression tests for the confirmed P4 dream-cycle review findings.

The lease-safety cluster (#3/#4/#6) is the load-bearing invariant — exactly
one dream mutates memory at a time — so it gets the most coverage.
"""

from __future__ import annotations

import pytest
from brain import llm as brain_llm
from brain.dream import lease, run_dream
from brain.dream.shift import Shift
from brain.store import db
from conftest import seed_memory

# -- finding #6 (BLOCKER): a holder that lost the lease must stop -------------

def test_keepalive_yields_when_lease_lost(conn):
    lease.acquire(conn, "dream", "holderA")
    shift = Shift(shift_id="s", conn=conn, config={}, holder="holderA")
    assert shift.keepalive() is True          # we hold it
    # Another process steals it (expire + take over).
    conn.execute("UPDATE brain_lease SET holder='holderB', "
                 "expires_at='2099-01-01T00:00:00.000Z' WHERE name='dream'")
    conn.commit()
    shift._last_renew = 0.0                    # force a real renew attempt
    assert shift.keepalive() is False          # lost -> must yield
    assert shift.preempted() is True
    assert shift.tick() is False


# -- finding #4: the lease is released even if setup raises -------------------

def test_lease_released_on_setup_error(conn, monkeypatch):
    from brain.dream import run as run_mod

    def boom(*a, **k):
        raise RuntimeError("open_shift blew up")

    monkeypatch.setattr(run_mod, "_open_shift", boom)
    out = run_dream(conn, {}, actor="test")        # never raises
    assert "error" in out
    assert lease.held_by(conn, "dream") is None    # released despite the error


# -- finding #5: dream-wide dry_run must not let flush WRITE memories ---------

def test_dry_run_flush_writes_nothing(conn):
    # Buffer a capturable turn so flush would extract if it ran active.
    from brain.capture.turns import TurnContext, capture_turn

    eid = capture_turn(conn, TurnContext(session_id="s", trust_tier="owner"),
                       "remember: the prod region is us-east-1", "noted")
    conn.execute("UPDATE episodes SET salience=0.9 WHERE id=?", (eid,))
    conn.execute("INSERT INTO ingest_buffer(kind,session_id,payload,ts) "
                 "VALUES('session_end_marker','s','{}',?)", (db.iso_now(),))
    conn.commit()
    before = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]

    brain_llm.set_llm_for_tests(lambda p, *, system=None, max_tokens=0: '[{"content":'
                                '"prod region is us-east-1","kind":"fact",'
                                '"about_user":false,"time_sensitive":false,'
                                '"instruction_shaped":false,"source_uids":[]}]')
    try:
        out = run_dream(conn, {}, dry_run=True, actor="test")
    finally:
        brain_llm.set_llm_for_tests(None)
    after = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
    assert after == before                     # dry_run wrote no memory
    assert out["strategies"]["flush"]["mode"] == "dry_run"


# -- finding #1: outcome reflects the real result, not always 'completed' ----

def test_close_shift_derives_outcome(conn):
    from brain.dream.run import _close_shift, _open_shift

    def outcome_for(summary):
        sid, _, _ = _open_shift(conn, None, "test")
        _close_shift(conn, sid, summary)
        return conn.execute(
            "SELECT outcome FROM shift_runs WHERE shift_id=?", (sid,)).fetchone()["outcome"]

    assert outcome_for({"strategies": {"a": {"mode": "active"}}}) == "completed"
    assert outcome_for({"strategies": {"a": {"skipped": "preempted"}}}) == "preempted"
    assert outcome_for({"strategies": {"a": {"error": "boom"}}}) == "failed"
    assert outcome_for({"strategies": {}, "aborted": "lease_lost"}) == "aborted"


# -- finding #7: consolidation never clusters across scope_user --------------

def test_consolidate_does_not_cluster_across_users(tmp_home):
    pytest.importorskip("sqlite_vec")
    from brain.dream import consolidate
    from brain.recall.embed import StubEmbedder
    from brain.store import vec as vec_store

    conn = db.connect(tmp_home)
    emb = StubEmbedder()
    vec_store.ensure_tables(conn, emb.dim, emb.name)
    text = "the deploy pipeline fails above 2GB heap"
    ids = []
    for scope in ("userA", "userB", "userA", "userB", "userA", "userB"):
        mid = seed_memory(conn, text, kind="warning", memory_type="semantic",
                          created_by="extraction", trust_tier="known_user")
        conn.execute("UPDATE memories SET scope_user=? WHERE id=?", (scope, mid))
        vec_store.upsert(conn, "mem_vec", mid, emb.encode_documents([text])[0])
        ids.append(mid)
    conn.commit()

    lease.acquire(conn, "dream", "test")
    captured = {}
    brain_llm.set_llm_for_tests(lambda p, *, system=None, max_tokens=0: '{}')
    try:
        shift = Shift(shift_id="s", conn=conn, config={"_forced_mode": "active"},
                      embedder=emb, holder="test")
        # Cluster directly and assert every cluster is single-scope.
        cands = consolidate._candidates(conn)
        blobs = consolidate._blobs(conn, [c["id"] for c in cands])
        for members in consolidate._cluster(shift, cands, blobs):
            scopes = {m["scope_user"] for m in members}
            captured[len(members)] = scopes
            assert len(scopes) == 1, f"cross-scope cluster: {scopes}"
    finally:
        brain_llm.set_llm_for_tests(None)
        conn.close()
