"""Regression tests for the confirmed P5 adversarial-review findings.

One test per fix, each written to FAIL against the pre-fix code — the point of
the review loop is that every fixed bug leaves a tripwire behind it.
"""

from __future__ import annotations

import io
import json
import time

import pytest
from brain.store import db
from conftest import seed_memory

# ===========================================================================
# BLOCKER — MCP server must not crash on valid-JSON-but-non-object input
# ===========================================================================

def test_mcp_survives_non_object_json(tmp_home):
    from brain.mcp_server import BrainMCPServer

    stdin = io.StringIO('5\n"foo"\ntrue\nnull\n[1,2]\n[]\n')
    stdout = io.StringIO()
    rc = BrainMCPServer(str(tmp_home)).serve(stdin=stdin, stdout=stdout)  # must NOT raise
    assert rc == 0
    replies = [json.loads(x) for x in stdout.getvalue().splitlines() if x.strip()]
    # Every non-object line gets a JSON-RPC error, never a crash.
    assert replies and all("error" in r for r in replies)
    assert all(r["error"]["code"] in (-32600, -32603) for r in replies)


def test_mcp_handle_rejects_non_dict():
    from brain.mcp_server import BrainMCPServer

    s = BrainMCPServer("nowhere")
    assert s.handle(5)["error"]["code"] == -32600
    assert s.handle("x")["error"]["code"] == -32600
    assert s.handle([1, 2])["error"]["code"] == -32600  # a list member is a non-dict


# ===========================================================================
# MAJOR — tune.py must NOT propose a "decrease" on identical rates (noise)
# ===========================================================================

def test_tune_does_not_fire_on_identical_rates(conn):
    from brain.dream import tune as tune_mod
    from brain.dream.lease import acquire
    from brain.dream.shift import Shift

    # 12 pinned + 12 unpinned, ALL with an identical 0.6 helpful-rate.
    for i in range(12):
        for pinned in (1, 0):
            mid = seed_memory(conn, f"m{i}-{pinned}", pinned=pinned)
            conn.execute("UPDATE memories SET helpful_count=3, harmful_count=2 "
                         "WHERE id=?", (mid,))
    conn.commit()
    acquire(conn, "dream", "t")
    result = tune_mod.run(Shift(shift_id="s", conn=conn,
                                config={"_forced_mode": "shadow"}, holder="t"))
    # Identical rates => no confident difference => no proposal in EITHER
    # direction. The pre-fix code emitted a spurious "decrease".
    assert result.get("proposed", 0) == 0, result
    assert conn.execute("SELECT count(*) AS n FROM proposals WHERE kind='tuning'"
                        ).fetchone()["n"] == 0


def test_tune_still_fires_on_a_real_difference(conn):
    from brain.dream import tune as tune_mod
    from brain.dream.lease import acquire
    from brain.dream.shift import Shift

    for i in range(12):
        mid = seed_memory(conn, f"good{i}", pinned=1)
        conn.execute("UPDATE memories SET helpful_count=9, harmful_count=1 WHERE id=?", (mid,))
    for i in range(12):
        mid = seed_memory(conn, f"bad{i}")
        conn.execute("UPDATE memories SET helpful_count=1, harmful_count=9 WHERE id=?", (mid,))
    conn.commit()
    acquire(conn, "dream", "t")
    result = tune_mod.run(Shift(shift_id="s", conn=conn,
                                config={"_forced_mode": "shadow"}, holder="t"))
    assert result.get("proposed", 0) >= 1
    payload = json.loads(conn.execute(
        "SELECT payload FROM proposals WHERE kind='tuning'").fetchone()["payload"])
    assert any(f["direction"] == "increase" for f in payload["features"])


# ===========================================================================
# MAJOR — assemble_episodes must NOT drop a session's trailing non-terminal
# episode when a later session exists
# ===========================================================================

def test_session_end_episode_is_not_dropped(tmp_home):
    from brain.dream.mine_state import assemble_episodes, open_state_ro

    from tests.test_episode_assembly import make_state_db

    now = time.time()
    # s1 ends NON-terminal but with a thumbs-up (a real success); s2 follows.
    state = open_state_ro(make_state_db(
        tmp_home,
        messages=[("s1", "user", "refactor the auth module", now - 300),
                  ("s2", "user", "later task", now - 100)],
        outcomes=[("s1", "t-1", now - 290, "completed_unverified", "reaction", "thumbs_up"),
                  ("s2", "t-2", now - 90, "verified", None, None)],
    ))
    try:
        eps = assemble_episodes(state)
    finally:
        state.close()
    by_session = {e.session_id: e for e in eps}
    assert "s1" in by_session, "s1's session-end episode was dropped"
    assert by_session["s1"].verdict == "success"  # thumbs-up overrode the label


# ===========================================================================
# MAJOR — distill watermark must not advance past unprocessed episodes
# ===========================================================================

def test_distill_watermark_does_not_skip_on_preempt(tmp_home, monkeypatch):
    pytest.importorskip("sqlite_vec")
    from brain.dream import distill as distill_mod
    from brain.dream import mine_state
    from brain.recall.embed import StubEmbedder
    from brain.store import vec as vec_store

    from tests.test_episode_assembly import make_state_db

    conn = db.connect(tmp_home)
    emb = StubEmbedder()
    if not vec_store.ensure_tables(conn, emb.dim, emb.name):
        pytest.skip("no sqlite-vec")

    now = time.time()
    make_state_db(
        tmp_home,
        messages=[("s1", "user", "deploy to staging with migration", now - 300),
                  ("s1", "assistant", "it failed", now - 295),
                  ("s2", "user", "deploy to staging with migration again", now - 100),
                  ("s2", "assistant", "worked", now - 95)],
        outcomes=[("s1", "t-1", now - 290, "failed", None, None),
                  ("s2", "t-2", now - 90, "verified", None, None)],
    )

    from brain.dream.lease import acquire
    from brain.dream.shift import Shift

    acquire(conn, "dream", "t")
    shift = Shift(shift_id="s", conn=conn,
                  config={"hermes_home": str(tmp_home), "_forced_mode": "active"},
                  embedder=emb, started_at=db.iso_now(),
                  activity_baseline="9999-12-31T00:00:00.000Z", holder="t")
    # Preempt immediately: the very first tick yields -> ZERO episodes processed.
    monkeypatch.setattr(shift, "tick", lambda: False)
    distill_mod.run(shift)

    # The fix's intent: after processing nothing, BOTH episodes must still be
    # re-assemblable at the new watermark (the pre-fix code jumped to the batch
    # max and lost s1 forever). Verify by re-assembling at the stored watermark.
    row = conn.execute("SELECT watermark FROM sweep_state WHERE key='distill:watermark'"
                       ).fetchone()
    stored = float(json.loads(row["watermark"])["ended_at"]) if row else 0.0
    state = mine_state.open_state_ro(tmp_home / "state.db")
    try:
        again = {e.session_id for e in
                 mine_state.assemble_episodes(state, since_epoch=stored)}
    finally:
        state.close()
    assert {"s1", "s2"} <= again, f"watermark {stored} dropped unprocessed episodes: {again}"
    conn.close()


# ===========================================================================
# MAJOR — exclude_kinds must be honored on the no-FTS5 LIKE fallback
# ===========================================================================

def test_like_fallback_honors_exclude_kinds(conn, monkeypatch):
    from brain.recall import search as search_mod

    # Force the LIKE path regardless of real FTS5 support.
    monkeypatch.setattr(search_mod.db, "capabilities", lambda c: {"fts5": False})

    seed_memory(conn, "always dry-run the staging migration", kind="guardrail",
                memory_type="procedural")
    seed_memory(conn, "the staging migration runbook is in the wiki", kind="fact")
    conn.commit()

    hits = search_mod.search(conn, "staging migration", trust_tier="owner",
                             exclude_kinds=("strategy", "guardrail", "case"),
                             include_episodes=False)
    kinds = {h.mkind for h in hits}
    assert "guardrail" not in kinds, "guidance-type memory leaked into LIKE facts"
    assert "fact" in kinds


# ===========================================================================
# MAJOR — promote is retry-safe: a leftover dest from a failed promote does
# not cause a false "name unavailable" rejection
# ===========================================================================

def test_promote_is_retry_safe_after_partial_failure(tmp_home):
    pytest.importorskip("sqlite_vec")
    from brain.skillforge import forge, skilltree

    conn = db.connect(tmp_home)
    # A validated skill_draft proposal + a draft on disk.
    name = "retry-skill"
    draft_dir = skilltree.drafts_root(tmp_home) / name
    draft_dir.mkdir(parents=True)
    (draft_dir / "SKILL.md").write_text(
        skilltree.build_skill_md(name, "do the thing", "## Procedure\n\nsteps"),
        encoding="utf-8")
    uid = db.new_ulid()
    conn.execute(
        "INSERT INTO proposals (uid, kind, target, title, status, payload,"
        " validation, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (uid, "skill_draft", name, "draft", "validated",
         json.dumps({"name": name, "dir": str(draft_dir), "evidence_count": 3,
                     "success_rate": 1.0}),
         json.dumps({"passed": True}), db.iso_now()))
    conn.commit()

    # Simulate a PRIOR promote that copied the file but crashed before the DB
    # commit: the dest SKILL.md is already on disk, proposal still 'validated'.
    dest = skilltree.skills_root(tmp_home) / name
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text("stale partial", encoding="utf-8")

    res = forge.promote_draft(conn, {"hermes_home": str(tmp_home)}, uid)
    assert res.get("promoted"), res   # must NOT false-reject on its own remnant
    assert conn.execute("SELECT status FROM proposals WHERE uid=?", (uid,)
                        ).fetchone()["status"] == "applied"
    # And a genuinely different bundled name is still refused.
    conn.close()


def test_promote_still_refuses_a_bundled_collision(tmp_home):
    pytest.importorskip("sqlite_vec")
    from brain.skillforge import forge, skilltree

    conn = db.connect(tmp_home)
    name = "bundled-clash"
    root = skilltree.skills_root(tmp_home)
    root.mkdir(parents=True)
    (root / ".bundled_manifest").write_text(f"{name}:deadbeef\n", encoding="utf-8")
    draft_dir = skilltree.drafts_root(tmp_home) / name
    draft_dir.mkdir(parents=True)
    (draft_dir / "SKILL.md").write_text(
        skilltree.build_skill_md(name, "x", "## Procedure\n\ny"), encoding="utf-8")
    uid = db.new_ulid()
    conn.execute(
        "INSERT INTO proposals (uid, kind, target, title, status, payload, created_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (uid, "skill_draft", name, "draft", "validated",
         json.dumps({"name": name, "dir": str(draft_dir)}), db.iso_now()))
    conn.commit()
    res = forge.promote_draft(conn, {"hermes_home": str(tmp_home)}, uid)
    assert not res.get("promoted")
    assert "bundled" in res["reason"]
    assert not (root / name / "SKILL.md").exists()
    conn.close()


# ===========================================================================
# MINOR — CLI review with an empty uid teaches instead of crashing
# ===========================================================================

def test_skills_approve_empty_uid_does_not_crash(tmp_home, monkeypatch):
    import argparse

    from brain import cli

    monkeypatch.setenv("HERMES_HOME", str(tmp_home))
    db.connect(tmp_home).close()
    rc = cli.cmd_skills(argparse.Namespace(skills_command="approve", uid=""))
    assert rc == 1  # teaches, does not raise AttributeError
