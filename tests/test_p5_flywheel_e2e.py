"""The learning flywheel, end to end — the plan's acceptance script (§5).

One concrete week, driven through the real modules (stub embedder for
determinism, injected LLM for the distill/draft calls):

  Mon   a task FAILS; the episode is captured as a case.
  Tue   the same task SUCCEEDS; another case.
  Thu   it succeeds again.
  Night the dream distills a guardrail from the failure, banks the cases,
        and the skill-forge drafts + auto-approves a skill from the cluster.
  Next  the guardrail is injected into a similar planning turn (lane 2),
        the skill is live in Hermes's tree, and mining credits a memory that
        helped a verified turn.

This is the "not just dinky .md files" claim, mechanized and asserted.
"""

from __future__ import annotations

import json
import time

import pytest
from brain import llm as brain_llm
from brain.store import db


@pytest.fixture
def week(tmp_home):
    pytest.importorskip("sqlite_vec")
    from brain.recall.embed import StubEmbedder
    from brain.store import vec as vec_store

    conn = db.connect(tmp_home)
    emb = StubEmbedder()
    if not vec_store.ensure_tables(conn, emb.dim, emb.name):
        pytest.skip("sqlite-vec not loadable")
    yield conn, emb, tmp_home
    conn.close()


def _state_db(home, rows):
    """rows: list of (session, turn_id, epoch, outcome, user_msg)."""
    from tests.test_episode_assembly import make_state_db

    messages, outcomes = [], []
    for sid, tid, ts, outcome, msg in rows:
        messages.append((sid, "user", msg, ts - 5))
        messages.append((sid, "assistant", f"handled: {msg}", ts - 2))
        outcomes.append((sid, tid, ts, outcome, None, None))
    make_state_db(home, messages=messages, outcomes=outcomes)


def _append_state(home, rows):
    """Insert more turns into an EXISTING state.db (no schema recreate)."""
    import sqlite3

    state = sqlite3.connect(str(home / "state.db"))
    try:
        for sid, tid, ts, outcome, msg in rows:
            state.execute("INSERT OR IGNORE INTO sessions (id, source, started_at)"
                          " VALUES (?,?,?)", (sid, "cli", ts - 60))
            state.execute("INSERT INTO messages (session_id, role, content, timestamp)"
                          " VALUES (?,?,?,?)", (sid, "user", msg, ts - 5))
            state.execute("INSERT INTO turn_outcomes (session_id, turn_id, created_at,"
                          " outcome) VALUES (?,?,?,?)", (sid, tid, ts, outcome))
        state.commit()
    finally:
        state.close()


_GUARDRAIL = json.dumps({
    "kind": "guardrail",
    "title": "Dry-run the DB migration before a staging deploy",
    "insight": "The staging deploy failed when the migration ran live against "
               "a bad column; re-running after a dry-run caught it and the "
               "deploy succeeded. Always dry-run the migration first.",
    "scope": "migration", "actionable": True,
})

# Deterministic case summary: the three attempts are the SAME task, so a fixed
# summary makes their (stub) embeddings identical and they cluster — the test
# must not depend on whether a real LLM is reachable to paraphrase each turn
# (a real embedder would cluster the paraphrases; the stub embedder can't).
_CASE = json.dumps({
    "summary": "deploy the payments service to staging with a schema migration",
    "plan": "1. dry-run the migration 2. deploy 3. smoke test",
})


def _skill_draft(prompt, *, system=None, tier=None):
    # Draft from the cluster text so the replay gate (embed vs cluster) passes.
    import re

    bodies = [re.sub(r"^\([a-z]+\)\s*", "", ln[2:].strip())
              for ln in prompt.splitlines() if ln.startswith("- ")]
    topic = bodies[0] if bodies else "task"
    return {
        "name": "staging-migration-deploy",
        "description": "Dry-run migrations before staging deploys",
        "when_and_why": f"Tasks like: {topic}. A dry-run catches schema errors "
                        "before they take staging down.",
        "procedure": "1. " + " 2. ".join(bodies[:3]),
        "exemplar": "Migration failed live; a dry-run next time caught it.",
    }


def test_one_concrete_week(week):
    conn, emb, home = week
    now = time.time()
    task = "deploy the payments service to staging with a schema migration"

    # -- the week's task history in Hermes's ledger --
    _state_db(home, [
        ("mon", "t-mon", now - 5 * 86400, "failed", task + " (attempt 1)"),
        ("tue", "t-tue", now - 4 * 86400, "verified", task + " (attempt 2)"),
        ("thu", "t-thu", now - 2 * 86400, "verified", task + " (attempt 3)"),
    ])

    cfg = {"hermes_home": str(home), "skill_auto_approve": True}

    from brain.dream import cases as cases_mod
    from brain.dream import distill as distill_mod
    from brain.dream.lease import acquire
    from brain.dream.shift import Shift

    def shift(mode):
        acquire(conn, "dream", "wk")
        return Shift(shift_id=db.new_ulid(), conn=conn,
                     config={**cfg, "_forced_mode": mode}, embedder=emb,
                     started_at=db.iso_now(),
                     activity_baseline="9999-12-31T00:00:00.000Z", holder="wk")

    # -- night: bank the cases (deterministic summary so the run is hermetic) --
    brain_llm.set_llm_for_tests(lambda p, *, system=None, max_tokens=0: _CASE)
    try:
        cases_result = cases_mod.run(shift("active"))
    finally:
        brain_llm.set_llm_for_tests(None)
    assert cases_result["written"] == 3, cases_result
    n_cases = conn.execute("SELECT count(*) AS n FROM memories WHERE kind='case'"
                          ).fetchone()["n"]
    assert n_cases == 3

    # -- night: distill a guardrail from the failure→success trajectory --
    brain_llm.set_llm_for_tests(lambda p, *, system=None, max_tokens=0: _GUARDRAIL)
    try:
        distilled = distill_mod.run(shift("active"))
    finally:
        brain_llm.set_llm_for_tests(None)
    assert distilled["distilled"] == 1, distilled
    guardrail = conn.execute(
        "SELECT id, uid, content FROM memories WHERE kind='guardrail' AND "
        "memory_type='procedural'").fetchone()
    assert guardrail is not None
    assert "dry-run" in guardrail["content"].lower()

    # -- night: forge + auto-approve a skill from the case cluster --
    from brain.skillforge import forge_once, skilltree

    forged = forge_once(conn, cfg, embedder=emb, shift_id="wk", llm_call=_skill_draft)
    assert forged["outcome"] == "promoted", forged
    skill_md = skilltree.skills_root(home) / forged["drafted"] / "SKILL.md"
    assert skill_md.exists()
    assert "created_by: hermes-brain" in skill_md.read_text(encoding="utf-8")
    # curator-safe usage record.
    rec = next(iter(skilltree.read_usage(home).values()))
    assert rec["created_by"] == "hermes-brain" and rec["pinned"] is True

    # -- next week: the guardrail is injected into a similar planning turn --
    from brain.recall.strategies import retrieve_guidance

    guidance = retrieve_guidance(
        conn, "deploy the payments service to staging", embedder=emb)
    assert any(g.id == guardrail["id"] and g.kind == "guardrail" for g in guidance), \
        "the learned guardrail must be injected into a similar task turn"

    # -- the loop closes: an injected memory that helped a verified turn --
    from brain.dream.mine_state import run as mine_run
    from brain.recall.search import log_retrieval, stamp_pending_injections

    class _Hit:
        kind = "memory"

        def __init__(self, mid, uid):
            self.id, self.uid, self.source, self.score = mid, uid, "guidance", 1.0

    # A future turn injects the guardrail, then ends 'verified' in state.db.
    fut = now - 86400
    _append_state(home, [("fri", "t-fri", fut, "verified",
                          "deploy payments to staging again")])
    log_retrieval(conn, "fri", "deploy payments",
                  [_Hit(guardrail["id"], guardrail["uid"])], {guardrail["uid"]})
    stamp_pending_injections(conn, "fri", 1, "deploy payments to staging again")
    conn.execute("UPDATE retrieval_log SET ts=? WHERE session_id='fri'",
                 (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(fut - 5)) + ".000Z",))
    conn.commit()

    mined = mine_run(shift("active"))
    assert mined["helpful"] >= 1, mined
    helped = conn.execute("SELECT helpful_count FROM memories WHERE id=?",
                          (guardrail["id"],)).fetchone()["helpful_count"]
    assert helped >= 1, "the guardrail that helped a verified turn must be credited"
