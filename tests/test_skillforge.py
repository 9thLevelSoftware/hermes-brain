"""Skill-forge: detect -> draft -> validate -> promote, and curator safety.

The headline test (test_three_session_trajectory_forges_a_skill) is the
plan's P5 acceptance for skills: a repeated 3-session task pattern produces a
validated draft that auto-approves into the skills tree with a curator-safe
usage record. Uses the stub embedder so clustering is real cosine behavior.
"""

from __future__ import annotations

import json

import pytest
from brain.skillforge import forge_once, promote_draft, skilltree
from brain.store import db


@pytest.fixture
def forge_env(tmp_home):
    pytest.importorskip("sqlite_vec")
    from brain.recall.embed import StubEmbedder
    from brain.store import vec as vec_store

    conn = db.connect(tmp_home)
    emb = StubEmbedder()
    if not vec_store.ensure_tables(conn, emb.dim, emb.name):
        pytest.skip("sqlite-vec not loadable")
    yield conn, emb, tmp_home
    conn.close()


def _seed_case(conn, emb, text, verdict, *, session_id="s"):
    """A case-bank row (kind='case') with an embedding, like dream/cases writes."""
    from brain.capture.symbols import symbols_field
    from brain.store import vec as vec_store

    now = db.iso_now()
    uid = db.new_ulid()
    meta = json.dumps({"verdict": verdict, "session_id": session_id})
    cur = conn.execute(
        "INSERT INTO memories (uid, epistemic, memory_type, kind, status, live,"
        " content, summary, content_hash, symbols, tags, token_len, trust_tier,"
        " created_by, scope_user, valid_from, recorded_at, importance, meta)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uid, "observation", "episodic", "case", "active", 1, text, text,
         db.content_hash(uid), symbols_field(text), "[]", db.approx_tokens(text),
         "agent", "distillation", None, now, now, 0.5, meta))
    mid = cur.lastrowid
    vec_store.upsert(conn, "mem_vec", mid, emb.encode_documents([text])[0])
    conn.commit()
    return mid


def _fake_llm(prompt, *, system=None, tier=None):
    """Draft FROM the cluster, like a real LLM: pull the member-case lines out
    of the prompt so the draft shares the cluster's vocabulary (the replay
    gate embeds the whole draft against the cluster)."""
    import re

    lines = [ln[2:].strip() for ln in prompt.splitlines() if ln.startswith("- ")]
    # Strip the "(verdict) " prefix each member line carries.
    bodies = [re.sub(r"^\([a-z]+\)\s*", "", ln) for ln in lines]
    topic = bodies[0] if bodies else "the task"
    return {
        "name": "forged-" + re.sub(r"[^a-z]+", "-", topic.lower())[:24].strip("-"),
        "description": ("Checklist for " + topic)[:60],
        "when_and_why": f"Any task like: {topic}. The successful runs share a "
                        f"repeatable approach worth capturing.",
        "procedure": "1. " + " 2. ".join(bodies[:3]),
        "exemplar": "A failure was fixed on a later run: " + topic,
    }


def test_three_session_trajectory_forges_a_skill(forge_env):
    """Monday failure, Tuesday + Thursday success on the same task class -> a
    validated draft skill that activates. The learning flywheel, end to end."""
    conn, emb, home = forge_env
    text = "deploy the payments service to staging with a db migration"
    _seed_case(conn, emb, text + " (attempt 1)", "failure", session_id="mon")
    _seed_case(conn, emb, text + " (attempt 2)", "success", session_id="tue")
    _seed_case(conn, emb, text + " (attempt 3)", "success", session_id="thu")

    result = forge_once(conn, {"hermes_home": str(home), "skill_auto_approve": True},
                        embedder=emb, shift_id="sh1", llm_call=_fake_llm)

    assert result.get("candidates") == 1, result
    assert result["outcome"] == "promoted", result
    assert result["validation"]["passed"]
    assert result["validation"]["gates"]["statistical"]["passed"]
    assert result["validation"]["gates"]["replay"]["passed"]

    # The SKILL.md is in the skills tree and is loadable (valid frontmatter).
    name = result["drafted"]
    skill_md = skilltree.skills_root(home) / name / "SKILL.md"
    assert skill_md.exists()
    content = skill_md.read_text(encoding="utf-8")
    assert content.startswith("---")
    assert f"name: {name}" in content
    assert "created_by: hermes-brain" in content
    assert "## When and why" in content

    # The proposal is applied and the archive record is preserved.
    prop = conn.execute("SELECT status FROM proposals WHERE kind='skill_draft'"
                        ).fetchone()
    assert prop["status"] == "applied"


def test_promoted_skill_is_curator_safe(forge_env):
    """The usage record keeps the skill out of the curator's auto-walk and
    pins it — a monthly-cadence skill can't be archived before its 2nd use."""
    conn, emb, home = forge_env
    text = "rotate the production api credentials safely"
    for i, v in enumerate(["failure", "success", "success"]):
        _seed_case(conn, emb, f"{text} run {i}", v, session_id=f"s{i}")
    forge_once(conn, {"hermes_home": str(home), "skill_auto_approve": True},
               embedder=emb, shift_id="sh2", llm_call=_fake_llm)

    usage = skilltree.read_usage(home)
    assert usage, "a .usage.json record must be written"
    rec = next(iter(usage.values()))
    # created_by != 'agent' => the curator's automatic archival walk skips it.
    assert rec["created_by"] == "hermes-brain"
    assert rec["pinned"] is True
    assert rec["created_at"] and rec["last_patched_at"]


def test_auto_approve_off_routes_to_review_queue(forge_env):
    conn, emb, home = forge_env
    text = "configure the ci pipeline caching for the monorepo"
    for i, v in enumerate(["success", "success", "success"]):
        _seed_case(conn, emb, f"{text} attempt {i}", v, session_id=f"s{i}")
    result = forge_once(conn, {"hermes_home": str(home), "skill_auto_approve": False},
                        embedder=emb, shift_id="sh3", llm_call=_fake_llm)

    assert result["outcome"] == "awaiting_approval"
    # Not on disk yet — the draft lives outside the skills tree.
    assert not (skilltree.skills_root(home) / result["drafted"] / "SKILL.md").exists()
    assert (skilltree.drafts_root(home) / result["drafted"] / "SKILL.md").exists()
    # CLI approval promotes it.
    uid = conn.execute("SELECT uid FROM proposals WHERE kind='skill_draft'"
                       ).fetchone()["uid"]
    promo = promote_draft(conn, {"hermes_home": str(home)}, uid)
    assert promo["promoted"]
    assert (skilltree.skills_root(home) / result["drafted"] / "SKILL.md").exists()


def test_no_candidate_without_enough_successes(forge_env):
    conn, emb, home = forge_env
    text = "debug the flaky integration test suite"
    # 3 similar cases but only ONE success -> below the _MIN_SUCCESS bar.
    for i, v in enumerate(["failure", "failure", "success"]):
        _seed_case(conn, emb, f"{text} {i}", v, session_id=f"s{i}")
    result = forge_once(conn, {"hermes_home": str(home)}, embedder=emb,
                        shift_id="sh4", llm_call=_fake_llm)
    assert result.get("candidates", 0) == 0


def test_dedup_against_existing_skill(forge_env):
    conn, emb, home = forge_env
    text = "provision a new postgres read replica"
    for i, v in enumerate(["success", "success", "success"]):
        _seed_case(conn, emb, f"{text} {i}", v, session_id=f"s{i}")
    # A live skill already covers this theme (same text => high cosine).
    existing = skilltree.skills_root(home) / "pg-replica"
    existing.mkdir(parents=True)
    (existing / "SKILL.md").write_text(
        skilltree.build_skill_md("pg-replica", text, "## Procedure\n\nsteps"),
        encoding="utf-8")
    result = forge_once(conn, {"hermes_home": str(home)}, embedder=emb,
                        shift_id="sh5", llm_call=_fake_llm)
    assert result.get("candidates", 0) == 0, "covered cluster must route to revision"


def test_name_collision_with_bundled_is_refused(forge_env):
    conn, emb, home = forge_env
    # Pre-seed a bundled manifest claiming the exact name the LLM will propose.
    root = skilltree.skills_root(home)
    root.mkdir(parents=True, exist_ok=True)
    (root / ".bundled_manifest").write_text(
        "collision-skill:abc123\n", encoding="utf-8")

    def fixed_name_llm(prompt, *, system=None, tier=None):
        return {"name": "collision-skill", "description": "x",
                "when_and_why": "y", "procedure": "z", "exemplar": "e"}

    text = "deploy the payments service to staging with a db migration"
    for i, v in enumerate(["failure", "success", "success"]):
        _seed_case(conn, emb, f"{text} {i}", v, session_id=f"s{i}")
    result = forge_once(conn, {"hermes_home": str(home), "skill_auto_approve": True},
                        embedder=emb, shift_id="sh6", llm_call=fixed_name_llm)
    # The draft name is unavailable (bundled) -> skipped at draft, never on disk.
    assert result.get("outcome") != "promoted"
    assert not (root / "collision-skill" / "SKILL.md").exists()


def test_forge_skips_without_vectors(tmp_home):
    conn = db.connect(tmp_home)
    try:
        result = forge_once(conn, {"hermes_home": str(tmp_home)},
                            embedder=None, shift_id="s")
        assert result.get("skipped") == "no_vectors"
    finally:
        conn.close()
