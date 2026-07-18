"""C4 skill degradation -> revision/retirement loop + B5 curator readback.

A brain-forged skill that proves net-harmful (its .usage.json shows it hurting
more turns than it helps, with statistical confidence) draws a reviewable
`skill_revision` proposal; a repeat offender (>= 2 already-rejected revisions)
draws a `skill_retire` proposal instead. The loop only ever PROPOSES — it
promotes/deletes nothing. Hermetic: every LLM step is faked via
brain.llm.set_llm_for_tests (agent is importable here, so an un-faked call
would hit a real model).
"""

from __future__ import annotations

import json

import pytest
from brain.skillforge import revise as revise_mod
from brain.skillforge import skilltree
from brain.skillforge.revise import revise_once
from brain.store import db

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def env(tmp_home):
    conn = db.connect(tmp_home)
    yield conn, tmp_home
    conn.close()


@pytest.fixture(autouse=True)
def _reset_llm():
    from brain import llm
    yield
    llm.set_llm_for_tests(None)


def _install_llm(fn):
    from brain import llm
    llm.set_llm_for_tests(fn)


def _good_revise_llm(prompt, *, system=None, max_tokens=None):
    """A well-formed revision delta, as the consolidate tier would return."""
    return json.dumps({
        "diagnosis": "the trigger is too broad, so it loads on unrelated tasks",
        "sections": [{"heading": "## Procedure",
                      "new_text": "1. confirm the task matches before applying"}],
        "summary": "narrow the trigger; add a match check to step 1",
    })


def _empty_llm(prompt, *, system=None, max_tokens=None):
    """Empty text -> llm.call_text raises LLMUnavailable (budget/degrade path)."""
    return ""


def _seed_skill(home, name, *, helped, hurt, neutral=0, created_by="hermes-brain"):
    """Write a SKILL.md (frontmatter provenance = `created_by`) plus a
    matching .usage.json outcome tally, like a forged skill in service."""
    md = skilltree.build_skill_md(
        name, f"do the {name} task", "## When and why\n\nx\n\n## Procedure\n\n1. y",
        frontmatter_extra={"created_by": created_by, "evidence_count": 3})
    root = skilltree.skills_root(home) / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(md, encoding="utf-8")
    skilltree.write_usage_record(home, name, {
        "created_by": created_by, "helped": helped, "hurt": hurt,
        "neutral": neutral,
        "outcome_counts": {"helped": helped, "hurt": hurt, "neutral": neutral}})


def _seed_rejected_revision(conn, name, n=1):
    for _ in range(n):
        conn.execute(
            "INSERT INTO proposals (uid, kind, target, title, rationale, payload,"
            " evidence, status, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (db.new_ulid(), "skill_revision", name, f"revise '{name}'", "r",
             "{}", "[]", "rejected", db.iso_now()))
    conn.commit()


def _props(conn, name, kind):
    return conn.execute(
        "SELECT * FROM proposals WHERE target=? AND kind=? ORDER BY created_at",
        (name, kind)).fetchall()


def _cfg(home):
    return {"hermes_home": str(home)}


# ---------------------------------------------------------------------------
# B5 readback helpers (skilltree)
# ---------------------------------------------------------------------------

def test_brain_owned_skills_filters_on_provenance(env):
    conn, home = env
    _seed_skill(home, "brain-one", helped=1, hurt=1)
    _seed_skill(home, "agent-one", helped=1, hurt=1, created_by="agent")

    owned = dict((n, p) for n, p in skilltree.brain_owned_skills(home))
    assert set(owned) == {"brain-one"}                 # 'agent' provenance excluded
    assert owned["brain-one"].name == "SKILL.md"


def test_skill_outcomes_reads_back_counts(env):
    conn, home = env
    _seed_skill(home, "s", helped=2, hurt=5, neutral=1)
    oc = skilltree.skill_outcomes(home, "s")
    assert oc == {"helped": 2, "hurt": 5, "neutral": 1, "total": 8,
                  "outcome_counts": {"helped": 2, "hurt": 5, "neutral": 1}}
    # Unknown skill -> zeros, never a KeyError.
    assert skilltree.skill_outcomes(home, "nope")["total"] == 0


# ---------------------------------------------------------------------------
# C4 revision
# ---------------------------------------------------------------------------

def test_net_harmful_skill_gets_revision_proposal(env):
    conn, home = env
    _seed_skill(home, "bad-skill", helped=1, hurt=4)
    _install_llm(_good_revise_llm)

    result = revise_once(conn, _cfg(home), shift_id="sh1")
    assert result["revisions"] == 1, result
    assert result["retirements"] == 0

    rows = _props(conn, "bad-skill", "skill_revision")
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "pending"                  # reviewable, not applied
    payload = json.loads(row["payload"])
    assert payload["revision"]["sections"][0]["heading"] == "## Procedure"
    assert payload["harm"]["harmful"] is True
    # Nothing was applied — the loop only proposes.
    assert conn.execute("SELECT count(*) AS n FROM proposals WHERE status='applied'"
                        ).fetchone()["n"] == 0


def test_healthy_skill_is_untouched(env):
    conn, home = env
    _seed_skill(home, "good-skill", helped=6, hurt=1)
    _install_llm(_good_revise_llm)

    result = revise_once(conn, _cfg(home), shift_id="sh2")
    assert result["healthy"] == 1
    assert result["revisions"] == 0
    assert _props(conn, "good-skill", "skill_revision") == []


def test_insufficient_samples_untouched(env):
    conn, home = env
    # hurt > helped, but only 3 total outcomes -> below _MIN_SAMPLES.
    _seed_skill(home, "young-skill", helped=0, hurt=3)
    _install_llm(_good_revise_llm)

    result = revise_once(conn, _cfg(home), shift_id="sh3")
    assert result["revisions"] == 0
    assert result["healthy"] == 1


def test_non_brain_skill_ignored(env):
    conn, home = env
    _seed_skill(home, "agent-skill", helped=1, hurt=9, created_by="agent")
    _install_llm(_good_revise_llm)

    result = revise_once(conn, _cfg(home), shift_id="sh4")
    assert result == {"skills": 0}                      # not brain-owned -> invisible
    assert _props(conn, "agent-skill", "skill_revision") == []


def test_open_proposal_is_idempotent(env):
    conn, home = env
    _seed_skill(home, "bad-skill", helped=1, hurt=4)
    _install_llm(_good_revise_llm)

    first = revise_once(conn, _cfg(home), shift_id="sh5a")
    assert first["revisions"] == 1
    # A second run while the first proposal is still open must NOT pile on.
    second = revise_once(conn, _cfg(home), shift_id="sh5b")
    assert second["revisions"] == 0
    assert second["skipped_open"] == 1
    assert len(_props(conn, "bad-skill", "skill_revision")) == 1


def test_llm_unavailable_defers_revision(env):
    conn, home = env
    _seed_skill(home, "bad-skill", helped=1, hurt=4)
    _install_llm(_empty_llm)                            # -> LLMUnavailable

    result = revise_once(conn, _cfg(home), shift_id="sh6")
    assert result["skipped_llm"] == 1
    assert result["revisions"] == 0
    # No half-baked proposal written — the harmful skill is retried next run.
    assert _props(conn, "bad-skill", "skill_revision") == []


# ---------------------------------------------------------------------------
# C4 retirement (repeat offender)
# ---------------------------------------------------------------------------

def test_repeat_rejections_trigger_retirement(env):
    conn, home = env
    _seed_skill(home, "bad-skill", helped=1, hurt=4)
    _seed_rejected_revision(conn, "bad-skill", n=2)     # two prior rejections
    _install_llm(_good_revise_llm)

    result = revise_once(conn, _cfg(home), shift_id="sh7")
    assert result["retirements"] == 1
    assert result["revisions"] == 0                     # no LLM draft, went to retire

    retire = _props(conn, "bad-skill", "skill_retire")
    assert len(retire) == 1
    assert retire[0]["status"] == "pending"
    payload = json.loads(retire[0]["payload"])
    assert payload["action"] == "mark_stale"
    assert payload["rejected_revisions"] == 2
    # Still only proposing — nothing marked stale/applied on disk here.
    assert (skilltree.skills_root(home) / "bad-skill" / "SKILL.md").exists()


def test_one_rejection_still_revises(env):
    conn, home = env
    _seed_skill(home, "bad-skill", helped=1, hurt=4)
    _seed_rejected_revision(conn, "bad-skill", n=1)     # below the retire threshold
    _install_llm(_good_revise_llm)

    result = revise_once(conn, _cfg(home), shift_id="sh8")
    assert result["revisions"] == 1
    assert result["retirements"] == 0


# ---------------------------------------------------------------------------
# never-raise + pipeline wiring
# ---------------------------------------------------------------------------

def test_no_home_and_no_skills_are_clean_skips(env):
    conn, home = env
    assert revise_once(conn, {}, shift_id="s")["skipped"] == "no_hermes_home"
    assert revise_once(conn, _cfg(home), shift_id="s") == {"skills": 0}


def test_revise_never_raises_on_bad_db(env, monkeypatch):
    conn, home = env
    _seed_skill(home, "bad-skill", helped=1, hurt=4)
    _install_llm(_good_revise_llm)

    def _boom(*a, **k):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(revise_mod, "_draft_revision", _boom)
    out = revise_once(conn, _cfg(home), shift_id="s")
    assert "error" in out and "db exploded" in out["error"]


def test_revise_strategy_mode_gating(env):
    from brain.dream.run import _strategy_fn
    from brain.dream.shift import Shift

    conn, home = env
    fn = _strategy_fn("revise")

    def _shift(mode):
        return Shift(shift_id="s", conn=conn,
                     config={"_forced_mode": mode, "hermes_home": str(home)},
                     started_at=db.iso_now(),
                     activity_baseline="9999-12-31T00:00:00.000Z", holder="t")

    assert fn(_shift("shadow")) == {"skipped": "shadow"}
    assert fn(_shift("off")) == {"skipped": "off"}
    # active with no brain skills -> clean {"skills": 0}
    assert fn(_shift("active")).get("skills") == 0


def test_revise_in_pipeline_after_forge():
    from brain.dream.shift import DEFAULT_MODES, PIPELINE

    assert "revise" in PIPELINE
    assert PIPELINE.index("revise") == PIPELINE.index("forge") + 1
    assert DEFAULT_MODES["revise"] == "active"
