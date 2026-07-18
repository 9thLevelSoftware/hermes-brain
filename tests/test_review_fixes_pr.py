"""Regression tests for the PR-review findings (Codex + Kilo) — each asserts
the specific defect is fixed:

  R1  peer_card must not be retrievable by a non-owner (the peer it describes)
  R2  the revise sample floor applies to DECISIVE outcomes, not neutrals
  R3  an empty/malformed revision delta creates no proposal
  R4  the budget gate counts unpriced token rows even alongside priced ones
  R6  a retired (stale) skill is skipped by the brain-owned scan
  R7  the archive filename is server-derived, never the caller's archived_at
"""

from __future__ import annotations

import pytest
from brain import llm
from brain.store import db
from conftest import seed_memory

# ---------------------------------------------------------------------------
# R1 (P1) — peer_card scope leak
# ---------------------------------------------------------------------------

def test_peer_card_hidden_from_the_peer_it_describes(conn):
    from brain.recall.search import search

    mem = seed_memory(conn, "bob prefers terse replies and skips every standup",
                      kind="peer_card", memory_type="semantic", trust_tier="known_user")
    conn.execute("UPDATE memories SET scope_user='peer-bob' WHERE id=?", (mem,))
    conn.commit()

    # Bob (non-owner, his own principal) would match scope_user='peer-bob' —
    # but a peer_card must never surface via generic recall for a non-owner.
    hits = search(conn, "bob terse replies standup", trust_tier="known_user",
                  principal_id="peer-bob", include_episodes=False)
    assert all(h.mkind != "peer_card" for h in hits)
    assert not any("terse replies" in h.text for h in hits)


# ---------------------------------------------------------------------------
# R2 (P2) — sample floor over decisives
# ---------------------------------------------------------------------------

def test_revise_neutral_heavy_is_not_harmful():
    from brain.skillforge.revise import _harm_verdict

    # 1 hurt, 0 helped, 4 neutral: total>=5 but only 1 DECISIVE -> not harmful.
    assert _harm_verdict({"helped": 0, "hurt": 1, "neutral": 4, "total": 5})["harmful"] is False
    # Enough decisives, hurt-dominant -> harmful.
    assert _harm_verdict({"helped": 1, "hurt": 5, "neutral": 0, "total": 6})["harmful"] is True


# ---------------------------------------------------------------------------
# R3 (P2) — reject empty revision deltas
# ---------------------------------------------------------------------------

def test_revise_empty_sections_writes_no_proposal(conn, tmp_home):
    from brain.skillforge.revise import _draft_revision

    llm.set_llm_for_tests(
        lambda p, *, system=None, max_tokens=0:
        '{"diagnosis":"x","sections":[],"summary":"y"}')
    try:
        res = _draft_revision(conn, {}, "skill-a", str(tmp_home / "nope.md"),
                              {"helped": 0, "hurt": 5, "neutral": 0, "total": 5},
                              {"harm_rate": 1.0}, "sh")
    finally:
        llm.set_llm_for_tests(None)
    assert res.get("skipped") == "no_usable_sections"
    assert conn.execute("SELECT count(*) FROM proposals WHERE kind='skill_revision'"
                        ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# R4 (P2) — mixed priced + unpriced budget
# ---------------------------------------------------------------------------

def test_budget_gate_counts_unpriced_rows_alongside_priced(conn):
    now = db.iso_now()
    # One cheap PRICED success ...
    conn.execute("INSERT INTO llm_ledger (strategy,model,tokens_in,tokens_out,est_usd,ts)"
                 " VALUES ('extract','claude',10,5,0.01,?)", (now,))
    # ... then unpriced (est_usd=0) failures piling up 900k tokens (~$2.25 proxy).
    for _ in range(3):
        conn.execute("INSERT INTO llm_ledger (strategy,model,tokens_in,tokens_out,est_usd,ts)"
                     " VALUES ('extract','local',300000,0,0.0,?)", (now,))
    conn.commit()
    # Old code returned after the priced check and let the unpriced sail through;
    # now the token-proxy for unpriced rows is counted and trips the gate.
    with pytest.raises(llm.LLMUnavailable, match="budget"):
        llm._budget_gate(conn, {"day_budget_usd": 1.5})


# ---------------------------------------------------------------------------
# R6 (P2) — retired skills skipped by the brain-owned scan
# ---------------------------------------------------------------------------

def test_brain_owned_skills_skips_retired(tmp_home):
    from brain.skillforge import skilltree

    d = skilltree.skills_root(tmp_home) / "old-skill"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        skilltree.build_skill_md("old-skill", "a skill", "## Procedure\nx"),
        encoding="utf-8")

    assert any(n == "old-skill" for n, _ in skilltree.brain_owned_skills(tmp_home))
    skilltree.mark_stale(tmp_home, "old-skill")     # retirement
    assert not any(n == "old-skill" for n, _ in skilltree.brain_owned_skills(tmp_home))


# ---------------------------------------------------------------------------
# R7 (warning) — archive filename never derives from caller input
# ---------------------------------------------------------------------------

def test_archive_ignores_malicious_archived_at(tmp_home):
    from brain.store import archive

    ref = archive.append(tmp_home, {"uid": "ULID9", "content": "secret",
                                    "archived_at": "../../../etc/passwd"})
    assert ref is not None
    filename = ref.rsplit(":", 1)[0]
    assert ".." not in filename and "/" not in filename and "\\" not in filename
    assert archive.recover_content(tmp_home, ref) == "secret"
    # Nothing was written outside the archive dir.
    escaped = list((archive.archive_dir(tmp_home) / "..").glob("*.jsonl.gz"))
    assert escaped == []
