"""Dream 'forget' strategy: value scoring + tiered demotion.

Covers: low-value stale row demoted active->summarized in active mode;
pinned/outcome/recalled rows immune; dry_run computing scores + auditing
would_demote without mutating; summarized rows past grace tombstoned;
tombstones past grace purged (content NULLed, provenance kept); and
importance written in active mode only.
"""

from __future__ import annotations

import json

from brain.dream.forget import run as forget_run
from brain.dream.shift import Shift
from brain.store import db
from conftest import iso_days_ago, seed_memory

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_shift(conn, config=None):
    from brain.dream import lease
    lease.acquire(conn, "dream", "test-holder")  # a real shift holds the lease
    return Shift(
        shift_id=db.new_ulid(), conn=conn, config=dict(config or {}),
        started_at=db.iso_now(), activity_baseline="9999-12-31T00:00:00.000Z",
        holder="test-holder",
    )


def seed_stale(conn, content, **kw):
    """A low-value row: old, never recalled, no outcome, default trust."""
    kw.setdefault("valid_from", iso_days_ago(400))
    return seed_memory(conn, content, **kw)


def mem_row(conn, mem_id):
    return conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()


def audit_actions(conn, action):
    return conn.execute(
        "SELECT target, detail FROM audit_log WHERE action=?", (action,)).fetchall()


def stamp_audit(conn, action, uid, ts):
    """Backdated tier-entry marker (the 'forgotten_at' tracking rows the
    strategy itself writes in active mode)."""
    conn.execute(
        "INSERT INTO audit_log (actor, action, target, detail, ts)"
        " VALUES ('dream:test',?,?,?,?)", (action, uid, "{}", ts))
    conn.commit()


ACTIVE = {"_forced_mode": "active"}
DRY = {"_forced_mode": "dry_run"}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_forget_demotes_low_value_stale_row(conn):
    mem = seed_stale(conn, "temporary note about a one-off script")
    result = forget_run(make_shift(conn, ACTIVE))

    assert result["scored"] >= 1
    assert result["demoted"] == 1
    row = mem_row(conn, mem)
    assert row["status"] == "summarized"
    assert row["content"] is not None            # distill-don't-delete
    assert row["importance"] is not None
    assert row["importance"] < 0.15

    audits = audit_actions(conn, "forget_demote")
    assert len(audits) == 1
    assert audits[0]["target"] == row["uid"]
    assert "score" in json.loads(audits[0]["detail"])


def test_forget_immunity_pinned_outcome_recalled(conn):
    pinned = seed_stale(conn, "pinned fact that must never demote", pinned=1)
    with_outcome = seed_stale(conn, "strategy row with a recorded outcome",
                              outcome="worked")
    recalled = seed_stale(conn, "old but recently useful row")
    conn.execute(
        "UPDATE memories SET recall_count=3, last_recalled_at=? WHERE id=?",
        (db.iso_now(), recalled))
    conn.commit()

    result = forget_run(make_shift(conn, ACTIVE))

    assert result["demoted"] == 0
    for mem_id in (pinned, with_outcome, recalled):
        assert mem_row(conn, mem_id)["status"] == "active"
    assert mem_row(conn, pinned)["importance"] == 1.0   # pinned pegs the score


def test_forget_fresh_or_valued_rows_survive(conn):
    fresh = seed_memory(conn, "brand new fact from this morning")
    old_but_young_half_life = seed_stale(
        conn, "decision with a long half-life", half_life_days=365.0)

    result = forget_run(make_shift(conn, ACTIVE))

    assert result["demoted"] == 0
    assert mem_row(conn, fresh)["status"] == "active"
    # age 400d < 2*365d half-life: not stale enough
    assert mem_row(conn, old_but_young_half_life)["status"] == "active"


def test_forget_dry_run_scores_but_mutates_nothing(conn):
    mem = seed_stale(conn, "stale row the dry run must not touch")
    result = forget_run(make_shift(conn, DRY))

    assert result["scored"] >= 1
    assert result["demoted"] == 1                # honest would-do counts
    row = mem_row(conn, mem)
    assert row["status"] == "active"             # nothing moved
    assert row["importance"] is None             # score written in active only

    audits = audit_actions(conn, "would_demote")
    assert len(audits) == 1
    assert audits[0]["target"] == row["uid"]
    detail = json.loads(audits[0]["detail"])
    assert detail["score"] < 0.15
    assert not audit_actions(conn, "forget_demote")


def test_forget_summarized_past_grace_becomes_tombstone(conn):
    mem = seed_stale(conn, "summarized row that stayed worthless",
                     status="summarized")
    uid = mem_row(conn, mem)["uid"]
    stamp_audit(conn, "forget_demote", uid, iso_days_ago(40))

    result = forget_run(make_shift(conn, ACTIVE))

    assert result["tombstoned"] == 1
    row = mem_row(conn, mem)
    assert row["status"] == "tombstone"
    assert row["content"] is not None            # purge is a later, separate step
    assert audit_actions(conn, "forget_tombstone")[0]["target"] == uid


def test_forget_summarized_within_grace_untouched(conn):
    mem = seed_stale(conn, "recently demoted row still in its grace period",
                     status="summarized")
    stamp_audit(conn, "forget_demote", mem_row(conn, mem)["uid"], iso_days_ago(5))

    result = forget_run(make_shift(conn, ACTIVE))

    assert result["tombstoned"] == 0
    assert mem_row(conn, mem)["status"] == "summarized"


def test_forget_purges_tombstone_past_grace_keeping_provenance(conn, tmp_home):
    mem = seed_stale(conn, "tombstoned row whose grace has fully lapsed",
                     status="tombstone")
    before = mem_row(conn, mem)
    stamp_audit(conn, "forget_tombstone", before["uid"], iso_days_ago(40))

    # A purge now archives the raw text first, so it needs a home to write to.
    result = forget_run(make_shift(conn, {**ACTIVE, "hermes_home": str(tmp_home)}))

    assert result["purged"] == 1
    row = mem_row(conn, mem)
    assert row["content"] is None                # content NULLed...
    assert row["status"] == "tombstone"          # ...but the row remains
    assert row["uid"] == before["uid"]           # provenance intact
    assert row["created_by"] == before["created_by"]
    # source_refs GAINS an archive ref (where the raw text went) — enhanced
    # provenance, not lost provenance.
    refs = json.loads(row["source_refs"] or "[]")
    assert any(r.startswith("archive:") for r in refs)
    assert row["summary"]                        # distilled stub kept
    assert audit_actions(conn, "forget_purge")[0]["target"] == before["uid"]


def test_forget_purge_dry_run_keeps_content(conn):
    mem = seed_stale(conn, "tombstone the dry run must not purge",
                     status="tombstone")
    stamp_audit(conn, "forget_tombstone", mem_row(conn, mem)["uid"], iso_days_ago(40))

    result = forget_run(make_shift(conn, DRY))

    assert result["purged"] == 1                 # honest count
    assert mem_row(conn, mem)["content"] is not None
    assert len(audit_actions(conn, "would_purge")) == 1


def test_forget_writes_importance_in_active_only(conn):
    keeper = seed_memory(conn, "healthy recent fact", pinned=0)
    forget_run(make_shift(conn, DRY))
    assert mem_row(conn, keeper)["importance"] is None

    forget_run(make_shift(conn, ACTIVE))
    imp = mem_row(conn, keeper)["importance"]
    assert imp is not None
    assert 0.0 <= imp <= 1.0
