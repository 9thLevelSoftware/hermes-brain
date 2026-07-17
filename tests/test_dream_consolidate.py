"""P4 dream: consolidate (episodic->semantic distillation) + contradict
(polarity conflict detection with supersede-don't-delete). Hermetic: fake
LLM via llm.set_llm_for_tests, StubEmbedder vectors, sqlite-vec required.
"""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("sqlite_vec")

from brain import llm  # noqa: E402
from brain.dream import consolidate, contradict  # noqa: E402
from brain.dream.shift import Shift  # noqa: E402
from brain.recall.embed import StubEmbedder  # noqa: E402
from brain.recall.search import search  # noqa: E402
from brain.store import db  # noqa: E402
from brain.store import vec as vec_store  # noqa: E402
from conftest import seed_memory  # noqa: E402

_ULID_RE = re.compile(r"\[([0-9A-HJKMNP-TV-Z]{26})\]")

# Near-identical extracted observations: pairwise stub-embedder cosine is
# ~10/11 (10 of 11 tokens shared) — comfortably above the 0.80 gate.
_CLUSTER_TEXTS = [
    "the user always prefers tabs over spaces in python code projects",
    "the user always prefers tabs over spaces in python code repositories",
    "the user always prefers tabs over spaces in python code generally",
]

_OLD_FACT = "the deploy target is the staging server alpha for project hermes"
_NEW_FACT = "the deploy target is not the staging server alpha for project hermes"


@pytest.fixture(autouse=True)
def _clean_llm():
    yield
    llm.set_llm_for_tests(None)


@pytest.fixture
def embedder():
    return StubEmbedder()


def _mk_shift(conn, mode, embedder):
    from brain.dream import lease
    # A real shift holds the lease (run_dream acquires it); strategies renew
    # it via keepalive() and yield if it was lost. Unit tests must acquire it
    # too or every strategy would correctly yield immediately.
    lease.acquire(conn, "dream", "test")
    return Shift(
        shift_id="01TESTSHIFTAAAAAAAAAAAAAAA",
        conn=conn,
        config={"_forced_mode": mode, "night_budget_usd": 1.0},
        embedder=embedder,
        started_at=db.iso_now(),
        activity_baseline="",
        holder="test",
    )


def _index(conn, embedder, mem_id, text):
    vec_store.upsert(conn, "mem_vec", mem_id, embedder.encode_documents([text])[0])


def _seed_cluster(conn, embedder):
    assert vec_store.ensure_tables(conn, embedder.dim)
    ids = []
    for text in _CLUSTER_TEXTS:
        mid = seed_memory(conn, text, created_by="extraction")
        _index(conn, embedder, mid, text)
        ids.append(mid)
    conn.commit()
    return ids


def _seed_conflict_pair(conn, embedder):
    assert vec_store.ensure_tables(conn, embedder.dim)
    old_id = seed_memory(conn, _OLD_FACT, created_by="extraction")
    conn.execute(
        "UPDATE memories SET recorded_at='2026-01-01T00:00:00.000Z',"
        " valid_from='2026-01-01T00:00:00.000Z' WHERE id=?", (old_id,))
    new_id = seed_memory(conn, _NEW_FACT, created_by="extraction")
    _index(conn, embedder, old_id, _OLD_FACT)
    _index(conn, embedder, new_id, _NEW_FACT)
    conn.commit()
    return old_id, new_id


def _audit_actions(conn, action):
    return conn.execute(
        "SELECT * FROM audit_log WHERE action=? ORDER BY id", (action,)).fetchall()


class ConsolidateFake:
    """Returns a lesson citing every uid it sees in the prompt."""

    def __init__(self, entity="python", actionable=True):
        self.calls = 0
        self.entity = entity
        self.actionable = actionable

    def __call__(self, prompt, *, system, max_tokens):
        self.calls += 1
        return json.dumps({
            "content": "The user consistently prefers tabs over spaces in python code.",
            "cites": _ULID_RE.findall(prompt),
            "entity": self.entity,
            "actionable": self.actionable,
        })


class ContradictFake:
    def __init__(self, winner="a", contradicts=True):
        self.calls = 0
        self.winner = winner
        self.contradicts = contradicts

    def __call__(self, prompt, *, system, max_tokens):
        self.calls += 1
        return json.dumps({"contradicts": self.contradicts,
                           "winner": self.winner, "why": "negated restatement"})


# ---------------------------------------------------------------------------
# consolidate
# ---------------------------------------------------------------------------

def test_consolidate_active_distills_cluster(conn, embedder):
    members = _seed_cluster(conn, embedder)
    fake = ConsolidateFake()
    llm.set_llm_for_tests(fake)

    result = consolidate.run(_mk_shift(conn, "active", embedder))

    assert result.get("distilled") == 1 and fake.calls == 1
    pattern = conn.execute(
        "SELECT * FROM memories WHERE created_by='consolidation'").fetchone()
    assert pattern is not None
    assert pattern["epistemic"] == "inference"
    assert pattern["memory_type"] == "semantic"
    assert pattern["kind"] == "insight"
    assert pattern["status"] == "active"
    assert pattern["half_life_days"] == 180.0
    # importance boosted +0.2 over the member mean (0.5 default) = 0.7
    assert pattern["importance"] == pytest.approx(0.7)

    # provenance: member uids + shift id in source_refs
    refs = json.loads(pattern["source_refs"])
    member_uids = {conn.execute("SELECT uid FROM memories WHERE id=?",
                                (m,)).fetchone()["uid"] for m in members}
    assert member_uids <= set(refs)
    assert any(r.startswith("shift:") for r in refs)

    # related_to edges pattern -> each member; members demoted, NOT deleted
    for mid in members:
        edge = conn.execute(
            "SELECT 1 FROM edges WHERE src_id=? AND dst_id=?"
            " AND edge_type='related_to'", (pattern["id"], mid)).fetchone()
        assert edge is not None
        row = conn.execute(
            "SELECT importance, status, valid_to FROM memories WHERE id=?",
            (mid,)).fetchone()
        assert row["status"] == "active" and row["valid_to"] is None
        assert row["importance"] == pytest.approx(0.5 * 0.7)

    # pattern embedded into mem_vec
    assert conn.execute("SELECT 1 FROM mem_vec WHERE id=?",
                        (pattern["id"],)).fetchone() is not None
    assert _audit_actions(conn, "consolidate_insert")

    # second run: members carry 'consolidated' markers — nothing to do
    fake2 = ConsolidateFake()
    llm.set_llm_for_tests(fake2)
    again = consolidate.run(_mk_shift(conn, "active", embedder))
    assert again.get("distilled", 0) == 0 and fake2.calls == 0


def test_consolidate_dry_run_audits_only(conn, embedder):
    members = _seed_cluster(conn, embedder)
    llm.set_llm_for_tests(ConsolidateFake())

    result = consolidate.run(_mk_shift(conn, "dry_run", embedder))

    assert result.get("distilled") == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM memories WHERE created_by='consolidation'"
    ).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0
    for mid in members:
        row = conn.execute("SELECT importance FROM memories WHERE id=?",
                           (mid,)).fetchone()
        assert row["importance"] is None  # untouched
    audits = _audit_actions(conn, "would_consolidate")
    assert len(audits) == 1
    detail = json.loads(audits[0]["detail"])
    member_uids = {conn.execute("SELECT uid FROM memories WHERE id=?",
                                (m,)).fetchone()["uid"] for m in members}
    assert set(detail["members"]) == member_uids and detail["content"]


def test_consolidate_specificity_gate_rejects_vague(conn, embedder):
    _seed_cluster(conn, embedder)
    # 'synergy' matches no entities row and appears in no member content.
    llm.set_llm_for_tests(ConsolidateFake(entity="synergy"))

    result = consolidate.run(_mk_shift(conn, "active", embedder))

    assert result.get("rejected") == 1 and result.get("distilled") == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM memories WHERE created_by='consolidation'"
    ).fetchone()[0] == 0
    assert _audit_actions(conn, "consolidate_reject_vague")


def test_consolidate_preemption_stops_before_llm(conn, embedder):
    _seed_cluster(conn, embedder)
    fake = ConsolidateFake()
    llm.set_llm_for_tests(fake)
    shift = _mk_shift(conn, "active", embedder)
    shift.preempted = lambda: True  # user came back

    result = consolidate.run(shift)

    assert fake.calls == 0
    assert result.get("preempted") is True
    assert conn.execute(
        "SELECT COUNT(*) FROM memories WHERE created_by='consolidation'"
    ).fetchone()[0] == 0


def test_consolidate_empty_feed_is_free(conn, embedder):
    fake = ConsolidateFake()
    llm.set_llm_for_tests(fake)
    result = consolidate.run(_mk_shift(conn, "active", embedder))
    assert result == {"clusters": 0} and fake.calls == 0


# ---------------------------------------------------------------------------
# contradict
# ---------------------------------------------------------------------------

def test_contradict_active_invalidates_older(conn, embedder):
    old_id, new_id = _seed_conflict_pair(conn, embedder)
    fake = ContradictFake(winner="a")
    llm.set_llm_for_tests(fake)

    result = contradict.run(_mk_shift(conn, "active", embedder))

    assert fake.calls == 1
    assert result.get("invalidated") == 1
    loser = conn.execute("SELECT * FROM memories WHERE id=?", (old_id,)).fetchone()
    winner = conn.execute("SELECT * FROM memories WHERE id=?", (new_id,)).fetchone()
    # supersede-don't-delete: loser closed bi-temporally, still a row
    assert loser["valid_to"] is not None
    assert loser["invalidated_by"] == new_id
    assert loser["status"] == "active"
    assert winner["valid_to"] is None

    edge = conn.execute(
        "SELECT * FROM edges WHERE edge_type='conflicts_with'").fetchone()
    assert edge is not None and {edge["src_id"], edge["dst_id"]} == {old_id, new_id}
    assert _audit_actions(conn, "contradict_invalidate")

    # loser dropped from current-truth recall
    hits = search(conn, "deploy target staging server alpha", embedder=embedder)
    assert all(not (h.kind == "memory" and h.id == old_id) for h in hits)

    # watermark advanced to the composite (recorded_at, id) of the last
    # fully-processed candidate (tie-safe cursor, review finding #9).
    ra, wm_id = contradict._get_watermark(conn)
    assert (ra, wm_id) >= (winner["recorded_at"], 0)


def test_contradict_dry_run_mutates_nothing(conn, embedder):
    old_id, new_id = _seed_conflict_pair(conn, embedder)
    fake = ContradictFake(winner="a")
    llm.set_llm_for_tests(fake)

    result = contradict.run(_mk_shift(conn, "dry_run", embedder))

    assert fake.calls == 1 and result.get("contradictions") == 1
    for mid in (old_id, new_id):
        row = conn.execute("SELECT valid_to, invalidated_by, needs_review"
                           " FROM memories WHERE id=?", (mid,)).fetchone()
        assert row["valid_to"] is None
        assert row["invalidated_by"] is None
        assert row["needs_review"] == 0
    assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0
    assert _audit_actions(conn, "would_contradict")
    # watermark must NOT advance in dry_run — re-see the pair next run
    assert conn.execute("SELECT 1 FROM sweep_state WHERE key=?",
                        ("contradict:watermark",)).fetchone() is None


def test_contradict_neither_flags_review_without_invalidation(conn, embedder):
    old_id, new_id = _seed_conflict_pair(conn, embedder)
    llm.set_llm_for_tests(ContradictFake(winner="neither"))

    result = contradict.run(_mk_shift(conn, "active", embedder))

    assert result.get("flagged") == 1 and result.get("invalidated") == 0
    for mid in (old_id, new_id):
        row = conn.execute("SELECT valid_to, needs_review FROM memories"
                           " WHERE id=?", (mid,)).fetchone()
        assert row["valid_to"] is None and row["needs_review"] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM edges WHERE edge_type='conflicts_with'"
    ).fetchone()[0] == 1


def test_contradict_preemption_stops_before_llm(conn, embedder):
    _seed_conflict_pair(conn, embedder)
    fake = ContradictFake()
    llm.set_llm_for_tests(fake)
    shift = _mk_shift(conn, "active", embedder)
    shift.preempted = lambda: True

    result = contradict.run(shift)

    assert fake.calls == 0 and result.get("preempted") is True
    assert conn.execute("SELECT 1 FROM sweep_state WHERE key=?",
                        ("contradict:watermark",)).fetchone() is None
