"""Phase C dream: contradict's deterministic knowledge-update-and-delete path.

When BOTH memories in a conflicting pair are triple-backed by facts sharing a
(subject, predicate) with DIFFERENT objects, it is a knowledge UPDATE, not a
genuine contradiction — resolved with ZERO LLM calls (config-gated on
``contradict_knowledge_update``, default True). The newer memory wins; the
fact layer moves forward and the stale row is demoted to 'summarized', never
tombstoned.

Hermetic: fake LLM via llm.set_llm_for_tests (with a call COUNTER), StubEmbedder
vectors, sqlite-vec required. The LLM is faked around EVERY step.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("sqlite_vec")

from brain import llm  # noqa: E402
from brain.dream import contradict  # noqa: E402
from brain.dream.shift import Shift  # noqa: E402
from brain.recall.embed import StubEmbedder  # noqa: E402
from brain.store import db  # noqa: E402
from brain.store import facts as facts_store  # noqa: E402
from brain.store import vec as vec_store  # noqa: E402
from conftest import seed_memory  # noqa: E402

# Two value-change statements: 10/12 tokens shared -> stub-embedder cosine well
# above the 0.82 neighbor gate, and NO negation token (a knowledge update, not
# a polarity flip), so only the fact-backed deterministic path can catch it.
_OLD_CITY = "the primary deploy target region for project hermes is us-east"
_NEW_CITY = "the primary deploy target region for project hermes is us-west"

# A genuine contradiction phrased with negation and NOT triple-backed by facts.
_OLD_NEG = "the deploy target is the staging server alpha for project hermes"
_NEW_NEG = "the deploy target is not the staging server alpha for project hermes"

_SUBJECT = "project:hermes"
_PREDICATE = "deploy_region"


@pytest.fixture(autouse=True)
def _clean_llm():
    yield
    llm.set_llm_for_tests(None)


@pytest.fixture
def embedder():
    return StubEmbedder()


class CountingFake:
    """Fake LLM that COUNTS calls; returns a contradiction verdict."""

    def __init__(self, winner="a", contradicts=True):
        self.calls = 0
        self.winner = winner
        self.contradicts = contradicts

    def __call__(self, prompt, *, system, max_tokens):
        self.calls += 1
        return json.dumps({"contradicts": self.contradicts,
                           "winner": self.winner, "why": "counted"})


def _mk_shift(conn, mode, embedder, config=None):
    from brain.dream import lease
    lease.acquire(conn, "dream", "test")
    cfg = {"_forced_mode": mode, "night_budget_usd": 1.0}
    if config:
        cfg.update(config)
    return Shift(
        shift_id="01TESTSHIFTAAAAAAAAAAAAAAA",
        conn=conn,
        config=cfg,
        embedder=embedder,
        started_at=db.iso_now(),
        activity_baseline="",
        holder="test",
    )


def _index(conn, embedder, mem_id, text):
    vec_store.upsert(conn, "mem_vec", mem_id, embedder.encode_documents([text])[0])


def _seed_update_pair(conn, embedder):
    """Two current-truth memories, each triple-backed by a fact on the same
    (subject, predicate) but with a different object. Older recorded/valid
    first so the newer one is the update winner."""
    assert vec_store.ensure_tables(conn, embedder.dim)
    old_id = seed_memory(conn, _OLD_CITY, created_by="extraction")
    conn.execute(
        "UPDATE memories SET recorded_at='2026-01-01T00:00:00.000Z',"
        " valid_from='2026-01-01T00:00:00.000Z' WHERE id=?", (old_id,))
    new_id = seed_memory(conn, _NEW_CITY, created_by="extraction")
    conn.execute(
        "UPDATE memories SET recorded_at='2026-06-01T00:00:00.000Z',"
        " valid_from='2026-06-01T00:00:00.000Z' WHERE id=?", (new_id,))
    _index(conn, embedder, old_id, _OLD_CITY)
    _index(conn, embedder, new_id, _NEW_CITY)
    # Each memory carries its own current fact for the SAME (subject,predicate).
    # supersede=False so both stay current (the pre-conflict state the dream
    # run must reconcile). valid_from mirrors each memory's timeline.
    facts_store.add_fact(conn, _SUBJECT, _PREDICATE, "us-east", memory_id=old_id,
                         source="extract", supersede=False,
                         valid_from="2026-01-01T00:00:00.000Z")
    facts_store.add_fact(conn, _SUBJECT, _PREDICATE, "us-west", memory_id=new_id,
                         source="extract", supersede=False,
                         valid_from="2026-06-01T00:00:00.000Z")
    conn.commit()
    return old_id, new_id


def _seed_negation_pair(conn, embedder, *, with_facts=False):
    assert vec_store.ensure_tables(conn, embedder.dim)
    old_id = seed_memory(conn, _OLD_NEG, created_by="extraction")
    conn.execute(
        "UPDATE memories SET recorded_at='2026-01-01T00:00:00.000Z',"
        " valid_from='2026-01-01T00:00:00.000Z' WHERE id=?", (old_id,))
    new_id = seed_memory(conn, _NEW_NEG, created_by="extraction")
    _index(conn, embedder, old_id, _OLD_NEG)
    _index(conn, embedder, new_id, _NEW_NEG)
    if with_facts:
        facts_store.add_fact(conn, _SUBJECT, _PREDICATE, "alpha", memory_id=old_id,
                             source="extract", supersede=False,
                             valid_from="2026-01-01T00:00:00.000Z")
        facts_store.add_fact(conn, _SUBJECT, _PREDICATE, "not-alpha", memory_id=new_id,
                             source="extract", supersede=False)
    conn.commit()
    return old_id, new_id


def _audit_actions(conn, action):
    return conn.execute(
        "SELECT * FROM audit_log WHERE action=? ORDER BY id", (action,)).fetchall()


# ---------------------------------------------------------------------------
# Deterministic knowledge-update path (ZERO LLM)
# ---------------------------------------------------------------------------

def test_knowledge_update_resolves_without_llm(conn, embedder):
    old_id, new_id = _seed_update_pair(conn, embedder)
    fake = CountingFake()
    llm.set_llm_for_tests(fake)

    result = contradict.run(_mk_shift(conn, "active", embedder))

    # ZERO LLM calls: the fact-backed update never touches the adjudicator.
    assert fake.calls == 0
    assert result.get("updated") == 1
    assert result.get("contradictions", 0) == 0
    assert result.get("invalidated", 0) == 0

    # Winner's fact is the single current truth for (subject, predicate).
    current = facts_store.query_facts(conn, subject=_SUBJECT, predicate=_PREDICATE)
    assert len(current) == 1
    assert current[0].object == "us-west"
    assert current[0].memory_id == new_id

    # Stale (older) memory demoted to 'summarized' — NOT tombstoned/deleted,
    # still a present, recoverable row.
    stale = conn.execute(
        "SELECT status, content FROM memories WHERE id=?", (old_id,)).fetchone()
    assert stale is not None                         # still present (recoverable)
    assert stale["status"] == "summarized"
    assert stale["status"] != "tombstone"
    assert stale["content"] == _OLD_CITY             # content preserved

    # Winner stays active current-truth.
    winner = conn.execute(
        "SELECT status, valid_to FROM memories WHERE id=?", (new_id,)).fetchone()
    assert winner["status"] == "active" and winner["valid_to"] is None

    assert _audit_actions(conn, "contradict_knowledge_update")


def test_knowledge_update_dry_run_mutates_nothing(conn, embedder):
    old_id, new_id = _seed_update_pair(conn, embedder)
    fake = CountingFake()
    llm.set_llm_for_tests(fake)

    result = contradict.run(_mk_shift(conn, "dry_run", embedder))

    assert fake.calls == 0 and result.get("updated") == 1
    # No mutation: both memories untouched, both facts still current.
    for mid in (old_id, new_id):
        row = conn.execute(
            "SELECT status, valid_to FROM memories WHERE id=?", (mid,)).fetchone()
        assert row["status"] == "active" and row["valid_to"] is None
    current = facts_store.query_facts(conn, subject=_SUBJECT, predicate=_PREDICATE)
    assert {f.object for f in current} == {"us-east", "us-west"}
    assert _audit_actions(conn, "would_knowledge_update")
    # dry_run must NOT advance the watermark.
    assert conn.execute("SELECT 1 FROM sweep_state WHERE key=?",
                        ("contradict:watermark",)).fetchone() is None


# ---------------------------------------------------------------------------
# The LLM adjudication path is untouched for non-fact-backed pairs / gate off
# ---------------------------------------------------------------------------

def test_genuine_contradiction_not_fact_backed_uses_llm(conn, embedder):
    old_id, new_id = _seed_negation_pair(conn, embedder, with_facts=False)
    fake = CountingFake(winner="a")
    llm.set_llm_for_tests(fake)

    result = contradict.run(_mk_shift(conn, "active", embedder))

    # No shared (subject,predicate) facts -> deterministic path skipped,
    # the LLM adjudicator decides.
    assert fake.calls == 1
    assert result.get("updated", 0) == 0
    assert result.get("invalidated") == 1
    loser = conn.execute(
        "SELECT valid_to, invalidated_by FROM memories WHERE id=?",
        (old_id,)).fetchone()
    assert loser["valid_to"] is not None and loser["invalidated_by"] == new_id


def test_gate_off_falls_back_to_llm_even_when_fact_backed(conn, embedder):
    old_id, new_id = _seed_negation_pair(conn, embedder, with_facts=True)
    fake = CountingFake(winner="a")
    llm.set_llm_for_tests(fake)

    # Gate OFF: byte-for-byte the current LLM path even though the pair is
    # fact-backed on the same (subject, predicate).
    shift = _mk_shift(conn, "active", embedder,
                      config={"contradict_knowledge_update": False})
    result = contradict.run(shift)

    assert fake.calls == 1
    assert result.get("updated", 0) == 0
    assert result.get("invalidated") == 1
    # The invalidation went through the LLM (bi-temporal close), not the
    # deterministic 'summarized' demotion.
    loser = conn.execute(
        "SELECT status, valid_to, invalidated_by FROM memories WHERE id=?",
        (old_id,)).fetchone()
    assert loser["valid_to"] is not None and loser["invalidated_by"] == new_id
    assert loser["status"] != "summarized"
    assert not _audit_actions(conn, "contradict_knowledge_update")
