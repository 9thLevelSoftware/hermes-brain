"""P(B) dream: facts (natural-language memories -> normalized temporal
triples via store/facts.add_fact). Hermetic: fake LLM via
llm.set_llm_for_tests around EVERY step; no embedder needed (the facts
strategy indexes text, it does not cluster vectors).

Mode invariant: shadow/dry_run write NOTHING to the facts table; only
active mutates. Knowledge-update conflicts (same subject+predicate,
different object) are detected and counted.
"""

from __future__ import annotations

import json
import re

import pytest
from brain import llm  # noqa: E402
from brain.dream import facts  # noqa: E402
from brain.dream.shift import Shift  # noqa: E402
from brain.store import db  # noqa: E402
from brain.store import facts as facts_store  # noqa: E402
from conftest import seed_memory  # noqa: E402

_ULID_RE = re.compile(r"\[([0-9A-HJKMNP-TV-Z]{26})\]")


@pytest.fixture(autouse=True)
def _clean_llm():
    yield
    llm.set_llm_for_tests(None)


def _mk_shift(conn, mode):
    from brain.dream import lease
    # A real shift holds the lease (run_dream acquires it); strategies renew
    # it via keepalive() and yield if it was lost. Unit tests must acquire it
    # too or every strategy would correctly yield immediately.
    lease.acquire(conn, "dream", "test")
    return Shift(
        shift_id="01TESTSHIFTAAAAAAAAAAAAAAA",
        conn=conn,
        config={"_forced_mode": mode, "night_budget_usd": 1.0,
                "day_budget_usd": 1.5},
        embedder=None,
        started_at=db.iso_now(),
        activity_baseline="",
        holder="test",
    )


def _audit_actions(conn, action):
    return conn.execute(
        "SELECT * FROM audit_log WHERE action=? ORDER BY id", (action,)).fetchall()


class FactsFake:
    """Emits one triple per uid it sees in the prompt, with configured
    subject/predicate/object (so tests can force a specific (s,p) conflict)."""

    def __init__(self, subject="the user", predicate="prefers", obj="tabs"):
        self.calls = 0
        self.subject = subject
        self.predicate = predicate
        self.obj = obj

    def __call__(self, prompt, *, system, max_tokens):
        self.calls += 1
        uids = _ULID_RE.findall(prompt)
        return json.dumps([
            {"uid": u, "subject": self.subject,
             "predicate": self.predicate, "object": self.obj}
            for u in uids
        ])


class EmptyFake:
    """Returns empty text -> llm.call_text raises LLMUnavailable."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, *, system, max_tokens):
        self.calls += 1
        return ""


def _dream_facts_count(conn):
    return conn.execute(
        "SELECT COUNT(*) FROM facts WHERE source='dream:facts'").fetchone()[0]


# ---------------------------------------------------------------------------
# shadow: compute + audit, write nothing
# ---------------------------------------------------------------------------

def test_facts_shadow_writes_nothing(conn):
    mid = seed_memory(conn, "the user prefers tabs over spaces in python")
    fake = FactsFake()
    llm.set_llm_for_tests(fake)

    result = facts.run(_mk_shift(conn, "shadow"))

    assert fake.calls == 1
    assert result["scanned"] == 1
    assert result["proposed"] == 1
    assert result["written"] == 0
    # LOAD-BEARING: shadow writes zero rows to the facts table.
    assert _dream_facts_count(conn) == 0
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
    # ...but it still audits what it WOULD write.
    audits = _audit_actions(conn, "would_write_fact")
    assert len(audits) == 1
    detail = json.loads(audits[0]["detail"])
    assert detail["subject"] == "the user"
    assert detail["predicate"] == "prefers"
    assert detail["object"] == "tabs"
    # the source memory is untouched (still current truth)
    row = conn.execute("SELECT valid_to, status FROM memories WHERE id=?",
                       (mid,)).fetchone()
    assert row["valid_to"] is None and row["status"] == "active"


def test_facts_dry_run_writes_nothing(conn):
    seed_memory(conn, "the deploy target is staging alpha for project hermes")
    llm.set_llm_for_tests(FactsFake(subject="project hermes",
                                    predicate="deploy_target", obj="staging alpha"))

    result = facts.run(_mk_shift(conn, "dry_run"))

    assert result["written"] == 0
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
    assert _audit_actions(conn, "would_write_fact")


# ---------------------------------------------------------------------------
# active: write triples via add_fact, linked to the source memory
# ---------------------------------------------------------------------------

def test_facts_active_writes_triples(conn):
    mid = seed_memory(conn, "the user prefers tabs over spaces in python")
    fake = FactsFake()
    llm.set_llm_for_tests(fake)

    result = facts.run(_mk_shift(conn, "active"))

    assert fake.calls == 1
    assert result["scanned"] == 1
    assert result["proposed"] == 1
    assert result["written"] == 1
    assert result["conflicts"] == 0

    fact = conn.execute(
        "SELECT * FROM facts WHERE source='dream:facts'").fetchone()
    assert fact is not None
    assert fact["subject"] == "the user"
    assert fact["predicate"] == "prefers"
    assert fact["object"] == "tabs"
    # the triple is an INDEX OVER the source memory (critique item 9)
    assert fact["memory_id"] == mid
    assert fact["valid_until"] is None            # current truth
    assert _audit_actions(conn, "fact_write")

    # current-truth read path sees it
    current = facts_store.current_facts_for(conn, "the user")
    assert len(current) == 1 and current[0].object == "tabs"

    # idempotent: the memory now has a triple, so a second run scans it out
    fake2 = FactsFake()
    llm.set_llm_for_tests(fake2)
    again = facts.run(_mk_shift(conn, "active"))
    assert again["scanned"] == 0 and fake2.calls == 0
    assert _dream_facts_count(conn) == 1


# ---------------------------------------------------------------------------
# knowledge update: same (subject, predicate), different object
# ---------------------------------------------------------------------------

def test_facts_active_detects_conflict(conn):
    # Prior current truth: the user prefers spaces.
    facts_store.add_fact(conn, "the user", "prefers", "spaces", source="extract")
    assert facts_store.current_facts_for(conn, "the user")[0].object == "spaces"

    seed_memory(conn, "the user now prefers tabs over spaces in python")
    llm.set_llm_for_tests(FactsFake(obj="tabs"))      # same s,p — different o

    result = facts.run(_mk_shift(conn, "active"))

    assert result["proposed"] == 1
    assert result["written"] == 1
    assert result["conflicts"] == 1

    # supersede-don't-delete: exactly one current-truth row, now 'tabs'
    current = facts_store.current_facts_for(conn, "the user")
    assert len(current) == 1 and current[0].object == "tabs"
    # the old triple is closed, not gone
    closed = conn.execute(
        "SELECT object, valid_until FROM facts WHERE object='spaces'").fetchone()
    assert closed["valid_until"] is not None

    detail = json.loads(_audit_actions(conn, "fact_write")[0]["detail"])
    assert detail["superseded"] == "spaces"


def test_facts_shadow_detects_conflict_without_writing(conn):
    facts_store.add_fact(conn, "the user", "prefers", "spaces", source="extract")
    seed_memory(conn, "the user now prefers tabs over spaces in python")
    llm.set_llm_for_tests(FactsFake(obj="tabs"))

    result = facts.run(_mk_shift(conn, "shadow"))

    assert result["conflicts"] == 1
    assert result["written"] == 0
    # still 'spaces' — shadow proposed nothing to the store
    current = facts_store.current_facts_for(conn, "the user")
    assert len(current) == 1 and current[0].object == "spaces"
    assert _dream_facts_count(conn) == 0


# ---------------------------------------------------------------------------
# LLMUnavailable -> clean skip, never raises
# ---------------------------------------------------------------------------

def test_facts_llm_unavailable_clean_skip(conn):
    seed_memory(conn, "the user prefers tabs over spaces in python")
    fake = EmptyFake()
    llm.set_llm_for_tests(fake)

    result = facts.run(_mk_shift(conn, "active"))

    assert result == {"skipped": "llm_unavailable"}
    assert fake.calls == 1
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0


def test_facts_empty_feed_is_free(conn):
    fake = FactsFake()
    llm.set_llm_for_tests(fake)

    result = facts.run(_mk_shift(conn, "active"))

    assert result["scanned"] == 0
    assert result["written"] == 0
    assert fake.calls == 0                 # cheap pre-check: no LLM on empty feed


def test_facts_off_mode_skips(conn):
    seed_memory(conn, "the user prefers tabs over spaces in python")
    fake = FactsFake()
    llm.set_llm_for_tests(fake)

    result = facts.run(_mk_shift(conn, "off"))

    assert result == {"skipped": "off"}
    assert fake.calls == 0
