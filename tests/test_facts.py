"""Tests for the temporal fact index (store/facts.py).

Single-current-truth per (subject, predicate); close-then-insert
supersession; as-of point-in-time queries; linked-memory retirement in
lockstep; provenance walk over the memories version chain.
"""

from __future__ import annotations

import json

from brain.store import facts
from conftest import seed_memory


def _current(conn, subject, predicate):
    rows = facts.query_facts(conn, subject=subject, predicate=predicate)
    return rows


# ---------------------------------------------------------------------------
# Single current truth + supersession
# ---------------------------------------------------------------------------

def test_exactly_one_current_row_after_readds(conn):
    facts.add_fact(conn, "maya", "assigned_to", "auth")
    facts.add_fact(conn, "maya", "assigned_to", "billing")
    facts.add_fact(conn, "maya", "assigned_to", "search")

    current = _current(conn, "maya", "assigned_to")
    assert len(current) == 1
    assert current[0].object == "search"
    assert current[0].valid_until is None

    # But history is preserved (three total rows).
    total = conn.execute(
        "SELECT COUNT(*) FROM facts WHERE subject='maya' AND predicate='assigned_to'"
    ).fetchone()[0]
    assert total == 3


def test_supersession_stamps_old_row(conn):
    first = facts.add_fact(conn, "user", "prefers", "dark")
    second = facts.add_fact(conn, "user", "prefers", "light")

    old = conn.execute(
        "SELECT valid_until, superseded_by FROM facts WHERE id=?", (first,)
    ).fetchone()
    assert old["valid_until"] is not None
    assert old["superseded_by"] == second


def test_supersede_false_keeps_priors_open(conn):
    facts.add_fact(conn, "user", "speaks", "english")
    facts.add_fact(conn, "user", "speaks", "spanish", supersede=False)

    current = _current(conn, "user", "speaks")
    assert {f.object for f in current} == {"english", "spanish"}


# ---------------------------------------------------------------------------
# As-of point-in-time query
# ---------------------------------------------------------------------------

def test_as_of_returns_historical_then_current(conn):
    facts.add_fact(conn, "maya", "team", "platform",
                   valid_from="2020-01-01T00:00:00.000Z")
    facts.add_fact(conn, "maya", "team", "infra",
                   valid_from="2026-01-01T00:00:00.000Z")

    # A moment before the second fact took effect -> the historical value.
    past = facts.query_facts(conn, subject="maya", predicate="team",
                             as_of="2021-06-01T00:00:00.000Z")
    assert len(past) == 1
    assert past[0].object == "platform"

    # No as_of -> current truth.
    now = facts.query_facts(conn, subject="maya", predicate="team")
    assert len(now) == 1
    assert now[0].object == "infra"


# ---------------------------------------------------------------------------
# Linked-memory retirement in lockstep
# ---------------------------------------------------------------------------

def test_new_memory_id_retires_old_memory(conn):
    mem_old = seed_memory(conn, "Maya is on the platform team", kind="fact")
    mem_new = seed_memory(conn, "Maya is on the infra team", kind="fact")

    facts.add_fact(conn, "maya", "team", "platform", memory_id=mem_old)
    facts.add_fact(conn, "maya", "team", "infra", memory_id=mem_new)

    old_row = conn.execute(
        "SELECT valid_to, superseded_by FROM memories WHERE id=?", (mem_old,)
    ).fetchone()
    assert old_row["valid_to"] is not None
    assert old_row["superseded_by"] == mem_new

    # The replacement memory stays current.
    new_row = conn.execute(
        "SELECT valid_to FROM memories WHERE id=?", (mem_new,)
    ).fetchone()
    assert new_row["valid_to"] is None


def test_same_memory_id_does_not_retire(conn):
    mem = seed_memory(conn, "stable fact", kind="fact")
    facts.add_fact(conn, "s", "p", "o1", memory_id=mem)
    facts.add_fact(conn, "s", "p", "o2", memory_id=mem)

    row = conn.execute(
        "SELECT valid_to FROM memories WHERE id=?", (mem,)
    ).fetchone()
    assert row["valid_to"] is None


# ---------------------------------------------------------------------------
# end_fact
# ---------------------------------------------------------------------------

def test_end_fact_closes_without_replacement(conn):
    facts.add_fact(conn, "task", "status", "open")
    closed = facts.end_fact(conn, "task", "status")
    assert closed == 1
    assert _current(conn, "task", "status") == []

    # Idempotent: nothing left to close.
    assert facts.end_fact(conn, "task", "status") == 0

    # The closed row has no superseded_by (nothing replaced it).
    row = conn.execute(
        "SELECT valid_until, superseded_by FROM facts "
        "WHERE subject='task' AND predicate='status'"
    ).fetchone()
    assert row["valid_until"] is not None
    assert row["superseded_by"] is None


# ---------------------------------------------------------------------------
# query_facts filters + current_facts_for
# ---------------------------------------------------------------------------

def test_query_facts_filters(conn):
    facts.add_fact(conn, "a", "likes", "x")
    facts.add_fact(conn, "a", "hates", "y")
    facts.add_fact(conn, "b", "likes", "x")

    assert {f.predicate for f in facts.query_facts(conn, subject="a")} == {"likes", "hates"}
    assert {f.subject for f in facts.query_facts(conn, predicate="likes")} == {"a", "b"}
    by_obj = facts.query_facts(conn, object="x")
    assert {f.subject for f in by_obj} == {"a", "b"}
    assert all(f.object == "x" for f in by_obj)


def test_current_facts_for(conn):
    facts.add_fact(conn, "alice", "role", "eng")
    facts.add_fact(conn, "alice", "city", "sf")
    facts.add_fact(conn, "alice", "role", "manager")  # supersedes eng
    facts.add_fact(conn, "bob", "role", "eng")

    result = facts.current_facts_for(conn, "alice")
    pairs = {(f.predicate, f.object) for f in result}
    assert pairs == {("role", "manager"), ("city", "sf")}


# ---------------------------------------------------------------------------
# reasoning_chain
# ---------------------------------------------------------------------------

def test_reasoning_chain_walks_supersedes(conn):
    mem_old = seed_memory(conn, "old belief", kind="fact")
    mem_new = seed_memory(conn, "new belief", kind="fact")
    conn.execute(
        "UPDATE memories SET supersedes_id=?, source_refs=? WHERE id=?",
        (mem_old, json.dumps(["ep-42"]), mem_new),
    )
    conn.execute(
        "UPDATE memories SET source_refs=? WHERE id=?",
        (json.dumps(["ep-1"]), mem_old),
    )
    conn.commit()

    chain = facts.reasoning_chain(conn, mem_new)
    assert [c["id"] for c in chain] == [mem_new, mem_old]
    assert chain[0]["content"] == "new belief"
    assert chain[0]["source_refs"] == ["ep-42"]
    assert chain[1]["source_refs"] == ["ep-1"]


def test_reasoning_chain_single_row(conn):
    mem = seed_memory(conn, "lonely fact", kind="fact")
    chain = facts.reasoning_chain(conn, mem)
    assert len(chain) == 1
    assert chain[0]["id"] == mem
    assert chain[0]["supersedes_id"] is None


def test_reasoning_chain_missing_memory(conn):
    assert facts.reasoning_chain(conn, 999999) == []
