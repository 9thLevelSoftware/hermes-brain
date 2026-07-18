"""A2 graph/PPR retrieval leg: entity population, Personalized PageRank over
the co-mention graph, and the end-to-end search integration where the graph
surfaces a memory that shares an entity with the keyword hits but contains
none of the query words.
"""

from __future__ import annotations

from brain.recall.graph import ppr_leg
from brain.recall.search import search
from brain.store import entities
from conftest import seed_memory


def _uid(conn, mem_id):
    return conn.execute("SELECT uid FROM memories WHERE id=?", (mem_id,)).fetchone()["uid"]


# ---------------------------------------------------------------------------
# entities module
# ---------------------------------------------------------------------------

def test_link_dedups_and_counts(conn):
    a = seed_memory(conn, "first mention of projectx")
    b = seed_memory(conn, "second doc about projectx")
    e1 = entities.link(conn, "ProjectX", a)
    e2 = entities.link(conn, "  projectx ", b)   # normalizes to the same entity
    assert e1 == e2                              # one global entity
    # re-linking the same (entity, memory) is idempotent
    entities.link(conn, "projectx", a)
    row = conn.execute("SELECT mention_count FROM entities WHERE id=?", (e1,)).fetchone()
    assert row["mention_count"] == 2             # two distinct memories, not three


def test_entities_and_memories_lookup(conn):
    a = seed_memory(conn, "alpha")
    b = seed_memory(conn, "beta")
    ent = entities.link(conn, "shared-thing", a)
    entities.link(conn, "shared-thing", b)
    assert entities.entities_of(conn, [a]) == {ent}
    assert set(entities.memories_of(conn, [ent])) == {a, b}
    assert entities.co_mentioned(conn, [a]) == [b]   # b shares the entity, a excluded


def test_link_rejects_blank(conn):
    a = seed_memory(conn, "x")
    assert entities.link(conn, "   ", a) is None
    assert entities.link(conn, "thing", None) is None


# ---------------------------------------------------------------------------
# PPR leg in isolation
# ---------------------------------------------------------------------------

def test_ppr_discovers_co_mentioned(conn):
    a = seed_memory(conn, "seed memory a")
    b = seed_memory(conn, "connected memory b")
    c = seed_memory(conn, "unrelated memory c")
    entities.link(conn, "topic", a)
    entities.link(conn, "topic", b)          # a and b share 'topic'; c does not
    ranked = ppr_leg(conn, [a])
    assert b in ranked
    assert c not in ranked
    assert a not in ranked                   # seeds are excluded from the leg


def test_ppr_empty_when_isolated(conn):
    a = seed_memory(conn, "lonely memory")   # no entities, no edges
    assert ppr_leg(conn, [a]) == []
    assert ppr_leg(conn, []) == []


# ---------------------------------------------------------------------------
# end-to-end through search()
# ---------------------------------------------------------------------------

def test_graph_leg_surfaces_entity_neighbor(conn):
    # 'hit' matches the query; 'neighbor' shares an entity but has NO query words.
    hit = seed_memory(conn, "deploy the staging runbook")
    neighbor = seed_memory(conn, "the alpha release checklist")
    entities.link(conn, "projectx", hit)
    entities.link(conn, "projectx", neighbor)

    with_graph = search(conn, "deploy staging", include_episodes=False, graph=True)
    without = search(conn, "deploy staging", include_episodes=False, graph=False)

    uids_with = {h.uid for h in with_graph}
    uids_without = {h.uid for h in without}
    nbr_uid = _uid(conn, neighbor)
    assert nbr_uid in uids_with          # graph discovered it
    assert nbr_uid not in uids_without   # keyword alone did not
    # provenance names the graph leg
    src = next(h.source for h in with_graph if h.uid == nbr_uid)
    assert "ppr" in src
