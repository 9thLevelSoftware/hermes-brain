"""Facts retrieval leg (`recall/facts_leg.py`): probing the current-truth
`facts` index for the DISTINCT backing memory ids of triples whose subject or
object mentions a query token, ranked by confidence x recency.

Facts are seeded by direct INSERTs into the `facts` table (schema.sql migration
003) so the test does not depend on a sibling agent's `store/facts.py`.
"""

from __future__ import annotations

from brain.recall.facts_leg import facts_leg
from brain.store import db
from conftest import iso_days_ago, seed_memory


def seed_fact(conn, subject, predicate, obj, *, memory_id=None, confidence=1.0,
              valid_from=None, valid_until=None):
    now = db.iso_now()
    cur = conn.execute(
        "INSERT INTO facts (subject, predicate, object, memory_id, confidence,"
        " source, valid_from, valid_until, recorded_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (subject, predicate, obj, memory_id, confidence, "test",
         valid_from or now, valid_until, now),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Empty / degenerate inputs never raise
# ---------------------------------------------------------------------------

def test_empty_db_returns_empty(conn):
    assert facts_leg(conn, "anything") == []


def test_empty_query_returns_empty(conn):
    m = seed_memory(conn, "some memory")
    seed_fact(conn, "alpha", "is", "beta", memory_id=m)
    assert facts_leg(conn, "") == []
    assert facts_leg(conn, "   ") == []
    # Punctuation-only query has no word-run tokens -> no probe.
    assert facts_leg(conn, "!!!") == []


# ---------------------------------------------------------------------------
# Core: a token matching subject/object surfaces the backing memory
# ---------------------------------------------------------------------------

def test_subject_token_match_returns_memory_id(conn):
    m = seed_memory(conn, "capital fact")
    seed_fact(conn, "Paris", "capital_of", "France", memory_id=m)
    assert facts_leg(conn, "Paris") == [m]


def test_object_token_match_returns_memory_id(conn):
    m = seed_memory(conn, "capital fact")
    seed_fact(conn, "Paris", "capital_of", "France", memory_id=m)
    # matches on the OBJECT column, not the subject
    assert facts_leg(conn, "France") == [m]


def test_non_matching_query_returns_empty(conn):
    m = seed_memory(conn, "capital fact")
    seed_fact(conn, "Paris", "capital_of", "France", memory_id=m)
    assert facts_leg(conn, "Tokyo") == []


# ---------------------------------------------------------------------------
# Current-truth only: superseded facts (valid_until set) are excluded
# ---------------------------------------------------------------------------

def test_superseded_fact_is_not_returned(conn):
    old = seed_memory(conn, "old employer fact")
    seed_fact(conn, "Ada", "works_at", "OldCorp", memory_id=old,
              valid_from=iso_days_ago(100), valid_until=iso_days_ago(1))
    # The only matching fact is closed -> nothing current to surface.
    assert facts_leg(conn, "Ada") == []


def test_current_truth_survives_when_older_version_superseded(conn):
    old = seed_memory(conn, "old employer fact")
    new = seed_memory(conn, "new employer fact")
    seed_fact(conn, "Ada", "works_at", "OldCorp", memory_id=old,
              valid_from=iso_days_ago(100), valid_until=iso_days_ago(1))
    seed_fact(conn, "Ada", "works_at", "NewCorp", memory_id=new,
              valid_from=iso_days_ago(1))
    assert facts_leg(conn, "Ada") == [new]


# ---------------------------------------------------------------------------
# Ranking: confidence and recency both push a fact up
# ---------------------------------------------------------------------------

def test_higher_confidence_ranks_first(conn):
    lo = seed_memory(conn, "low confidence fact")
    hi = seed_memory(conn, "high confidence fact")
    vf = iso_days_ago(5)  # equal recency -> confidence decides
    seed_fact(conn, "topic", "rel", "lo-answer", memory_id=lo,
              confidence=0.2, valid_from=vf)
    seed_fact(conn, "topic", "rel", "hi-answer", memory_id=hi,
              confidence=0.95, valid_from=vf)
    assert facts_leg(conn, "topic") == [hi, lo]


def test_more_recent_ranks_first(conn):
    old = seed_memory(conn, "older fact")
    new = seed_memory(conn, "newer fact")
    conf = 0.8  # equal confidence -> recency decides
    seed_fact(conn, "topic", "rel", "old-answer", memory_id=old,
              confidence=conf, valid_from=iso_days_ago(300))
    seed_fact(conn, "topic", "rel", "new-answer", memory_id=new,
              confidence=conf, valid_from=iso_days_ago(1))
    assert facts_leg(conn, "topic") == [new, old]


# ---------------------------------------------------------------------------
# NULL memory_id facts are skipped (leg emits backing memory ids only)
# ---------------------------------------------------------------------------

def test_null_memory_id_is_skipped(conn):
    m = seed_memory(conn, "backed fact")
    seed_fact(conn, "orphan", "rel", "value", memory_id=None)
    seed_fact(conn, "orphan", "rel2", "value2", memory_id=m)
    # Only the fact with a backing memory contributes an id.
    assert facts_leg(conn, "orphan") == [m]


def test_all_null_memory_ids_returns_empty(conn):
    seed_fact(conn, "orphan", "rel", "value", memory_id=None)
    assert facts_leg(conn, "orphan") == []


# ---------------------------------------------------------------------------
# De-duplication and limit
# ---------------------------------------------------------------------------

def test_distinct_memory_id_dedup(conn):
    m = seed_memory(conn, "one memory, two facts")
    seed_fact(conn, "widget", "color", "blue", memory_id=m)
    seed_fact(conn, "widget", "size", "large", memory_id=m)
    # Two matching facts, one backing memory -> a single id, not duplicated.
    assert facts_leg(conn, "widget") == [m]


def test_limit_is_honored(conn):
    for i in range(6):
        mi = seed_memory(conn, f"fact number {i}")
        seed_fact(conn, "shared", "idx", str(i), memory_id=mi,
                  valid_from=iso_days_ago(i))
    out = facts_leg(conn, "shared", limit=3)
    assert len(out) == 3
