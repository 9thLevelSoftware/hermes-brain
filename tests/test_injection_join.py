"""The injection -> outcome join, driven through the REAL hook sequence.

This is the loop-closing edge the whole learning system rests on, and it was
broken in a way the P4 tests could not see: they called `log_injection()`
directly with a real hash, while production went through
`provider._do_retrieve` -> `log_retrieval(..., user_msg=None)`, writing
`user_msg_hash = sha256("")` on every row. The miner joins on that hash
against state.db `messages`, so it matched nothing, forever — with green
tests either side of the gap.

So these tests drive the host's documented call order instead of hand-built
rows (memory_manager.py docstring; the single-worker executor at :376-379
guarantees it):

    turn N:  prefetch_all(msg_N)          <- serves the cached block
             sync_all(msg_N, resp_N)      <- capture
             queue_prefetch_all(msg_N)    <- brain computes the NEXT block

The block computed after turn N is injected into turn N+1. Attribution must
follow the injection, not the query that produced it.
"""

from __future__ import annotations

from brain.recall.search import Hit, log_retrieval, stamp_pending_injections
from brain.store import db
from conftest import seed_memory


def _hit(conn, mem_id):
    row = conn.execute("SELECT uid FROM memories WHERE id=?", (mem_id,)).fetchone()
    return Hit(kind="memory", id=mem_id, uid=row["uid"], text="t", summary=None,
               memory_type="semantic", mkind="fact", ts=db.iso_now(),
               platform="cli", score=1.0, source="fts")


def _rows(conn):
    return conn.execute(
        "SELECT * FROM retrieval_log ORDER BY id").fetchall()


# -- the regression: a logged row must be joinable at all ---------------------

def test_logged_row_is_pending_not_hashed_to_empty_string(conn):
    """The original bug: every row carried sha256("") and could never match."""
    mem = seed_memory(conn, "the prod region is us-east-1")
    hit = _hit(conn, mem)
    log_retrieval(conn, "s1", "which region", [hit], {hit.uid})

    row = _rows(conn)[0]
    assert row["injected"] == 1
    assert row["user_msg_hash"] is None, "row must land pending, not pre-hashed"
    assert row["user_msg_hash"] != db.content_hash(""), "the P4 bug is back"


def test_stamp_attributes_the_block_to_the_next_turn(conn):
    """The off-by-one: a block computed from msg_N is injected into turn N+1."""
    mem = seed_memory(conn, "always cap the JVM heap at 2GB")
    hit = _hit(conn, mem)

    # Turn 1 completes -> brain computes the block for turn 2.
    log_retrieval(conn, "s1", "deploy staging", [hit], {hit.uid})
    # Turn 2 arrives: THIS is the message that saw the block.
    stamped = stamp_pending_injections(conn, "s1", 2, "now bump the heap and redeploy")
    conn.commit()

    assert stamped == 1
    row = _rows(conn)[0]
    assert row["user_turn_count"] == 2, "credited the wrong turn"
    assert row["user_msg_hash"] == db.content_hash("now bump the heap and redeploy")
    # ...and specifically NOT the query that produced the candidates.
    assert row["user_msg_hash"] != db.content_hash("deploy staging")


def test_stamp_only_touches_pending_rows_of_its_own_session(conn):
    mem = seed_memory(conn, "x")
    hit = _hit(conn, mem)
    log_retrieval(conn, "s1", "q", [hit], {hit.uid})
    log_retrieval(conn, "s2", "q", [hit], {hit.uid})
    stamp_pending_injections(conn, "s1", 2, "msg for s1")
    conn.commit()

    by_session = {r["session_id"]: r for r in _rows(conn)}
    assert by_session["s1"]["user_msg_hash"] == db.content_hash("msg for s1")
    assert by_session["s2"]["user_msg_hash"] is None


def test_already_stamped_rows_are_never_restamped(conn):
    """A later turn must not re-attribute an earlier turn's injection."""
    mem = seed_memory(conn, "x")
    hit = _hit(conn, mem)
    log_retrieval(conn, "s1", "q", [hit], {hit.uid})
    stamp_pending_injections(conn, "s1", 2, "turn two message")
    conn.commit()
    first = db.content_hash("turn two message")

    # Turn 3 arrives with no new retrieve in between: nothing pending.
    assert stamp_pending_injections(conn, "s1", 3, "turn three message") == 0
    conn.commit()
    row = _rows(conn)[0]
    assert row["user_turn_count"] == 2
    assert row["user_msg_hash"] == first


def test_superseded_block_is_demoted_not_credited(conn):
    """Only the most recent block is served, so an older pending batch was
    never injected and must not be stamped into the next turn."""
    old = seed_memory(conn, "stale candidate")
    new = seed_memory(conn, "fresh candidate")
    h_old, h_new = _hit(conn, old), _hit(conn, new)

    log_retrieval(conn, "s1", "first query", [h_old], {h_old.uid})
    log_retrieval(conn, "s1", "second query", [h_new], {h_new.uid})   # supersedes
    stamp_pending_injections(conn, "s1", 2, "the turn that saw the block")
    conn.commit()

    by_mem = {r["memory_id"]: r for r in _rows(conn)}
    assert by_mem[old]["injected"] == 0, "superseded block was credited anyway"
    assert by_mem[new]["injected"] == 1


def test_unconsumed_block_stays_unresolvable(conn):
    """Session ends before the next turn: the block was never injected, so it
    must stay pending (the miner tolerates it) rather than credit anything."""
    mem = seed_memory(conn, "x")
    hit = _hit(conn, mem)
    log_retrieval(conn, "s1", "q", [hit], {hit.uid})
    conn.commit()
    row = _rows(conn)[0]
    assert row["user_msg_hash"] is None
    assert row["user_turn_count"] is None


def test_empty_user_message_stamps_nothing(conn):
    mem = seed_memory(conn, "x")
    hit = _hit(conn, mem)
    log_retrieval(conn, "s1", "q", [hit], {hit.uid})
    assert stamp_pending_injections(conn, "s1", 2, "   ") == 0
    assert _rows(conn)[0]["user_msg_hash"] is None


# -- end to end: the miner can now actually resolve a real provider trace -----

def test_miner_resolves_a_trace_written_by_the_real_provider_path(conn, tmp_home):
    """The proof the P4 suite missed: rows written the way the provider writes
    them, mined against a real state.db, must resolve and credit."""
    import time as _time

    from brain.dream.mine_state import run as mine_run

    from tests.test_dream_mine import make_shift, make_state_db

    now = _time.time()
    mem = seed_memory(conn, "always cap the JVM heap at 2GB")
    hit = _hit(conn, mem)
    user_msg = "deploy the payments service to staging"

    # state.db: the user message + the turn outcome it produced.
    make_state_db(
        tmp_home,
        messages=[("s1", "user", user_msg, now - 300)],
        outcomes=[("s1", "t-42", now - 280, "verified", None, None)],
    )

    # The provider's real sequence: block computed after the previous turn...
    log_retrieval(conn, "s1", "previous turn query", [hit], {hit.uid})
    # ...then this turn arrives and stamps it with the RAW user text.
    stamp_pending_injections(conn, "s1", 1, user_msg)
    conn.execute("UPDATE retrieval_log SET ts=?",
                 (_time.strftime("%Y-%m-%dT%H:%M:%S",
                                 _time.gmtime(now - 300)) + ".000Z",))
    conn.commit()

    result = mine_run(make_shift(conn, {"hermes_home": str(tmp_home),
                                        "_forced_mode": "active"}))
    assert result == {"resolved": 1, "helpful": 1, "harmful": 0, "unresolved": 0}
    counts = conn.execute(
        "SELECT helpful_count FROM memories WHERE id=?", (mem,)).fetchone()
    assert counts["helpful_count"] == 1
    assert _rows(conn)[0]["resolved_turn_id"] == "t-42"
