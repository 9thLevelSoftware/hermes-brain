"""Dream 'mine' strategy: retrieval_log ⋈ state.db turn_outcomes.

Fixtures build a real-column state.db (sessions/messages/turn_outcomes per
hermes-agent hermes_state.py) plus injected retrieval_log rows whose
user_msg_hash matches a state.db user message. Covers: helpful++ for a
'verified' turn, harmful++ for a 'failed' turn, resolved_turn_id set,
unresolved rows tolerated, dry_run mutating nothing, and the skip paths.
"""

from __future__ import annotations

import sqlite3
import time

from brain.dream.mine_state import run as mine_run
from brain.dream.shift import Shift
from brain.store import db
from conftest import seed_memory

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_STATE_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    timestamp REAL NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE turn_outcomes (
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    outcome TEXT NOT NULL,
    outcome_reason TEXT,
    feedback_kind TEXT,
    feedback_value TEXT,
    PRIMARY KEY (session_id, turn_id)
);
"""


def make_shift(conn, config=None):
    from brain.dream import lease
    lease.acquire(conn, "dream", "test-holder")  # a real shift holds the lease
    return Shift(
        shift_id=db.new_ulid(), conn=conn, config=dict(config or {}),
        started_at=db.iso_now(), activity_baseline="9999-12-31T00:00:00.000Z",
        holder="test-holder",
    )


def make_state_db(home, *, messages=(), outcomes=(), with_outcomes_table=True):
    """messages: (session_id, role, content, epoch). outcomes: (session_id,
    turn_id, created_at, outcome, feedback_kind, feedback_value)."""
    path = home / "state.db"
    state = sqlite3.connect(str(path))
    try:
        schema = _STATE_SCHEMA
        if not with_outcomes_table:
            schema = schema.split("CREATE TABLE turn_outcomes")[0]
        state.executescript(schema)
        sessions = {m[0] for m in messages} | {o[0] for o in outcomes}
        for sid in sessions:
            state.execute(
                "INSERT OR IGNORE INTO sessions (id, source, started_at, ended_at)"
                " VALUES (?,?,?,?)", (sid, "cli", time.time() - 3600, time.time()))
        for sid, role, content, ts in messages:
            state.execute(
                "INSERT INTO messages (session_id, role, content, timestamp)"
                " VALUES (?,?,?,?)", (sid, role, content, ts))
        for sid, turn_id, created_at, outcome, fk, fv in outcomes:
            state.execute(
                "INSERT INTO turn_outcomes (session_id, turn_id, created_at,"
                " outcome, feedback_kind, feedback_value) VALUES (?,?,?,?,?,?)",
                (sid, turn_id, created_at, outcome, fk, fv))
        state.commit()
    finally:
        state.close()
    return path


def log_injection(conn, session_id, user_msg, memory_id, *, ts=None, injected=1):
    cur = conn.execute(
        "INSERT INTO retrieval_log (session_id, user_turn_count, query_hash,"
        " user_msg_hash, ts, memory_id, leg, rank_score, injected)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (session_id, 1, db.content_hash(user_msg), db.content_hash(user_msg),
         ts or db.iso_now(), memory_id, "fts", 1.0, injected))
    conn.commit()
    return cur.lastrowid


def counters(conn, mem_id):
    row = conn.execute(
        "SELECT helpful_count, harmful_count FROM memories WHERE id=?",
        (mem_id,)).fetchone()
    return row["helpful_count"], row["harmful_count"]


def seed_loop_fixture(conn, tmp_home):
    """Two resolvable injections (verified + failed turns) and one that can
    never resolve. Returns (m_good, m_bad, rlog ids)."""
    now = time.time()
    m_good = seed_memory(conn, "user prefers uv over pip")
    m_bad = seed_memory(conn, "the deploy script lives in /opt/deploy")
    make_state_db(
        tmp_home,
        messages=[
            ("s1", "user", "please fix the failing import", now - 600),
            ("s1", "assistant", "done, tests pass", now - 590),
            ("s1", "user", "now deploy it to staging", now - 300),
            ("s1", "assistant", "deploy attempted", now - 290),
        ],
        outcomes=[
            ("s1", "t-1", now - 580, "verified", None, None),
            ("s1", "t-2", now - 280, "failed", None, None),
        ],
    )
    r1 = log_injection(conn, "s1", "please fix the failing import", m_good,
                       ts=_iso(now - 600))
    r2 = log_injection(conn, "s1", "now deploy it to staging", m_bad,
                       ts=_iso(now - 300))
    r3 = log_injection(conn, "s1", "a query that matches no message", m_good,
                       ts=_iso(now - 200))
    return m_good, m_bad, (r1, r2, r3)


def _iso(epoch):
    ms = int((epoch % 1) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(epoch)) + f".{ms:03d}Z"


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_mine_credits_helpful_and_harmful(conn, tmp_home):
    m_good, m_bad, (r1, r2, r3) = seed_loop_fixture(conn, tmp_home)
    shift = make_shift(conn, {"hermes_home": str(tmp_home),
                              "_forced_mode": "active"})
    result = mine_run(shift)

    assert result == {"resolved": 2, "helpful": 1, "harmful": 1, "unresolved": 1}
    assert counters(conn, m_good) == (1, 0)
    assert counters(conn, m_bad) == (0, 1)

    resolved = {row["id"]: row["resolved_turn_id"] for row in conn.execute(
        "SELECT id, resolved_turn_id FROM retrieval_log")}
    assert resolved[r1] == "t-1"
    assert resolved[r2] == "t-2"
    assert resolved[r3] is None  # unresolved tolerated

    audits = conn.execute(
        "SELECT action FROM audit_log WHERE action='mine_credit'").fetchall()
    assert len(audits) == 1


def test_mine_never_double_counts(conn, tmp_home):
    m_good, m_bad, _ = seed_loop_fixture(conn, tmp_home)
    cfg = {"hermes_home": str(tmp_home), "_forced_mode": "active"}
    mine_run(make_shift(conn, cfg))
    result = mine_run(make_shift(conn, cfg))  # second run: nothing new

    assert result["resolved"] == 0
    assert result["helpful"] == 0 and result["harmful"] == 0
    assert counters(conn, m_good) == (1, 0)
    assert counters(conn, m_bad) == (0, 1)


def test_mine_negative_feedback_overrides_outcome(conn, tmp_home):
    now = time.time()
    mem = seed_memory(conn, "always run ruff before committing")
    make_state_db(
        tmp_home,
        messages=[("s2", "user", "clean up the linting", now - 100)],
        outcomes=[("s2", "t-9", now - 90, "completed_unverified",
                   "reaction", "thumbs_down")],
    )
    log_injection(conn, "s2", "clean up the linting", mem, ts=_iso(now - 100))
    result = mine_run(make_shift(conn, {"hermes_home": str(tmp_home),
                                        "_forced_mode": "active"}))
    # thumbs_down overrides the neutral outcome label
    assert result == {"resolved": 1, "helpful": 0, "harmful": 1, "unresolved": 0}
    assert counters(conn, mem) == (0, 1)


def test_mine_dry_run_mutates_nothing(conn, tmp_home):
    m_good, m_bad, (r1, r2, r3) = seed_loop_fixture(conn, tmp_home)
    result = mine_run(make_shift(conn, {"hermes_home": str(tmp_home),
                                        "_forced_mode": "dry_run"}))

    # counts are honest...
    assert result == {"resolved": 2, "helpful": 1, "harmful": 1, "unresolved": 1}
    # ...but nothing moved
    assert counters(conn, m_good) == (0, 0)
    assert counters(conn, m_bad) == (0, 0)
    assert conn.execute(
        "SELECT count(*) AS n FROM retrieval_log WHERE resolved_turn_id"
        " IS NOT NULL").fetchone()["n"] == 0
    assert conn.execute(
        "SELECT count(*) AS n FROM sweep_state WHERE key='mine:watermark'"
    ).fetchone()["n"] == 0
    # intent recorded
    assert conn.execute(
        "SELECT count(*) AS n FROM audit_log WHERE action='would_credit'"
    ).fetchone()["n"] == 1


def test_mine_skips_without_state_db(conn, tmp_home):
    assert mine_run(make_shift(conn, {"hermes_home": str(tmp_home)})) == \
        {"skipped": "no_state_db"}


def test_mine_skips_without_hermes_home(conn):
    assert mine_run(make_shift(conn, {})) == {"skipped": "no_hermes_home"}


def test_mine_tolerates_missing_outcomes_table(conn, tmp_home):
    make_state_db(tmp_home, messages=[("s1", "user", "hi", time.time())],
                  with_outcomes_table=False)
    assert mine_run(make_shift(conn, {"hermes_home": str(tmp_home)})) == \
        {"skipped": "no_state_db"}
