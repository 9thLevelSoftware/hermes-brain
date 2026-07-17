"""assemble_episodes: the shared task-episode assembler (mine_state.py).

Both P5 learning strategies read episodes, so its boundary rules — terminal
outcome closes, session change closes, a long gap closes, feedback overrides
the machine label — are load-bearing. Fixtures build a real-column state.db.
"""

from __future__ import annotations

import sqlite3
import time

from brain.dream.mine_state import assemble_episodes, episode_verdict, open_state_ro

# The full turn_outcomes DDL (hermes_state.py:815-838). The P4 mine fixture
# uses a 6-column subset; episode assembly reads skills_loaded/model/api_calls
# etc., so it needs the real 21-column shape.
_STATE_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, user_id TEXT,
    started_at REAL NOT NULL, ended_at REAL
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
    timestamp REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE turn_outcomes (
    session_id TEXT NOT NULL, turn_id TEXT NOT NULL, created_at REAL NOT NULL,
    outcome TEXT NOT NULL, outcome_reason TEXT, turn_exit_reason TEXT,
    api_calls INTEGER NOT NULL DEFAULT 0, tool_iterations INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0, guardrail_halt TEXT,
    cost_usd_delta REAL NOT NULL DEFAULT 0, input_tokens_delta INTEGER NOT NULL DEFAULT 0,
    output_tokens_delta INTEGER NOT NULL DEFAULT 0, cache_read_tokens_delta INTEGER NOT NULL DEFAULT 0,
    skills_loaded TEXT, model TEXT, feedback_kind TEXT, feedback_value TEXT,
    feedback_source TEXT, feedback_at REAL, feedback_event_id TEXT,
    PRIMARY KEY (session_id, turn_id)
);
"""


def make_state_db(tmp_home, *, messages=(), outcomes=(), with_outcomes_table=True,
                  source="cli"):
    """messages: (session_id, role, content, epoch).
    outcomes: (session_id, turn_id, created_at, outcome, feedback_kind,
    feedback_value[, skills_loaded, model, tool_iterations, cost])."""
    path = tmp_home / "state.db"
    state = sqlite3.connect(str(path))
    try:
        schema = _STATE_SCHEMA
        if not with_outcomes_table:
            schema = schema.split("CREATE TABLE turn_outcomes")[0]
        state.executescript(schema)
        sessions = {m[0] for m in messages} | {o[0] for o in outcomes}
        for sid in sessions:
            state.execute("INSERT OR IGNORE INTO sessions (id, source, started_at)"
                          " VALUES (?,?,?)", (sid, source, time.time() - 3600))
        for sid, role, content, ts in messages:
            state.execute("INSERT INTO messages (session_id, role, content, timestamp)"
                          " VALUES (?,?,?,?)", (sid, role, content, ts))
        for o in outcomes:
            sid, turn_id, created_at, outcome, fk, fv = o[:6]
            skills, model, tools_it, cost = (o + (None, None, 0, 0.0))[6:10]
            state.execute(
                "INSERT INTO turn_outcomes (session_id, turn_id, created_at, outcome,"
                " feedback_kind, feedback_value, skills_loaded, model, tool_iterations,"
                " cost_usd_delta) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (sid, turn_id, created_at, outcome, fk, fv, skills, model,
                 tools_it, cost))
        state.commit()
    finally:
        state.close()
    return path


def _episodes(tmp_home, *, include_open=False, **kw):
    state = open_state_ro(make_state_db(tmp_home, **kw))
    try:
        return assemble_episodes(state, include_open=include_open)
    finally:
        state.close()


def test_terminal_outcome_closes_an_episode(tmp_home):
    now = time.time()
    eps = _episodes(
        tmp_home,
        messages=[("s1", "user", "add retry logic", now - 100),
                  ("s1", "assistant", "done", now - 95)],
        outcomes=[("s1", "t-1", now - 90, "verified", None, None)],
    )
    assert len(eps) == 1
    ep = eps[0]
    assert ep.verdict == "success"
    assert ep.closing_turn_id == "t-1"
    assert ep.user_goal == "add retry logic"
    assert ("assistant", "done") in ep.transcript


def test_feedback_overrides_the_machine_label(tmp_home):
    now = time.time()
    eps = _episodes(
        tmp_home,
        messages=[("s1", "user", "ship it", now - 50)],
        outcomes=[("s1", "t-1", now - 40, "completed_unverified",
                   "reaction", "thumbs_down")],
    )
    # completed_unverified is not terminal, so this only appears as an open
    # run; but the verdict must reflect the thumbs-down.
    assert episode_verdict("completed_unverified", "reaction", "thumbs_down") == "failure"
    assert episode_verdict("verified", None, None) == "success"
    assert episode_verdict("partial", None, None) == "ambiguous"
    assert episode_verdict("completed_unverified", "reaction", "thumbs_up") == "success"
    # verified but thumbs_down -> the human wins.
    assert episode_verdict("verified", "reaction", "\U0001f44e") == "failure"
    assert eps == [] or eps[0].verdict == "failure"


def test_two_sessions_never_merge(tmp_home):
    now = time.time()
    eps = _episodes(
        tmp_home,
        messages=[("s1", "user", "task A", now - 200),
                  ("s2", "user", "task B", now - 100)],
        outcomes=[("s1", "t-1", now - 190, "verified", None, None),
                  ("s2", "t-2", now - 90, "failed", None, None)],
    )
    assert len(eps) == 2
    by_session = {e.session_id: e for e in eps}
    assert by_session["s1"].verdict == "success"
    assert by_session["s2"].verdict == "failure"


def test_multi_turn_episode_accumulates_until_it_closes(tmp_home):
    now = time.time()
    eps = _episodes(
        tmp_home,
        messages=[("s1", "user", "start the migration", now - 300),
                  ("s1", "user", "now roll it back", now - 200),
                  ("s1", "user", "ok try again", now - 100)],
        outcomes=[("s1", "t-1", now - 290, "partial", None, None),
                  ("s1", "t-2", now - 190, "partial", None, None),
                  ("s1", "t-3", now - 90, "verified", None, None)],
    )
    assert len(eps) == 1
    ep = eps[0]
    assert ep.turn_ids == ("t-1", "t-2", "t-3")
    assert ep.verdict == "success"          # the CLOSING turn's label
    assert ep.user_goal == "start the migration"


def test_a_long_gap_splits_one_session_into_two_episodes(tmp_home):
    now = time.time()
    eps = _episodes(
        tmp_home,
        messages=[("s1", "user", "morning task", now - 10000),
                  ("s1", "user", "evening task", now - 100)],
        # first partial, big gap, then a terminal — the gap must split them
        outcomes=[("s1", "t-1", now - 9990, "partial", None, None),
                  ("s1", "t-2", now - 90, "verified", None, None)],
    )
    # t-1's run is open (partial, never terminal) and separated by >2h;
    # t-2 closes its own episode. Two distinct episodes, not one spanning 3h.
    assert len(eps) == 2
    assert [e.closing_turn_id for e in eps] == ["t-1", "t-2"]


def test_open_run_excluded_by_default_included_on_request(tmp_home):
    now = time.time()
    state = open_state_ro(make_state_db(
        tmp_home,
        messages=[("s1", "user", "in-flight task", now - 50)],
        outcomes=[("s1", "t-1", now - 40, "partial", None, None)],
    ))
    try:
        assert assemble_episodes(state) == []               # open run hidden
        opened = assemble_episodes(state, include_open=True)
        assert len(opened) == 1 and opened[0].verdict == "ambiguous"
    finally:
        state.close()


def test_since_epoch_watermark_is_respected(tmp_home):
    now = time.time()
    state = open_state_ro(make_state_db(
        tmp_home,
        messages=[("s1", "user", "old", now - 5000),
                  ("s2", "user", "new", now - 100)],
        outcomes=[("s1", "t-1", now - 4990, "verified", None, None),
                  ("s2", "t-2", now - 90, "verified", None, None)],
    ))
    try:
        recent = assemble_episodes(state, since_epoch=now - 1000)
        assert [e.closing_turn_id for e in recent] == ["t-2"]
    finally:
        state.close()


def test_no_state_tables_returns_empty(tmp_home):
    state = open_state_ro(make_state_db(tmp_home, messages=[
        ("s1", "user", "hi", time.time())], with_outcomes_table=False))
    try:
        assert assemble_episodes(state) == []
    finally:
        state.close()
