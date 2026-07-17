"""Capture-path tests: episodes + buffer writes, salience ordering, memory
writes, boundary markers, and the never-raise guarantee (hard rule 5 — a
memory bug must never break the agent's turn).
"""

from __future__ import annotations

from brain.capture.turns import (
    TurnContext,
    capture_delegation,
    capture_memory_write,
    capture_pre_compress,
    capture_session_end,
    capture_turn,
)
from brain.store import db


def _ctx(session_id="s1", turn_no=1):
    return TurnContext(session_id=session_id, turn_no=turn_no, platform="cli",
                       trust_tier="owner")


def _count(conn, table, where="1=1", args=()):
    return conn.execute(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", args).fetchone()["n"]


def test_capture_turn_writes_episode_buffer_and_fts(conn):
    episode_id = capture_turn(
        conn, _ctx(),
        "How do I debug the flux capacitor overload?",
        "Check `flux_capacitor.py` for the overload guard and rerun the bench.",
    )
    assert episode_id is not None

    episodes = conn.execute("SELECT * FROM episodes").fetchall()
    assert len(episodes) == 1
    row = episodes[0]
    assert row["session_id"] == "s1"
    assert "flux capacitor" in row["user_content"]
    assert row["token_len"] > 0

    buffer_rows = conn.execute(
        "SELECT kind, episode_id FROM ingest_buffer WHERE kind='turn'"
    ).fetchall()
    assert len(buffer_rows) == 1
    assert buffer_rows[0]["episode_id"] == episode_id

    if db.capabilities(conn).get("fts5"):
        hits = conn.execute(
            "SELECT rowid FROM episode_fts WHERE episode_fts MATCH ?", ("capacitor",)
        ).fetchall()
        assert len(hits) >= 1


def test_empty_content_skipped(conn):
    assert capture_turn(conn, _ctx(turn_no=1), "", "") is None
    assert capture_turn(conn, _ctx(turn_no=2), "   \n ", "") is None
    assert _count(conn, "episodes") == 0
    assert _count(conn, "ingest_buffer") == 0


def test_salience_correction_beats_pleasantry(conn):
    correction_user = "No, that's wrong — use the retry queue, not a sleep loop."
    capture_turn(conn, _ctx(turn_no=1), correction_user,
                 "You're right, switching to the retry queue.")
    capture_turn(conn, _ctx(turn_no=2), "thanks!", "anytime!")

    def salience_of(user_content):
        row = conn.execute(
            "SELECT salience FROM episodes WHERE user_content=?", (user_content,)
        ).fetchone()
        assert row is not None, f"turn not captured: {user_content!r}"
        return row["salience"]

    assert salience_of(correction_user) > salience_of("thanks!")


def test_memory_write_add_then_duplicate_bumps_verification(conn):
    content = "User prefers tabs over spaces in Makefiles."

    capture_memory_write(conn, _ctx(), "add", "preferences", content, None)
    rows_after_first = _count(conn, "memories")
    assert rows_after_first >= 1

    capture_memory_write(conn, _ctx(), "add", "preferences", content, None)
    assert _count(conn, "memories") == rows_after_first, "duplicate add must not add a row"

    row = conn.execute(
        "SELECT verification_count FROM memories WHERE content_hash=? AND valid_to IS NULL",
        (db.content_hash(content),),
    ).fetchone()
    assert row is not None
    assert row["verification_count"] == 2


def test_memory_write_remove_tombstones(conn):
    content = "The staging server lives at 10.0.0.5."
    capture_memory_write(conn, _ctx(), "add", "infra", content, None)
    capture_memory_write(conn, _ctx(), "remove", "infra", content, None)

    h = db.content_hash(content)
    active = _count(conn, "memories",
                    "content_hash=? AND status='active' AND valid_to IS NULL", (h,))
    assert active == 0, "removed memory must not stay active"
    assert _count(conn, "memories", "status='tombstone'") >= 1


def test_boundary_captures_land_with_right_kinds(conn):
    messages = [{"role": "user", "content": "wrap it up"},
                {"role": "assistant", "content": "done — summary above"}]
    capture_session_end(conn, "s1")
    capture_delegation(conn, _ctx(), "summarize the error logs",
                       "3 recurring KeyErrors found in api/users.py", "child-1")
    capture_pre_compress(conn, "s1", messages)

    kinds = {r["kind"] for r in conn.execute("SELECT kind FROM ingest_buffer").fetchall()}
    assert {"session_end_marker", "delegation", "pre_compress"} <= kinds


def test_capture_never_raises_on_closed_connection(tmp_home):
    closed = db.connect(tmp_home)
    closed.close()

    # Hard rule 5: capture must catch, log, and return — never raise.
    assert capture_turn(closed, _ctx(), "does this explode?", "it must not.") is None
    assert capture_memory_write(closed, _ctx(), "add", "x", "never raises", None) is None
    capture_session_end(closed, "s1")
    capture_delegation(closed, _ctx(), "t", "r", "c")
    capture_pre_compress(closed, "s1", [])
