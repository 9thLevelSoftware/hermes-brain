"""Bootstrap tests: §-file import, state.db backfill (watermarks, pairing,
rate cap), Daem0n migration mapping, and the never-raise orchestration.

All source data is fabricated on disk (no network, no models beyond the
StubEmbedder, sqlite_vec only where importorskip'd) and asserted READ-ONLY
by re-running every stage and expecting zero new rows.
"""

from __future__ import annotations

import sqlite3

import pytest
from brain.bootstrap import run_bootstrap
from brain.bootstrap.daemon_import import import_daemon_db
from brain.bootstrap.memory_md import import_memory_files
from brain.bootstrap.state_db import backfill_sessions
from conftest import seed_memory

DELIM = "\n§\n"


# ---------------------------------------------------------------------------
# Fabrication helpers
# ---------------------------------------------------------------------------

def write_memory_files(home, memory_entries=(), user_entries=()):
    mem_dir = home / "memories"
    mem_dir.mkdir(exist_ok=True)
    if memory_entries:
        (mem_dir / "MEMORY.md").write_text(DELIM.join(memory_entries), encoding="utf-8")
    if user_entries:
        (mem_dir / "USER.md").write_text(DELIM.join(user_entries), encoding="utf-8")


def build_state_db(home, sessions):
    """sessions: list of (session_id, source, started_at, messages[, extra]).

    ``extra`` is a dict with optional ``user_id`` and ``ended_at`` (an
    explicit ``ended_at=None`` fabricates a still-running session; the
    default is started_at + 60). Each message is (role, content) or
    (role, content, timestamp) or (role, content, timestamp, active) —
    mirroring hermes_state.py's real columns.
    """
    path = home / "state.db"
    con = sqlite3.connect(str(path))
    con.executescript(
        "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, source TEXT,"
        " started_at REAL, user_id TEXT, ended_at REAL);"
        "CREATE TABLE IF NOT EXISTS messages (session_id TEXT, role TEXT, content TEXT,"
        " timestamp REAL, active INTEGER NOT NULL DEFAULT 1);"
    )
    for entry in sessions:
        sid, source, started_at, messages = entry[:4]
        extra = entry[4] if len(entry) > 4 else {}
        ended_at = extra["ended_at"] if "ended_at" in extra else (started_at or 0) + 60.0
        con.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?)",
            (sid, source, started_at, extra.get("user_id"), ended_at),
        )
        for i, msg in enumerate(messages):
            role, content = msg[0], msg[1]
            ts = msg[2] if len(msg) > 2 else (started_at or 0) + float(i)
            active = msg[3] if len(msg) > 3 else 1
            con.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, active)"
                " VALUES (?,?,?,?,?)",
                (sid, role, content, ts, active),
            )
    con.commit()
    con.close()
    return path


def _session_messages(n_turns, tag):
    msgs = [("system", "you are hermes")]
    for i in range(1, n_turns + 1):
        msgs += [
            ("user", f"{tag} question {i} about the flux capacitor"),
            ("tool", '{"result": "tool noise to be skipped"}'),
            ("assistant", f"{tag} answer {i}: check the overload guard"),
        ]
    return msgs


def two_session_home(tmp_home):
    build_state_db(tmp_home, [
        ("sess-a", "cli", 100.0, _session_messages(3, "alpha")),
        ("sess-b", "telegram", 200.0, _session_messages(3, "beta")),
    ])
    return tmp_home


def build_daemon_db(tmp_path, rows, project="MyProject"):
    """rows: dicts with the Daem0n memories columns used by the importer."""
    db_dir = tmp_path / project / ".daem0nmcp"
    db_dir.mkdir(parents=True)
    path = db_dir / "memory.db"
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE memories (id INTEGER PRIMARY KEY, category TEXT, content TEXT,"
        " rationale TEXT, tags TEXT, outcome TEXT, worked INTEGER, pinned INTEGER,"
        " archived INTEGER, created_at TEXT)"
    )
    for r in rows:
        con.execute(
            "INSERT INTO memories (category, content, rationale, tags, outcome, worked,"
            " pinned, archived, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (r.get("category"), r.get("content"), r.get("rationale"), r.get("tags"),
             r.get("outcome"), r.get("worked"), r.get("pinned", 0),
             r.get("archived", 0), r.get("created_at")),
        )
    con.commit()
    con.close()
    return path


def _count(conn, table, where="1=1", args=()):
    return conn.execute(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", args).fetchone()["n"]


# ---------------------------------------------------------------------------
# memory_md
# ---------------------------------------------------------------------------

def test_memory_md_import_kinds_and_dedup(conn, tmp_home):
    seed_memory(conn, "Owner prefers tabs over spaces")  # pre-existing duplicate
    write_memory_files(
        tmp_home,
        memory_entries=[
            "Owner prefers tabs over spaces",          # dup -> skipped
            "The staging box is 10.0.0.7",
            "Deploys go through `make ship`,\nnever raw rsync",  # multiline entry
        ],
        user_entries=["Name: Devil", "Timezone: America/Chicago"],
    )
    counts = import_memory_files(conn, tmp_home)
    assert counts == {"memory": 2, "user": 2, "skipped": 1}

    fact = conn.execute(
        "SELECT * FROM memories WHERE content LIKE 'The staging box%'"
    ).fetchone()
    assert fact["kind"] == "fact"
    assert fact["memory_type"] == "semantic"
    assert fact["epistemic"] == "observation"
    assert fact["trust_tier"] == "owner"
    assert fact["created_by"] == "bootstrap"
    assert fact["tags"] == '["builtin-import"]'
    assert fact["scope_user"] is None

    profile = conn.execute(
        "SELECT * FROM memories WHERE content = 'Name: Devil'"
    ).fetchone()
    assert profile["kind"] == "profile"
    assert profile["scope_user"] == "owner"

    multiline = conn.execute(
        "SELECT 1 FROM memories WHERE content LIKE 'Deploys go through%raw rsync'"
    ).fetchone()
    assert multiline is not None


def test_memory_md_idempotent(conn, tmp_home):
    write_memory_files(tmp_home, memory_entries=["a fact", "another fact"])
    assert import_memory_files(conn, tmp_home)["memory"] == 2
    again = import_memory_files(conn, tmp_home)
    assert again == {"memory": 0, "user": 0, "skipped": 2}
    assert _count(conn, "memories") == 2


def test_memory_md_missing_files(conn, tmp_home):
    assert import_memory_files(conn, tmp_home) == {"memory": 0, "user": 0, "skipped": 0}
    assert _count(conn, "memories") == 0


# ---------------------------------------------------------------------------
# state_db backfill
# ---------------------------------------------------------------------------

def test_backfill_pairs_turns_and_skips_tool_rows(conn, tmp_home):
    two_session_home(tmp_home)
    counts = backfill_sessions(conn, tmp_home)
    assert counts["sessions"] == 2
    assert counts["turns"] == 6
    assert counts["skipped"] == 0

    assert _count(conn, "episodes") == 6
    row = conn.execute(
        "SELECT * FROM episodes WHERE session_id='sess-a' AND turn_no=2"
    ).fetchone()
    assert row["user_content"] == "alpha question 2 about the flux capacitor"
    assert row["assistant_content"].startswith("alpha answer 2")
    assert row["platform"] == "cli"
    assert row["trust_tier"] == "owner"
    assert _count(conn, "episodes", "user_content LIKE '%tool noise%'") == 0
    assert _count(conn, "episodes", "platform='telegram'") == 3

    # capture_turn reuse: buffer work units exist for the sweep to chew.
    assert _count(conn, "ingest_buffer", "kind='turn'") == 6
    # watermarks, one per session
    assert _count(conn, "sweep_state", "key LIKE 'bootstrap:%'") == 2


def test_backfill_idempotent(conn, tmp_home):
    two_session_home(tmp_home)
    backfill_sessions(conn, tmp_home)
    again = backfill_sessions(conn, tmp_home)
    assert again["sessions"] == 0
    assert again["turns"] == 0
    assert again["skipped"] == 2
    assert _count(conn, "episodes") == 6


def test_backfill_watermark_resume_new_session(conn, tmp_home):
    two_session_home(tmp_home)
    backfill_sessions(conn, tmp_home)

    # A third session appears later (agent kept running) — only it imports.
    build_state_db(tmp_home, [("sess-c", "discord", 300.0, _session_messages(2, "gamma"))])
    counts = backfill_sessions(conn, tmp_home)
    assert counts["sessions"] == 1
    assert counts["turns"] == 2
    assert counts["skipped"] == 2
    assert _count(conn, "episodes", "session_id='sess-c'") == 2
    assert _count(conn, "episodes") == 8


def test_backfill_max_sessions_cap_resumes_next_run(conn, tmp_home):
    two_session_home(tmp_home)
    first = backfill_sessions(conn, tmp_home, max_sessions=1)
    assert first["sessions"] == 1
    # Oldest first: sess-a (started_at 100) before sess-b (200).
    assert _count(conn, "episodes", "session_id='sess-a'") == 3
    assert _count(conn, "episodes", "session_id='sess-b'") == 0

    second = backfill_sessions(conn, tmp_home, max_sessions=1)
    assert second["sessions"] == 1
    assert second["skipped"] == 1
    assert _count(conn, "episodes") == 6


def test_backfill_missing_state_db(conn, tmp_home):
    counts = backfill_sessions(conn, tmp_home)
    assert counts["sessions"] == 0 and counts["turns"] == 0
    assert "note" in counts


def test_backfill_not_a_state_db(conn, tmp_home):
    con = sqlite3.connect(str(tmp_home / "state.db"))
    con.execute("CREATE TABLE unrelated (x TEXT)")
    con.commit()
    con.close()
    counts = backfill_sessions(conn, tmp_home)
    assert counts["sessions"] == 0 and counts["turns"] == 0
    assert "not a state.db" in counts["note"]
    assert _count(conn, "episodes") == 0


def test_backfill_embeds_episodes_with_stub(conn, tmp_home):
    pytest.importorskip("sqlite_vec")
    from brain.recall.embed import StubEmbedder
    from brain.store import vec

    two_session_home(tmp_home)
    counts = backfill_sessions(conn, tmp_home, embedder=StubEmbedder())
    assert counts["turns"] == 6
    stats = vec.stats(conn)
    assert stats and stats["epi_vec"] == 6


def test_backfill_decodes_structured_json_content(conn, tmp_home):
    """'\\x00json:' multimodal content: text parts joined, image parts
    replaced by '[image]' — base64 payloads never reach episodes."""
    import json as _json

    structured = "\x00json:" + _json.dumps([
        {"type": "text", "text": "what is in this picture?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 5000}},
    ])
    build_state_db(tmp_home, [
        ("sess-json", "cli", 100.0, [
            ("user", structured),
            ("assistant", "\x00json:" + _json.dumps(
                [{"type": "text", "text": "a flux capacitor"}])),
        ]),
    ])
    counts = backfill_sessions(conn, tmp_home)
    assert counts["turns"] == 1
    row = conn.execute("SELECT * FROM episodes WHERE session_id='sess-json'").fetchone()
    assert row["user_content"] == "what is in this picture?\n[image]"
    assert row["assistant_content"] == "a flux capacitor"
    assert "base64" not in row["user_content"]
    assert len(row["user_content"]) < 100


def test_backfill_decode_fallback_on_bad_json(conn, tmp_home):
    build_state_db(tmp_home, [
        ("sess-bad", "cli", 100.0, [
            ("user", "\x00json:{not valid json" + "x" * 10_000),
            ("assistant", "still answered"),
        ]),
    ])
    counts = backfill_sessions(conn, tmp_home)
    assert counts["turns"] == 1
    row = conn.execute("SELECT * FROM episodes WHERE session_id='sess-bad'").fetchone()
    assert row["user_content"].startswith("{not valid json")
    assert len(row["user_content"]) <= 4000  # truncated, no sentinel prefix


def test_backfill_trust_tiers_from_source_and_identities(conn, tmp_home):
    from brain.store import db as store_db

    # Enroll one telegram identity as the owner (critique item 33).
    conn.execute(
        "INSERT INTO identities (principal_id, platform, platform_user_id, is_owner,"
        " added_at, added_by) VALUES ('owner','telegram','777',1,?, 'cli')",
        (store_db.iso_now(),),
    )
    conn.commit()

    build_state_db(tmp_home, [
        ("s-cli", "cli", 100.0, _session_messages(1, "local")),
        ("s-known", "telegram", 200.0, _session_messages(1, "stranger"),
         {"user_id": "555"}),
        ("s-owner", "telegram", 300.0, _session_messages(1, "boss"),
         {"user_id": "777"}),
        ("s-nouser", "discord", 400.0, _session_messages(1, "ghost")),
    ])
    counts = backfill_sessions(conn, tmp_home)
    assert counts["sessions"] == 4

    def episode(sid):
        return conn.execute("SELECT * FROM episodes WHERE session_id=?", (sid,)).fetchone()

    local = episode("s-cli")
    assert local["trust_tier"] == "owner" and local["principal_id"] == "owner"

    known = episode("s-known")  # gateway user, NOT enrolled: never owner
    assert known["trust_tier"] == "known_user"
    assert known["principal_id"] is None
    assert known["source_author"] == "555"

    boss = episode("s-owner")  # enrolled with is_owner=1
    assert boss["trust_tier"] == "owner" and boss["principal_id"] == "owner"
    assert boss["source_author"] == "777"

    ghost = episode("s-nouser")  # gateway row without a user_id
    assert ghost["trust_tier"] == "known_user"
    assert ghost["principal_id"] is None and ghost["source_author"] is None


def test_backfill_historical_timestamps(conn, tmp_home):
    """Episodes keep the ORIGINAL conversation time; the buffer row keeps
    queue-arrival time (now)."""
    epoch_2001 = 999_999_999.0  # 2001-09-09T01:46:39Z
    build_state_db(tmp_home, [
        ("sess-old", "cli", epoch_2001, [
            ("user", "what year is it?", epoch_2001 + 1.0),
            ("assistant", "it is 2001", epoch_2001 + 2.0),
        ]),
    ])
    counts = backfill_sessions(conn, tmp_home)
    assert counts["turns"] == 1
    row = conn.execute("SELECT ts FROM episodes WHERE session_id='sess-old'").fetchone()
    assert row["ts"].startswith("2001-09-09T")

    import time as _time
    buf = conn.execute(
        "SELECT ts FROM ingest_buffer WHERE session_id='sess-old'").fetchone()
    assert buf["ts"].startswith(_time.strftime("%Y", _time.gmtime()))


def test_backfill_session_started_at_fallback_ts(conn, tmp_home):
    """Messages without usable timestamps fall back to sessions.started_at."""
    path = build_state_db(tmp_home, [("sess-fb", "cli", 999_999_999.0, [])])
    con = sqlite3.connect(str(path))
    con.execute("INSERT INTO messages (session_id, role, content, timestamp, active)"
                " VALUES ('sess-fb','user','hello',NULL,1)")
    con.execute("INSERT INTO messages (session_id, role, content, timestamp, active)"
                " VALUES ('sess-fb','assistant','hi there',NULL,1)")
    con.commit()
    con.close()
    backfill_sessions(conn, tmp_home)
    row = conn.execute("SELECT ts FROM episodes WHERE session_id='sess-fb'").fetchone()
    assert row["ts"].startswith("2001-09-09T")


def test_backfill_skips_inactive_messages(conn, tmp_home):
    build_state_db(tmp_home, [
        ("sess-act", "cli", 100.0, [
            ("user", "compacted away", 101.0, 0),
            ("assistant", "compacted reply", 102.0, 0),
            ("user", "live question", 103.0, 1),
            ("assistant", "live answer", 104.0),  # active defaults to 1
        ]),
    ])
    counts = backfill_sessions(conn, tmp_home)
    assert counts["turns"] == 1
    assert _count(conn, "episodes", "user_content='live question'") == 1
    assert _count(conn, "episodes", "user_content='compacted away'") == 0


def test_backfill_skips_live_session_without_watermark(conn, tmp_home):
    """ended_at IS NULL means still running: no import AND no watermark —
    a watermark now would freeze the partial transcript forever."""
    build_state_db(tmp_home, [
        ("sess-live", "cli", 100.0, _session_messages(2, "live"),
         {"ended_at": None}),
    ])
    counts = backfill_sessions(conn, tmp_home)
    assert counts["sessions"] == 0 and counts["turns"] == 0
    assert _count(conn, "episodes") == 0
    assert _count(conn, "sweep_state", "key='bootstrap:sess-live'") == 0

    # The session ends later — the next run picks it up in full.
    con = sqlite3.connect(str(tmp_home / "state.db"))
    con.execute("UPDATE sessions SET ended_at=200.0 WHERE id='sess-live'")
    con.commit()
    con.close()
    counts = backfill_sessions(conn, tmp_home)
    assert counts["sessions"] == 1 and counts["turns"] == 2
    assert _count(conn, "sweep_state", "key='bootstrap:sess-live'") == 1


def test_backfill_ancient_schema_without_projected_columns(conn, tmp_home):
    """Pre-user_id/ended_at/active schemas import via the SELECT * fallback."""
    con = sqlite3.connect(str(tmp_home / "state.db"))
    con.executescript(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, started_at REAL);"
        "CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT);"
    )
    con.execute("INSERT INTO sessions VALUES ('sess-old-schema','cli',100.0)")
    con.execute("INSERT INTO messages VALUES ('sess-old-schema','user','ancient q')")
    con.execute("INSERT INTO messages VALUES ('sess-old-schema','assistant','ancient a')")
    con.commit()
    con.close()
    counts = backfill_sessions(conn, tmp_home)
    assert counts["sessions"] == 1 and counts["turns"] == 1
    row = conn.execute(
        "SELECT * FROM episodes WHERE session_id='sess-old-schema'").fetchone()
    assert row["user_content"] == "ancient q"
    # No message timestamp column -> falls back to sessions.started_at (1970).
    assert row["ts"].startswith("1970-01-01T")


# ---------------------------------------------------------------------------
# daemon import
# ---------------------------------------------------------------------------

DAEMON_ROWS = [
    {"category": "decision", "content": "Use SQLite over Postgres",
     "rationale": "single-user, zero ops", "tags": '["storage"]',
     "worked": 1, "created_at": "2024-03-05 10:20:30.123456"},
    {"category": "pattern", "content": "Always gate optional deps behind lazy import",
     "tags": '["style"]'},
    {"category": "warning", "content": "Never call executescript inside a transaction",
     "pinned": 1, "outcome": "corrupted the WAL once"},
    {"category": "learning", "content": "Batch embedding halves backfill time", "worked": 0},
    {"category": "pattern", "content": "This one was retired", "archived": 1},
]


def test_daemon_import_mapping(conn, tmp_path):
    path = build_daemon_db(tmp_path, DAEMON_ROWS)
    counts = import_daemon_db(conn, path)
    assert counts == {"imported": 4, "skipped": 1}
    assert _count(conn, "memories", "content LIKE '%retired%'") == 0  # archived skipped

    decision = conn.execute(
        "SELECT * FROM memories WHERE kind='decision'"
    ).fetchone()
    assert decision["content"] == "Use SQLite over Postgres\nRationale: single-user, zero ops"
    assert decision["memory_type"] == "episodic"
    assert decision["half_life_days"] == 90.0
    assert decision["outcome"] == "worked"
    assert decision["valid_from"] == "2024-03-05T10:20:30.123456Z"
    assert decision["created_by"] == "migration"
    assert decision["trust_tier"] == "owner"
    assert decision["scope_project"] == "MyProject"
    assert '"storage"' in decision["tags"] and '"daem0n-import"' in decision["tags"]

    warning = conn.execute("SELECT * FROM memories WHERE kind='warning'").fetchone()
    assert warning["memory_type"] == "semantic"
    assert warning["half_life_days"] is None
    assert warning["pinned"] == 1
    assert warning["outcome"] is None
    assert warning["outcome_note"] == "corrupted the WAL once"

    insights = conn.execute(
        "SELECT memory_type, outcome FROM memories WHERE kind='insight' ORDER BY id"
    ).fetchall()
    assert [(r["memory_type"], r["outcome"]) for r in insights] == [
        ("semantic", None),        # pattern
        ("episodic", "failed"),    # learning, worked=0
    ]


def test_daemon_import_idempotent(conn, tmp_path):
    path = build_daemon_db(tmp_path, DAEMON_ROWS)
    import_daemon_db(conn, path)
    again = import_daemon_db(conn, path)
    assert again["imported"] == 0
    assert again["skipped"] == 5  # 1 archived + 4 content-hash dups
    assert _count(conn, "memories") == 4


def test_daemon_import_wrong_path(conn, tmp_path):
    counts = import_daemon_db(conn, tmp_path / "nope" / "memory.db")
    assert counts["imported"] == 0
    assert ".daem0nmcp" in counts["error"]


def test_daemon_import_not_a_daemon_db(conn, tmp_path):
    bogus = tmp_path / "memory.db"
    con = sqlite3.connect(str(bogus))
    con.execute("CREATE TABLE other (x TEXT)")
    con.commit()
    con.close()
    counts = import_daemon_db(conn, bogus)
    assert counts["imported"] == 0
    assert "memories table" in counts["error"]


# ---------------------------------------------------------------------------
# run_bootstrap orchestration
# ---------------------------------------------------------------------------

def test_run_bootstrap_merged_counts(conn, tmp_home, tmp_path):
    write_memory_files(tmp_home, memory_entries=["fact one"], user_entries=["Name: Devil"])
    two_session_home(tmp_home)
    daemon = build_daemon_db(tmp_path, DAEMON_ROWS)

    counts = run_bootstrap(conn, tmp_home, {}, daemon_db=daemon)
    assert counts["memory_md"] == 1
    assert counts["user_md"] == 1
    assert counts["sessions"] == 2
    assert counts["turns"] == 6
    assert counts["daemon_imported"] == 4
    assert "error" not in counts

    # Fully idempotent end-to-end.
    again = run_bootstrap(conn, tmp_home, {}, daemon_db=daemon)
    assert again["memory_md"] == 0 and again["user_md"] == 0
    assert again["sessions"] == 0 and again["turns"] == 0
    assert again["daemon_imported"] == 0


def test_run_bootstrap_respects_config_gate(conn, tmp_home):
    write_memory_files(tmp_home, memory_entries=["should not import"])
    counts = run_bootstrap(conn, tmp_home, {"bootstrap_import": False})
    assert counts.get("disabled") is True
    assert _count(conn, "memories") == 0


def test_run_bootstrap_never_raises(tmp_home):
    from brain.store import db

    broken = db.connect(tmp_home)
    broken.close()
    counts = run_bootstrap(broken, tmp_home, {})
    assert isinstance(counts, dict)
    assert "error" in counts


def test_run_bootstrap_daemon_error_reported_not_raised(conn, tmp_home, tmp_path):
    counts = run_bootstrap(conn, tmp_home, {}, daemon_db=tmp_path / "missing.db")
    assert counts["daemon_imported"] == 0
    assert ".daem0nmcp" in counts["error"]


def test_run_bootstrap_forwards_max_sessions(conn, tmp_home):
    two_session_home(tmp_home)
    counts = run_bootstrap(conn, tmp_home, {}, max_sessions=1)
    assert counts["sessions"] == 1
    # None means "state_db's default cap" — the rest imports next run.
    counts = run_bootstrap(conn, tmp_home, {}, max_sessions=None)
    assert counts["sessions"] == 1
    assert _count(conn, "episodes") == 6


# ---------------------------------------------------------------------------
# CLI: hermes brain bootstrap (regression for the daemon_db= keyword)
# ---------------------------------------------------------------------------

def test_cmd_bootstrap_via_cli_returns_zero(tmp_home, monkeypatch, capsys):
    """argparse.Namespace(daemon=None, max_sessions=None) must round-trip
    through cli.brain_command -> run_bootstrap without a TypeError."""
    import argparse

    from brain import cli
    from brain import config as brain_config

    monkeypatch.setattr(cli, "_hermes_home", lambda: tmp_home)
    brain_config.save_config(tmp_home, {"mode": "fts-only"})  # hermetic: no embedder
    write_memory_files(tmp_home, memory_entries=["cli fact"])
    two_session_home(tmp_home)

    args = argparse.Namespace(daemon=None, max_sessions=None, brain_command="bootstrap")
    assert cli.brain_command(args) == 0
    out = capsys.readouterr().out
    assert "sessions" in out and "turns" in out

    from brain.store import db
    conn = db.connect(tmp_home)
    try:
        assert _count(conn, "episodes") == 6
        assert _count(conn, "memories", "content='cli fact'") == 1
    finally:
        conn.close()
