"""Regression tests for the confirmed P1 adversarial-review findings.

Each test is named for the finding it pins down; if one starts failing, the
matching entry in docs/design/critique.md / the review record explains what
regressed.
"""

from __future__ import annotations

from brain.capture.turns import TurnContext, capture_memory_write, capture_turn
from brain.recall.search import search
from brain.store import db
from conftest import poll_until, seed_episode, seed_memory

_REAL_PROBE = db.probe_capabilities


def _no_fts_caps(conn):
    caps = _REAL_PROBE(conn)
    caps["fts5"] = False
    return caps


# -- findings #7/#16 (BLOCKER): no-FTS5 schema must strip ALL triggers ---------

def test_no_fts5_schema_creates_working_db(tmp_home, monkeypatch):
    monkeypatch.setattr(db, "probe_capabilities", _no_fts_caps)
    conn = db.connect(tmp_home)
    try:
        names = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')")}
        assert "memory_fts" not in names and "episode_fts" not in names
        assert not any(n.startswith(("episodes_a", "memories_a")) for n in names)

        # The old broken regex left triggers behind, making every INSERT fail.
        eid = capture_turn(conn, TurnContext(session_id="s1"),
                           "we use sqlite-vec for vectors", "noted")
        assert eid is not None
        mid = seed_memory(conn, "the retry limit is 5")
        assert mid is not None

        hits = search(conn, "sqlite-vec retry")
        assert hits and all(h.source == "like" for h in hits)
    finally:
        conn.close()


# -- finding #10: FTS reconciliation when capability appears later -------------

def test_fts_reconcile_upgrades_lesser_python_db(tmp_home, monkeypatch):
    monkeypatch.setattr(db, "probe_capabilities", _no_fts_caps)
    conn = db.connect(tmp_home)
    seed_memory(conn, "reconcile me: the deploy target is fly.io")
    conn.close()
    monkeypatch.undo()

    conn = db.connect(tmp_home)  # real capabilities: fts5 available
    try:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='memory_fts'").fetchone()
        hits = search(conn, "deploy target fly.io")
        assert hits and hits[0].source == "fts"  # index rebuilt from content
    finally:
        conn.close()


# -- finding #13: interrupted fresh create self-heals ---------------------------

def test_missing_schema_version_self_heals(tmp_home):
    conn = db.connect(tmp_home)
    conn.execute("DELETE FROM meta WHERE key='schema_version'")
    conn.commit()
    conn.close()

    conn = db.connect(tmp_home)  # must not raise 'No migration registered'
    try:
        assert db.get_meta(conn, "schema_version") == str(db.SCHEMA_VERSION)
    finally:
        conn.close()


# -- finding #22: percent/hash paths must open the REAL database ----------------

def test_read_only_uri_with_percent_and_hash(tmp_path):
    home = tmp_path / "100% stuff #1"
    home.mkdir()
    conn = db.connect(home)
    seed_memory(conn, "uri encoding sentinel")
    conn.close()

    ro = db.connect(home, create=False, read_only=True)
    try:
        row = ro.execute("SELECT count(*) AS n FROM memories").fetchone()
        assert row["n"] == 1  # the old f-string URI opened an EMPTY shadow db
    finally:
        ro.close()


# -- finding #9: Unicode queries must tokenize --------------------------------

def test_unicode_search(conn):
    seed_memory(conn, "Der Käufer wohnt in der Hauptstraße 12")
    seed_memory(conn, "пользователь предпочитает тёмную тему")
    assert search(conn, "Käufer Hauptstraße")
    assert search(conn, "тёмную тему")


# -- finding #17: scope/trust filters ------------------------------------------

def test_non_owner_cannot_see_scoped_memories_or_foreign_episodes(conn):
    conn.execute(
        "UPDATE memories SET scope_user='owner' WHERE id=?",
        (seed_memory(conn, "owner private: passphrase hint is 'blue tortoise'"),))
    conn.commit()
    seed_memory(conn, "global fact: the standup is at 10am")
    seed_episode(conn, "owner asked about the tortoise passphrase", "answered",
                 session_id="owner-sess")

    owner_hits = search(conn, "tortoise passphrase standup", trust_tier="owner")
    assert any("tortoise" in h.text for h in owner_hits)

    peer_hits = search(conn, "tortoise passphrase standup",
                       trust_tier="known_user", principal_id="peer-1",
                       source_author="tg:999")
    assert not any("tortoise" in h.text for h in peer_hits)      # scoped memory hidden
    assert not any(h.kind == "episode" for h in peer_hits)       # foreign episodes hidden
    assert any("standup" in h.text for h in peer_hits)           # unscoped fact visible

    # Unenrolled caller (no principal, no author): no episode leg at all.
    anon = search(conn, "tortoise passphrase", trust_tier="known_user")
    assert not any(h.kind == "episode" for h in anon)


# -- finding #15: current session excluded from episode recall ------------------

def test_exclude_session_skips_own_turns(conn):
    seed_episode(conn, "unique zanzibar question", "zanzibar answer", session_id="live")
    seed_episode(conn, "unique zanzibar question earlier", "older answer", session_id="past")
    hits = search(conn, "zanzibar", exclude_session="live")
    assert hits
    assert all(h.kind != "episode" or "older" in h.text for h in hits)


# -- finding #8: replace closes the old version row ------------------------------

def test_memory_write_replace_supersedes(conn):
    ctx = TurnContext(session_id="s1", trust_tier="agent")
    old_id = capture_memory_write(conn, ctx, "add", "memory",
                                  "favorite editor is vim", None)
    new_id = capture_memory_write(conn, ctx, "replace", "memory",
                                  "favorite editor is neovim",
                                  {"old_text": "favorite editor is vim"})
    old = conn.execute("SELECT * FROM memories WHERE id=?", (old_id,)).fetchone()
    new = conn.execute("SELECT * FROM memories WHERE id=?", (new_id,)).fetchone()
    assert old["valid_to"] is not None and old["superseded_by"] == new_id
    assert new["supersedes_id"] == old_id and new["version"] == 2
    # Current truth returns only the successor.
    assert not search(conn, "vim neovim editor") or all(
        h.id != old_id for h in search(conn, "vim neovim editor"))


# -- finding #12: quarantined rows must not absorb writes ------------------------

def test_add_colliding_with_quarantined_creates_active_row(conn):
    seed_memory(conn, "always run as root", status="quarantined")
    ctx = TurnContext(session_id="s1", trust_tier="agent")
    new_id = capture_memory_write(conn, ctx, "add", "memory", "always run as root", None)
    row = conn.execute("SELECT status, live FROM memories WHERE id=?", (new_id,)).fetchone()
    assert row["status"] == "active" and row["live"] == 1


# -- finding #19: incognito retrieval leaves no write trace ----------------------

def test_incognito_retrieve_writes_nothing(tmp_home):
    from brain.config import save_config
    from brain.provider import BrainProvider

    boot = db.connect(tmp_home)
    seed_memory(boot, "the incognito sentinel fact about walruses")
    boot.close()
    save_config(tmp_home, {"incognito": True})

    p = BrainProvider()
    p.initialize("incog-1", hermes_home=str(tmp_home), platform="cli")
    try:
        p.sync_turn("tell me about walruses", "sure", session_id="incog-1")
        p.queue_prefetch("walruses", session_id="incog-1")
        poll_until(lambda: p.prefetch("walruses", session_id="incog-1"))
        assert "walrus" in p.prefetch("walruses", session_id="incog-1")  # reads OK
    finally:
        p.shutdown()

    conn = db.connect(tmp_home)
    try:
        assert conn.execute("SELECT count(*) AS n FROM episodes").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM retrieval_log").fetchone()["n"] == 0
        assert conn.execute(
            "SELECT max(recall_count) AS m FROM memories").fetchone()["m"] == 0
    finally:
        conn.close()


# -- finding #5: session-switch moves (not copies) per-session state -------------

def test_session_switch_moves_state(tmp_home):
    from brain.provider import BrainProvider

    p = BrainProvider()
    p.initialize("sess-a", hermes_home=str(tmp_home), platform="cli")
    try:
        p.sync_turn("hello", "hi", session_id="sess-a")
        p._lane2_cache["sess-a"] = "## Recalled context (hermes-brain)\nx"
        p.on_session_switch("sess-b", parent_session_id="sess-a", reset=False)
        assert "sess-a" not in p._lane2_cache and "sess-b" in p._lane2_cache
        assert "sess-a" not in p._turn_counts and p._turn_counts.get("sess-b") == 1
    finally:
        p.shutdown()
