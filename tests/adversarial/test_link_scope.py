"""Adversarial: cross-profile links must not become an escalation path.

The operator chose FULL OWNER ACCESS across links — a linked profile is
searched as its owner, `peer_card` rows included. That is defensible when it is
you reading your own other profile. It is not defensible for a gateway peer or
an MCP `tool`-trust session, so the guarantee this file exists to pin is:

    **links are traversed ONLY for an owner-trust caller.**

Everything else follows from that one comparison in `recall/linked.py`. The
second guarantee is that links are read-only in fact, not merely by intent:
`Hit.id` is a rowid, so a linked hit's id addresses a DIFFERENT database, and
any local write keyed on it would corrupt an unrelated memory.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from brain.recall.linked import local_only, search_linked
from brain.recall.search import log_retrieval, search
from brain.store import db
from brain.store import links as links_mod
from conftest import seed_memory

_SECRET = "vaulttoken hunter2 launchcodes"


@pytest.fixture
def other_home(tmp_path):
    """A second profile with a distinctive memory in it."""
    home = tmp_path / "other_home"
    home.mkdir()
    conn = db.connect(home)
    try:
        seed_memory(conn, f"the other profile knows: {_SECRET}")
        seed_memory(conn, "the other profile also knows about deploy pipelines")
    finally:
        conn.close()
    return home


def _link(conn, other_home, name="other"):
    return links_mod.add(conn, name, str(other_home))


def _text(hits) -> str:
    return json.dumps([h.text for h in hits])


# ---------------------------------------------------------------------------
# THE guarantee: owner-only traversal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tier", ["known_user", "tool", "untrusted", "agent"])
def test_non_owner_callers_never_traverse_a_link(conn, other_home, tier):
    _link(conn, other_home)
    local = search(conn, "profile knows", trust_tier=tier)
    merged = search_linked(conn, "profile knows", local_hits=local, trust_tier=tier)

    assert merged == local, f"{tier} must not reach a linked profile at all"
    assert _SECRET not in _text(merged)


def test_owner_does_traverse(conn, other_home):
    _link(conn, other_home)
    merged = search_linked(conn, "profile knows",
                           local_hits=search(conn, "profile knows", trust_tier="owner"),
                           trust_tier="owner")
    assert _SECRET in _text(merged)
    assert any(h.profile == "other" for h in merged)


def test_linked_hits_are_labelled_with_their_profile(conn, other_home):
    _link(conn, other_home, name="coder")
    merged = search_linked(conn, "deploy pipelines",
                           local_hits=[], trust_tier="owner")
    assert merged and all(h.profile == "coder" for h in merged)
    # ...and the rendered line says so, so it can never read as local memory.
    from brain.recall.render import index_line

    assert "@coder" in index_line(merged[0])


def test_the_mcp_tool_surface_never_traverses_links(conn, other_home, tmp_home):
    """MCP runs at 'tool' trust — the cross-platform read surface must not
    become a cross-profile read surface."""
    from brain import tools
    from brain.config import DEFAULTS

    _link(conn, other_home)
    ctx = tools.ToolContext(session_id="mcp-1", principal_id=None,
                            trust_tier="tool", platform="mcp",
                            config=dict(DEFAULTS), hermes_home=str(tmp_home))
    out = tools.dispatch(conn, "brain_recall", {"query": "profile knows"}, ctx=ctx)
    assert _SECRET not in out


# ---------------------------------------------------------------------------
# read-only in fact, not just intent
# ---------------------------------------------------------------------------

def test_linked_hits_never_reach_the_retrieval_log(conn, other_home):
    """`Hit.id` is a rowid. Logging a linked hit would credit — and eventually
    reweight and forget — whatever local memory happens to share the number."""
    local_mem = seed_memory(conn, "a local memory about deploy pipelines")
    _link(conn, other_home)
    merged = search_linked(conn, "deploy pipelines",
                           local_hits=search(conn, "deploy pipelines",
                                             trust_tier="owner"),
                           trust_tier="owner")
    assert any(h.profile for h in merged), "sanity: a linked hit is present"

    log_retrieval(conn, "sess-1", "deploy pipelines", merged,
                  {h.uid for h in merged})
    logged = conn.execute("SELECT memory_id FROM retrieval_log").fetchall()
    local_ids = {r["id"] for r in conn.execute("SELECT id FROM memories")}
    for row in logged:
        assert row["memory_id"] in local_ids, "a linked rowid was logged locally"
    assert local_mem in local_ids


def test_local_only_filters_linked_hits(conn, other_home):
    _link(conn, other_home)
    merged = search_linked(conn, "deploy pipelines", local_hits=[],
                           trust_tier="owner")
    assert merged and local_only(merged) == []


def test_a_link_cannot_write_to_the_other_profile(conn, other_home):
    """Connections are mode=ro at the SQLite level, so a bug cannot write."""
    remote = links_mod.open_link({"name": "o", "hermes_home": str(other_home)})
    assert remote is not None
    try:
        with pytest.raises(sqlite3.OperationalError):
            remote.execute("INSERT INTO memories (uid, content) VALUES ('X','y')")
    finally:
        remote.close()


def test_mutating_a_linked_uid_is_refused_with_the_profile_named(conn, other_home,
                                                                 tmp_home):
    from brain import tools
    from brain.config import DEFAULTS

    _link(conn, other_home, name="personal")
    remote = links_mod.open_link({"name": "p", "hermes_home": str(other_home)})
    foreign_uid = remote.execute("SELECT uid FROM memories LIMIT 1").fetchone()["uid"]
    remote.close()

    ctx = tools.ToolContext(session_id="s", principal_id="owner", trust_tier="owner",
                            platform="cli", config=dict(DEFAULTS),
                            hermes_home=str(tmp_home))
    out = json.loads(tools.dispatch(
        conn, "brain_manage",
        {"action": "forget", "id": foreign_uid[:10], "reason": "nope"}, ctx=ctx))
    assert "error" in out
    assert "personal" in out["error"], "the error must name the owning profile"
    assert "recovery_hint" in out


# ---------------------------------------------------------------------------
# link registration guards
# ---------------------------------------------------------------------------

def test_self_link_is_refused(conn, tmp_home):
    """It would double every local hit and let rows vote for themselves."""
    with pytest.raises(ValueError, match="itself"):
        links_mod.add(conn, "me", str(tmp_home))


def test_linking_a_nonexistent_profile_is_refused(conn, tmp_path):
    with pytest.raises(ValueError, match="no brain.db"):
        links_mod.add(conn, "ghost", str(tmp_path / "nope"))


def test_linking_a_future_schema_db_is_refused(conn, tmp_path):
    """Its rows may use columns this code does not know; reading a silent
    subset is worse than refusing."""
    home = tmp_path / "future"
    home.mkdir()
    other = db.connect(home)
    try:
        db.set_meta(other, "schema_version", str(db.SCHEMA_VERSION + 5))
        other.commit()
    finally:
        other.close()
    with pytest.raises(ValueError, match="schema v"):
        links_mod.add(conn, "future", str(home))


def test_duplicate_link_names_are_refused(conn, other_home):
    _link(conn, other_home)
    with pytest.raises(ValueError, match="already exists"):
        _link(conn, other_home)


def test_link_count_is_capped(conn, tmp_path):
    for i in range(links_mod.MAX_LINKS):
        home = tmp_path / f"p{i}"
        home.mkdir()
        db.connect(home).close()
        links_mod.add(conn, f"p{i}", str(home))
    home = tmp_path / "one-too-many"
    home.mkdir()
    db.connect(home).close()
    with pytest.raises(ValueError, match="at most"):
        links_mod.add(conn, "extra", str(home))


# ---------------------------------------------------------------------------
# degradation — a link must never break a turn
# ---------------------------------------------------------------------------

def test_a_deleted_linked_profile_degrades_to_local_only(conn, other_home):
    import shutil

    _link(conn, other_home)
    seed_memory(conn, "a local memory about deploy pipelines")
    shutil.rmtree(other_home)

    local = search(conn, "deploy pipelines", trust_tier="owner")
    merged = search_linked(conn, "deploy pipelines", local_hits=local,
                           trust_tier="owner")
    assert merged == local


def test_a_corrupt_linked_db_degrades_to_local_only(conn, other_home):
    _link(conn, other_home)
    seed_memory(conn, "a local memory about deploy pipelines")
    db.db_path(other_home).write_bytes(b"not a database")

    local = search(conn, "deploy pipelines", trust_tier="owner")
    merged = search_linked(conn, "deploy pipelines", local_hits=local,
                           trust_tier="owner")
    assert merged == local


def test_one_broken_link_does_not_cost_the_others(conn, other_home, tmp_path):
    import shutil

    good = tmp_path / "good"
    good.mkdir()
    good_conn = db.connect(good)
    try:
        seed_memory(good_conn, "the good profile knows about deploy pipelines")
    finally:
        good_conn.close()

    _link(conn, other_home, name="broken")
    links_mod.add(conn, "good", str(good))
    shutil.rmtree(other_home)

    merged = search_linked(conn, "deploy pipelines", local_hits=[],
                           trust_tier="owner")
    assert any(h.profile == "good" for h in merged)


def test_empty_query_returns_local_unchanged(conn, other_home):
    _link(conn, other_home)
    assert search_linked(conn, "", local_hits=[], trust_tier="owner") == []


def test_no_links_returns_local_unchanged(conn):
    seed_memory(conn, "a local memory about deploy pipelines")
    local = search(conn, "deploy pipelines", trust_tier="owner")
    assert search_linked(conn, "deploy pipelines", local_hits=local,
                         trust_tier="owner") == local
