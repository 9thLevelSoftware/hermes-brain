"""The `memories` tool (integration.md §3.1 tool #5) — the Anthropic-shaped
virtual file interface over brain storage.

Functional behavior: path grammar, the three views, and the six commands
mapping onto real memories. Scope/trust invariants live in
tests/adversarial/test_memfs_scope.py.
"""

from __future__ import annotations

import json

import pytest
from brain import tools
from brain.config import DEFAULTS
from conftest import seed_memory


@pytest.fixture
def ctx(tmp_home):
    return tools.ToolContext(
        session_id="sess-memfs", principal_id="owner", trust_tier="owner",
        platform="cli", config=dict(DEFAULTS), hermes_home=str(tmp_home),
    )


def _call(conn, ctx, **args):
    return json.loads(tools.dispatch(conn, "memories", args, ctx=ctx))


# ---------------------------------------------------------------------------
# Path grammar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,expected", [
    ("/memories", ("root", "")),
    ("/memories/", ("root", "")),
    ("/memories/profile.md", ("profile", "")),
    ("/memories/index.md", ("index", "")),
    ("/memories/topics", ("topics", "")),
    ("/memories/topics/deploy.md", ("topic", "deploy")),
    ("/memories/topics/DEPLOY.md", ("topic", "deploy")),
    ("\\memories\\topics\\ci.md", ("topic", "ci")),
])
def test_parse_path_accepts_the_grammar(path, expected):
    from brain.memfs import parse_path

    assert parse_path(path) == expected


@pytest.mark.parametrize("path", [
    "", None, 42, "/etc/passwd", "memories/../etc/passwd",
    "/memories/../../secrets", "/memories/topics/../../etc/passwd",
    "/memories/nope.md", "/memories/topics/deploy", "/memories/a/b/c.md",
    "/memories/topics/.md",
])
def test_parse_path_rejects_everything_else(path):
    from brain.memfs import parse_path
    from brain.tools import _ToolError

    with pytest.raises(_ToolError):
        parse_path(path)


# ---------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------

def test_view_root_lists_the_namespace(conn, ctx):
    out = _call(conn, ctx, command="view", path="/memories")
    assert out["kind"] == "directory"
    assert "/memories/profile.md" in out["entries"]


def test_view_profile_shows_standing_facts(conn, ctx):
    seed_memory(conn, "User prefers terse answers", kind="preference")
    seed_memory(conn, "The VPS is a Hetzner CX22", kind="fact")
    seed_memory(conn, "an episode-ish insight", kind="insight")

    out = _call(conn, ctx, command="view", path="/memories/profile.md")
    assert "terse answers" in out["content"]
    assert "Hetzner CX22" in out["content"]
    # profile.md is facts+preferences only
    assert "episode-ish insight" not in out["content"]
    assert out["count"] == 2


def test_view_topic_filters_by_tag(conn, ctx):
    seed_memory(conn, "staging deploys need the VPN", tags=["deploy"])
    seed_memory(conn, "unrelated fact about coffee", tags=["kitchen"])

    out = _call(conn, ctx, command="view", path="/memories/topics/deploy.md")
    assert "VPN" in out["content"] and "coffee" not in out["content"]


def test_view_topic_does_not_substring_match_tags(conn, ctx):
    """A LIKE prefilter on the serialized JSON must be verified exactly, or
    tag 'ci' would match 'cinema'."""
    seed_memory(conn, "movie night is friday", tags=["cinema"])
    out = _call(conn, ctx, command="view", path="/memories/topics/ci.md")
    assert "movie night" not in out["content"]


def test_view_topics_directory_lists_tags(conn, ctx):
    seed_memory(conn, "a", tags=["deploy"])
    seed_memory(conn, "b", tags=["kitchen", "deploy"])
    out = _call(conn, ctx, command="view", path="/memories/topics")
    assert sorted(out["entries"]) == ["/memories/topics/deploy.md",
                                      "/memories/topics/kitchen.md"]


def test_view_index_is_readonly(conn, ctx):
    out = _call(conn, ctx, command="view", path="/memories/index.md")
    assert out["readonly"] is True

    err = _call(conn, ctx, command="create", path="/memories/index.md",
                file_text="- nope")
    assert "read-only" in err["error"]
    assert "recovery_hint" in err


def test_empty_view_says_so_rather_than_erroring(conn, ctx):
    out = _call(conn, ctx, command="view", path="/memories/profile.md")
    assert "(empty)" in out["content"]


# ---------------------------------------------------------------------------
# create / insert
# ---------------------------------------------------------------------------

def test_create_writes_one_memory_per_line(conn, ctx):
    out = _call(conn, ctx, command="create", path="/memories/topics/deploy.md",
                file_text="# Topic: deploy\n\n- staging needs the VPN\n"
                          "- prod deploys are Tuesdays\n\n")
    assert len(out["created"]) == 2

    view = _call(conn, ctx, command="view", path="/memories/topics/deploy.md")
    assert "VPN" in view["content"] and "Tuesdays" in view["content"]


def test_create_round_trips_a_viewed_file(conn, ctx):
    """A model that views a file, edits it, and writes it back must not turn
    the rendered `[id]` prefixes into memory text."""
    seed_memory(conn, "staging needs the VPN", tags=["deploy"])
    viewed = _call(conn, ctx, command="view", path="/memories/topics/deploy.md")

    _call(conn, ctx, command="create", path="/memories/topics/deploy2.md",
          file_text=viewed["content"])
    again = _call(conn, ctx, command="view", path="/memories/topics/deploy2.md")
    assert "staging needs the VPN" in again["content"]
    assert "[" not in again["content"].split("\n")[2].split("]")[1]


def test_create_with_no_memory_lines_teaches(conn, ctx):
    out = _call(conn, ctx, command="create", path="/memories/topics/x.md",
                file_text="# heading only\n\n")
    assert "no memory lines" in out["error"]


def test_insert_adds_one_memory(conn, ctx):
    out = _call(conn, ctx, command="insert", path="/memories/profile.md",
                insert_line=0, insert_text="- prefers no emojis")
    assert out["id"]
    view = _call(conn, ctx, command="view", path="/memories/profile.md")
    assert "no emojis" in view["content"]


def test_insert_says_line_numbers_are_advisory(conn, ctx):
    out = _call(conn, ctx, command="insert", path="/memories/profile.md",
                insert_line=99, insert_text="prefers dark mode")
    assert "advisory" in out["note"]


# ---------------------------------------------------------------------------
# str_replace
# ---------------------------------------------------------------------------

def test_str_replace_supersedes_rather_than_mutating(conn, ctx):
    seed_memory(conn, "The VPS is a Hetzner CX22", kind="fact")

    out = _call(conn, ctx, command="str_replace", path="/memories/profile.md",
                old_str="CX22", new_str="CX32")
    assert out["id"] and out["replaced"]

    view = _call(conn, ctx, command="view", path="/memories/profile.md")
    assert "CX32" in view["content"] and "CX22" not in view["content"]

    # versions-are-rows: the old text is tombstoned, not erased
    row = conn.execute(
        "SELECT status FROM memories WHERE content LIKE '%CX22%'").fetchone()
    assert row["status"] == "tombstone"


def test_str_replace_ambiguous_match_teaches(conn, ctx):
    seed_memory(conn, "deploy step one is slow", kind="fact")
    seed_memory(conn, "deploy step two is slow", kind="fact")

    out = _call(conn, ctx, command="str_replace", path="/memories/profile.md",
                old_str="is slow", new_str="is fast")
    assert "matches 2 memories" in out["error"]
    assert "longer" in out["recovery_hint"]


def test_str_replace_miss_teaches(conn, ctx):
    out = _call(conn, ctx, command="str_replace", path="/memories/profile.md",
                old_str="nothing like this", new_str="x")
    assert "not found" in out["error"]


def test_str_replace_refuses_to_empty_a_memory(conn, ctx):
    seed_memory(conn, "CX22", kind="fact")
    out = _call(conn, ctx, command="str_replace", path="/memories/profile.md",
                old_str="CX22", new_str="")
    assert "empty the memory" in out["error"]
    assert "brain_manage" in out["recovery_hint"]


# ---------------------------------------------------------------------------
# delete / rename
# ---------------------------------------------------------------------------

def test_delete_is_a_soft_tombstone(conn, ctx):
    seed_memory(conn, "staging needs the VPN", tags=["deploy"])
    out = _call(conn, ctx, command="delete", path="/memories/topics/deploy.md")
    assert len(out["deleted"]) == 1
    assert "soft" in out["note"]

    row = conn.execute("SELECT status, content FROM memories").fetchone()
    assert row["status"] == "tombstone"
    assert "VPN" in row["content"], "content must survive a soft delete"


def test_delete_refuses_the_whole_profile(conn, ctx):
    seed_memory(conn, "User prefers terse answers", kind="preference")
    out = _call(conn, ctx, command="delete", path="/memories/profile.md")
    assert "refusing" in out["error"]
    row = conn.execute("SELECT status FROM memories").fetchone()
    assert row["status"] == "active"


def test_rename_retags(conn, ctx):
    seed_memory(conn, "staging needs the VPN", tags=["ci", "infra"])
    out = _call(conn, ctx, command="rename", path="/memories/topics/ci.md",
                new_path="/memories/topics/deploy.md")
    assert out["renamed"] == 1

    view = _call(conn, ctx, command="view", path="/memories/topics/deploy.md")
    assert "VPN" in view["content"]
    old = _call(conn, ctx, command="view", path="/memories/topics/ci.md")
    assert "VPN" not in old["content"]
    # unrelated tags survive
    tags = json.loads(conn.execute("SELECT tags FROM memories").fetchone()["tags"])
    assert "infra" in tags


def test_rename_only_moves_topic_files(conn, ctx):
    out = _call(conn, ctx, command="rename", path="/memories/profile.md",
                new_path="/memories/topics/x.md")
    assert "only moves topic files" in out["error"]


# ---------------------------------------------------------------------------
# gating + error shape
# ---------------------------------------------------------------------------

def test_unknown_command_teaches(conn, ctx):
    out = _call(conn, ctx, command="chmod", path="/memories")
    assert "unknown command" in out["error"]
    assert "view" in out["recovery_hint"]


def test_memories_tool_can_be_switched_off(conn, tmp_home):
    off = tools.ToolContext(
        session_id="s", principal_id="owner", trust_tier="owner", platform="cli",
        config={**DEFAULTS, "memories_tool": False}, hermes_home=str(tmp_home))
    out = json.loads(tools.dispatch(conn, "memories", {"command": "view",
                                                       "path": "/memories"}, ctx=off))
    assert "unknown tool" in out["error"]


def test_dispatch_never_raises_on_garbage(conn, ctx):
    for args in ({}, {"command": "view"}, {"command": "view", "path": 5},
                 {"command": "create", "path": "/memories/topics/a.md"}):
        out = json.loads(tools.dispatch(conn, "memories", args, ctx=ctx))
        assert "error" in out and "recovery_hint" in out
