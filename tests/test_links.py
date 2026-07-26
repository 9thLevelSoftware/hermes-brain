"""Cross-profile linking: registration, RRF merge, and the CLI verbs.

Scope/safety invariants live in tests/adversarial/test_link_scope.py — this
file is about the mechanics being right.
"""

from __future__ import annotations

import argparse

import pytest
from brain import cli
from brain.recall.linked import _merge, search_linked
from brain.recall.search import Hit, search
from brain.store import db
from brain.store import links as links_mod
from conftest import seed_memory


def _run(argv):
    parser = argparse.ArgumentParser(prog="brain")
    cli.register_cli(parser)
    return cli.brain_command(parser.parse_args(argv))


@pytest.fixture
def home(tmp_home, monkeypatch):
    monkeypatch.setattr(cli, "_hermes_home", lambda: tmp_home)
    return tmp_home


@pytest.fixture
def other_home(tmp_path):
    home = tmp_path / "other"
    home.mkdir()
    conn = db.connect(home)
    try:
        seed_memory(conn, "the coder profile pins the buildkite deploy runbook")
    finally:
        conn.close()
    return home


def _hit(uid, score, profile=None, kind="memory"):
    return Hit(kind=kind, id=1, uid=uid, text=uid, summary=None,
               memory_type=None, mkind=None, ts="2026-01-01", platform="cli",
               score=score, source="fts", profile=profile)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def test_add_list_remove(conn, other_home):
    assert links_mod.load(conn) == []
    entry = links_mod.add(conn, "coder", str(other_home))
    assert entry["name"] == "coder" and entry["enabled"] is True
    assert [link["name"] for link in links_mod.load(conn)] == ["coder"]
    assert links_mod.remove(conn, "coder") is True
    assert links_mod.load(conn) == []
    assert links_mod.remove(conn, "coder") is False


def test_disabled_links_are_not_searched(conn, other_home):
    links_mod.add(conn, "coder", str(other_home))
    links_mod.set_enabled(conn, "coder", False)
    assert links_mod.enabled(conn) == []
    assert search_linked(conn, "buildkite", local_hits=[], trust_tier="owner") == []


def test_a_link_needs_a_name(conn, other_home):
    with pytest.raises(ValueError, match="needs a name"):
        links_mod.add(conn, "  ", str(other_home))


def test_corrupt_link_metadata_reads_as_no_links(conn):
    db.set_meta(conn, links_mod.META_KEY, "{not json")
    conn.commit()
    assert links_mod.load(conn) == []


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------

def test_merge_keeps_local_and_linked_and_reranks_by_rrf():
    local = [_hit("LOCAL1", 0.9), _hit("LOCAL2", 0.5)]
    linked = [_hit("REMOTE1", 0.95, profile="coder")]
    merged = _merge(local, [("coder", linked)], 0.85, 10)

    uids = [h.uid for h in merged]
    assert set(uids) == {"LOCAL1", "LOCAL2", "REMOTE1"}
    # Rank 1 locally beats rank 1 remotely at a 0.85 discount.
    assert uids[0] == "LOCAL1"


def test_merge_scores_are_rewritten_to_the_fused_scale():
    """Per-corpus scores are min-max normalized WITHIN a corpus, so carrying
    them forward would compare incomparable numbers."""
    local = [_hit("LOCAL1", 0.9)]
    linked = [_hit("REMOTE1", 0.95, profile="coder")]
    merged = _merge(local, [("coder", linked)], 0.85, 10)
    assert all(h.score < 0.1 for h in merged), "RRF scores, not the originals"
    assert merged[0].score > merged[1].score


def test_merge_does_not_collapse_the_same_uid_from_two_profiles():
    """Two profiles can hold the same memory (an import, a sync). Collapsing
    on uid alone would let one row vote for itself twice."""
    local = [_hit("SAME", 0.9)]
    linked = [_hit("SAME", 0.9, profile="coder")]
    merged = _merge(local, [("coder", linked)], 0.85, 10)
    assert len(merged) == 2
    assert {h.profile for h in merged} == {None, "coder"}


def test_merge_respects_the_limit():
    local = [_hit(f"L{i}", 0.9) for i in range(10)]
    linked = [_hit(f"R{i}", 0.9, profile="c") for i in range(10)]
    assert len(_merge(local, [("c", linked)], 0.85, 5)) == 5


def test_link_weight_of_one_makes_profiles_equal_peers():
    local = [_hit("LOCAL1", 0.5)]
    linked = [_hit("REMOTE1", 0.5, profile="coder")]
    merged = _merge(local, [("coder", linked)], 1.0, 10)
    assert merged[0].score == pytest.approx(merged[1].score)


def test_search_linked_end_to_end(conn, other_home):
    links_mod.add(conn, "coder", str(other_home))
    seed_memory(conn, "the local profile notes the buildkite token rotation")

    local = search(conn, "buildkite", trust_tier="owner")
    merged = search_linked(conn, "buildkite", local_hits=local, trust_tier="owner")
    profiles = {h.profile for h in merged}
    assert None in profiles and "coder" in profiles


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_link_unlink_links(home, other_home, capsys):
    assert _run(["links"]) == 0
    assert "no linked profiles" in capsys.readouterr().out

    assert _run(["link", "coder", "--home", str(other_home)]) == 0
    out = capsys.readouterr().out
    assert "linked" in out and "coder" in out
    assert "OWNER-trust" in out, "the safety rule should be stated at link time"

    assert _run(["links"]) == 0
    out = capsys.readouterr().out
    assert "coder" in out and "ok" in out

    assert _run(["unlink", "coder"]) == 0
    assert "unlinked" in capsys.readouterr().out
    assert _run(["unlink", "coder"]) == 1


def test_cli_link_reports_a_missing_target(home, tmp_path, capsys):
    assert _run(["link", "ghost", "--home", str(tmp_path / "nope")]) == 1
    assert "no brain.db" in capsys.readouterr().err


def test_cli_links_flags_an_unreachable_target(home, other_home, capsys):
    import shutil

    _run(["link", "coder", "--home", str(other_home)])
    capsys.readouterr()
    shutil.rmtree(other_home)
    assert _run(["links"]) == 0
    assert "MISSING" in capsys.readouterr().out


def test_cli_search_includes_linked_profiles(home, other_home, capsys):
    _run(["link", "coder", "--home", str(other_home)])
    capsys.readouterr()
    assert _run(["search", "buildkite"]) == 0
    out = capsys.readouterr().out
    assert "@coder" in out, "linked hits must be labelled in search output"
