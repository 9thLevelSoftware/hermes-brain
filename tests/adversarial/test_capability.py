"""Phase 1 (pytest half) — the degradation matrix at the capability level.

Proves the brain degrades, never crashes, when FTS5 / sqlite-vec are absent and
when a brain.db migrates between interpreters of differing capability — and that
a config typo for the tier resolves to a working tier rather than disabling
retrieval. The Docker half (docker/adversarial/) runs the same behavioral golden
on the real floor/full images; this half simulates the missing capabilities so
it runs on any interpreter.

Seams: store/db.py (probe_capabilities, _reconcile_fts, _create_fresh),
store/sysinfo.py (resolve_mode), recall/search.py (_like_search + the fts5 gate
at ~200), store/vec.py (ensure_tables embedder-identity key).
"""

from __future__ import annotations

from unittest import mock

import pytest

from conftest import seed_memory
from faults import simulate_no_fts5


def _search(conn, query, **kw):
    from brain.recall.search import search

    kw.setdefault("principal_id", "owner")
    kw.setdefault("trust_tier", "owner")
    return search(conn, query, limit=8, **kw)


# ---------------------------------------------------------------------------
# tier resolution: a config typo must degrade, never disable retrieval
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["garbage", "", "   ", "FULL", " Auto ", "xyzzy", None])
def test_resolve_mode_never_raises_and_returns_valid_tier(value):
    from brain.store import sysinfo

    mode = sysinfo.resolve_mode(value)  # type: ignore[arg-type]
    assert mode in ("full", "lite", "fts-only", "stub")


def test_resolve_mode_stub_is_config_only_never_auto():
    from brain.store import sysinfo

    # 'auto' must never resolve to the test-only 'stub' tier.
    assert sysinfo.resolve_mode("auto") != "stub"
    # but an explicit stub is honored.
    assert sysinfo.resolve_mode("stub") == "stub"


# ---------------------------------------------------------------------------
# no-fts5: LIKE fallback still recalls, still honors exclude_kinds, no injection
# ---------------------------------------------------------------------------

def test_like_fallback_recalls_after_fts5_removed(conn):
    seed_memory(conn, "the staging database is postgres 14 on fly.io", kind="fact")
    simulate_no_fts5(conn)
    from brain.store import db

    assert db.capabilities(conn).get("fts5") is False
    hits = _search(conn, "staging database")
    assert any("staging" in (h.text or "").lower() for h in hits)


def test_like_fallback_still_excludes_guidance_kinds(conn):
    """The guidance-leak guard (search.py docstring ~481-487): the LIKE leg must
    honor exclude_kinds exactly like FTS, or strategy/guardrail rows leak into
    the facts block on the floor tier."""
    seed_memory(conn, "always run the migration before deploying", kind="guardrail",
                memory_type="procedural")
    seed_memory(conn, "the deploy runbook lives in docs/deploy.md", kind="fact")
    simulate_no_fts5(conn)
    hits = _search(conn, "deploy", exclude_kinds=("strategy", "guardrail", "case", "peer_card"))
    kinds = {h.mkind for h in hits}
    assert "guardrail" not in kinds
    assert any("runbook" in (h.text or "").lower() for h in hits)


def test_like_escape_neutralizes_metacharacters():
    """The escape helper must neutralize the three LIKE metacharacters so a
    token can never inject a wildcard into the pattern."""
    from brain.recall.search import _like_escape

    assert _like_escape("a%b_c\\d") == "a\\%b\\_c\\\\d"
    assert _like_escape("100%") == "100\\%"


def test_metacharacter_queries_are_safe_on_like_tier(conn):
    """On the no-fts5 tier: a pure-metacharacter query tokenizes to nothing and
    returns [] (never a match-all), and a metachar-laden query never raises and
    only matches on its real alphanumeric token — not the whole table."""
    seed_memory(conn, "the widget subsystem is 50% assembled", kind="fact")
    seed_memory(conn, "the gadget inventory count reached 200", kind="fact")
    simulate_no_fts5(conn)

    # pure metacharacters -> no tokens -> [] (not a wildcard match-all)
    for q in ("%", "_", "\\", "%_%", "  %  "):
        assert _search(conn, q) == [], f"{q!r} should not match anything"

    # a real token with adjacent metachars matches only its row, and never raises
    hits = _search(conn, "widget%_")
    texts = " ".join((h.text or "").lower() for h in hits)
    assert "widget" in texts and "gadget" not in texts


# ---------------------------------------------------------------------------
# cross-interpreter reconcile (store/db._reconcile_fts) — both directions
# ---------------------------------------------------------------------------

def test_fresh_create_without_fts5_captures_and_recalls(tmp_home):
    """A brain.db BORN on a Python without FTS5 (probe returns fts5=False):
    _create_fresh strips the FTS DDL; capture still works and recall degrades to
    LIKE."""
    from brain.store import db

    caps = {"sqlite_version": "3", "fts5": False, "load_extension": False, "vec": False}
    with mock.patch.object(db, "probe_capabilities", return_value=caps):
        conn = db.connect(tmp_home)
    try:
        assert db.capabilities(conn).get("fts5") is False
        # no memory_fts trigger exists, so a plain insert must not error
        seed_memory(conn, "cold-tier fact about the staging host neptune", kind="fact")
        hits = _search(conn, "staging host neptune")
        assert any("neptune" in (h.text or "").lower() for h in hits)
    finally:
        conn.close()


def test_reopen_on_lesser_interpreter_drops_triggers_keeps_capture(tmp_home):
    """An fts5-born brain.db reopened on a no-fts5 Python: _reconcile_fts drops
    the INSERT triggers (which would otherwise fail every capture) and search
    degrades to LIKE. The DB must stay usable — never unopenable."""
    from brain.store import db

    # 1. born with real FTS
    conn = db.connect(tmp_home)
    seed_memory(conn, "an fts-born fact about the deploy pipeline", kind="fact")
    assert db.capabilities(conn).get("fts5") is True
    conn.close()

    # 2. reopen as if on a Python without FTS5
    caps = {"sqlite_version": "3", "fts5": False, "load_extension": False, "vec": False}
    with mock.patch.object(db, "probe_capabilities", return_value=caps):
        conn = db.connect(tmp_home)
    try:
        trig = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name='memories_ai'"
        ).fetchone()
        assert trig is None, "no-fts5 reopen must drop the FTS insert triggers"
        # capture still works (no trigger to fail) and recall finds it via LIKE
        seed_memory(conn, "a fact captured after the downgrade", kind="fact")
        assert db.capabilities(conn).get("fts5") is False
        assert any("downgrade" in (h.text or "").lower()
                   for h in _search(conn, "downgrade captured"))
    finally:
        conn.close()


def test_reopen_recreates_missing_fts_and_rebuilds(tmp_home):
    """A brain.db whose FTS objects are missing (born lesser) reopened on an
    fts5 Python: _reconcile_fts recreates the tables + triggers and rebuilds the
    index from content, so a pre-existing row becomes FTS-findable."""
    from brain.store import db

    conn = db.connect(tmp_home)
    # simulate "born without FTS": drop the FTS tables + triggers, then insert a
    # row directly (no trigger to mirror it into any index)
    for name in ("episodes_ai", "episodes_ad", "episodes_au",
                 "memories_ai", "memories_ad", "memories_au"):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.execute("DROP TABLE IF EXISTS memory_fts")
    conn.execute("DROP TABLE IF EXISTS episode_fts")
    conn.commit()
    seed_memory(conn, "a pre-reconcile fact about the rebuild path", kind="fact")
    conn.close()

    # reopen on a capable (real) interpreter — reconcile recreates + rebuilds
    conn = db.connect(tmp_home)
    try:
        assert db.capabilities(conn).get("fts5") is True
        have = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_fts'"
        ).fetchone()
        assert have is not None, "reconcile must recreate memory_fts"
        assert any("rebuild path" in (h.text or "").lower()
                   for h in _search(conn, "rebuild path"))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# embedder-identity compatibility key (full tier only)
# ---------------------------------------------------------------------------

def test_embedder_identity_mismatch_disables_vec_without_destroying(conn):
    """Two embedders sharing 256d must NOT silently compare across spaces: a
    name mismatch with allow_rebuild=False disables the vector legs (returns
    False) and leaves the existing index intact (never destroyed mid-session)."""
    from brain.store import vec

    if not vec.load_extension(conn):
        pytest.skip("sqlite-vec not available on this tier")

    assert vec.ensure_tables(conn, 256, "embedder-A", allow_rebuild=False) is True
    vec.upsert(conn, "mem_vec", 1, [0.1] * 256)
    conn.commit()
    before = conn.execute("SELECT count(*) FROM mem_vec").fetchone()[0]
    assert before == 1

    # a DIFFERENT embedder identity, same dim, no rebuild -> disabled, intact
    assert vec.ensure_tables(conn, 256, "embedder-B", allow_rebuild=False) is False
    after = conn.execute("SELECT count(*) FROM mem_vec").fetchone()[0]
    assert after == 1, "the vector index must survive an identity mismatch"
