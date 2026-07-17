"""D1: learned fusion weights — PROPOSED, never auto-applied (memory-engine §3.7).

recall/fit_weights.fit fits a pure-Python logistic model over labeled
retrieval_log rows (features = per-leg membership + fused rank_score; label =
helpful/harmful from the injection->outcome join against state.db turn_outcomes)
and returns proposed convex leg weights — or None on cold start. dream/tune.py
folds a returned fit into its SINGLE shadow `proposals` row and never applies it,
even under a forced 'active'. RRF stays the live default and the fallback.

Hermetic: no LLM (fitting is pure compute), a real-column state.db, and
synthetic retrieval_log rows carrying a planted fts->helpful / vec->harmful
signal.
"""

from __future__ import annotations

import json

from brain.dream import lease
from brain.dream import tune as tune_mod
from brain.dream.mine_state import open_state_ro
from brain.dream.shift import Shift
from brain.recall import fit_weights
from brain.recall.fusion import rrf, weighted_rrf
from brain.store import db
from conftest import seed_memory

from tests.test_dream_mine import make_state_db

# Two turns whose outcomes give the two labels the miner would assign.
_OUTCOMES = [
    ("s1", "t-help", 0.0, "verified", None, None),   # credit() -> helpful
    ("s1", "t-harm", 0.0, "failed", None, None),     # credit() -> harmful
]


def _seed_rlog(conn, mem_id, specs):
    """specs: list of (leg, resolved_turn_id, count) injected+resolved rows."""
    rows = []
    for leg, turn_id, count in specs:
        rows += [("s1", "qh", db.iso_now(), mem_id, leg, 0.8, 1, turn_id)] * count
    conn.executemany(
        "INSERT INTO retrieval_log (session_id, query_hash, ts, memory_id, leg,"
        " rank_score, injected, resolved_turn_id) VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def _state(tmp_home):
    make_state_db(tmp_home, outcomes=_OUTCOMES)
    return open_state_ro(tmp_home / "state.db")


# ---------------------------------------------------------------------------
# fit(): cold start vs. learnable signal
# ---------------------------------------------------------------------------

def test_fit_returns_none_below_min_rows(conn, tmp_home):
    """< min_rows labeled rows => None => RRF stays the default."""
    mem = seed_memory(conn, "the prod region is us-east-1")
    _seed_rlog(conn, mem, [("fts", "t-help", 300), ("vec", "t-harm", 100)])  # 400
    state = _state(tmp_home)
    try:
        assert fit_weights.fit(conn, state, min_rows=500) is None
    finally:
        state.close()


def test_fit_returns_none_without_state(conn):
    """No state.db => no labels => None (never guesses)."""
    mem = seed_memory(conn, "x")
    _seed_rlog(conn, mem, [("fts", "t-help", 600)])
    assert fit_weights.fit(conn, None, min_rows=500) is None


def test_fit_learns_leg_weights_from_signal(conn, tmp_home):
    """>= 500 labeled rows with a planted signal => proposed per-leg weights.

    fts candidates land on the verified (helpful) turn, vec candidates on the
    failed (harmful) turn — so the fit should up-weight fts and down-weight vec.
    ppr never appears, so it is not identifiable and stays neutral at 1.0.
    """
    mem = seed_memory(conn, "always cap the JVM heap at 2GB")
    _seed_rlog(conn, mem, [("fts", "t-help", 300), ("vec", "t-harm", 300)])
    state = _state(tmp_home)
    try:
        result = fit_weights.fit(conn, state, min_rows=500)
    finally:
        state.close()

    assert result is not None
    assert result["n_labeled"] == 600
    assert result["n_helpful"] == 300 and result["n_harmful"] == 300
    assert set(result["weights"]) == {"fts", "vec", "ppr"}
    # planted direction recovered...
    assert result["weights"]["fts"] > 1.0 > result["weights"]["vec"]
    assert result["coefficients"]["fts"] > result["coefficients"]["vec"]
    # ...ppr had no variation, so it is neutral (not invented).
    assert result["weights"]["ppr"] == 1.0
    assert "fts" in result["varied_legs"] and "vec" in result["varied_legs"]
    assert "ppr" not in result["varied_legs"]


def test_fit_returns_none_when_only_one_leg_varies(conn, tmp_home):
    """Leg weights are RELATIVE — with only one leg varying there is no
    identifiable contrast, so the fit declines rather than propose a no-op."""
    mem = seed_memory(conn, "x")
    # Every labeled row is fts (some helpful, some harmful); vec/ppr never vary.
    _seed_rlog(conn, mem, [("fts", "t-help", 300), ("fts", "t-harm", 300)])
    state = _state(tmp_home)
    try:
        assert fit_weights.fit(conn, state, min_rows=500) is None
    finally:
        state.close()


# ---------------------------------------------------------------------------
# tune(): the fit rides the SINGLE shadow proposal, never applied
# ---------------------------------------------------------------------------

def _tune_shift(conn, tmp_home, mode):
    lease.acquire(conn, "dream", "t")
    return Shift(shift_id="s", conn=conn, holder="t",
                 config={"_forced_mode": mode, "hermes_home": str(tmp_home)})


def test_tune_records_fit_as_shadow_proposal(conn, tmp_home):
    mem = seed_memory(conn, "always cap the JVM heap at 2GB")
    _seed_rlog(conn, mem, [("fts", "t-help", 300), ("vec", "t-harm", 300)])
    make_state_db(tmp_home, outcomes=_OUTCOMES)   # tune opens this read-only

    result = tune_mod.run(_tune_shift(conn, tmp_home, "shadow"))

    assert result.get("fusion_fit") == 600
    prop = conn.execute(
        "SELECT status, payload FROM proposals WHERE kind='tuning'").fetchone()
    assert prop is not None and prop["status"] == "shadow"
    payload = json.loads(prop["payload"])
    assert "fusion_weights" in payload
    assert payload["fusion_weights"]["weights"]["fts"] > \
        payload["fusion_weights"]["weights"]["vec"]
    # A shadow audit was written; nothing was applied.
    assert conn.execute("SELECT count(*) AS n FROM audit_log WHERE "
                        "action='would_tune'").fetchone()["n"] == 1
    assert conn.execute("SELECT count(*) AS n FROM proposals WHERE "
                        "status='applied'").fetchone()["n"] == 0


def test_tune_never_applies_even_when_forced_active(conn, tmp_home):
    """The hard v1 invariant: even a forced 'active' only PROPOSES."""
    mem = seed_memory(conn, "always cap the JVM heap at 2GB")
    _seed_rlog(conn, mem, [("fts", "t-help", 300), ("vec", "t-harm", 300)])
    make_state_db(tmp_home, outcomes=_OUTCOMES)

    tune_mod.run(_tune_shift(conn, tmp_home, "active"))

    props = conn.execute(
        "SELECT status FROM proposals WHERE kind='tuning'").fetchall()
    assert props and all(p["status"] == "shadow" for p in props)
    assert conn.execute("SELECT count(*) AS n FROM proposals WHERE "
                        "status='applied'").fetchone()["n"] == 0
    # And no weights leaked into live-retrieval config (meta) either.
    assert db.get_meta(conn, "fusion_weights") is None


def test_tune_without_fit_still_proposes_feature_contrast(conn, tmp_home):
    """The fit is additive: with no labeled rows (fit=None) but a per-memory
    contrast signal, tune still writes its original shadow proposal."""
    for i in range(12):
        mid = seed_memory(conn, f"pinned fact {i}", pinned=1)
        conn.execute("UPDATE memories SET helpful_count=6 WHERE id=?", (mid,))
    for i in range(12):
        mid = seed_memory(conn, f"plain fact {i}")
        conn.execute("UPDATE memories SET harmful_count=6 WHERE id=?", (mid,))
    conn.commit()
    make_state_db(tmp_home, outcomes=_OUTCOMES)   # exists, but no injected rows

    result = tune_mod.run(_tune_shift(conn, tmp_home, "shadow"))

    assert result.get("proposed", 0) >= 1
    assert "fusion_fit" not in result            # no labeled rows -> no fit
    prop = conn.execute(
        "SELECT payload FROM proposals WHERE kind='tuning'").fetchone()
    assert prop is not None
    assert "fusion_weights" not in json.loads(prop["payload"])


# ---------------------------------------------------------------------------
# weighted_rrf(): dormant plumbing
# ---------------------------------------------------------------------------

def test_weighted_rrf_unit_weights_equal_rrf():
    rankings = [["a", "b", "c"], ["b", "c", "d"]]
    assert weighted_rrf(rankings, [1.0, 1.0]) == rrf(rankings)


def test_weighted_rrf_upweights_a_leg():
    # Leg 0 ranks x first; leg 1 ranks y first. Up-weighting leg 0 lifts x.
    rankings = [["x", "y"], ["y", "x"]]
    scores = weighted_rrf(rankings, [2.0, 0.5])
    assert scores["x"] > scores["y"]


def test_weighted_rrf_missing_weight_defaults_to_one():
    rankings = [["a"], ["b"], ["c"]]
    scores = weighted_rrf(rankings, [1.0])   # legs 1,2 default to 1.0
    assert scores["b"] == scores["c"]
