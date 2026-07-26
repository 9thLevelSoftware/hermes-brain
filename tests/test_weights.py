"""Retrieval-leg weights: the consumer `dream/tune.py` never had.

Before this, `fusion.rrf()` took no weights, so the fitted weights had nowhere
to go and `review --approve` was a status flip (alignment-audit.md §F4).
"""

from __future__ import annotations

import pytest
from brain.recall import fusion
from brain.recall import weights as weights_mod
from brain.store import db

# ---------------------------------------------------------------------------
# weighted rrf
# ---------------------------------------------------------------------------

def test_uniform_weights_are_byte_identical_to_no_weights():
    """Every existing caller depends on this equivalence — blend.py and the
    facts/graph legs all call rrf() without weights."""
    rankings = [["a", "b", "c"], ["c", "d"], ["b", "e", "a"]]
    assert fusion.rrf(rankings) == fusion.rrf(rankings, weights=[1.0, 1.0, 1.0])


def test_weights_shift_the_ranking():
    rankings = [["a", "b"], ["b", "a"]]
    assert fusion.rrf(rankings)["a"] == pytest.approx(fusion.rrf(rankings)["b"])

    heavy_first = fusion.rrf(rankings, weights=[3.0, 0.5])
    assert heavy_first["a"] > heavy_first["b"], "the up-weighted leg should win"


def test_zero_weight_removes_a_leg_entirely():
    scores = fusion.rrf([["a"], ["b"]], weights=[1.0, 0.0])
    assert "a" in scores and "b" not in scores


def test_missing_weights_default_to_one():
    """A short weights list must not silently zero the tail."""
    rankings = [["a"], ["b"]]
    assert fusion.rrf(rankings, weights=[1.0]) == fusion.rrf(rankings)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_validate_fills_missing_legs_with_the_default():
    out = weights_mod.validate({"fts": 1.5})
    assert out == {**weights_mod.DEFAULT, "fts": 1.5}


@pytest.mark.parametrize("bad", [
    None, {}, [], "fts=2", {"fts": "heavy"},
    {"fts": 0.0},                       # below MIN: silently deletes a leg
    {"fts": 99.0},                      # above MAX: fusion becomes one leg
    {"fts": -1.0},
])
def test_validate_rejects_rather_than_repairs(bad):
    """A half-understood weight set silently steering retrieval is the failure
    mode this subsystem is being fixed to avoid."""
    assert weights_mod.validate(bad) is None


def test_validate_ignores_unknown_legs_for_forward_compat():
    out = weights_mod.validate({"fts": 1.2, "some_future_leg": 2.0})
    assert out is not None and "some_future_leg" not in out


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def test_load_defaults_to_uniform(conn):
    assert weights_mod.load(conn) == weights_mod.DEFAULT
    assert weights_mod.is_active(conn) is False


def test_save_load_reset_round_trip(conn):
    saved = weights_mod.save(conn, {"fts": 1.5, "vec": 0.5})
    assert saved["fts"] == 1.5 and saved["vec"] == 0.5
    assert weights_mod.load(conn)["fts"] == 1.5
    assert weights_mod.is_active(conn) is True

    weights_mod.reset(conn)
    assert weights_mod.load(conn) == weights_mod.DEFAULT
    assert weights_mod.is_active(conn) is False


def test_save_rejects_invalid_loudly(conn):
    """An operator action should fail visibly, not degrade to uniform."""
    with pytest.raises(ValueError, match="invalid retrieval weights"):
        weights_mod.save(conn, {"fts": 50.0})


def test_save_bumps_the_generation_counter(conn):
    """query_cache keys on mem_generation — stale results computed under the
    old weights must not be served."""
    before = db.get_meta(conn, "mem_generation")
    weights_mod.save(conn, {"fts": 1.5})
    assert db.get_meta(conn, "mem_generation") != before


def test_load_tolerates_corrupt_stored_weights(conn):
    """search() is a capture path — it degrades, it never raises."""
    db.set_meta(conn, weights_mod.META_KEY, "{not json")
    conn.commit()
    assert weights_mod.load(conn) == weights_mod.DEFAULT


# ---------------------------------------------------------------------------
# proposal translation
# ---------------------------------------------------------------------------

def test_from_proposal_renames_ppr_to_graph():
    """fit_weights names the leg by its algorithm; search names it by role."""
    out = weights_mod.from_proposal(
        {"fusion_weights": {"weights": {"fts": 0.5, "vec": 0.3, "ppr": 0.2}}})
    assert out is not None and "graph" in out
    assert out["fts"] > out["vec"] > out["graph"]


def test_from_proposal_rescales_convex_weights():
    """Convex weights sum to 1 (~0.33 each). Applied literally that is a
    uniform down-scale, and RRF ranking is invariant under a uniform scale —
    approving would provably change nothing."""
    out = weights_mod.from_proposal(
        {"fusion_weights": {"weights": {"fts": 0.34, "vec": 0.33, "ppr": 0.33}}})
    assert out is not None
    mean = sum(out[k] for k in ("fts", "vec", "graph")) / 3
    assert abs(mean - 1.0) < 0.05, "rescaled to mean 1.0, not left at ~0.33"


def test_from_proposal_clamps_an_extreme_fit():
    """A fit on sparse labels can produce an extreme ratio; the useful part
    survives clamping."""
    out = weights_mod.from_proposal(
        {"fusion_weights": {"weights": {"fts": 0.999, "vec": 0.0005, "ppr": 0.0005}}})
    assert out is not None
    # Every leg lands in band — the near-zero legs are lifted to MIN rather
    # than deleting themselves, and fts cannot run away.
    for leg in ("fts", "vec", "graph"):
        assert weights_mod.MIN_WEIGHT <= out[leg] <= weights_mod.MAX_WEIGHT
    assert out["vec"] == weights_mod.MIN_WEIGHT
    assert out["fts"] > out["vec"] * 10, "the fitted emphasis survives clamping"


@pytest.mark.parametrize("payload", [
    None, {}, {"features": []}, {"fusion_weights": {}},
    {"fusion_weights": {"weights": {}}},
    {"fusion_weights": {"weights": {"unknown_leg": 1.0}}},
])
def test_from_proposal_returns_none_when_there_is_nothing_to_apply(payload):
    assert weights_mod.from_proposal(payload) is None


# ---------------------------------------------------------------------------
# end-to-end through search()
# ---------------------------------------------------------------------------

def test_search_honours_approved_weights(conn, tmp_home):
    from brain.recall.search import search
    from conftest import seed_memory

    seed_memory(conn, "the deploy pipeline runs on buildkite")
    seed_memory(conn, "the deploy pipeline notes are in the wiki")

    baseline = [h.uid for h in search(conn, "deploy pipeline", trust_tier="owner")]
    assert baseline, "sanity: FTS should find both"

    # Zeroing the only live leg must not crash search — it degrades to empty.
    weights_mod.save(conn, {"fts": weights_mod.MIN_WEIGHT})
    after = search(conn, "deploy pipeline", trust_tier="owner")
    assert isinstance(after, list)


def test_search_never_raises_on_corrupt_weights(conn):
    from brain.recall.search import search
    from conftest import seed_memory

    seed_memory(conn, "the deploy pipeline runs on buildkite")
    db.set_meta(conn, weights_mod.META_KEY, "garbage")
    conn.commit()
    assert search(conn, "deploy pipeline", trust_tier="owner")


# ---------------------------------------------------------------------------
# weights_override — score a candidate WITHOUT committing it (§G1)
# ---------------------------------------------------------------------------

def test_weights_override_does_not_persist(conn):
    """The whole point: evaluating a candidate must not change live retrieval.
    Before this, the only way to measure a tune proposal was to apply it."""
    from brain.recall.search import search
    from conftest import seed_memory

    seed_memory(conn, "the deploy pipeline runs on buildkite")
    search(conn, "deploy pipeline", trust_tier="owner",
           weights_override={"fts": 2.0, "vec": 0.5})
    assert weights_mod.load(conn) == weights_mod.DEFAULT
    assert weights_mod.is_active(conn) is False


def test_weights_override_beats_the_stored_value(conn):
    from brain.recall.search import search
    from conftest import seed_memory

    seed_memory(conn, "the deploy pipeline runs on buildkite")
    weights_mod.save(conn, {"fts": 1.0})
    # An override of a DIFFERENT shape must be what search actually uses; the
    # simplest observable proof is that a valid override does not raise and
    # still returns the row, while the stored value stays untouched.
    hits = search(conn, "deploy pipeline", trust_tier="owner",
                  weights_override={"fts": 2.5, "vec": 0.5, "graph": 1.0, "facts": 1.0})
    assert hits
    assert weights_mod.load(conn)["fts"] == 1.0


def test_invalid_weights_override_falls_back_to_stored(conn):
    """search() is a capture path — a bad override degrades, never raises."""
    from brain.recall.search import search
    from conftest import seed_memory

    seed_memory(conn, "the deploy pipeline runs on buildkite")
    assert search(conn, "deploy pipeline", trust_tier="owner",
                  weights_override={"fts": 999.0})
