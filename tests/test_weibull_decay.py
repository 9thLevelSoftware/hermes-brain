"""Weibull per-kind decay shapes in the forget value score (Phase C).

Covers: with shape k=1 the Weibull recency reproduces the legacy
``0.5 ** (age/half_life)`` exactly (equivalence proof); different kinds with
different shapes DIVERGE at the same age (a 'warning' retains more value than
an ephemeral kind); NULL half_life is untouched by the gate (never routed
through per-kind decay); the gate OFF reproduces the legacy kind-independent
score; and dream/weibull.py imports with numpy unavailable (guarded).
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time

import pytest
from brain.dream.forget import _decay_shape, _value_score
from brain.dream.weibull import halflife_survival, weibull_survival
from conftest import seed_memory


def mem_row(conn, mem_id):
    return conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()


# A fixed "now" reference (epoch) matched to the seeded ages below.
NOW = 1_800_000_000.0


def _seed_aged(conn, *, kind, age_days, half_life_days):
    """Seed a bare low-signal row of a given kind/age/half-life so the value
    score is dominated by the recency term (all other terms are 0)."""
    valid_from = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.gmtime(NOW - age_days * 86400.0)) + ".000Z"
    mem = seed_memory(conn, f"{kind} row aged {age_days}d", kind=kind,
                      half_life_days=half_life_days, valid_from=valid_from)
    return mem_row(conn, mem)


# ---------------------------------------------------------------------------
# unit: k=1 reproduces the legacy exponential exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("age_days,half_life", [
    (0.0, 30.0), (5.0, 30.0), (30.0, 30.0), (60.0, 30.0),
    (100.0, 45.0), (365.0, 90.0),
])
def test_halflife_survival_k1_equals_legacy(age_days, half_life):
    """S(t) with scale=half_life/ln2 and k=1 == 0.5 ** (age/half_life)."""
    legacy = 0.5 ** (age_days / half_life)
    assert halflife_survival(age_days, half_life, 1.0) == pytest.approx(legacy)


def test_halflife_survival_half_life_point_is_one_half():
    # At age == half_life with k=1 survival is exactly 0.5 by construction.
    assert halflife_survival(30.0, 30.0, 1.0) == pytest.approx(0.5)


def test_weibull_survival_core_math():
    # S(t) = exp(-(t/scale)^k); degenerate inputs clamp, never raise.
    assert weibull_survival(0.0, 10.0, 1.0) == 1.0
    assert weibull_survival(-5.0, 10.0, 1.0) == 1.0
    assert weibull_survival(10.0, 0.0, 1.0) == 0.0     # non-positive scale
    assert weibull_survival(10.0, 10.0, 0.0) == 0.0    # non-positive shape
    assert weibull_survival(10.0, 10.0, 1.0) == pytest.approx(math.exp(-1.0))


# ---------------------------------------------------------------------------
# shape map sanity
# ---------------------------------------------------------------------------

def test_decay_shapes_directionality():
    # Durable/safety kinds have heavy tails (k < 1); ephemeral kinds k > 1;
    # unmapped kinds fall back to k=1 (legacy).
    assert _decay_shape("warning") < 1.0
    assert _decay_shape("guardrail") < 1.0
    assert _decay_shape("preference") < 1.0
    assert _decay_shape("event") > 1.0
    assert _decay_shape("request") > 1.0
    assert _decay_shape("decision") == 1.0
    assert _decay_shape("some_unmapped_kind") == 1.0
    assert _decay_shape(None) == 1.0


# ---------------------------------------------------------------------------
# integration: k=1 kinds reproduce the legacy score under the score fn
# ---------------------------------------------------------------------------

def test_value_score_k1_kind_matches_legacy_path(conn):
    # A k=1 kind ('decision') must score identically whether the Weibull gate
    # is on or off — k=1 is the legacy exponential.
    row = _seed_aged(conn, kind="decision", age_days=75.0, half_life_days=30.0)
    on = _value_score(row, NOW, True)
    off = _value_score(row, NOW, False)
    assert on == pytest.approx(off)


def test_value_score_unmapped_kind_matches_legacy_path(conn):
    # Unmapped kind -> default shape 1.0 -> legacy-equivalent under the gate.
    row = _seed_aged(conn, kind="mystery_kind", age_days=75.0, half_life_days=30.0)
    assert _value_score(row, NOW, True) == pytest.approx(_value_score(row, NOW, False))


# ---------------------------------------------------------------------------
# integration: different kinds -> divergent decay at the same age
# ---------------------------------------------------------------------------

def test_divergent_decay_warning_outlives_ephemeral(conn):
    # Same age (well past the half-life) and same half-life; only the kind
    # differs. With the gate ON, the heavy-tailed 'warning' retains more value
    # than the ephemeral 'event' (k>1 falls off a cliff).
    age, hl = 90.0, 30.0   # 3 half-lives old
    warning = _seed_aged(conn, kind="warning", age_days=age, half_life_days=hl)
    event = _seed_aged(conn, kind="event", age_days=age, half_life_days=hl)

    w_on = _value_score(warning, NOW, True)
    e_on = _value_score(event, NOW, True)
    assert w_on > e_on

    # Sanity: the recency term itself diverges in the same direction.
    assert (halflife_survival(age, hl, _decay_shape("warning"))
            > halflife_survival(age, hl, _decay_shape("event")))


def test_gate_off_is_kind_independent(conn):
    # With the gate OFF, kind is ignored: warning and event score identically
    # (the legacy 0.5**(age/half_life) path, byte-for-byte kind-agnostic).
    age, hl = 90.0, 30.0
    warning = _seed_aged(conn, kind="warning", age_days=age, half_life_days=hl)
    event = _seed_aged(conn, kind="event", age_days=age, half_life_days=hl)
    assert _value_score(warning, NOW, False) == pytest.approx(
        _value_score(event, NOW, False))


# ---------------------------------------------------------------------------
# NULL half_life is never routed through per-kind decay
# ---------------------------------------------------------------------------

def test_null_half_life_untouched_by_gate(conn):
    # A row with NULL half_life takes the no-decay reference branch regardless
    # of the gate — Weibull never touches it. On == Off for every kind.
    for kind in ("warning", "event", "fact"):
        row = _seed_aged(conn, kind=kind, age_days=90.0, half_life_days=None)
        assert _value_score(row, NOW, True) == pytest.approx(
            _value_score(row, NOW, False))


# ---------------------------------------------------------------------------
# floor-tier: imports with numpy unavailable (guarded)
# ---------------------------------------------------------------------------

def test_weibull_imports_without_numpy():
    """dream/weibull.py must import at the stdlib floor tier (no numpy)."""

    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == "numpy" or name.startswith("numpy."):
                raise ImportError("numpy blocked for test")
            return None

    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "numpy" or k.startswith("numpy.")}
    try:
        path = os.path.join(os.path.dirname(__file__), "..", "dream", "weibull.py")
        spec = importlib.util.spec_from_file_location("brain_weibull_no_numpy", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # must not raise with numpy blocked
        assert mod._HAVE_NUMPY is False
        assert mod.halflife_survival(10.0, 30.0, 1.0) == pytest.approx(
            0.5 ** (10.0 / 30.0))
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
