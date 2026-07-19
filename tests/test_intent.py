"""Shadow-only query-intent classifier (recall/intent.py).

Covers: representative queries classify to expected intents; ``propose_weights``
returns a delta dict; classification has NO retrieval side effects; and the
module imports/classifies cleanly with numpy unavailable (floor tier).
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest
from brain.recall.intent import (
    Intent,
    adjust_weights,
    classify,
    propose_weights,
    record_proposal,
)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("what happened last Monday", "temporal"),
        ("what is the database password", "factual"),
        ("I prefer dark roast coffee", "preference"),
        ("how do I deploy the service", "procedural"),
        ("tell me about the quarterly plan", "entity"),
        ("banana sandwiches", "general"),
    ],
)
def test_representative_queries_classify(query, expected):
    intent = classify(query)
    assert isinstance(intent, Intent)
    assert intent.category == expected


def test_classify_is_deterministic_and_side_effect_free():
    # Same input -> same output; pure function, nothing external touched.
    a = classify("how do I configure the build")
    b = classify("how do I configure the build")
    assert (a.category, a.confidence, a.signals) == (b.category, b.confidence, b.signals)
    assert a.category == "procedural"
    assert 0.0 <= a.confidence <= 1.0


def test_propose_weights_returns_delta_dict():
    intent = classify("what happened yesterday")
    proposal = propose_weights(intent)
    assert isinstance(proposal, dict)
    assert set(proposal) == {
        "category",
        "confidence",
        "vec_bias",
        "fts_bias",
        "importance_bias",
    }
    # Temporal boosts FTS over vector (a proposal, never applied).
    assert proposal["category"] == "temporal"
    assert proposal["fts_bias"] > proposal["vec_bias"]


def test_general_proposal_is_neutral():
    proposal = propose_weights(classify("banana sandwiches"))
    assert proposal["vec_bias"] == 1.0
    assert proposal["fts_bias"] == 1.0
    assert proposal["importance_bias"] == 1.0
    assert propose_weights(None)["category"] == "general"


def test_adjust_weights_normalizes_to_one():
    vw, fw, iw = adjust_weights(intent=classify("how do I deploy"))
    assert vw + fw + iw == pytest.approx(1.0)


def test_no_retrieval_side_effects(monkeypatch):
    # Classifying must not import or touch the live retrieval / search path.
    import brain.recall.intent as intent_mod

    def _boom(*a, **k):  # pragma: no cover - only fires on a regression
        raise AssertionError("intent classification touched the search path")

    monkeypatch.setattr(intent_mod, "record_proposal", _boom, raising=True)
    intent = classify("what is the api key")
    # propose_weights is pure math over the Intent value object — no DB, no
    # search, no audit write unless record_proposal is called explicitly.
    proposal = propose_weights(intent)
    assert proposal["category"] == "factual"


def test_record_proposal_writes_audit_row(conn):
    proposal = record_proposal(conn, "what happened last week")
    assert proposal["category"] == "temporal"
    row = conn.execute(
        "SELECT actor, action, detail FROM audit_log WHERE action='intent_proposal'"
    ).fetchone()
    assert row is not None
    assert row["actor"] == "shadow"
    assert row["action"] == "intent_proposal"
    assert "temporal" in row["detail"]


def test_imports_cleanly_without_numpy(monkeypatch):
    # Simulate the stdlib floor tier: numpy unavailable. The guarded import
    # must degrade so the module still imports and classifies.
    real_import = builtins.__import__

    def _no_numpy(name, *args, **kwargs):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("simulated floor tier: numpy is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "brain.recall.intent", raising=False)
    monkeypatch.setattr(builtins, "__import__", _no_numpy)
    mod = importlib.import_module("brain.recall.intent")
    assert mod._HAS_NUMPY is False
    assert mod.classify("how do I install this").category == "procedural"
