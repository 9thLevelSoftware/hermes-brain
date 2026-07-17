"""A1 reranker: the late-interaction stage reorders the fused candidates,
and degrades to the fused order on every skip path (no reranker, no model,
no score separation, blown budget). No model download needed — a fake and
the StubReranker exercise the logic deterministically.
"""

from __future__ import annotations

from brain.recall.rerank import StubReranker, get_reranker, rerank_scores
from brain.recall.search import search
from conftest import seed_memory


class _Prefer:
    """Fake reranker that forces docs containing ``needle`` to the top."""

    name = "prefer"

    def __init__(self, needle: str) -> None:
        self.needle = needle

    def score(self, query, docs):
        return [100.0 if self.needle in d else 1.0 for d in docs]


class _Flat:
    """Reranker with no separation — every doc scores the same."""

    name = "flat"

    def score(self, query, docs):
        return [5.0] * len(list(docs))


# ---------------------------------------------------------------------------
# StubReranker + get_reranker factory
# ---------------------------------------------------------------------------

def test_stub_reranker_scores_by_overlap():
    rr = StubReranker()
    scores = rr.score("deploy staging database", ["deploy the staging box", "unrelated text"])
    assert scores[0] > scores[1]


def test_get_reranker_modes():
    # Disabled explicitly, or on any non-ONNX tier -> no stage.
    assert get_reranker({"rerank": "off"}, "full") is None
    assert get_reranker({}, "lite") is None
    assert get_reranker({}, "fts-only") is None
    # The stub is config-only but tier-independent (test tier).
    assert isinstance(get_reranker({"rerank_model": "stub"}, "lite"), StubReranker)
    # Full tier, real model requested but not downloaded and download disabled:
    # degrades to None via ModelDownloadError, never raises.
    assert get_reranker({}, "full", allow_download=False) is None


# ---------------------------------------------------------------------------
# rerank_scores helper degrade paths
# ---------------------------------------------------------------------------

def test_rerank_scores_degrades():
    cands = [("m:1", "deploy alpha"), ("m:2", "deploy omega")]
    assert rerank_scores(None, "deploy", cands) is None            # no reranker
    assert rerank_scores(StubReranker(), "deploy", cands[:1]) is None  # <2 candidates
    assert rerank_scores(StubReranker(), "", cands) is None        # empty query
    # A negative budget always trips the guard -> keep fused order.
    assert rerank_scores(StubReranker(), "deploy", cands, budget_s=-1.0) is None


# ---------------------------------------------------------------------------
# End-to-end through search()
# ---------------------------------------------------------------------------

def test_rerank_drives_top_hit(conn):
    seed_memory(conn, "deploy runbook alpha path")
    seed_memory(conn, "deploy runbook omega path")
    top_alpha = search(conn, "deploy runbook", include_episodes=False, reranker=_Prefer("alpha"))
    top_omega = search(conn, "deploy runbook", include_episodes=False, reranker=_Prefer("omega"))
    assert top_alpha and "alpha" in top_alpha[0].text
    assert top_omega and "omega" in top_omega[0].text
    assert top_alpha[0].uid != top_omega[0].uid  # the reranker flipped the winner


def test_reranker_none_is_noop(conn):
    seed_memory(conn, "deploy runbook alpha path")
    seed_memory(conn, "deploy runbook omega path")
    hits = search(conn, "deploy runbook", include_episodes=False, reranker=None)
    assert len(hits) == 2  # unchanged behavior, no error


def test_flat_rerank_keeps_fused_order(conn):
    seed_memory(conn, "deploy runbook alpha path")
    seed_memory(conn, "deploy runbook omega path")
    base = [h.uid for h in search(conn, "deploy runbook", include_episodes=False)]
    flat = [h.uid for h in search(conn, "deploy runbook", include_episodes=False,
                                  reranker=_Flat())]
    assert base == flat  # no score separation -> fused order preserved


def test_budget_skip_keeps_fused_order(conn):
    seed_memory(conn, "deploy runbook alpha path")
    seed_memory(conn, "deploy runbook omega path")
    base = [h.uid for h in search(conn, "deploy runbook", include_episodes=False)]
    budgeted = [h.uid for h in search(conn, "deploy runbook", include_episodes=False,
                                      reranker=_Prefer("omega"), rerank_budget_s=-1.0)]
    assert base == budgeted  # budget guard tripped -> rerank skipped
