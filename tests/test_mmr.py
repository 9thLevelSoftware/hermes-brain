"""MMR diversity re-ranking: pure math, no DB, no LLM.

Vendored Jaccard-based Maximal Marginal Relevance (recall/mmr.py). Covers the
tradeoff knob (lambda_=1.0 is pure relevance order; a lower lambda demotes a
near-duplicate of the top item), plus the empty / single / k-truncation edges.
"""

from __future__ import annotations

from brain.recall.mmr import MMRCandidate, jaccard_similarity, mmr


def _ids(ranked):
    return [c[0] for c in ranked]


def test_lambda_one_is_pure_relevance_order():
    cands = [
        MMRCandidate("a", 0.9, "postgres connection pool tuning"),
        MMRCandidate("b", 0.8, "postgres connection pool sizing"),  # near-dup of a
        MMRCandidate("c", 0.5, "office coffee machine broken"),
    ]
    # lambda_=1.0 ignores similarity: output is exactly relevance-desc order.
    assert _ids(mmr(cands, lambda_=1.0)) == ["a", "b", "c"]


def test_lower_lambda_demotes_near_duplicate():
    cands = [
        MMRCandidate("a", 0.9, "postgres connection pool tuning guide"),
        MMRCandidate("b", 0.8, "postgres connection pool tuning notes"),  # near-dup of a
        MMRCandidate("c", 0.5, "cats are great debugging companions"),
    ]
    # Pure relevance would keep b (the near-dup) in slot 2. A diversity-weighted
    # pass promotes the novel c ahead of b.
    ranked = _ids(mmr(cands, lambda_=0.3))
    assert ranked[0] == "a"          # top item still anchors
    assert ranked[1] == "c"          # novel item beats the near-duplicate
    assert ranked[2] == "b"


def test_empty_input_returns_empty():
    assert mmr([]) == []
    assert mmr([], k=5) == []


def test_single_candidate():
    cands = [MMRCandidate("solo", 0.42, "only one here")]
    assert _ids(mmr(cands)) == ["solo"]
    assert _ids(mmr(cands, k=10)) == ["solo"]


def test_k_truncation():
    cands = [
        MMRCandidate("a", 0.9, "alpha topic one"),
        MMRCandidate("b", 0.8, "beta topic two"),
        MMRCandidate("c", 0.7, "gamma topic three"),
        MMRCandidate("d", 0.6, "delta topic four"),
    ]
    ranked = mmr(cands, k=2)
    assert len(ranked) == 2
    assert ranked[0][0] == "a"       # highest relevance anchors slot 0
    # k=0 (and negative) yields nothing.
    assert mmr(cands, k=0) == []


def test_accepts_plain_tuples_and_token_sets():
    # The public surface accepts bare (id, relevance, text) triples, and text
    # may already be a token set rather than a raw string.
    cands = [
        ("a", 0.9, {"postgres", "pool", "tuning"}),
        ("b", 0.8, {"postgres", "pool", "sizing"}),
        ("c", 0.5, {"coffee", "machine"}),
    ]
    ranked = _ids(mmr(cands, lambda_=1.0))
    assert ranked == ["a", "b", "c"]


def test_jaccard_similarity_basics():
    assert jaccard_similarity("a b c", "a b c") == 1.0
    assert jaccard_similarity("a b", "c d") == 0.0
    assert jaccard_similarity("", "a b") == 0.0
    # two shared of three-unique-total -> 0.5 (a b | a c => {a,b,c}, ∩={a})
    assert jaccard_similarity("a b", "a c") == 1 / 3
