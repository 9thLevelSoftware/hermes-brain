"""Reciprocal Rank Fusion — lifted from Daem0n-MCP fusion.py semantics
(k=60, rank starts at 1), the module that was correct there but never
wired in (docs/research/daem0n-memory-core.md). Here it IS the fusion path:
FTS and vector legs each contribute a ranked list; RRF combines them
without needing their scores to share a scale.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

RRF_K = 60


def rrf(rankings: Sequence[list[Hashable]], k: int = RRF_K,
        weights: Sequence[float] | None = None) -> dict[Hashable, float]:
    """score(item) = Σ_legs w_leg / (k + rank_in_leg), rank starting at 1.

    ``weights`` is aligned positionally with ``rankings``; omit it (or pass all
    1.0) for the uniform behavior this function had before weights existed —
    that equivalence is asserted in tests, because every existing caller
    depends on it.

    This parameter is the piece `dream/tune.py` was always missing: it fitted
    per-leg weights from the injection→outcome labels, wrote them to a
    proposal, let you approve it, and had nowhere to put them
    (docs/design/alignment-audit.md §F4). Weights are still NEVER applied
    automatically — see recall/weights.py.
    """
    scores: dict[Hashable, float] = {}
    for index, ranking in enumerate(rankings):
        weight = 1.0
        if weights is not None and index < len(weights):
            weight = float(weights[index])
        if weight == 0.0:
            continue
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return scores


def normalized(scores: dict[Hashable, float], floor: float = 0.2) -> dict[Hashable, float]:
    """Min-max to [floor, 1] (same band discipline as the bm25 path)."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {key: 1.0 for key in scores}
    span = hi - lo
    return {key: floor + (1.0 - floor) * (val - lo) / span for key, val in scores.items()}
