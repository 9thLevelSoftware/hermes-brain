"""MMR (Maximal Marginal Relevance) diversity re-ranking.

Vendored from mnemosyne-oss/mnemosyne (``mnemosyne/core/mmr.py``),
MIT License, Copyright (c) 2026 Abdias J, adapted for hermes-brain.
See CLAUDE.md for how vendored MIT code is attributed in-file.

MMR balances relevance (a high fusion/rerank score) against novelty
(dissimilarity to already-selected results), so the top-k of a recall
result is not a run of near-duplicates. ``lambda_`` is the tradeoff knob:
1.0 = pure relevance order, 0.0 = pure diversity; the default 0.7 leans on
relevance with a diversity penalty. Similarity defaults to word-level
Jaccard overlap (fast, stdlib-only, no embeddings needed) but a custom
``similarity_fn`` may be supplied.

Pure-Python math, stdlib-only by design (this module sits in the retrieval
path and must load without the ONNX/vector tier). Callers on the capture
path must not let this raise into a turn — it is deterministic and total on
well-formed input, but treat it like every other retrieval stage: degrade,
don't raise.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple


class MMRCandidate(NamedTuple):
    """A candidate for diversity re-ranking.

    ``text`` may be a raw string (tokenized by lowercase whitespace split) or
    an already-tokenized iterable of terms — either is accepted so a caller
    that already holds a token set need not re-join and re-split it.
    """

    id: object
    relevance: float
    text: object


def _tokens(text: object) -> set[str]:
    """Word-level token set: lowercase whitespace split for strings, or the
    terms of an already-tokenized iterable (anything but a bare string)."""
    if isinstance(text, str):
        return set(text.lower().split())
    if isinstance(text, Iterable):
        return {str(tok).lower() for tok in text}
    return set()


def jaccard_similarity(text_a: object, text_b: object) -> float:
    """Jaccard similarity between two texts, using word-level overlap."""
    words_a = _tokens(text_a)
    words_b = _tokens(text_b)

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    return len(intersection) / len(union)


def mmr(
    candidates: Sequence[MMRCandidate | tuple],
    *,
    lambda_: float = 0.7,
    k: int | None = None,
    similarity_fn: Callable[[object, object], float] | None = None,
) -> list:
    """Re-rank ``candidates`` for diversity using Maximal Marginal Relevance.

    Args:
        candidates: sequence of ``(id, relevance, text)`` triples (an
            :class:`MMRCandidate` or any indexable of the same shape). ``text``
            is a string or a pre-tokenized iterable of terms.
        lambda_: relevance vs. diversity tradeoff in ``[0.0, 1.0]``. 1.0 is
            pure relevance order; lower values penalize similarity to
            already-selected items. Default 0.7.
        k: number of candidates to return. ``None`` (default) returns all,
            fully reordered.
        similarity_fn: custom ``(text_a, text_b) -> float`` similarity.
            Defaults to word-level Jaccard overlap.

    Returns:
        A new list of the selected candidate objects (the same objects passed
        in), reordered by MMR and truncated to ``k``. Never mutates the input.
    """
    if k is None:
        k = len(candidates)
    if k <= 0 or not candidates:
        return []
    if len(candidates) == 1:
        return list(candidates[:k])

    if similarity_fn is None:
        similarity_fn = jaccard_similarity

    # Sort by relevance initially; the top item always anchors the selection.
    sorted_candidates = sorted(candidates, key=lambda c: c[1], reverse=True)

    selected = [sorted_candidates[0]]
    remaining = sorted_candidates[1:]

    while remaining and len(selected) < k:
        mmr_scores = []
        for candidate in remaining:
            relevance = candidate[1]

            # Max similarity to any already-selected result.
            max_sim = max(
                similarity_fn(candidate[2], chosen[2]) for chosen in selected
            )

            # MMR formula: λ * relevance - (1-λ) * max_similarity
            score = lambda_ * relevance - (1.0 - lambda_) * max_sim
            mmr_scores.append(score)

        # Select the best MMR-scored candidate.
        best_idx = mmr_scores.index(max(mmr_scores))
        selected.append(remaining.pop(best_idx))

    # Fill from the relevance-ordered tail if the loop exhausted early.
    if len(selected) < k:
        selected.extend(remaining[: k - len(selected)])

    return selected
