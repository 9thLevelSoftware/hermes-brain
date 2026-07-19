"""Query-intent classifier — SHADOW-ONLY retrieval-weight proposals.

Vendored from mnemosyne-oss/mnemosyne (``mnemosyne/core/query_intent.py``,
MIT, (c) 2026 Abdias J), adapted for hermes-brain; used SHADOW-ONLY.

A regex classifier that labels a query's intent (temporal / factual / entity /
preference / procedural / general) and *proposes* per-leg retrieval weight
biases per intent. In hermes-brain this is observability scaffolding only: it
mirrors how the ``tune`` strategy treats learned fusion weights — it may
PROPOSE, it must NEVER apply. Nothing in this module reads or mutates live
retrieval state, and ``search()`` does not import it. The provider/dream side
may log a proposal to ``audit_log`` for shadow analysis (see
``record_proposal``); v1 never feeds these deltas into RRF fusion or ranking.

The donor is pure-stdlib (``re`` + ``dataclasses``); it needs no numpy, so the
module imports cleanly at the stdlib floor tier by construction. An optional
numpy import is guarded below purely so that, if a future confidence path wants
it, the floor tier still imports and degrades to the stdlib path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

try:  # optional extra; floor tier has no numpy — degrade, never fail to import
    import numpy as _np  # noqa: F401  (reserved for an optional confidence path)

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - exercised only at floor tier
    _np = None
    _HAS_NUMPY = False


@dataclass
class Intent:
    """Classification result for a query (non-mutating value object)."""

    category: str  # temporal, factual, entity, preference, procedural, general
    confidence: float  # 0.0 - 1.0
    signals: list = field(default_factory=list)  # which pattern categories matched

    # Proposed weight adjustments (multipliers) — SHADOW-ONLY, never applied.
    vec_bias: float = 1.0
    fts_bias: float = 1.0
    importance_bias: float = 1.0


# Backwards-compatible alias for the donor's type name.
QueryIntent = Intent


# Regex patterns per intent category (verbatim from the donor).
INTENT_PATTERNS = [
    # TEMPORAL — "when", "last week", "yesterday", dates, etc.
    ("temporal", [
        r"\b(when|last|yesterday|today|tomorrow|ago|before|after|since|until|during|recently|lately)\b",
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(this|next|last)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b\d+\s+(day|week|month|year|hour|minute)s?\s+(ago|from now|later|earlier)\b",
    ]),

    # FACTUAL — "what is", "who is", "where is", concrete facts
    ("factual", [
        r"\bwhat\s+is\b",
        r"\bwho\s+is\b",
        r"\bwhere\s+is\b",
        r"\b(definition|define|explain|meaning)\b",
        r"\bhow\s+(many|much|long|far)\b",
    ]),

    # ENTITY — seeking info about a person/place/thing
    ("entity", [
        r"\b(tell\s+me\s+about|what\s+do\s+you\s+know\s+about)\b",
        r"\b(who\s+is|what\s+does)\s+[a-z]+\b",
        r"\b(about|regarding|concerning)\s+[a-z]+\b",
    ]),

    # PREFERENCE — likes, dislikes, preferences
    ("preference", [
        r"\b(prefer|like|dislike|want|hate|love|enjoy|favorite|best|worst)\b",
        r"\b(should\s+i|would\s+you|do\s+you\s+recommend)\b",
        r"\b(choose|pick|select|option|choice|decide)\b",
    ]),

    # PROCEDURAL — "how to", "how do I", steps/processes
    ("procedural", [
        r"\bhow\s+(to|do|can|should|would)\b",
        r"\b(step|process|procedure|workflow|guide|tutorial)\b",
        r"\b(setup|install|configure|build|deploy|run|execute|start|stop)\b",
    ]),
]


# Proposed weight biases per intent (SHADOW-ONLY multipliers).
INTENT_WEIGHTS = {
    "temporal": {"vec_bias": 0.6, "fts_bias": 1.5, "importance_bias": 0.8},
    "factual": {"vec_bias": 1.0, "fts_bias": 1.2, "importance_bias": 0.9},
    "entity": {"vec_bias": 1.1, "fts_bias": 1.0, "importance_bias": 1.3},
    "preference": {"vec_bias": 0.9, "fts_bias": 0.8, "importance_bias": 1.5},
    "procedural": {"vec_bias": 1.3, "fts_bias": 0.9, "importance_bias": 0.7},
    "general": {"vec_bias": 1.0, "fts_bias": 1.0, "importance_bias": 1.0},
}


def classify(query: str) -> Intent:
    """Classify the search intent of a query. Pure function — no side effects.

    Args:
        query: The user's search query.

    Returns:
        An ``Intent`` with category, confidence, matched signals, and the
        PROPOSED per-leg biases for that category. Nothing is applied here.
    """
    query_lower = (query or "").lower()
    best_intent = "general"
    best_score = 0.0
    all_signals = []

    for category, patterns in INTENT_PATTERNS:
        matches = 0
        for pattern in patterns:
            if re.search(pattern, query_lower):
                matches += 1
                all_signals.append(category)

        if matches > 0:
            # Score: base 0.3 + 0.15 per match, capped at 1.0.
            score = min(0.3 + matches * 0.15, 1.0)
            if score > best_score:
                best_score = score
                best_intent = category

    weights = INTENT_WEIGHTS.get(best_intent, INTENT_WEIGHTS["general"])

    return Intent(
        category=best_intent,
        confidence=best_score,
        signals=all_signals,
        vec_bias=weights["vec_bias"],
        fts_bias=weights["fts_bias"],
        importance_bias=weights["importance_bias"],
    )


# Donor spelling, kept for parity with the upstream import surface.
classify_intent = classify


def propose_weights(intent: Intent) -> dict:
    """Return the PROPOSED weight deltas for an intent as a plain dict.

    SHADOW-ONLY: the returned dict is a *proposal*. It is never read by
    ``recall/search.py`` or fusion; a caller that wants observability logs it
    via ``record_proposal``. v1 never applies these to live retrieval.

    Returns:
        ``{"category", "confidence", "vec_bias", "fts_bias",
        "importance_bias"}`` — multipliers relative to the live default
        weights (all 1.0 == the neutral/general proposal).
    """
    if intent is None:
        intent = Intent(category="general", confidence=0.0)
    return {
        "category": intent.category,
        "confidence": round(float(intent.confidence), 4),
        "vec_bias": intent.vec_bias,
        "fts_bias": intent.fts_bias,
        "importance_bias": intent.importance_bias,
    }


def adjust_weights(
    base_vec: float = 0.5,
    base_fts: float = 0.3,
    base_importance: float = 0.2,
    intent: Intent | None = None,
) -> tuple[float, float, float]:
    """Normalized (vec, fts, importance) weights for a proposal (donor helper).

    Pure math over ``propose_weights`` biases, kept near-verbatim from the
    donor for parity. SHADOW-ONLY: it is a preview of what the proposal WOULD
    yield; no live retrieval reads its output in v1.
    """
    if intent is None:
        intent = Intent(category="general", confidence=0.0)

    vw = base_vec * intent.vec_bias
    fw = base_fts * intent.fts_bias
    iw = base_importance * intent.importance_bias

    total = vw + fw + iw
    if total > 0:
        vw, fw, iw = vw / total, fw / total, iw / total

    return (vw, fw, iw)


def record_proposal(conn, query: str, intent: Intent | None = None,
                    *, actor: str = "shadow") -> dict:
    """Log a shadow intent proposal to ``audit_log`` and return the proposal.

    shadow-only: this records what the classifier WOULD propose for a query so
    the deltas can be analyzed offline; the deltas are NEVER applied to live
    retrieval in v1. The insert mirrors the existing convention
    (``dream/shift.py:Shift.audit``): ``(actor, action, target, detail, ts)``
    with ``detail`` as JSON. Best-effort — this is capture-adjacent, so it
    swallows storage errors rather than raising into any path.
    """
    import json

    from brain.store import db

    if intent is None:
        intent = classify(query)
    proposal = propose_weights(intent)
    detail = dict(proposal)
    detail["signals"] = intent.signals
    try:
        conn.execute(
            "INSERT INTO audit_log (actor, action, target, detail, ts)"
            " VALUES (?,?,?,?,?)",
            (actor, "intent_proposal", None, json.dumps(detail), db.iso_now()),
        )
        conn.commit()
    except Exception:  # pragma: no cover - shadow observability must never raise
        pass
    return proposal
