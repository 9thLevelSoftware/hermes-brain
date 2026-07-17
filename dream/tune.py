"""Dream strategy 'tune': shadow-only retrieval-weight self-tuning
(learning-system.md §2 prompt-section optimization / §3 ship-inert).

This is the loop that tunes the loop — and the one the user most explicitly
asked to keep on a leash: "shadow-logged self-tuning ... tuning shadow-log
reviewed before activation." So `tune` NEVER changes a live weight. Its only
output is a `proposals` row (kind='tuning', status='shadow') carrying the
evidence and the suggested nudge, surfaced by `hermes brain review`. Applying
it is a deliberate, human, out-of-band act (there is intentionally no
auto-apply path in v1 — not even with skill_auto_approve).

The signal is correlational and honest about it: over memories that have been
injected and resolved enough times, does a scoring feature (pinned, warning
kind, failed-outcome boost) actually predict helpful injections? Each feature
is a bucket contrast — helpful-rate WITH the feature vs WITHOUT — gated by a
Wilson lower bound on the difference so noise can't propose a change. The
co-injection confound (critique item 28) is respected: only memories with
>= _MIN_SAMPLES resolved injections count.

A SECOND, independent signal rides the same shadow proposal (memory-engine.md
§3.7): once >= _MIN_FIT_ROWS labeled retrieval_log rows exist, recall/fit_weights
fits a logistic model over per-leg features -> PROPOSED convex fusion weights.
It obeys the same leash — it PROPOSES weights folded into the one tuning
proposal, and RRF stays the live default and the permanent fallback. Neither
signal is ever auto-applied, even under a forced 'active'.
"""

from __future__ import annotations

import json
import logging
import sqlite3

from ..store import db
from .shift import Shift
from .stats import wilson_diff_lower_bound

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 5             # per-memory resolved injections to count (item 28)
_MIN_BUCKET = 8             # memories on each side of a contrast
_ACCEPT_DELTA = 0.05        # Wilson lower bound on the rate diff to propose
_MIN_FIT_ROWS = 500         # labeled retrieval_log rows before a weight fit (§3.7)
_PROMPT_VERSION = "tune-v1"

# The live weights `recall/search._modulate` uses, as the tuning baseline.
# tune proposes deltas against these; it never writes them back in v1.
_BASELINE = {"pinned": 1.3, "warning_kind": 1.2, "failed_outcome": 1.5}

# feature -> (predicate SQL over a memories row aliased `m`, human label).
_FEATURES = {
    "pinned": ("m.pinned = 1", "pinned"),
    "warning_kind": ("m.kind = 'warning'", "kind='warning'"),
    "failed_outcome": ("m.outcome = 'failed'", "outcome='failed'"),
}


def run(shift: Shift) -> dict:
    """Never raises. Shadow by default and by design — active is refused."""
    try:
        return _run(shift)
    except Exception as e:
        logger.warning("tune: failed: %s", e, exc_info=True)
        try:
            shift.conn.rollback()
        except sqlite3.Error:
            pass
        return {"error": str(e)}


def _run(shift: Shift) -> dict:
    conn = shift.conn
    mode = shift.config.get("_forced_mode") or shift.mode("tune")
    # Hard guard: tune has no active path in v1. Even a forced 'active' only
    # proposes — the human reviews the shadow log before any weight moves.
    if mode == "off":
        return {"skipped": "off"}

    # Signal 1 (memory-engine.md §3.7): learned convex fusion weights over
    # labeled retrieval_log rows. Shadow only; None on cold start (< 500 labeled
    # rows, no state.db, or no learnable contrast) — RRF stays the default.
    fusion_weights = _fit_fusion_weights(shift)

    # Signal 2: per-memory helpful/harmful over resolved injections (aggregate
    # counters set by dream/mine_state.py). One row per current memory that has
    # enough signal to count.
    rows = conn.execute(
        "SELECT m.id, m.pinned, m.kind, m.outcome, m.helpful_count AS h,"
        " m.harmful_count AS harm FROM memories m"
        " WHERE m.valid_to IS NULL AND m.status='active'"
        " AND (m.helpful_count + m.harmful_count) >= ?", (_MIN_SAMPLES,),
    ).fetchall()
    proposals = _feature_contrasts(rows) if len(rows) >= 2 * _MIN_BUCKET else []

    counts: dict = {"samples": len(rows), "proposed": len(proposals)}
    if fusion_weights:
        counts["fusion_fit"] = fusion_weights["n_labeled"]

    # Nothing to say: preserve the original "insufficient_evidence" contract
    # when the per-memory contrast had too little data AND no weight fit fired.
    if not proposals and not fusion_weights:
        if len(rows) < 2 * _MIN_BUCKET:
            return {"samples": len(rows), "skipped": "insufficient_evidence"}
        return counts

    _record_proposal(shift, proposals, fusion_weights, counts)
    return counts


def _feature_contrasts(rows) -> list:
    """The per-feature Wilson-gated helpful-rate contrasts (the original signal)."""
    proposals = []
    for feature, (_pred, label) in _FEATURES.items():
        contrast = _bucket_contrast(rows, feature)
        if contrast is None:
            continue
        with_rate, without_rate, inc_lb, dec_lb, nw, nwo = contrast
        # BOTH directions use a genuine one-sided Wilson lower bound so noise
        # cannot propose a change (the decrease test must be its OWN lower
        # bound on without-minus-with, NOT the negation of the increase
        # bound — negating an increase LOWER bound gives an UPPER bound on
        # the reverse, which fires whenever the intervals are merely wide).
        if inc_lb >= _ACCEPT_DELTA:
            direction, lb = "increase", inc_lb
        elif dec_lb >= _ACCEPT_DELTA:
            direction, lb = "decrease", dec_lb
        else:
            continue
        proposals.append({
            "feature": feature, "label": label,
            "current_weight": _BASELINE[feature],
            "direction": direction,
            "helpful_rate_with": round(with_rate, 3),
            "helpful_rate_without": round(without_rate, 3),
            "wilson_diff_lb": round(lb, 3),
            "n_with": nw, "n_without": nwo,
        })
    return proposals


def _fit_fusion_weights(shift: Shift) -> dict | None:
    """Fit PROPOSED convex fusion weights over labeled retrieval_log rows.

    Opens Hermes's state.db READ-ONLY through the shared reader (mine_state —
    critique item 9: one state.db surface) to supply the injection->outcome
    labels, then delegates to recall/fit_weights. Shadow only: returns a dict or
    None and NEVER applies weights to live retrieval. Never raises.
    """
    try:
        from ..recall import fit_weights
        from .mine_state import open_state_ro, state_db_path

        path = state_db_path(shift.config)
        if path is None:
            return None
        state = open_state_ro(path)
        try:
            return fit_weights.fit(shift.conn, state, min_rows=_MIN_FIT_ROWS)
        finally:
            state.close()
    except Exception as e:
        logger.warning("tune: fusion-weight fit skipped: %s", e)
        return None


def _record_proposal(shift: Shift, proposals: list, fusion_weights: dict | None,
                     counts: dict) -> None:
    """Record ONE tuning proposal (status=shadow) + a shadow audit, folding in
    BOTH signals. Never supersede a still-open one — accumulate across nights.

    This is the sole proposal-writing path, shared by the feature contrasts and
    the learned fusion weights: both are shadow, both are reviewed before any
    weight moves, and neither is ever written to live retrieval here.
    """
    conn = shift.conn
    open_row = conn.execute(
        "SELECT uid FROM proposals WHERE kind='tuning' AND status='shadow'"
        " ORDER BY created_at DESC LIMIT 1").fetchone()
    payload_obj: dict = {"baseline": _BASELINE, "features": proposals}
    evidence = [p["feature"] for p in proposals]
    if fusion_weights:
        payload_obj["fusion_weights"] = fusion_weights
        evidence.append("fusion_weights")
    payload = json.dumps(payload_obj)
    signals = len(proposals) + (1 if fusion_weights else 0)
    now = db.iso_now()
    if open_row is None:
        uid = db.new_ulid()
        conn.execute(
            "INSERT INTO proposals (uid, kind, target, title, rationale, payload,"
            " evidence, status, shift_id, created_at) VALUES"
            " (?,?,?,?,?,?,?,?,?,?)",
            (uid, "tuning", "retrieval_weights",
             f"retrieval weight tuning ({signals} signal(s))",
             "correlational helpful-rate contrast + logistic leg-weight fit over "
             "resolved injections; SHADOW — review before applying",
             payload,
             json.dumps(evidence), "shadow",
             shift.shift_id, now))
        counts["proposal"] = uid[:8]
    else:
        conn.execute("UPDATE proposals SET payload=?, shift_id=? WHERE uid=?",
                     (payload, shift.shift_id, open_row["uid"]))
        counts["proposal"] = open_row["uid"][:8]
    audit_detail: dict = {"features": proposals}
    if fusion_weights:
        audit_detail["fusion_weights"] = fusion_weights["weights"]
    shift.audit("would_tune", None, audit_detail)
    conn.commit()


def _bucket_contrast(rows, feature):
    """(rate_with, rate_without, inc_lb, dec_lb, n_with, n_without) or None.

    inc_lb = Wilson lower bound on (with_rate - without_rate); dec_lb = Wilson
    lower bound on (without_rate - with_rate). They are computed separately —
    dec_lb is NOT -inc_lb — so each direction has genuine one-sided evidence.
    """
    pred = _PREDICATES[feature]
    with_s = with_n = without_s = without_n = 0
    for r in rows:
        helpful, total = r["h"], r["h"] + r["harm"]
        if total == 0:
            continue
        if pred(r):
            with_s += helpful
            with_n += total
        else:
            without_s += helpful
            without_n += total
    n_with_mem = sum(1 for r in rows if pred(r))
    n_without_mem = len(rows) - n_with_mem
    if n_with_mem < _MIN_BUCKET or n_without_mem < _MIN_BUCKET:
        return None
    if with_n == 0 or without_n == 0:
        return None
    inc_lb = wilson_diff_lower_bound(with_s, with_n, without_s, without_n)
    dec_lb = wilson_diff_lower_bound(without_s, without_n, with_s, with_n)
    return (with_s / with_n, without_s / without_n, inc_lb, dec_lb,
            n_with_mem, n_without_mem)


_PREDICATES = {
    "pinned": lambda r: bool(r["pinned"]),
    "warning_kind": lambda r: r["kind"] == "warning",
    "failed_outcome": lambda r: r["outcome"] == "failed",
}
