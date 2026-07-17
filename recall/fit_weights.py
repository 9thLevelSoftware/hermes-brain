"""Learned fusion weights — PROPOSED, never auto-applied (memory-engine.md §3.7).

Today retrieval fuses its legs with fixed RRF (recall/fusion.py, k=60, every
leg equal). §3.7's "log now, fit later" hook says: once enough labeled data
exists in ``retrieval_log``, fit a logistic model over per-leg features to
estimate how much each leg actually PREDICTS a helpful injection, and turn the
learned coefficients into proposed convex leg weights. This module is the fit.

Three properties are load-bearing:

  * **Cold-start safe.** With fewer than ``min_rows`` labeled rows ``fit``
    returns ``None`` and RRF stays the default (and the permanent fallback).
  * **Shadow only.** ``fit`` computes weights; it NEVER writes them anywhere
    or touches live retrieval. dream/tune.py records the result as a SHADOW
    ``proposals`` row for human review — the one place weights could ever be
    applied, and only by a deliberate out-of-band act (design §2/§3).
  * **Never raises.** A fit failure degrades to ``None`` (RRF), like every
    other capture/learning-path function.

Features (bounded by what the v1 log schema actually stores — see the REPORT:
``retrieval_log`` keeps the FUSED leg-membership string and the fused
``rank_score``, not per-leg reciprocal ranks, so a true (r_fts, r_vec, r_ppr,
s_rerank) vector would need a schema change that is out of scope for this
slice). Per labeled candidate row: leg-membership indicators (fts / vec / ppr,
parsed from the ``leg`` column) plus the standardized fused ``rank_score``.
The learned coefficient on each leg-membership indicator IS that leg's proposed
convex weight — which is exactly what recall/fusion.py:weighted_rrf consumes.

Label (helpful=1 / harmful=0): the injection -> outcome join the whole learning
system rests on. A row is labeled by resolving ``resolved_turn_id`` (filled by
the nightly miner) against Hermes's ``state.db`` ``turn_outcomes`` via the
SHARED read-only reader (dream/mine_state.py — critique item 9: one state.db
surface) and reusing ``mine_state.credit`` so the label is byte-identical to
the one the miner used. Neutral rows (resolved but no verdict) carry no signal
and are dropped.

Pure-Python (no numpy / sklearn): a handful of features and a few hundred
full-batch gradient-descent steps are microseconds and keep the floor tier
dependency-free.
"""

from __future__ import annotations

import logging
import math
import sqlite3

logger = logging.getLogger(__name__)

# The three RRF fusion legs weighted_rrf can consume. `guidance` (separate
# injection channel, rank_score NULL) and `like` (no-FTS5 degraded fallback)
# are NOT fusion legs, so their rows are excluded from the fit.
_FUSED_LEGS = ("fts", "vec", "ppr")

_MAX_ROWS = 20_000          # labeled rows scanned from retrieval_log (bounded)
_FIT_CAP = 8_000            # rows fed to gradient descent (stride-subsampled)
_SESSION_CHUNK = 400        # session ids per turn_outcomes IN () query
_MIN_CLASS = 10             # need this many of EACH class to fit
_MIN_VARIED_LEGS = 2        # need contrast among >= 2 legs (weights are relative)

_ITERS = 800
_LR = 0.5
_L2 = 1e-3
_WEIGHT_LO, _WEIGHT_HI = 0.5, 2.0   # clamp band for proposed weights (conservative)


def fit(conn: sqlite3.Connection, state: sqlite3.Connection | None = None, *,
        min_rows: int = 500) -> dict | None:
    """Fit proposed per-leg fusion weights over labeled ``retrieval_log`` rows.

    conn:  the brain.db connection (holds retrieval_log).
    state: an OPEN read-only ``state.db`` connection (holds turn_outcomes) — the
           label source. ``None`` (standalone / no state.db) => no labels =>
           ``None``. dream/tune.py supplies this via mine_state's shared reader.
    min_rows: cold-start floor — below this many LABELED rows, return ``None``
              (RRF stays the default).

    Returns a proposal dict (per-leg ``weights`` + fit diagnostics) or ``None``.
    Never raises, never applies anything.
    """
    try:
        return _fit(conn, state, min_rows)
    except Exception as e:  # learning path — degrade to RRF, never raise
        logger.warning("fit_weights.fit failed: %s", e)
        return None


def _fit(conn: sqlite3.Connection, state: sqlite3.Connection | None,
         min_rows: int) -> dict | None:
    if state is None:
        return None
    # Reuse the single state.db surface: same table probe + label mapping the
    # miner uses, so a fitted label can never diverge from a mined one.
    from ..dream.mine_state import credit, has_table

    if not has_table(state, "turn_outcomes"):
        return None

    rows = conn.execute(
        "SELECT session_id, resolved_turn_id, leg, rank_score FROM retrieval_log"
        " WHERE injected=1 AND resolved_turn_id IS NOT NULL"
        " ORDER BY id LIMIT ?", (_MAX_ROWS,),
    ).fetchall()
    if not rows:
        return None

    labels = _resolve_labels(state, rows, credit)

    # Build (features, y). features = [has_fts, has_vec, has_ppr, rank_score|None]
    samples: list[tuple[list[float | None], float]] = []
    n_helpful = n_harmful = 0
    for r in rows:
        legs = set((r["leg"] or "").split("+"))
        if legs.isdisjoint(_FUSED_LEGS):
            continue  # guidance / like rows are not fusion legs
        verdict = labels.get((str(r["session_id"]), str(r["resolved_turn_id"])))
        if verdict == "helpful":
            y = 1.0
            n_helpful += 1
        elif verdict == "harmful":
            y = 0.0
            n_harmful += 1
        else:
            continue  # neutral (resolved but no verdict) — no training signal
        rank = r["rank_score"]
        samples.append((
            [1.0 if "fts" in legs else 0.0,
             1.0 if "vec" in legs else 0.0,
             1.0 if "ppr" in legs else 0.0,
             float(rank) if rank is not None else None],
            y,
        ))

    n_labeled = len(samples)
    if n_labeled < min_rows:
        return None                       # cold start — RRF stays the default
    if n_helpful < _MIN_CLASS or n_harmful < _MIN_CLASS:
        return None                       # single-class-ish — nothing to learn

    # Which legs actually vary in the data — only those are identifiable, and
    # leg weights are RELATIVE, so we need contrast among >= 2 of them.
    varied = [leg for i, leg in enumerate(_FUSED_LEGS)
              if 0 < sum(1 for f, _ in samples if f[i]) < n_labeled]
    if len(varied) < _MIN_VARIED_LEGS:
        return None

    fit_samples = _subsample(samples, _FIT_CAP)
    x, y = _design_matrix(fit_samples)
    w, iterations, converged = _fit_logreg(x, y, iters=_ITERS, lr=_LR, l2=_L2)

    coefs = {"fts": w[1], "vec": w[2], "ppr": w[3]}
    weights = _coefs_to_weights(coefs, set(varied))
    if weights is None:
        return None

    return {
        "weights": weights,                         # PROPOSED per-leg convex weights
        "baseline": {leg: 1.0 for leg in _FUSED_LEGS},  # RRF = every leg equal
        "n_labeled": n_labeled,
        "n_helpful": n_helpful,
        "n_harmful": n_harmful,
        "n_fit": len(fit_samples),
        "varied_legs": varied,
        "coefficients": {
            "bias": round(w[0], 4), "fts": round(w[1], 4), "vec": round(w[2], 4),
            "ppr": round(w[3], 4), "rank_score": round(w[4], 4),
        },
        "iterations": iterations,
        "converged": converged,
        "note": ("logistic fit over leg-membership + fused rank_score; PROPOSED "
                 "only — RRF remains the live default and the fallback"),
    }


# ---------------------------------------------------------------------------
# labels (the injection -> outcome join, via the shared state.db reader)
# ---------------------------------------------------------------------------

def _resolve_labels(state: sqlite3.Connection, rows, credit) -> dict:
    """{(session_id, turn_id): 'helpful'|'harmful'|None} for the rows' turns.

    Batched by session (chunked IN () to respect SQLite's variable limit); the
    verdict is produced by mine_state.credit — the SAME mapping the miner used.
    """
    sessions = sorted({str(r["session_id"]) for r in rows if r["session_id"]})
    out: dict[tuple[str, str], str | None] = {}
    for i in range(0, len(sessions), _SESSION_CHUNK):
        chunk = sessions[i:i + _SESSION_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        try:
            outcome_rows = state.execute(
                "SELECT session_id, turn_id, outcome, feedback_kind, feedback_value"
                f" FROM turn_outcomes WHERE session_id IN ({placeholders})", chunk,
            ).fetchall()
        except sqlite3.Error:
            continue
        for o in outcome_rows:
            out[(str(o["session_id"]), str(o["turn_id"]))] = credit(o)
    return out


# ---------------------------------------------------------------------------
# pure-Python logistic regression
# ---------------------------------------------------------------------------

def _subsample(samples: list, cap: int) -> list:
    """Deterministic stride subsample so gradient descent stays bounded."""
    if len(samples) <= cap:
        return samples
    step = len(samples) // cap
    return samples[::step][:cap]


def _design_matrix(samples: list) -> tuple[list[list[float]], list[float]]:
    """[bias, fts, vec, ppr, z(rank_score)] with rank_score standardized.

    A NULL/absent rank_score (guidance never reaches here, but be defensive)
    lands at the mean (z=0), so it contributes nothing rather than skewing.
    """
    ranks = [f[3] for f, _ in samples if f[3] is not None]
    if ranks:
        mean = sum(ranks) / len(ranks)
        var = sum((r - mean) ** 2 for r in ranks) / len(ranks)
        std = math.sqrt(var) if var > 1e-12 else 1.0
    else:
        mean, std = 0.0, 1.0
    x: list[list[float]] = []
    y: list[float] = []
    for feats, label in samples:
        z = 0.0 if feats[3] is None else (feats[3] - mean) / std
        x.append([1.0, feats[0], feats[1], feats[2], z])
        y.append(label)
    return x, y


def _sigmoid(t: float) -> float:
    if t < -35.0:
        return 0.0
    if t > 35.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-t))


def _fit_logreg(x: list[list[float]], y: list[float], *, iters: int, lr: float,
                l2: float) -> tuple[list[float], int, bool]:
    """Full-batch gradient descent on L2-regularized logistic loss.

    Returns (weights, iterations_run, converged). Bias (index 0) is NOT
    regularized. Converges when the mean loss stops improving.
    """
    n = len(x)
    d = len(x[0])
    w = [0.0] * d
    prev = float("inf")
    for it in range(iters):
        grad = [0.0] * d
        loss = 0.0
        for xi, yi in zip(x, y, strict=True):
            p = _sigmoid(sum(w[j] * xi[j] for j in range(d)))
            err = p - yi
            for j in range(d):
                grad[j] += err * xi[j]
            loss -= yi * math.log(p + 1e-12) + (1.0 - yi) * math.log(1.0 - p + 1e-12)
        for j in range(d):
            reg = 0.0 if j == 0 else l2 * w[j]
            w[j] -= lr * (grad[j] / n + reg)
        loss /= n
        if abs(prev - loss) < 1e-6:
            return w, it + 1, True
        prev = loss
    return w, iters, False


def _coefs_to_weights(coefs: dict[str, float], varied: set[str]) -> dict | None:
    """Map learned leg coefficients to conservative convex weights.

    A varied leg's weight is exp(coef) (its odds multiplier), the whole varied
    set renormalized so its MEAN is 1.0 (an RRF-centered nudge, not an absolute
    scale), then clamped to [_WEIGHT_LO, _WEIGHT_HI]. Non-varied legs (not
    identifiable) stay neutral at 1.0. Returns None if the proposal collapses to
    a no-op (nothing to review).
    """
    raw = {leg: math.exp(coefs[leg]) if leg in varied else 1.0 for leg in _FUSED_LEGS}
    varied_vals = [raw[leg] for leg in _FUSED_LEGS if leg in varied]
    mean = sum(varied_vals) / len(varied_vals)
    if mean <= 0:
        return None
    weights = {}
    for leg in _FUSED_LEGS:
        val = raw[leg] / mean if leg in varied else 1.0
        weights[leg] = round(min(_WEIGHT_HI, max(_WEIGHT_LO, val)), 3)
    if all(abs(v - 1.0) < 1e-3 for v in weights.values()):
        return None  # degenerate no-op — do not propose RRF-as-RRF
    return weights
