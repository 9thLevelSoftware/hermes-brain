"""Active retrieval-leg weights (docs/design/alignment-audit.md §F4).

`dream/tune.py` fits per-leg weights from the injection→outcome labels — which
retrieved memories actually helped. Before this module those weights had no
consumer at all: `fusion.rrf()` took no weight parameter, and approving a tune
proposal set `status='approved'` and did nothing. The brain learned what
worked, wrote it down, asked permission, and discarded the answer.

**Weights are never applied automatically.** `tune` stays a `shadow` strategy —
that is a hard invariant, not a default — and the only path from a fitted
weight to live retrieval is an explicit `hermes brain review --approve`. This
module is the store for what a human approved, nothing more.

Stored as one JSON `meta` row so it needs no migration, and a change bumps the
memories generation counter so `recall/query_cache.py` cannot serve results
computed under the old weights.
"""

from __future__ import annotations

import json
import logging
import sqlite3

from ..store import db

logger = logging.getLogger(__name__)

META_KEY = "retrieval_weights"

# The legs `recall/search.py` fuses. Memory and episode variants of a leg share
# one weight: "trust keyword search more than vectors" is the question an
# operator actually has, and splitting it four ways invites overfitting on the
# handful of labels a real brain accumulates.
LEGS: tuple[str, ...] = ("fts", "vec", "graph", "facts")

# Uniform — exactly what rrf() did before weights existed.
DEFAULT: dict[str, float] = dict.fromkeys(LEGS, 1.0)

# A weight outside this band is a bug or a bad fit, not a preference: 0 silently
# deletes a leg and a huge value makes fusion a one-leg ranking with extra steps.
MIN_WEIGHT = 0.1
MAX_WEIGHT = 3.0


def validate(raw: object) -> dict[str, float] | None:
    """Coerce stored/proposed weights to a full, in-band mapping, or None.

    Rejects rather than repairs anything structurally wrong: a half-understood
    weight set silently steering retrieval is precisely the failure mode this
    whole subsystem is being fixed to avoid.
    """
    if not isinstance(raw, dict) or not raw:
        return None
    out = dict(DEFAULT)
    for leg, value in raw.items():
        if leg not in LEGS:
            continue  # unknown legs are ignored, not fatal (forward compat)
        try:
            weight = float(value)
        except (TypeError, ValueError):
            return None
        if not (MIN_WEIGHT <= weight <= MAX_WEIGHT):
            return None
        out[leg] = weight
    return out


def load(conn: sqlite3.Connection) -> dict[str, float]:
    """Active weights, defaulting to uniform. Never raises — this is on the
    capture path (``search()`` must degrade, not fail)."""
    try:
        stored = db.get_meta(conn, META_KEY)
    except Exception:
        return dict(DEFAULT)
    if not stored:
        return dict(DEFAULT)
    try:
        parsed = json.loads(stored)
    except (ValueError, TypeError):
        logger.warning("retrieval weights unparseable; using uniform")
        return dict(DEFAULT)
    return validate(parsed) or dict(DEFAULT)


def save(conn: sqlite3.Connection, weights: dict[str, float]) -> dict[str, float]:
    """Persist approved weights. Raises ValueError on anything invalid — this
    is an operator action, so it should fail loudly rather than degrade."""
    checked = validate(weights)
    if checked is None:
        raise ValueError(
            f"invalid retrieval weights {weights!r}: every leg must be a number "
            f"in [{MIN_WEIGHT}, {MAX_WEIGHT}]")
    db.set_meta(conn, META_KEY, json.dumps(checked, sort_keys=True))
    # Recall results computed under the old weights must not be served.
    db.bump_generation(conn, "mem")
    conn.commit()
    return checked


def reset(conn: sqlite3.Connection) -> None:
    """Back to uniform — the documented escape hatch when an approved weight
    set turns out to be worse than what it replaced."""
    try:
        conn.execute("DELETE FROM meta WHERE key=?", (META_KEY,))
        db.bump_generation(conn, "mem")
        conn.commit()
    except sqlite3.Error as e:
        logger.warning("could not reset retrieval weights: %s", e)


def is_active(conn: sqlite3.Connection) -> bool:
    try:
        return bool(db.get_meta(conn, META_KEY))
    except Exception:
        return False


def for_legs(weights: dict[str, float], leg_names: list[str]) -> list[float]:
    """Positional weights for `fusion.rrf(rankings, weights=...)`."""
    return [float(weights.get(name, 1.0)) for name in leg_names]


# recall/fit_weights.py names the graph leg by its algorithm (Personalized
# PageRank); recall/search.py names it by its role. One rename, here, rather
# than in both.
_FIT_LEG_ALIASES = {"ppr": "graph"}


def from_proposal(payload: object) -> dict[str, float] | None:
    """Extract applicable weights from a `kind='tuning'` proposal payload.

    Two translations, both load-bearing:

    * ``fit_weights`` emits CONVEX weights (they sum to 1, so ~0.33 each over
      three legs). Applied literally that is just a uniform down-scale, and RRF
      ranking is invariant under a uniform scale — approving would provably
      change nothing. Rescaling to mean 1.0 preserves the fitted *relative*
      emphasis, which is the entire signal.
    * ``ppr`` -> ``graph`` (see above).

    Returns None when the payload carries no usable fit, so the caller can say
    so instead of applying an empty dict.
    """
    if not isinstance(payload, dict):
        return None
    fit = payload.get("fusion_weights")
    if not isinstance(fit, dict):
        return None
    raw = fit.get("weights")
    if not isinstance(raw, dict) or not raw:
        return None

    renamed: dict[str, float] = {}
    for leg, value in raw.items():
        name = _FIT_LEG_ALIASES.get(str(leg), str(leg))
        if name not in LEGS:
            continue
        try:
            renamed[name] = float(value)
        except (TypeError, ValueError):
            return None
    if not renamed:
        return None

    mean = sum(renamed.values()) / len(renamed)
    if mean <= 0:
        return None
    rescaled = {leg: round(val / mean, 4) for leg, val in renamed.items()}
    # Clamp rather than reject: a fit on sparse labels can produce an extreme
    # ratio, and the useful part of it survives clamping.
    clamped = {leg: min(MAX_WEIGHT, max(MIN_WEIGHT, val))
               for leg, val in rescaled.items()}
    return validate(clamped)
