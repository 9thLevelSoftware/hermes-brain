"""Facts retrieval leg — a structured index over the NL memory rows.

Phase B: `store/facts.py` distills (subject, predicate, object) triples out of
memories; each current-truth triple carries the `memory_id` of the NL row it
was extracted from, so the set of facts is a lightweight structured *index*
over memories. This leg probes that index: given a query, it finds
current-truth facts whose subject OR object mentions a query token and returns
the DISTINCT backing memory ids, ranked.

Like the graph leg (`recall/graph.py`), it returns candidate memory ids ONLY —
`search()` re-fetches those rows through `_memories_by_ids`, which applies the
same trust/scope/kind filters as every other leg, so this leg cannot leak a row
the caller wouldn't otherwise see and does not re-implement scoping.

Safety rules honored here (mirrors `recall/search.py`):
  * Raw query text is NEVER interpolated into SQL — the query is tokenized
    (Unicode-aware) and each token is a bound LIKE parameter with ESCAPE, so
    injection is impossible.
  * Current truth only: every probe is filtered to `valid_until IS NULL`, so a
    superseded fact (a closed row) can never surface.
  * Never raises: this is on the recall/capture path, so any failure is logged
    and degrades to no leg ([]).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from .search import _age_days, _like_escape, _tokens

logger = logging.getLogger(__name__)

# Recency half-life for the ranking factor. A fact's relevance decays gently
# over its `valid_from` age with the same shape as search._modulate's decay
# (0.5 ** age/half_life, floored so an old-but-matching fact isn't annihilated).
_HALF_LIFE_DAYS = 90.0
_RECENCY_FLOOR = 0.3
# Bound the candidate pool pulled from SQLite before Python-side ranking so a
# hot subject/object can't drag in an unbounded row set.
_POOL_CAP = 500


def facts_leg(conn, query, *, limit: int = 8) -> list[int]:
    """Ranked current-truth fact-backed MEMORY ids relevant to the query.

    Tokenizes the query, probes current-truth facts (`valid_until IS NULL`)
    whose subject OR object LIKE any token, ranks the matches by
    ``confidence * recency(valid_from)``, de-duplicates to distinct non-NULL
    ``memory_id`` (keeping each memory's best-ranked fact), and returns up to
    ``limit`` ids in ranked order. Empty on no match / no tokens. Never raises.
    """
    try:
        return _facts_leg(conn, query, limit)
    except Exception as e:
        logger.warning("facts leg failed: %s", e)
        return []


def _facts_leg(conn, query, limit) -> list[int]:
    toks = _tokens(query)
    if not toks:
        return []

    # One OR-group per token: a token matches a fact if it appears in either the
    # subject or the object. Groups are OR-joined (recall-oriented — any token
    # match surfaces the fact), always filtered to current truth and a present
    # backing memory. All token text is bound, never interpolated.
    group = "(subject LIKE ? ESCAPE '\\' OR object LIKE ? ESCAPE '\\')"
    where = " OR ".join(group for _ in toks)
    params: list = []
    for t in toks:
        pat = f"%{_like_escape(t)}%"
        params.extend([pat, pat])
    sql = (
        "SELECT memory_id, confidence, valid_from FROM facts "
        f"WHERE valid_until IS NULL AND memory_id IS NOT NULL AND ({where}) "
        "ORDER BY valid_from DESC LIMIT ?"
    )
    params.append(_POOL_CAP)

    now = datetime.now(UTC)
    best: dict[int, float] = {}
    for row in conn.execute(sql, params).fetchall():
        mid = row["memory_id"]
        if mid is None:
            continue
        conf = row["confidence"]
        conf = 1.0 if conf is None else float(conf)
        recency = max(_RECENCY_FLOOR,
                      0.5 ** (_age_days(row["valid_from"], now) / _HALF_LIFE_DAYS))
        score = conf * recency
        mid = int(mid)
        # De-dup to distinct memory_id, keeping the best-scoring fact per memory.
        if score > best.get(mid, float("-inf")):
            best[mid] = score

    ranked = sorted(best, key=lambda m: best[m], reverse=True)
    return ranked[:limit]
