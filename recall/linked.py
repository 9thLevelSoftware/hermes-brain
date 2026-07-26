"""Cross-profile recall (docs/design/alignment-audit.md §G2).

Each linked profile is searched through its OWN `search()` call on its own
read-only connection, and the ranked lists are merged with `fusion.rrf()`.

Why not ATTACH + UNION: every statement in `recall/search.py` would have to
become attach-aware, including the FTS5 external-content tables and the
sqlite-vec virtual tables, for a feature most of whose value is on-demand.
Rank-based fusion is also the *correct* merge — scores are min-max normalized
WITHIN a corpus, so a 0.9 from a 50-row profile and a 0.9 from a 5000-row
profile do not mean the same thing, and averaging them would be nonsense. RRF
compares positions, which is exactly the invariant that survives.

Safety, in one place:

* **Owner-trust callers only.** The operator chose full owner access across
  links, so a linked profile IS searched as its owner (peer_cards included).
  That is defensible when it is you reading your own other profile; it is not
  defensible for a gateway peer or an MCP `tool` session, and those never reach
  a link. A link widens what the owner sees and nothing else.
* **Read-only, always.** Connections are opened `mode=ro`.
* **Never raises.** This is reached from the capture path; a broken link logs
  and degrades to local-only results.
"""

from __future__ import annotations

import logging
from typing import Any

from . import fusion

logger = logging.getLogger(__name__)

# A linked profile is relevant but it is not THIS conversation's context, so it
# is fused at a slight discount rather than as an equal peer.
DEFAULT_LINK_WEIGHT = 0.85


def search_linked(
    conn,
    query: str,
    *,
    local_hits: list,
    trust_tier: str = "owner",
    limit: int = 8,
    link_weight: float = DEFAULT_LINK_WEIGHT,
    embedder=None,
    reranker=None,
    **search_kwargs: Any,
) -> list:
    """Merge local hits with hits from every enabled linked profile.

    Returns ``local_hits`` unchanged for any non-owner caller, for an empty
    query, or when nothing is linked.
    """
    if trust_tier != "owner":
        # The load-bearing line in this module. Do not soften it into "scope
        # the linked search instead" — a second scope path is a second thing to
        # keep correct forever, and this one is a single comparison.
        return local_hits
    if not query or not query.strip():
        return local_hits

    try:
        from ..store import links as links_mod

        registered = links_mod.enabled(conn)
    except Exception:
        logger.debug("links unavailable; local only", exc_info=True)
        return local_hits
    if not registered:
        return local_hits

    from .search import search as _search

    per_profile: list[tuple[str, list]] = []
    for link in registered:
        remote = None
        try:
            remote = links_mod.open_link(link)
            if remote is None:
                continue
            hits = _search(
                remote, query,
                limit=limit,
                trust_tier="owner",
                embedder=embedder,
                reranker=reranker,
                **search_kwargs,
            )
            for hit in hits:
                hit.profile = link["name"]
            if hits:
                per_profile.append((link["name"], hits))
        except Exception as e:
            # One unreachable profile must not cost the caller its local recall.
            logger.warning("link %s: search failed (%s); skipping",
                           link.get("name"), e)
        finally:
            if remote is not None:
                try:
                    remote.close()
                except Exception:
                    pass

    if not per_profile:
        return local_hits
    return _merge(local_hits, per_profile, link_weight, limit)


def _merge(local_hits: list, per_profile: list[tuple[str, list]],
           link_weight: float, limit: int) -> list:
    """RRF over per-profile ranked lists, keyed by (profile, uid).

    Keyed on the PAIR, not the uid alone: two profiles can hold the same
    memory (an import, a sync), and collapsing them would let one row's
    duplicate vote for itself twice.
    """
    def key(hit):
        return (hit.profile or "", hit.uid)

    rankings = [[key(h) for h in local_hits]]
    weights = [1.0]
    by_key = {key(h): h for h in local_hits}
    for _name, hits in per_profile:
        rankings.append([key(h) for h in hits])
        weights.append(float(link_weight))
        for hit in hits:
            by_key.setdefault(key(hit), hit)

    fused = fusion.rrf(rankings, weights=weights)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    out = []
    for item_key, score in ordered:
        hit = by_key.get(item_key)
        if hit is None:
            continue
        # Rewrite the score to the fused value so downstream renderers and any
        # caller sorting by score see one consistent scale. The per-corpus
        # scores that produced these ranks are not comparable and must not be
        # carried forward as if they were.
        hit.score = score
        out.append(hit)
    return out[:limit]


def local_only(hits: list) -> list:
    """The subset safe to write against the local database.

    `Hit.id` is a rowid, so a linked hit's id addresses a DIFFERENT database's
    row. Any local write keyed on it — the recall-count bump in
    `search.log_retrieval` above all — would silently corrupt an unrelated
    memory. Filtering is the whole guard.
    """
    return [h for h in hits if getattr(h, "profile", None) is None]
