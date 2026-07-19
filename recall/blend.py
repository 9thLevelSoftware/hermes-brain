"""Working-representation blend for lane-2 prefetch.

Composes lane-2 candidates from THREE ranked legs and fuses them with RRF,
then RE-FETCHES the winners through the centralized scoping helper so no
blended hit can bypass trust/scope/kind filtering:

  1. Semantic leg   — hybrid ``recall.search.search`` (FTS + vector + graph),
                      degrades to nothing without an embedder/FTS match.
  2. Reinforced leg — current-truth memories ordered by proven usefulness
                      ``(verification_count + helpful_count) DESC``. Plain
                      SQL; works on every tier (no embedder needed).
  3. Recent leg     — current-truth memories inside the recency window,
                      ``valid_from DESC``. Plain SQL; every tier.

The three ranked id-lists are fused with ``fusion.rrf`` (k=60) over one
keyspace (bare memory ids), then the fused order is re-materialized via
``search._memories_by_ids`` — the SAME access filters the main FTS/vector/
graph legs apply (finding #17: scoping is enforced at row fetch, per leg).
Legs 2/3 also pre-filter by scope so a foreign row never even reaches the
candidate pool; the re-fetch is the authoritative belt-and-braces.

Tier degradation: only leg 1 needs an embedder. With no embedder (fts-only /
stub tier) the semantic leg is empty and the blend is the reinforced+recent
fusion. On the FTS-only floor ``search`` still contributes its LIKE/BM25 leg.

This is a capture-path function: it NEVER raises. Any failure logs at
warning and returns [] so a memory bug can never break the turn.
"""

from __future__ import annotations

import logging
import sqlite3

from . import fusion
from . import search as search_mod

logger = logging.getLogger(__name__)

Hit = search_mod.Hit

# Same internal kinds the generic facts path excludes (provider.py:712):
# guidance-type memories and the owner's private peer cards are never
# surfaced as recalled facts.
DEFAULT_EXCLUDE_KINDS: tuple[str, ...] = ("strategy", "guardrail", "case", "peer_card")

# Current-truth predicate (store/schema.sql §5; mirrors recall/search.py).
_CURRENT_TRUTH = "m.valid_to IS NULL AND m.status = 'active' AND m.live = 1"


def blend(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 8,
    principal_id: str | None = None,
    source_author: str | None = None,
    trust_tier: str = "owner",
    scope_project: str | None = None,
    embedder=None,
    reranker=None,
    recent_days: int = 14,
    exclude_kinds: tuple[str, ...] = DEFAULT_EXCLUDE_KINDS,
    include_episodes: bool = True,
    episode_limit: int = 4,
    exclude_session: str | None = None,
    facts: bool = True,
) -> list[Hit]:
    """Blended lane-2 recall: semantic + reinforced + recent, RRF-fused.

    Returns ``Hit`` objects in blended order — the fused MEMORY working
    representation first, then (unless ``include_episodes`` is False) the raw
    episode hits the semantic leg already surfaced, appended below the
    memories (raw turns rank under distilled memories, the 0.6x discipline).
    Episodes carry the recall on tiers where nothing has been extracted yet
    (stub/floor), so dropping them would silently blind lane 2 there.
    Capped at ``limit`` memories + ``episode_limit`` episodes, with the same
    trust/scope/kind guarantees as ``search``. Never raises.
    """
    try:
        pool = max(limit * 3, limit)

        # -- Leg 1: semantic (hybrid) — the only embedder-dependent leg. It
        #    also carries the episode leg (raw turns), which we retain. --
        semantic_ids: list[int] = []
        episode_hits: list[Hit] = []
        try:
            hits = search_mod.search(
                conn, query, limit=pool,
                exclude_kinds=exclude_kinds,
                scope_project=scope_project,
                principal_id=principal_id,
                source_author=source_author,
                trust_tier=trust_tier,
                include_episodes=include_episodes,
                episode_limit=episode_limit,
                exclude_session=exclude_session,
                facts=facts,
                embedder=embedder,
                reranker=reranker,
            )
            semantic_ids = [h.id for h in hits if h.kind == "memory"]
            if include_episodes:
                episode_hits = [h for h in hits if h.kind == "episode"][:episode_limit]
        except Exception as e:  # search() shouldn't raise, but stay defensive
            logger.warning("blend semantic leg failed (%s); continuing", e)

        # -- Legs 2 & 3: plain SQL, tier-independent. --
        reinforced_ids = _reinforced_ids(
            conn, pool, exclude_kinds, scope_project, principal_id, trust_tier)
        recent_ids = _recent_ids(
            conn, pool, recent_days, exclude_kinds, scope_project,
            principal_id, trust_tier)

        fused = fusion.rrf([semantic_ids, reinforced_ids, recent_ids])
        ordered = [mid for mid, _ in sorted(
            fused.items(), key=lambda kv: kv[1], reverse=True)]

        # -- Authoritative re-fetch: applies trust/scope/kind filters again so
        #    a blended hit can NEVER bypass scoping (finding #17). Empty when
        #    no memory leg fired (e.g. stub/floor with nothing extracted yet) —
        #    the episode leg below still carries the recall. --
        rows = search_mod._memories_by_ids(
            conn, ordered, None, scope_project, principal_id, trust_tier,
            exclude_kinds=exclude_kinds) if ordered else {}

        results: list[Hit] = []
        seen: set[int] = set()
        for mid in ordered:
            if mid in seen:
                continue
            row = rows.get(mid)
            if row is None:  # filtered out by scoping/kind at re-fetch
                continue
            seen.add(mid)
            results.append(Hit(
                kind="memory", id=row["id"], uid=row["uid"],
                text=row["content"] or "", summary=row["summary"],
                memory_type=row["memory_type"], mkind=row["kind"],
                ts=row["valid_from"], platform=row["source_platform"],
                score=fused[mid], source="blend",
            ))
            if len(results) >= limit:
                break
        # Raw episodes rank below the distilled memory blend (0.6x discipline).
        results.extend(episode_hits)
        return results
    except Exception as e:  # capture path — a memory bug must never break a turn
        logger.warning("brain blend failed for %r: %s", query, e)
        return []


def _reinforced_ids(
    conn: sqlite3.Connection,
    n: int,
    exclude_kinds: tuple[str, ...],
    scope_project: str | None,
    principal_id: str | None,
    trust_tier: str,
) -> list[int]:
    """Current-truth memory ids by proven usefulness (no embedder needed)."""
    sql = f"SELECT m.id FROM memories m WHERE {_CURRENT_TRUTH}"
    params: list = []
    sql = _apply_filters(sql, params, exclude_kinds, scope_project,
                         principal_id, trust_tier)
    sql += " ORDER BY (m.verification_count + m.helpful_count) DESC, m.valid_from DESC LIMIT ?"
    params.append(n)
    return [r["id"] for r in conn.execute(sql, params).fetchall()]


def _recent_ids(
    conn: sqlite3.Connection,
    n: int,
    recent_days: int,
    exclude_kinds: tuple[str, ...],
    scope_project: str | None,
    principal_id: str | None,
    trust_tier: str,
) -> list[int]:
    """Current-truth memory ids within the recency window (no embedder)."""
    sql = f"SELECT m.id FROM memories m WHERE {_CURRENT_TRUTH}"
    params: list = []
    # SQLite date math over the ISO-8601 valid_from string. A non-positive
    # window disables the recency floor entirely (all current-truth rows).
    if recent_days and recent_days > 0:
        # datetime() on both sides normalizes the ISO 'T'/'Z' form stored in
        # valid_from to SQLite's space-separated form so the compare is on
        # equal footing (a bare string compare would trip on 'T' vs ' ').
        sql += " AND datetime(m.valid_from) >= datetime('now', ?)"
        params.append(f"-{int(recent_days)} days")
    sql = _apply_filters(sql, params, exclude_kinds, scope_project,
                         principal_id, trust_tier)
    sql += " ORDER BY m.valid_from DESC LIMIT ?"
    params.append(n)
    return [r["id"] for r in conn.execute(sql, params).fetchall()]


def _apply_filters(
    sql: str,
    params: list,
    exclude_kinds: tuple[str, ...],
    scope_project: str | None,
    principal_id: str | None,
    trust_tier: str,
) -> str:
    """Kind/project filters + the centralized trust/scope helper (finding #17).

    Re-uses ``search._scope_memories`` so legs 2/3 enforce the exact same
    non-owner scoping rule (unscoped-or-own, never a peer_card) as the main
    path — the final ``_memories_by_ids`` re-fetch remains authoritative."""
    if exclude_kinds:
        sql += f" AND (m.kind IS NULL OR m.kind NOT IN ({','.join('?' * len(exclude_kinds))}))"
        params.extend(exclude_kinds)
    if scope_project is not None:
        sql += " AND (m.scope_project IS NULL OR m.scope_project = ?)"
        params.append(scope_project)
    return search_mod._scope_memories(sql, params, principal_id, trust_tier)
