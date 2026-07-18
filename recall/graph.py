"""Graph retrieval leg — HippoRAG-2-style Personalized PageRank.

memory-engine.md §1.6: after the keyword + vector legs surface a seed set, we
propagate relevance across the memory graph so a memory connected to the seeds
by a shared entity or an explicit edge — but missed by keyword/vector — still
surfaces. The graph is the entity co-mention structure (`entity_mentions`) plus
the typed `edges` table; PPR is seeded (personalized) on the fused candidates.

Deliberately NO scipy/networkx (learning-system.md §0) and, unlike the design's
numpy-CSR sketch, NO numpy either: the neighborhood is hard-bounded to a few
hundred nodes, so a pure-Python sparse power iteration is fast and lets the
graph leg work on every tier (it needs entities, not the ONNX stack). The leg
returns candidate memory ids only — search() re-fetches rows through the same
access-scoping filters as the other legs, so nothing here can leak scope.

Never raises: a graph failure degrades to no leg ([]).
"""

from __future__ import annotations

import logging

from ..store import entities as ent

logger = logging.getLogger(__name__)

_DAMPING = 0.5
_ITERS = 25
# Keep every dynamic IN-clause well under SQLite's default 999-variable limit,
# so a hot seed/entity set can't raise and (via the outer except) silently drop
# the whole graph leg. _edge_neighbors binds 2x the seeds, so seeds get half.
_MAX_SQL_VARS = 900


def ppr_leg(conn, seed_ids, *, limit: int = 24, max_nodes: int = 400,
            per_entity_cap: int = 60) -> list[int]:
    """Ranked memory ids discovered by Personalized PageRank from the seeds
    (seeds themselves excluded — they are already in the FTS/vec legs). Empty
    when the seeds touch no graph structure. Never raises."""
    try:
        return _ppr(conn, seed_ids, limit, max_nodes, per_entity_cap)
    except Exception as e:
        logger.warning("graph ppr leg failed: %s", e)
        return []


def _ppr(conn, seed_ids, limit, max_nodes, per_entity_cap) -> list[int]:
    seeds = [int(s) for s in dict.fromkeys(seed_ids)
             if s is not None][:_MAX_SQL_VARS // 2]
    if not seeds:
        return []
    ent_ids = ent.entities_of(conn, seeds)
    if len(ent_ids) > _MAX_SQL_VARS:
        ent_ids = set(sorted(ent_ids)[:_MAX_SQL_VARS])  # deterministic bound
    edge_nbrs = _edge_neighbors(conn, seeds)
    if not ent_ids and not edge_nbrs:
        return []  # seeds are graph-isolated — nothing to propagate over

    candidates = set(seeds) | edge_nbrs
    candidates |= set(ent.memories_of(conn, ent_ids, limit=max_nodes * 2))
    nodes = set(seeds)
    for m in candidates:
        if len(nodes) >= max_nodes:
            break
        nodes.add(m)
    if len(nodes) <= 1:
        return []

    adj = _build_adjacency(conn, nodes, ent_ids, per_entity_cap, max_nodes)
    if not adj:
        return []
    scores = _pagerank(adj, seeds)
    seed_set = set(seeds)
    ranked = sorted(scores, key=lambda n: scores[n], reverse=True)
    return [n for n in ranked if n not in seed_set][:limit]


def _edge_neighbors(conn, seeds) -> set[int]:
    q = ",".join("?" * len(seeds))
    rows = conn.execute(
        f"SELECT src_id, dst_id FROM edges WHERE valid_to IS NULL"
        f" AND (src_id IN ({q}) OR dst_id IN ({q}))", [*seeds, *seeds]).fetchall()
    out: set[int] = set()
    for r in rows:
        out.add(r["src_id"])
        out.add(r["dst_id"])
    return out


def _build_adjacency(conn, nodes, ent_ids, per_entity_cap, max_nodes):
    node_set = set(nodes)
    adj: dict[int, dict[int, float]] = {n: {} for n in node_set}

    def _add(a: int, b: int, w: float) -> None:
        if a == b:
            return
        adj[a][b] = adj[a].get(b, 0.0) + w
        adj[b][a] = adj[b].get(a, 0.0) + w

    # co-mention edges: two memories sharing an entity are linked (weight += 1).
    if ent_ids:
        eq = ",".join("?" * len(ent_ids))
        rows = conn.execute(
            f"SELECT entity_id, memory_id FROM entity_mentions"
            f" WHERE entity_id IN ({eq}) LIMIT ?",
            [*ent_ids, max_nodes * 20]).fetchall()
        by_ent: dict[int, list[int]] = {}
        for r in rows:
            if r["memory_id"] in node_set:
                by_ent.setdefault(r["entity_id"], []).append(r["memory_id"])
        for mems in by_ent.values():
            mems = mems[:per_entity_cap]
            for i in range(len(mems)):
                for j in range(i + 1, len(mems)):
                    _add(mems[i], mems[j], 1.0)

    # explicit typed edges within the node set (weight = confidence).
    nq = ",".join("?" * len(node_set))
    node_list = list(node_set)
    erows = conn.execute(
        f"SELECT src_id, dst_id, confidence FROM edges WHERE valid_to IS NULL"
        f" AND src_id IN ({nq}) AND dst_id IN ({nq})",
        [*node_list, *node_list]).fetchall()
    for r in erows:
        if r["src_id"] in adj and r["dst_id"] in adj:
            _add(r["src_id"], r["dst_id"], float(r["confidence"] or 1.0))

    return {n: list(nbrs.items()) for n, nbrs in adj.items() if nbrs}


def _pagerank(adj, seeds) -> dict[int, float]:
    """Personalized PageRank via sparse power iteration; teleport mass returns
    to the seeds (or uniformly if none are in the graph)."""
    nodes = list(adj.keys())
    if not nodes:
        return {}
    seed_in = [s for s in seeds if s in adj]
    base = seed_in or nodes
    teleport = dict.fromkeys(nodes, 0.0)
    for n in base:
        teleport[n] = 1.0 / len(base)
    wsum = {n: (sum(w for _, w in adj[n]) or 1.0) for n in nodes}
    score = dict(teleport)
    for _ in range(_ITERS):
        nxt = {n: (1.0 - _DAMPING) * teleport[n] for n in nodes}
        for n in nodes:
            sc = score[n]
            if sc == 0.0:
                continue
            share = _DAMPING * sc / wsum[n]
            for m, w in adj[n]:
                nxt[m] += share * w
        score = nxt
    return score
