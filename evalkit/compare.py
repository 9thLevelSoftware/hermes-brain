"""Score a query set across retrieval leg configurations.

Reports P@k and MRR per configuration AND paired win/loss/tie against the FTS
baseline. The paired view is the one that matters at realistic sample sizes:
the first measurement of this stack produced a mean difference that read as a
3.7% regression and was, paired, two queries out of 71 with 87% of results
byte-identical.
"""

from __future__ import annotations

import logging
import sqlite3
import statistics
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

K = 5
SEARCH_LIMIT = 10


class Config(NamedTuple):
    name: str
    vec: bool
    rerank: bool
    graph: bool
    facts: bool


# Ordered so each row adds exactly one leg to the row above where possible —
# a table you can read down to see what each leg bought.
CONFIGURATIONS: tuple[Config, ...] = (
    Config("fts-only", False, False, False, False),
    Config("+vec", True, False, False, False),
    Config("+vec+rerank", True, True, False, False),
    Config("+vec+graph", True, False, True, False),
    Config("+vec+facts", True, False, False, True),
    Config("all", True, True, True, True),
)

BASELINE = "fts-only"


class Result(NamedTuple):
    name: str
    precision_at_k: float
    mrr: float
    n: int
    reciprocal_ranks: list[float]
    skipped: str | None = None


def _reciprocal_rank(hit_uids: list[str], gold: set[str]) -> float:
    for i, uid in enumerate(hit_uids, start=1):
        if uid in gold:
            return 1.0 / i
    return 0.0


def _vector_index_present(conn) -> bool:
    """An embedder alone is not a vector leg — there must also be an index.

    Without this a run with zero embedded rows reports `+vec` as having
    executed and scored identically to FTS, which reads as "vectors add
    nothing" when it means "vectors were never consulted". That is precisely
    the failure this whole module exists to stop making.
    """
    from ..store import db

    try:
        if db.get_meta(conn, "vec_dim") is None:
            return False
        row = conn.execute("SELECT count(*) FROM epi_vec").fetchone()
        if row and row[0]:
            return True
        row = conn.execute("SELECT count(*) FROM mem_vec").fetchone()
        return bool(row and row[0])
    except Exception:
        return False


def score_weights(conn, queries, weights: dict[str, float] | None, *,
                  embedder=None, reranker=None) -> Result:
    """Score ONE candidate weight set over the query set, without applying it.

    This is what makes approving a tune proposal an evidence-based decision
    rather than a leap: `dream/tune.py` fits weights and can now say what they
    would actually do, measured, before anyone commits them. ``weights=None``
    scores whatever is currently active.
    """
    cfg = _best_available(conn, embedder=embedder, reranker=reranker)
    return _run_one(conn, queries, cfg, embedder=embedder, reranker=reranker,
                    weights_override=weights)


def _best_available(conn, *, embedder, reranker) -> Config:
    """The richest configuration this install can actually run right now.

    A fixed `all` row is unreachable whenever any leg's model is missing, which
    reproduces the exact failure evalkit exists to prevent: you cannot see your
    real stack, so a leg that never ran is indistinguishable from a leg that
    did not help.
    """
    vec = embedder is not None and _vector_index_present(conn)
    return Config("best-available", vec, vec and reranker is not None, True, True)


def _run_one(conn, queries, cfg: Config, *, embedder, reranker,
             weights_override: dict[str, float] | None = None) -> Result:
    from ..recall.search import search

    if cfg.vec and embedder is None:
        return Result(cfg.name, 0.0, 0.0, 0, [], skipped="no embedder for this tier")
    if cfg.vec and not _vector_index_present(conn):
        return Result(cfg.name, 0.0, 0.0, 0, [],
                      skipped="no vector index (run 'hermes brain reindex')")
    if cfg.rerank and reranker is None:
        return Result(cfg.name, 0.0, 0.0, 0, [],
                      skipped="rerank model absent (hermes brain models --download)")

    precisions: list[float] = []
    rrs: list[float] = []
    for item in queries:
        gold = set(item.get("gold") or [])
        query = str(item.get("query") or "").strip()
        if not query or not gold:
            continue
        try:
            hits = search(
                conn, query,
                limit=SEARCH_LIMIT,
                include_episodes=True,
                episode_limit=SEARCH_LIMIT,
                trust_tier="owner",
                embedder=embedder if cfg.vec else None,
                reranker=reranker if cfg.rerank else None,
                graph=cfg.graph,
                facts=cfg.facts,
                weights_override=weights_override,
            )
        except Exception as e:  # search() is a capture path; be equally safe
            logger.warning("eval: search failed for %r: %s", query[:40], e)
            continue
        uids = [h.uid for h in hits]
        precisions.append(len(set(uids[:K]) & gold) / min(K, len(gold)))
        rrs.append(_reciprocal_rank(uids, gold))

    if not rrs:
        return Result(cfg.name, 0.0, 0.0, 0, [], skipped="no scorable queries")
    return Result(cfg.name, statistics.fmean(precisions), statistics.fmean(rrs),
                  len(rrs), rrs)


def _paired(baseline: Result, other: Result) -> dict[str, int]:
    """Win/loss/tie on reciprocal rank, query by query."""
    win = loss = tie = 0
    for a, b in zip(baseline.reciprocal_ranks, other.reciprocal_ranks, strict=False):
        if b > a:
            win += 1
        elif b < a:
            loss += 1
        else:
            tie += 1
    return {"win": win, "loss": loss, "tie": tie}


def run_comparison(
    conn: sqlite3.Connection,
    queries: list[dict[str, Any]],
    *,
    embedder=None,
    reranker=None,
    weights_override: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score every configuration over the same queries.

    Returns ``{"k": K, "results": [...], "paired": {name: {win,loss,tie}}}``.
    A configuration whose leg is unavailable is reported as skipped rather than
    silently scoring as the baseline — an absent reranker producing identical
    numbers is exactly how the first run reported a stage that never executed.
    """
    # `best-available` is computed from the legs actually present, so the real
    # stack is always visible. The fixed `all` row below stays, and stays
    # SKIPPED when something is missing — the gap must still be reported, not
    # hidden behind a row that happens to run.
    configs = [*CONFIGURATIONS, _best_available(conn, embedder=embedder,
                                               reranker=reranker)]
    results = [_run_one(conn, queries, cfg, embedder=embedder, reranker=reranker,
                        weights_override=weights_override)
               for cfg in configs]
    by_name = {r.name: r for r in results}
    base = by_name.get(BASELINE)
    paired: dict[str, dict[str, int]] = {}
    if base is not None and base.reciprocal_ranks:
        for r in results:
            if r.name != BASELINE and r.reciprocal_ranks:
                paired[r.name] = _paired(base, r)
    from .store import queryset_fingerprint

    return {
        "k": K,
        "queryset": queryset_fingerprint(queries),
        "results": [
            {"name": r.name, "p_at_k": r.precision_at_k, "mrr": r.mrr,
             "n": r.n, "skipped": r.skipped}
            for r in results
        ],
        "paired": paired,
    }


def format_report(report: dict[str, Any], baseline: dict[str, Any] | None = None) -> str:
    """Human-readable table. Paired counts sit next to the means on purpose:
    reading a mean without them is how noise gets mistaken for a finding."""
    k = report.get("k", K)
    prior = {}
    mismatched = False
    if baseline:
        # Only comparable when both runs scored the SAME queries.
        same = (baseline.get("queryset") or "") == (report.get("queryset") or "")
        if same:
            prior = {r["name"]: r for r in baseline.get("results", [])
                     if not r.get("skipped")}
        else:
            mismatched = True
    delta_head = "   ΔMRR vs saved" if prior else ""
    lines = [
        f"{'configuration':<16} {'P@' + str(k):>7} {'MRR':>7} {'n':>5}   "
        f"vs {BASELINE} (win/loss/tie){delta_head}",
        "-" * (74 + len(delta_head)),
    ]
    for row in report.get("results", []):
        if row.get("skipped"):
            lines.append(f"{row['name']:<16} {'—':>7} {'—':>7} {'—':>5}   "
                         f"SKIPPED: {row['skipped']}")
            continue
        pair = report.get("paired", {}).get(row["name"])
        cell = ""
        if pair:
            cell = f"{pair['win']} / {pair['loss']} / {pair['tie']}"
        delta = ""
        was = prior.get(row["name"])
        if was:
            delta = f"   {row['mrr'] - was['mrr']:+.4f}"
        lines.append(f"{row['name']:<16} {row['p_at_k']:>7.3f} {row['mrr']:>7.3f} "
                     f"{row['n']:>5}   {cell:<26}{delta}")
    if mismatched:
        lines.append("")
        lines.append("NOTE: the saved baseline was measured on a DIFFERENT query "
                     "set, so no delta is shown. Re-save one with "
                     "'--compare --save-baseline'.")
    n = next((r["n"] for r in report.get("results", []) if not r.get("skipped")), 0)
    if 0 < n < 100:
        lines.append("")
        lines.append(f"NOTE: n={n} is small. Read the paired counts, not the means — "
                     f"a few queries can move a mean by several percent.")
    return "\n".join(lines)
