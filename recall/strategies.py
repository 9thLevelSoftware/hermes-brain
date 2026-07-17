"""Learned-guidance retrieval for lane 2 (learning-system.md §1.2d step 4).

Strategy/guardrail items (procedural memories) and Memento cases (episodic
kind='case') are injected into the per-turn ephemeral fence — NOT via the
keyword search that serves recalled facts, but by semantic similarity to the
task, ranked by proven usefulness:

    rank = similarity * (0.5 + 0.5 * wilson_lower_bound(helpful, helpful+harmful))

A brand-new item (no outcome evidence yet) still surfaces at half weight so
it gets a chance to prove itself and feed `dream/mine_state.py`; an item that
has proven harmful (harmful > helpful, n >= 5) is dropped — the same
auto-deprecation the distiller applies, enforced again at read time.

Cases are only injected for TASK-LIKE queries (an imperative verb + object),
because "similar past task" is noise on a chit-chat turn. Failed cases rank
ABOVE successful ones at equal similarity — a past failure is the louder
warning.

Never raises: this rides the prefetch capture path and degrades to []
(no vec / no embedder / any error) exactly like recall.search.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import struct
from dataclasses import dataclass

from ..dream.stats import wilson_lower_bound
from ..store import vec as vec_store

logger = logging.getLogger(__name__)

_STRATEGY_KINDS = ("strategy", "guardrail")
_MIN_SIM = 0.35              # below this, "similar" is a lie
_CASE_MIN_SIM = 0.45
_DEPRECATE_MIN_N = 5         # harmful>helpful only deprecates with evidence
_KNN_K = 16


@dataclass
class Guidance:
    uid: str
    id: int
    kind: str                   # strategy | guardrail | case
    title: str
    verdict: str | None         # cases: success|failure; items: None
    score: float


# Imperative verbs that mark a "do a task" query (case-bank trigger). Kept
# small and boring on purpose — a 20-line heuristic, never an LLM (§1.2e).
_TASK_VERBS = frozenset(["add", "build", "change", "check", "clean", "configure", "connect", "create", "debug", "delete", "deploy", "diagnose", "edit", "enable", "disable", "fix", "generate", "implement", "import", "install", "integrate", "make", "migrate", "move", "optimize", "patch", "publish", "refactor", "release", "remove", "rename", "replace", "resolve", "run", "set", "setup", "ship", "test", "update", "upgrade", "write"])
_WORD_RE = re.compile(r"[a-z]+")


def is_task_like(query: str) -> bool:
    words = _WORD_RE.findall((query or "").casefold())
    if not words:
        return False
    # An imperative verb in the first few words, with something after it.
    return any(w in _TASK_VERBS and i < len(words) - 1
               for i, w in enumerate(words[:4]))


def retrieve_guidance(
    conn: sqlite3.Connection,
    query: str,
    *,
    embedder=None,
    limit_strategies: int = 3,
    limit_cases: int = 2,
    scope_user: str | None = None,
    trust_tier: str = "owner",
) -> list[Guidance]:
    """Top learned-guidance items for a query. [] when vectors are absent."""
    try:
        if embedder is None or not vec_store.vec_available(conn):
            return []
        qvec = embedder.encode_query(query)
        neighbors = vec_store.knn(conn, "mem_vec", qvec, _KNN_K)
        if not neighbors:
            return []
        # knn distance is L2 over int8 vectors at 127-scale; recompute an
        # honest cosine from the stored blobs (same derivation as
        # consolidate/distill) rather than back out a scale-dependent formula.
        q_int8 = _quantize(qvec)
        rows = _fetch(conn, [mid for mid, _ in neighbors], scope_user, trust_tier)

        out: list[Guidance] = []
        for row in rows:
            sim = _blob_cosine(q_int8, row["emb"]) if row["emb"] else 0.0
            kind = row["kind"]
            if kind in _STRATEGY_KINDS:
                g = _score_strategy(row, sim)
            elif kind == "case":
                g = _score_case(row, sim)
            else:
                g = None
            if g is not None:
                out.append(g)

        strategies = sorted((g for g in out if g.kind in _STRATEGY_KINDS),
                            key=lambda g: g.score, reverse=True)[:limit_strategies]
        cases = []
        if is_task_like(query):
            cases = sorted((g for g in out if g.kind == "case"),
                           key=lambda g: g.score, reverse=True)[:limit_cases]
        return strategies + cases
    except Exception as e:
        logger.warning("guidance retrieval failed for %r: %s", query, e)
        return []


def _fetch(conn, ids, scope_user, trust_tier) -> list[sqlite3.Row]:
    if not ids:
        return []
    sql = (
        f"SELECT m.id, m.uid, m.kind, m.summary, m.content, m.helpful_count,"
        f" m.harmful_count, m.importance, m.meta, v.emb"
        f" FROM memories m LEFT JOIN mem_vec v ON v.id = m.id"
        f" WHERE m.id IN ({','.join('?' * len(ids))})"
        " AND m.valid_to IS NULL AND m.status='active' AND m.live=1"
        " AND m.kind IN ('strategy','guardrail','case')"
    )
    params: list = list(ids)
    # Guidance is agent-owned, but honor scoping symmetrically with search:
    # a non-owner session never sees another principal's scoped items.
    if trust_tier != "owner":
        sql += " AND (m.scope_user IS NULL OR m.scope_user = ?)"
        params.append(scope_user or "")
    return conn.execute(sql, params).fetchall()


def _score_strategy(row: sqlite3.Row, sim: float) -> Guidance | None:
    if sim < _MIN_SIM:
        return None
    helpful, harmful = row["helpful_count"], row["harmful_count"]
    n = helpful + harmful
    if n >= _DEPRECATE_MIN_N and harmful > helpful:
        return None  # proven net-harmful — drop (read-time deprecation)
    wilson = wilson_lower_bound(helpful, n)
    score = sim * (0.5 + 0.5 * wilson)
    return Guidance(uid=row["uid"], id=row["id"], kind=row["kind"],
                    title=(row["summary"] or row["content"] or "").strip(),
                    verdict=None, score=score)


def _score_case(row: sqlite3.Row, sim: float) -> Guidance | None:
    if sim < _CASE_MIN_SIM:
        return None
    verdict = None
    try:
        verdict = (json.loads(row["meta"] or "{}") or {}).get("verdict")
    except (json.JSONDecodeError, TypeError):
        pass
    # Failed cases are the more useful warning: nudge them up at equal sim.
    bonus = 1.15 if verdict == "failure" else 1.0
    return Guidance(uid=row["uid"], id=row["id"], kind="case",
                    title=(row["summary"] or row["content"] or "").strip(),
                    verdict=verdict, score=sim * bonus)


def _quantize(vec) -> bytes:
    return struct.pack(f"{len(vec)}b",
                       *[max(-127, min(127, int(round(x * 127)))) for x in vec])


def _blob_cosine(a: bytes, b: bytes) -> float:
    va = struct.unpack(f"{len(a)}b", a)
    vb = struct.unpack(f"{len(b)}b", b)
    n = min(len(va), len(vb))
    dot = sum(va[i] * vb[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in va)) or 1.0
    nb = math.sqrt(sum(x * x for x in vb)) or 1.0
    return dot / (na * nb)
