"""In-process recall cache for the brain-bg worker thread.

Tier logic adapted from mnemosyne-oss/mnemosyne
(``mnemosyne/core/query_cache.py``, MIT, (c) 2026 Abdias J); re-shaped onto
brain's ``meta.mem_generation`` counter.

The donor cache invalidated per-query via an ever-growing version map and
persisted to its own SQLite file. hermes-brain is simpler AND stronger: a
single global generation counter (``meta.mem_generation``, bumped by
``store.db.bump_generation`` on every memories write) is the one
cache-invalidation primitive. When it moves, the whole cache is dropped. No
persistence, no cross-process coherence problem — this cache lives entirely
on the single owned brain-bg worker thread that holds the long-lived
connection, so a plain in-process dict is correct.

Tiers, tried in order (first hit wins):
    1. exact     — the raw query string matches a cached entry verbatim
    2. normalized — casefold + whitespace-collapse of the query matches
    3. jaccard   — token-set Jaccard similarity >= 0.9 against a cached entry
    4. semantic  — cosine over cached query embeddings; ACTIVE ONLY when an
                   ``embedder`` is supplied. Without one it is skipped
                   silently, so the fts-only / stub tiers still work.

This sits on the recall (capture) path, so it NEVER raises: any internal
error degrades to a cache miss (``get`` -> ``None``, ``put`` -> no-op) and is
logged at warning level.
"""

from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

log = logging.getLogger("brain.recall.query_cache")

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+")

DEFAULT_MAX_SIZE = 256
JACCARD_THRESHOLD = 0.9
SEMANTIC_THRESHOLD = 0.93


def _normalize(query: str) -> str:
    """casefold + whitespace-collapse — the tier-2 (normalized) key."""
    return _WS.sub(" ", query.strip()).casefold()


def _tokens(norm_query: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(norm_query))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _kinds_key(kinds: Iterable[str] | None) -> tuple[str, ...]:
    if not kinds:
        return ()
    return tuple(sorted(str(k) for k in kinds))


class _Entry:
    __slots__ = ("raw", "norm", "tokens", "hits", "emb")

    def __init__(self, raw: str, norm: str, tokens: frozenset[str], hits: Any) -> None:
        self.raw = raw
        self.norm = norm
        self.tokens = tokens
        self.hits = hits
        self.emb: list[float] | None = None  # lazily filled by the semantic tier


class QueryCache:
    """Bounded, generation-invalidated, in-process recall cache.

    Keyed on ``(normalized_query, kinds_tuple, scope)``. Eviction is LRU
    (least-recently-used entry dropped first) over an ``OrderedDict``.
    """

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE) -> None:
        self.max_size = max(1, int(max_size))
        self._store: OrderedDict[tuple, _Entry] = OrderedDict()
        self._generation: str | None = None
        # lightweight stats (never load-bearing; safe to read for probes)
        self.hits = 0
        self.misses = 0
        self.exact_hits = 0
        self.normalized_hits = 0
        self.jaccard_hits = 0
        self.semantic_hits = 0

    # -- generation / invalidation ----------------------------------------

    def check_generation(self, conn: Any) -> bool:
        """Sync to ``meta.mem_generation``; clear the cache if it moved.

        Returns True if the cache was cleared (generation changed or could
        not be read). Callers invoke this before a lookup; ``get`` also
        calls it internally, so a caller that forgets is still correct.
        """
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key='mem_generation'"
            ).fetchone()
            current = row[0] if row is not None else None
        except Exception as exc:  # pragma: no cover - defensive
            # Can't confirm freshness -> never serve stale. Drop everything.
            log.warning("query_cache: generation read failed (%s); clearing", exc)
            self._store.clear()
            self._generation = None
            return True

        if self._generation is None:
            # First sync: adopt the baseline. Entries can only have been put
            # by this same worker thread after it observed the DB, so an unset
            # generation has not "moved" — priming must not drop them.
            self._generation = current
            return False

        if current != self._generation:
            self._store.clear()
            self._generation = current
            return True
        return False

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    # -- lookup -----------------------------------------------------------

    def get(
        self,
        conn: Any,
        query: str,
        kinds: Iterable[str] | None = None,
        scope: Any = None,
        *,
        embedder: Any | None = None,
    ) -> Any | None:
        """Return cached hits for ``query`` or ``None`` on a miss.

        Tries the four tiers in order. Never raises — any failure is a miss.
        """
        try:
            self.check_generation(conn)

            norm = _normalize(query)
            kk = _kinds_key(kinds)
            key = (norm, kk, scope)

            # Tier 1/2: exact + normalized both resolve to one dict lookup,
            # because entries are keyed on the normalized query. We
            # distinguish only for stats.
            entry = self._store.get(key)
            if entry is not None:
                self._store.move_to_end(key)
                self.hits += 1
                if entry.raw == query:
                    self.exact_hits += 1
                else:
                    self.normalized_hits += 1
                return entry.hits

            qtokens = _tokens(norm)

            # Tier 3: Jaccard token-set near-match (same kinds + scope only).
            best: _Entry | None = None
            best_key: tuple | None = None
            best_j = JACCARD_THRESHOLD
            for cand_key, e in self._store.items():
                _n, k, s = cand_key
                if k != kk or s != scope:
                    continue
                j = _jaccard(qtokens, e.tokens)
                if j >= best_j:
                    best_j = j
                    best = e
                    best_key = cand_key
                    if j == 1.0:
                        break
            if best is not None:
                self._store.move_to_end(best_key)  # a fuzzy hit refreshes recency too
                self.hits += 1
                self.jaccard_hits += 1
                return best.hits

            # Tier 4: semantic — ACTIVE ONLY when an embedder is supplied.
            if embedder is not None:
                hit = self._semantic_lookup(query, kk, scope, embedder)
                if hit is not None:
                    self.hits += 1
                    self.semantic_hits += 1
                    return hit

            self.misses += 1
            return None
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("query_cache.get degraded to miss: %s", exc)
            return None

    def _semantic_lookup(
        self, query: str, kk: tuple, scope: Any, embedder: Any
    ) -> Any | None:
        encode: Callable[[str], Sequence[float]] | None = getattr(
            embedder, "encode_query", None
        )
        if not callable(encode):
            return None
        qemb = encode(query)
        if not qemb:
            return None

        best: _Entry | None = None
        best_key: tuple | None = None
        best_c = SEMANTIC_THRESHOLD
        for cand_key, e in self._store.items():
            _n, k, s = cand_key
            if k != kk or s != scope:
                continue
            if e.emb is None:
                # Lazily embed the cached query once, memoized on the entry.
                e.emb = list(encode(e.raw) or [])
            c = _cosine(qemb, e.emb)
            if c >= best_c:
                best_c = c
                best = e
                best_key = cand_key
        if best is None:
            return None
        self._store.move_to_end(best_key)  # a semantic hit refreshes recency too
        return best.hits

    # -- insert -----------------------------------------------------------

    def put(
        self,
        query: str,
        kinds: Iterable[str] | None = None,
        scope: Any = None,
        hits: Any = None,
    ) -> None:
        """Store ``hits`` for ``query``. Never raises."""
        try:
            norm = _normalize(query)
            key = (norm, _kinds_key(kinds), scope)
            self._store[key] = _Entry(query, norm, _tokens(norm), hits)
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)  # drop least-recently-used
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("query_cache.put skipped: %s", exc)
