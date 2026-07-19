"""In-process recall cache: tier logic + generation invalidation.

Uses the ``conn`` fixture (tests/conftest.py), which initializes ``meta``
with ``mem_generation='0'`` the way the real store does.
"""

from __future__ import annotations

from brain.recall.query_cache import QueryCache  # noqa: E402
from brain.store import db  # noqa: E402


class FakeEmbedder:
    """Deterministic 2-d embedder: 'network-ish' text -> [1,0], else [0,1].

    Lets two token-disjoint phrasings collide in embedding space so the
    semantic tier can be exercised without any model.
    """

    _TOPIC = ("wifi", "password", "network", "credentials")

    def encode_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if any(w in text.lower() for w in self._TOPIC) else [0.0, 1.0]


def test_exact_hit_and_miss(conn):
    cache = QueryCache()
    cache.put("foo bar", hits=["result"])

    assert cache.get(conn, "foo bar") == ["result"]
    assert cache.exact_hits == 1
    assert cache.get(conn, "totally unrelated") is None


def test_bump_generation_invalidates(conn):
    cache = QueryCache()
    cache.put("cache me", hits=[1, 2, 3])
    assert cache.get(conn, "cache me") == [1, 2, 3]

    db.bump_generation(conn)
    conn.commit()

    assert cache.get(conn, "cache me") is None
    assert len(cache) == 0


def test_normalized_match_case_and_whitespace(conn):
    cache = QueryCache()
    cache.put("Hello   World", hits=["hw"])

    # different case + collapsed whitespace, but not byte-identical -> tier 2
    assert cache.get(conn, "hello world") == ["hw"]
    assert cache.normalized_hits == 1
    assert cache.exact_hits == 0


def test_jaccard_near_match(conn):
    cache = QueryCache()
    cache.put("one two three four five six seven eight nine ten", hits=["j"])

    # drop one token, reorder the rest: inter=9, union=10 -> jaccard 0.9
    hit = cache.get(conn, "ten nine eight seven six five four three two")
    assert hit == ["j"]
    assert cache.jaccard_hits == 1


def test_no_embedder_semantic_tier_absent(conn):
    cache = QueryCache()
    cache.put("what is the wifi password", hits=["secret"])

    # token-disjoint query; without an embedder the semantic tier is skipped
    # and this is a clean miss (no crash).
    assert cache.get(conn, "network credentials") is None
    assert cache.semantic_hits == 0


def test_semantic_tier_hits_with_embedder(conn):
    cache = QueryCache()
    cache.put("what is the wifi password", hits=["secret"])

    hit = cache.get(conn, "network credentials", embedder=FakeEmbedder())
    assert hit == ["secret"]
    assert cache.semantic_hits == 1


def test_bounded_lru_eviction(conn):
    cache = QueryCache(max_size=2)
    cache.put("query A", hits=["a"])
    cache.put("query B", hits=["b"])
    cache.put("query C", hits=["c"])  # evicts least-recently-used (A)

    assert len(cache) == 2
    assert cache.get(conn, "query A") is None
    assert cache.get(conn, "query B") == ["b"]
    assert cache.get(conn, "query C") == ["c"]


def test_scope_and_kinds_partition_the_keyspace(conn):
    cache = QueryCache()
    cache.put("same words", kinds=["fact"], scope="owner", hits=["owned"])

    # same query, different scope -> distinct key, miss
    assert cache.get(conn, "same words", kinds=["fact"], scope="peer") is None
    # kinds order does not matter (normalized to a sorted tuple)
    cache.put("k words", kinds=["b", "a"], hits=["kk"])
    assert cache.get(conn, "k words", kinds=["a", "b"]) == ["kk"]
