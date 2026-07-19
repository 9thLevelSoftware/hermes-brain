"""Working-representation blend (recall/blend.py): semantic + reinforced +
recent legs fused with RRF, re-fetched through the centralized scoping helper.

Tier-independent by design: legs 2/3 are plain SQL, so the reinforced+recent
fusion works with embedder=None (fts-only/stub floor). The semantic leg is
exercised via the stub embedder + sqlite-vec where available.
"""

from __future__ import annotations

import pytest
from brain.recall.blend import blend  # noqa: E402
from conftest import iso_days_ago, seed_memory  # noqa: E402


def _bump(conn, mem_id, *, verification=1, helpful=0):
    conn.execute(
        "UPDATE memories SET verification_count=?, helpful_count=? WHERE id=?",
        (verification, helpful, mem_id))
    conn.commit()


def test_blend_never_raises_on_empty_db(conn):
    assert blend(conn, "anything at all") == []


def test_reinforced_and_recent_legs_work_without_embedder(conn):
    """The whole point of tier degradation: no embedder, blend still returns
    the reinforced + recent fusion (legs 2 and 3 are plain SQL)."""
    reinforced = seed_memory(conn, "the deploy pipeline uses github actions",
                             valid_from=iso_days_ago(90))
    _bump(conn, reinforced, verification=10, helpful=8)  # heavily reinforced
    recent = seed_memory(conn, "we switched the cache to redis yesterday",
                         valid_from=iso_days_ago(1))     # brand new
    stale = seed_memory(conn, "an old note nobody reinforced",
                        valid_from=iso_days_ago(365))

    hits = blend(conn, "unrelated query text", embedder=None, limit=8)
    ids = [h.id for h in hits]
    # Both the reinforced and the recent leg surface even with no semantic
    # signal at all; the untouched stale row is not favored by either leg.
    assert reinforced in ids
    assert recent in ids
    assert all(h.source == "blend" for h in hits)
    # The stale row loses to both loaded legs (it is neither recent nor
    # reinforced), so it ranks below them.
    if stale in ids:
        assert ids.index(stale) > ids.index(reinforced)
        assert ids.index(stale) > ids.index(recent)


def test_recent_leg_honors_the_window(conn):
    from brain.recall.blend import _recent_ids

    fresh = seed_memory(conn, "fresh fact within the window",
                        valid_from=iso_days_ago(2))
    old = seed_memory(conn, "ancient fact outside the window",
                      valid_from=iso_days_ago(400))
    # The recent leg itself only nominates rows inside the window.
    recent = _recent_ids(conn, 8, 14, (), None, None, "owner")
    assert fresh in recent
    assert old not in recent

    # In the full blend, the reinforced leg still nominates every current-truth
    # row (both have the default verification_count), but the fresh row wins
    # the recency leg too, so it ranks strictly above the old one.
    ids = [h.id for h in blend(conn, "no keyword overlap", embedder=None,
                               recent_days=14)]
    assert ids.index(fresh) < ids.index(old)


def test_three_legs_fuse_into_blended_order(conn):
    pytest.importorskip("sqlite_vec")
    from brain.recall.embed import StubEmbedder
    from brain.store import vec as vec_store

    embedder = StubEmbedder()
    assert vec_store.ensure_tables(conn, embedder.dim)

    # A row that is semantically on-topic AND recent AND reinforced should
    # win all three legs and top the blend.
    winner = seed_memory(conn, "postgres connection pool size is 20",
                         valid_from=iso_days_ago(1))
    _bump(conn, winner, verification=9, helpful=5)
    # An on-topic-but-nothing-else row and pure noise.
    other = seed_memory(conn, "postgres vacuum settings need tuning",
                        valid_from=iso_days_ago(200))
    noise = seed_memory(conn, "the office coffee machine is broken again",
                        valid_from=iso_days_ago(200))
    for mid, text in ((winner, "postgres connection pool size is 20"),
                      (other, "postgres vacuum settings need tuning"),
                      (noise, "the office coffee machine is broken again")):
        vec_store.upsert(conn, "mem_vec", mid, embedder.encode_documents([text])[0])
    conn.commit()

    hits = blend(conn, "postgres pool size", embedder=embedder, limit=8)
    ids = [h.id for h in hits]
    assert ids and ids[0] == winner
    assert other in ids  # semantic leg still surfaces the related row


def test_scope_enforcement_hides_foreign_principal_rows(conn):
    """A non-owner caller must never see another principal's scoped rows via
    the blend — the re-fetch through _memories_by_ids re-applies scoping, and
    legs 2/3 pre-filter too (finding #17)."""
    secret = seed_memory(conn, "owner secret: totp backup codes in the safe",
                         valid_from=iso_days_ago(1))
    _bump(conn, secret, verification=20, helpful=15)  # maximally reinforced
    conn.execute("UPDATE memories SET scope_user='owner' WHERE id=?", (secret,))
    conn.commit()

    # A foreign, non-owner caller: even though the row is the single most
    # reinforced AND most recent, scoping keeps it out of the blend entirely.
    peer = blend(conn, "totp backup codes safe", embedder=None,
                 trust_tier="known_user", principal_id="peer-9")
    assert all(h.id != secret for h in peer)

    # The owner does see it (both by keyword-free reinforced/recent legs).
    owner = blend(conn, "totp backup codes safe", embedder=None,
                  trust_tier="owner")
    assert any(h.id == secret for h in owner)


def test_peer_card_never_leaks_to_non_owner(conn):
    """peer_card is excluded both by the default exclude_kinds and by the
    non-owner scoping rule — belt and braces."""
    card = seed_memory(conn, "peer profile: prefers terse answers",
                       kind="peer_card", valid_from=iso_days_ago(1))
    _bump(conn, card, verification=20, helpful=20)
    conn.execute("UPDATE memories SET scope_user='peer-9' WHERE id=?", (card,))
    conn.commit()

    peer = blend(conn, "peer profile answers", embedder=None,
                 trust_tier="known_user", principal_id="peer-9")
    assert all(h.id != card for h in peer)
    # Even the owner never gets a peer_card through the generic blend
    # (default exclude_kinds drops it, same as the facts path).
    owner = blend(conn, "peer profile answers", embedder=None)
    assert all(h.id != card for h in owner)
