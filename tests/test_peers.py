"""D3 group-chat theory-of-mind peer modeling (dream/peers.py).

Hermetic: fake LLM via brain.llm.set_llm_for_tests, temp HERMES_HOME. Group
episodes are seeded through the REAL capture path (capture_turn + TurnContext)
so the group-chat signal (>= 2 distinct authors in a source_channel) and the
per-principal scoping are exercised exactly as production writes them.

Covers: capture_peers off => no card; owner is never modeled as a peer; a
non-owner peer in a group chat => exactly one peer_card scoped to them; a 1:1
DM is not modeled; a second run supersedes (still one current card); dry_run
writes nothing; and a non-owner caller (including the very peer the card is
about) can never retrieve another peer's card.
"""

from __future__ import annotations

import json

import pytest
from brain import llm as brain_llm
from brain.capture.turns import TurnContext, capture_turn
from brain.dream import lease
from brain.dream import peers as peers_mod
from brain.dream.shift import DEFAULT_MODES, PIPELINE, Shift
from brain.store import db
from conftest import seed_memory

_GROUP = "grp-1"
_SESSION = "s-grp"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_shift(conn, mode="active", config=None):
    lease.acquire(conn, "dream", "test-holder")
    cfg = {"_forced_mode": mode}
    if config:
        cfg.update(config)
    return Shift(shift_id=db.new_ulid(), conn=conn, config=cfg,
                 started_at=db.iso_now(),
                 activity_baseline="9999-12-31T00:00:00.000Z", holder="test-holder")


def _turn(conn, *, author, principal, tier, user, asst, channel=_GROUP,
          session=_SESSION, turn_no=1, platform="telegram"):
    return capture_turn(
        conn,
        TurnContext(session_id=session, turn_no=turn_no, platform=platform,
                    source_channel=channel, source_author=author,
                    principal_id=principal, trust_tier=tier),
        user, asst)


def _seed_group(conn):
    """A real group chat: the owner AND a non-owner peer both speak in one
    channel (2 distinct authors => group)."""
    _turn(conn, author="tg:owner", principal="owner", tier="owner", turn_no=1,
          user="hey team, what's the plan for the release?",
          asst="let's sync on the timeline")
    _turn(conn, author="tg:bob", principal="peer-bob", tier="known_user", turn_no=2,
          user="I want to ship friday and I really dislike long status meetings",
          asst="noted — friday it is")


def _fake(profile, headline="Bob: ship-fast, terse", usable=True):
    payload = json.dumps({"profile": profile, "headline": headline, "usable": usable})
    return lambda p, *, system=None, max_tokens=0: payload


def _sequence_fake(payloads):
    it = iter(payloads)
    last = {"v": payloads[-1]}

    def fn(p, *, system=None, max_tokens=0):
        try:
            last["v"] = next(it)
        except StopIteration:
            pass
        return last["v"]

    return fn


def _current_cards(conn):
    return conn.execute(
        "SELECT * FROM memories WHERE kind='peer_card' AND valid_to IS NULL"
    ).fetchall()


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def test_peers_registered_active_in_pipeline():
    assert "peers" in PIPELINE
    assert DEFAULT_MODES["peers"] == "active"


# ---------------------------------------------------------------------------
# write / scope / owner-exclusion
# ---------------------------------------------------------------------------

def test_peer_card_written_and_owner_never_modeled(conn):
    _seed_group(conn)
    brain_llm.set_llm_for_tests(_fake("Bob ships fast and dislikes long meetings."))
    try:
        res = peers_mod.run(make_shift(conn, "active"))
    finally:
        brain_llm.set_llm_for_tests(None)

    assert "error" not in res, res
    assert res["written"] == 1
    cards = _current_cards(conn)
    assert len(cards) == 1
    c = cards[0]
    assert c["scope_user"] == "peer-bob"          # scoped to the OBSERVED peer
    assert c["kind"] == "peer_card"
    assert c["memory_type"] == "semantic"          # 'profile' is a kind, not a type
    assert c["epistemic"] == "inference"
    assert c["trust_tier"] == "known_user"         # the observed peer's tier
    assert c["instruction_shaped"] == 0
    # The owner is never a peer.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE kind='peer_card'"
        " AND scope_user='owner'").fetchone()["n"] == 0


def test_capture_peers_off_writes_nothing(conn):
    _seed_group(conn)
    brain_llm.set_llm_for_tests(_fake("Bob is terse."))
    try:
        res = peers_mod.run(make_shift(conn, "active", {"capture_peers": False}))
    finally:
        brain_llm.set_llm_for_tests(None)
    assert res.get("skipped") == "capture_peers_off"
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE kind='peer_card'").fetchone()["n"] == 0


def test_one_on_one_dm_peer_not_modeled(conn):
    # Only bob authors this channel => a DM, not a group => no card.
    _turn(conn, author="tg:bob", principal="peer-bob", tier="known_user",
          user="hey can you help me with something", asst="sure",
          channel="dm-bob", session="s-dm")
    brain_llm.set_llm_for_tests(_fake("Bob"))
    try:
        res = peers_mod.run(make_shift(conn, "active"))
    finally:
        brain_llm.set_llm_for_tests(None)
    assert res == {"peers": 0}
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE kind='peer_card'").fetchone()["n"] == 0


def test_dry_run_writes_no_card_but_audits(conn):
    _seed_group(conn)
    brain_llm.set_llm_for_tests(_fake("Bob is terse."))
    try:
        res = peers_mod.run(make_shift(conn, "dry_run"))
    finally:
        brain_llm.set_llm_for_tests(None)
    assert res.get("would_write", 0) >= 1
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE kind='peer_card'").fetchone()["n"] == 0
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM audit_log WHERE action='would_write_peer_card'"
    ).fetchone()["n"] >= 1


# ---------------------------------------------------------------------------
# supersede (versions-are-rows): still exactly one current card
# ---------------------------------------------------------------------------

def test_second_run_supersedes_keeping_one_current_card(conn):
    _seed_group(conn)
    fake = _sequence_fake([
        json.dumps({"profile": "Bob ships fast and dislikes long meetings.",
                    "headline": "Bob v1", "usable": True}),
        json.dumps({"profile": "Bob has warmed to careful sprint planning.",
                    "headline": "Bob v2", "usable": True}),
    ])
    brain_llm.set_llm_for_tests(fake)
    try:
        peers_mod.run(make_shift(conn, "active"))
        v1 = _current_cards(conn)
        assert len(v1) == 1 and v1[0]["version"] == 1
        # New group activity for the peer => the card is rebuilt.
        _turn(conn, author="tg:bob", principal="peer-bob", tier="known_user",
              turn_no=3, user="actually let's plan the sprint carefully",
              asst="great, planning it is")
        res2 = peers_mod.run(make_shift(conn, "active"))
    finally:
        brain_llm.set_llm_for_tests(None)

    assert res2["updated"] == 1
    current = _current_cards(conn)
    assert len(current) == 1                         # STILL one current card
    assert current[0]["version"] == 2
    assert current[0]["scope_user"] == "peer-bob"
    # The old version is closed and chained to the new one.
    old = conn.execute(
        "SELECT * FROM memories WHERE kind='peer_card' AND valid_to IS NOT NULL"
    ).fetchall()
    assert len(old) == 1
    assert old[0]["superseded_by"] == current[0]["id"]
    assert old[0]["version"] == 1


def test_rerun_without_new_activity_is_idempotent(conn):
    _seed_group(conn)
    brain_llm.set_llm_for_tests(_fake("Bob ships fast and dislikes long meetings."))
    try:
        peers_mod.run(make_shift(conn, "active"))
        res2 = peers_mod.run(make_shift(conn, "active"))   # no new episodes
    finally:
        brain_llm.set_llm_for_tests(None)
    assert res2["unchanged"] == 1
    assert res2["written"] == 0 and res2["updated"] == 0
    assert len(_current_cards(conn)) == 1


# ---------------------------------------------------------------------------
# retrieval scoping: a non-owner can NEVER retrieve a peer card
# ---------------------------------------------------------------------------

@pytest.fixture
def vec_conn(tmp_home):
    pytest.importorskip("sqlite_vec")
    from brain.recall.embed import StubEmbedder
    from brain.store import vec as vec_store

    connection = db.connect(tmp_home)
    emb = StubEmbedder()
    if not vec_store.ensure_tables(connection, emb.dim, emb.name):
        pytest.skip("sqlite-vec not loadable")
    yield connection, emb
    connection.close()


def _seed_card(conn, emb, principal, text):
    from brain.store import vec as vec_store

    mid = seed_memory(conn, text, kind="peer_card", memory_type="semantic",
                      epistemic="inference", created_by="distillation")
    conn.execute(
        "UPDATE memories SET summary=?, scope_user=?, trust_tier='known_user'"
        " WHERE id=?", (text, principal, mid))
    vec_store.upsert(conn, "mem_vec", mid, emb.encode_documents([text])[0])
    conn.commit()
    return mid


def test_peer_card_retrieval_owner_only(vec_conn):
    from brain.recall.strategies import retrieve_guidance

    conn, emb = vec_conn
    text = "Bob is terse, dislikes long meetings, and prefers async updates"
    mid = _seed_card(conn, emb, "peer-bob", text)
    query = "what does Bob think about meetings and async updates"

    # The OWNER (observer) sees the peer card.
    owner = retrieve_guidance(conn, query, embedder=emb,
                              trust_tier="owner", scope_user="owner")
    assert any(g.id == mid and g.kind == "peer_card" for g in owner)

    # The very peer the card is ABOUT must NOT retrieve it (the leak test:
    # scope_user would match, but peer cards are owner-only).
    as_bob = retrieve_guidance(conn, query, embedder=emb,
                               trust_tier="known_user", scope_user="peer-bob")
    assert all(g.id != mid for g in as_bob)

    # Another peer must not retrieve it either.
    as_carol = retrieve_guidance(conn, query, embedder=emb,
                                 trust_tier="known_user", scope_user="peer-carol")
    assert all(g.id != mid for g in as_carol)


def test_owner_never_sees_another_persons_card_leak_across_peers(vec_conn):
    from brain.recall.strategies import retrieve_guidance

    conn, emb = vec_conn
    bob = _seed_card(conn, emb, "peer-bob",
                     "Bob is a blunt backend engineer who hates meetings")
    carol = _seed_card(conn, emb, "peer-carol",
                       "Carol is a meticulous designer who loves detailed specs")
    # A non-owner (Bob) never retrieves Carol's card, even on a Carol-shaped query.
    as_bob = retrieve_guidance(conn, "Carol detailed design specs", embedder=emb,
                               trust_tier="known_user", scope_user="peer-bob")
    assert all(g.id not in (bob, carol) for g in as_bob)
