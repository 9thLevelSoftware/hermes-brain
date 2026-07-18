"""Adversarial Phase 7 — anti-spam / quality gates.

Every test here asserts a load-bearing QUALITY-GATE invariant HOLDS under an
adversarial or low-quality LLM: the brain must REFUSE to ratchet vague, spammy,
malformed, or merely-lucky learning into memory, and only admit well-formed,
statistically-supported items. We drive the REAL gate functions with crafted
inputs (more robust than end-to-end pipeline drives) — the deterministic guards
in ``capture/extract.py``, ``dream/consolidate._validate`` +
``_entity_is_concrete``, ``skillforge/forge._validate`` (+ the Wilson helper),
``skillforge/revise._harm_verdict``, and ``recall/strategies._score_strategy``.

Vectors: sqlite-vec + a tiny deterministic in-test embedder (``VecEmbedder``)
give full control of cosine similarity, so a "paraphrase at 0.96" or an
"incoherent 3/3 cluster" is exact, not hoped-for.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager

import pytest
from conftest import seed_memory  # noqa: F401  (kept for parity/future use)

# ---------------------------------------------------------------------------
# in-test helpers
# ---------------------------------------------------------------------------

class VecEmbedder:
    """Deterministic embedder with FULL control of the returned vector.

    Interface matches recall/embed.py's embedders (``.name``, ``.dim``,
    ``.encode_documents([...]) -> [vec]``, ``.encode_query(text) -> vec``).
    Exact strings in ``registry`` return their mapped 256-d unit vector; any
    other string gets a stable hash-based fallback (never the binding case in
    these single/low-cardinality tests).
    """

    name = "fake-vec:256"
    dim = 256

    def __init__(self, registry: dict[str, list[float]] | None = None) -> None:
        self.registry = registry or {}

    def _vec(self, text: str) -> list[float]:
        if text in self.registry:
            return self.registry[text]
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        v = [(b / 127.5) - 1.0 for b in h][: self.dim]
        v += [0.0] * (self.dim - len(v))
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def encode_documents(self, texts):
        return [self._vec(t) for t in texts]

    def encode_query(self, text):
        return self._vec(text)


def _unit(pairs) -> list[float]:
    """A 256-d unit vector with the given (index, value) components set."""
    v = [0.0] * 256
    for i, val in pairs:
        v[i] = val
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _insert(conn, content, **cols) -> int:
    """Insert one memories row with sensible current-truth defaults; ``cols``
    overrides any column. Returns the rowid."""
    from brain.store import db

    base = {
        "uid": db.new_ulid(), "epistemic": "observation", "memory_type": "semantic",
        "kind": "fact", "status": "active", "live": 1, "content": content,
        "content_hash": db.content_hash(content), "symbols": "", "tags": "[]",
        "token_len": db.approx_tokens(content), "trust_tier": "owner",
        "created_by": "test", "valid_from": db.iso_now(), "recorded_at": db.iso_now(),
    }
    base.update(cols)
    keys = list(base.keys())
    cur = conn.execute(
        f"INSERT INTO memories ({','.join(keys)}) VALUES ({','.join('?' * len(keys))})",
        [base[k] for k in keys],
    )
    conn.commit()
    return cur.lastrowid


def _ensure_vec(conn, emb) -> None:
    from brain.store import vec as vec_store

    if not vec_store.ensure_tables(conn, dim=256, embedder_name=emb.name):
        pytest.skip("sqlite-vec unavailable — vector-gate tests need the vec extension")


def _rows(conn, ids):
    return [conn.execute("SELECT * FROM memories WHERE id=?", (i,)).fetchone()
            for i in ids]


@contextmanager
def fake_llm(reply):
    """Install a brain LLM fake for the duration of the block.

    ``reply`` is either a str (returned verbatim) or a callable matching
    llm.set_llm_for_tests's contract: ``fn(prompt, *, system, max_tokens)->str``.
    Cleared unconditionally on exit so no later test hits the real aux client
    (which IS importable in this dev env — see CLAUDE.md testing gotcha).
    """
    from brain import llm

    fn = reply if callable(reply) else (
        lambda prompt, *, system=None, max_tokens=0: reply)
    llm.set_llm_for_tests(fn)
    try:
        yield
    finally:
        llm.set_llm_for_tests(None)


# ===========================================================================
# capture/extract.py — deterministic item guards (~514-535)
# ===========================================================================

def _extract_ctx(session="s1"):
    from brain.capture import extract

    return {
        "session_id": session, "epi_by_uid8": {}, "batch_floor": "owner",
        "single_principal": True, "principal": "owner", "platform": "cli",
        "aids_max": extract._MAX_AIDS,
    }


def test_extract_item_guards_reject_malformed_items(conn):
    """A noisy 12-item batch mixing every decoy shape yields ONLY the
    well-formed, whitelisted, non-echo items — the rest are dropped, never
    written."""
    from brain.capture import extract

    echo = "An empty array is a perfectly good answer."
    assert echo in extract._EXTRACT_SYSTEM  # prompt-echo guard fixture

    good = [
        {"content": "the staging db is postgres 15 on fly.io", "kind": "fact"},
        {"content": "deploy runs via scripts/ship.sh at 0200 utc", "kind": "decision"},
        {"content": "user prefers dark mode in all dashboards", "kind": "preference"},
    ]
    decoys = [
        {"content": "short", "kind": "fact"},                     # < 10 chars
        {"content": "x" * 401, "kind": "fact"},                   # > 400 chars
        {"content": "a perfectly valid length statement here", "kind": "rumor"},  # bad kind
        {"content": echo, "kind": "fact"},                        # prompt echo
        "not a dict",                                             # non-dict
        None,                                                     # non-dict
        42,                                                       # non-dict
        {"content": "", "kind": "fact"},                          # empty content
        {"kind": "fact"},                                         # no content key
    ]
    result = good + decoys                                        # exactly 12 items
    counts = {"items": 0, "inserted": 0, "merged": 0, "quarantined": 0}

    extract._apply_items(conn, result, _extract_ctx(), embedder=None,
                         shadow=False, actor="t", counts=counts)

    stored = {r["content"] for r in conn.execute(
        "SELECT content FROM memories WHERE created_by='extraction'").fetchall()}
    assert stored == {g["content"] for g in good}
    assert counts["items"] == 3 and counts["inserted"] == 3
    # Not one decoy leaked through.
    assert "short" not in stored and echo not in stored
    assert not any("rumor" in (c or "") for c in stored)


def test_extract_item_cap_limits_to_12(conn):
    """Even 13 perfectly-valid, novel items only admit the first 12 —
    ``result[:_MAX_ITEMS_PER_BATCH]`` is a hard anti-flood wall."""
    from brain.capture import extract

    kinds = sorted(extract._KIND_WHITELIST)
    result = [{"content": f"unique durable fact number {i} about the system",
               "kind": kinds[i % len(kinds)]} for i in range(13)]
    counts = {"items": 0, "inserted": 0, "merged": 0, "quarantined": 0}

    extract._apply_items(conn, result, _extract_ctx(), embedder=None,
                         shadow=False, actor="t", counts=counts)

    stored = {r["content"] for r in conn.execute(
        "SELECT content FROM memories WHERE created_by='extraction'").fetchall()}
    assert len(stored) == extract._MAX_ITEMS_PER_BATCH == 12
    # The 13th (index 12) item is beyond the cap and never considered.
    assert "unique durable fact number 12 about the system" not in stored


def test_extract_search_aids_guards(conn):
    """``_search_aids`` caps the count and drops content-echoes, prompt-echoes,
    dupes, and out-of-bounds lengths — a noisy model can't turn aids into
    spam."""
    from brain.capture import extract

    good = ["reset the router credentials", "guest access code lookup",
            "connect a new laptop", "printer on the lan setup"]
    # Guard fixtures: the good aids must be novel; the echo must be a real
    # substring of the extraction prompt.
    for g in good:
        assert g not in extract._EXTRACT_SYSTEM
    prompt_echo = "NO duplicates of each other."
    assert prompt_echo in extract._EXTRACT_SYSTEM

    content = "the office wifi password is hunter2 for the network"
    item = {"search_aids": [
        good[0],                      # keep (1)
        good[1],                      # keep (2)
        good[0].upper(),             # casefold dupe of #1 -> drop
        good[2],                      # keep (3)
        good[3],                      # keep (4 -> cap reached)
        "spare backup phrase entry",  # 5th valid -> dropped by the cap
        content,                      # content echo -> drop
        "x",                          # 1 char (< 2) -> drop
        "z" * 81,                     # 81 chars (> 80) -> drop
        prompt_echo,                  # prompt echo -> drop
        999, None,                    # non-string -> skip
    ]}

    aids = extract._search_aids(item, content, max_aids=extract._MAX_AIDS)

    assert aids == good
    assert len(aids) <= extract._MAX_AIDS == 4
    assert content not in aids and "spare backup phrase entry" not in aids
    # max_aids=0 (config kill-switch) -> no aids at all.
    assert extract._search_aids(item, content, max_aids=0) == []


def test_extract_near_dup_merges_same_scope(conn):
    """A paraphrase at cos ~0.98 in the SAME scope MERGES onto the older row
    (verification_count++), never a second row."""
    from brain.capture import extract
    from brain.store import db
    from brain.store import vec as vec_store

    emb = VecEmbedder()
    _ensure_vec(conn, emb)

    content_a = "the office wifi password is hunter2"
    content_b = "office network passphrase hunter2 wifi"          # cos ~0.98 to A
    emb.registry[content_b] = _unit([(0, 0.98), (1, 0.198997)])
    id_a = _insert(conn, content_a, scope_user="userX", created_by="extraction")
    vec_store.upsert(conn, "mem_vec", id_a, _unit([(0, 1.0)]))
    conn.commit()

    merged = extract._try_merge(
        conn, db.content_hash(content_b), content_b, "userX",
        embedder=emb, shadow=False, actor="t", session="s", now=db.iso_now())

    assert merged is True
    assert conn.execute("SELECT verification_count FROM memories WHERE id=?",
                        (id_a,)).fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE action='extract_merge'"
    ).fetchone()[0] == 1


def test_extract_no_merge_across_scope(conn):
    """The SAME paraphrase under a DIFFERENT scope_user must NOT merge —
    cross-scope merge would leak one user's fact into another's recall."""
    from brain.capture import extract
    from brain.store import db
    from brain.store import vec as vec_store

    emb = VecEmbedder()
    _ensure_vec(conn, emb)

    content_b = "office network passphrase hunter2 wifi"
    emb.registry[content_b] = _unit([(0, 0.98), (1, 0.198997)])
    id_a = _insert(conn, "the office wifi password is hunter2",
                   scope_user="userX", created_by="extraction")
    vec_store.upsert(conn, "mem_vec", id_a, _unit([(0, 1.0)]))
    conn.commit()

    merged = extract._try_merge(
        conn, db.content_hash(content_b), content_b, "userY",  # different scope
        embedder=emb, shadow=False, actor="t", session="s", now=db.iso_now())

    assert merged is False
    assert conn.execute("SELECT verification_count FROM memories WHERE id=?",
                        (id_a,)).fetchone()[0] == 1  # untouched


def test_extract_no_merge_below_cosine_threshold(conn):
    """A merely-related item (cos ~0.50, below _MERGE_COSINE=0.95) is treated
    as novel — the merge is a tight near-dup gate, not a loose bucket."""
    from brain.capture import extract
    from brain.store import db
    from brain.store import vec as vec_store

    emb = VecEmbedder()
    _ensure_vec(conn, emb)

    content_c = "unrelated note about the coffee machine downstairs"
    emb.registry[content_c] = _unit([(0, 0.5), (1, 0.8660254)])   # cos ~0.50
    id_a = _insert(conn, "the office wifi password is hunter2",
                   scope_user="userX", created_by="extraction")
    vec_store.upsert(conn, "mem_vec", id_a, _unit([(0, 1.0)]))
    conn.commit()

    merged = extract._try_merge(
        conn, db.content_hash(content_c), content_c, "userX",
        embedder=emb, shadow=False, actor="t", session="s", now=db.iso_now())

    assert merged is False
    assert conn.execute("SELECT verification_count FROM memories WHERE id=?",
                        (id_a,)).fetchone()[0] == 1


# ===========================================================================
# dream/consolidate.py — specificity gate (~256-287) + single-scope clusters
# ===========================================================================

def _members(conn, contents, **cols):
    ids = [_insert(conn, c, created_by="extraction", **cols) for c in contents]
    return _rows(conn, ids)


def test_consolidate_validate_rejects_vague_entity(conn):
    """A lesson whose ``entity`` is a vague abstraction not present in the
    entities table nor verbatim in any member is rejected (returns None)."""
    from brain.dream import consolidate

    members = _members(conn, ["reorganized the kanban board this week again",
                              "moved three tickets to done on friday",
                              "closed the sprint with two carryover items"])
    proposal = {"content": "the user values getting things done efficiently",
                "cites": [members[0]["uid"]], "entity": "productivity",
                "actionable": True}
    assert consolidate._validate(conn, proposal, members) is None


def test_consolidate_validate_rejects_non_actionable(conn):
    """``actionable: false`` is rejected even with a concrete, verbatim
    entity and a valid cite — a non-actionable lesson is dream-spam."""
    from brain.dream import consolidate

    members = _members(conn, ["set FLY_API_TOKEN before the fly.io deploy",
                              "the fly.io deploy failed without FLY_API_TOKEN",
                              "FLY_API_TOKEN lives in the .env for fly.io"])
    proposal = {"content": "fly.io deploys need FLY_API_TOKEN present",
                "cites": [members[0]["uid"]], "entity": "FLY_API_TOKEN",
                "actionable": False}
    assert consolidate._validate(conn, proposal, members) is None


def test_consolidate_validate_rejects_cites_outside_members(conn):
    """Citations must be actual member uids; a lesson that cites only
    non-members has no evidence and is rejected."""
    from brain.dream import consolidate

    members = _members(conn, ["set FLY_API_TOKEN before the fly.io deploy",
                              "the fly.io deploy failed without FLY_API_TOKEN",
                              "FLY_API_TOKEN lives in the .env for fly.io"])
    proposal = {"content": "fly.io deploys need FLY_API_TOKEN present",
                "cites": ["NOTAMEMBERUID0000000000000", "ALSOFAKE00000000000000000"],
                "entity": "FLY_API_TOKEN", "actionable": True}
    assert consolidate._validate(conn, proposal, members) is None


def test_consolidate_validate_rejects_malformed_shape(conn):
    """Non-dict, empty-content, and >140-word proposals are all rejected."""
    from brain.dream import consolidate

    members = _members(conn, ["set FLY_API_TOKEN before the fly.io deploy",
                              "the fly.io deploy failed without FLY_API_TOKEN",
                              "FLY_API_TOKEN lives in the .env for fly.io"])
    assert consolidate._validate(conn, "not a dict", members) is None
    assert consolidate._validate(conn, ["also", "not"], members) is None
    assert consolidate._validate(
        conn, {"content": "", "cites": [members[0]["uid"]],
               "entity": "FLY_API_TOKEN", "actionable": True}, members) is None
    too_long = " ".join(["word"] * 141)
    assert consolidate._validate(
        conn, {"content": too_long, "cites": [members[0]["uid"]],
               "entity": "FLY_API_TOKEN", "actionable": True}, members) is None


def test_consolidate_validate_accepts_concrete_verbatim(conn):
    """A well-formed lesson — concrete entity appearing VERBATIM in a member,
    actionable, real cite, under the word wall — is accepted (positive
    control: the gate is not merely always-reject)."""
    from brain.dream import consolidate

    members = _members(conn, ["set FLY_API_TOKEN before the fly.io deploy",
                              "the fly.io deploy failed without FLY_API_TOKEN",
                              "FLY_API_TOKEN lives in the .env for fly.io"])
    proposal = {"content": "always set FLY_API_TOKEN before deploying to fly.io",
                "cites": [members[0]["uid"], members[1]["uid"]],
                "entity": "FLY_API_TOKEN", "actionable": True}
    lesson = consolidate._validate(conn, proposal, members)
    assert lesson is not None
    assert lesson["entity"] == "FLY_API_TOKEN"
    assert set(lesson["cites"]) <= {m["uid"] for m in members}


def test_consolidate_validate_accepts_entity_from_table(conn):
    """The specificity gate also accepts an entity that exact-matches the
    ``entities`` table (COLLATE NOCASE) even if not verbatim in a member."""
    from brain.dream import consolidate
    from brain.store import entities

    members = _members(conn, ["the primary datastore choice was locked last sprint",
                              "we standardized the backend store this quarter",
                              "the team agreed on one relational store"])
    entities.link(conn, "Postgres", members[0]["id"])
    conn.commit()
    proposal = {"content": "standardize on Postgres for the primary datastore",
                "cites": [members[0]["uid"]], "entity": "Postgres",
                "actionable": True}
    lesson = consolidate._validate(conn, proposal, members)
    assert lesson is not None and lesson["entity"] == "Postgres"


def test_consolidate_cluster_never_merges_across_scope(conn):
    """``_cluster`` must never put two scopes in one cluster — two users'
    near-identical private facts must not distill into one leaking pattern."""
    import types

    from brain.dream import consolidate
    from brain.store import vec as vec_store

    emb = VecEmbedder()
    _ensure_vec(conn, emb)

    ids = []
    for scope in ("userA", "userA", "userA", "userB", "userB", "userB"):
        mid = _insert(conn, "we always deploy on fridays after standup",
                      scope_user=scope, created_by="extraction")
        vec_store.upsert(conn, "mem_vec", mid, _unit([(0, 1.0)]))  # identical vectors
        ids.append(mid)
    conn.commit()

    candidates = _rows(conn, ids)
    blobs = {r["id"]: conn.execute("SELECT emb FROM mem_vec WHERE id=?",
                                   (r["id"],)).fetchone()["emb"] for r in candidates}
    shift = types.SimpleNamespace(conn=conn, tick=lambda: True)

    clusters = consolidate._cluster(shift, candidates, blobs)

    # Without the scope guard these 6 identical-vector rows would form ONE
    # cross-scope cluster; with it, every cluster is single-scope.
    for cl in clusters:
        assert len({m["scope_user"] for m in cl}) == 1
    assert not any({"userA", "userB"} <= {m["scope_user"] for m in cl}
                   for cl in clusters)


def test_consolidate_run_vague_rejected_audits_and_writes_nothing(conn, tmp_home):
    """End-to-end: an adversarial LLM returns a vague lesson for a real
    cluster -> ``consolidate_reject_vague`` audit row, nothing distilled."""
    import time as _time

    from brain.dream import consolidate
    from brain.dream.shift import Shift
    from brain.store import db
    from brain.store import vec as vec_store

    emb = VecEmbedder()
    _ensure_vec(conn, emb)

    ids = []
    for _ in range(3):
        mid = _insert(conn, "reorganized the kanban board again this week",
                      scope_user="userA", created_by="extraction")
        vec_store.upsert(conn, "mem_vec", mid, _unit([(0, 1.0)]))
        ids.append(mid)
    conn.commit()
    a_uid = conn.execute("SELECT uid FROM memories WHERE id=?", (ids[0],)).fetchone()[0]

    vague = json.dumps({"content": "the user values being productive",
                        "cites": [a_uid], "entity": "productivity",
                        "actionable": True})
    shift = Shift(shift_id="t7", conn=conn, config={}, embedder=emb,
                  started_at=db.iso_now(), activity_baseline="", holder="h")
    shift._last_renew = _time.monotonic()  # short-circuit lease renewal in tick()

    with fake_llm(vague):
        result = consolidate.run(shift)

    assert result.get("rejected", 0) >= 1
    assert result.get("distilled", 0) == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE action='consolidate_reject_vague'"
    ).fetchone()[0] >= 1
    # Nothing was distilled into a new semantic pattern.
    assert conn.execute(
        "SELECT COUNT(*) FROM memories WHERE created_by='consolidation'"
    ).fetchone()[0] == 0


# ===========================================================================
# skillforge/forge.py — statistical + replay gates (~467-504)
# ===========================================================================

def test_forge_wilson_lower_bound_anchors(conn):
    """The Wilson lower bound the ratchet leans on punishes small/noisy
    samples: 1/3 sits below the 0.20 floor while 3/3 clears it. (These are
    the anchors forge._WILSON_MIN and the harm gate both key off of.)"""
    from brain.dream.stats import wilson_lower_bound as w
    from brain.skillforge import forge

    assert w(1, 3) < forge._WILSON_MIN < w(3, 3)
    assert w(2, 3) == pytest.approx(0.208, abs=0.01)  # a gold 2/3 barely clears
    assert w(0, 5) == 0.0                              # no successes -> no floor


def _forge_cluster(conn, emb, n, successes, *, member_vec, draft_vec):
    """Build a case cluster of ``n`` members embedded at ``member_vec`` and a
    draft embedded at ``draft_vec``; return (cluster, draft)."""
    from brain.store import vec as vec_store

    members = []
    for i in range(n):
        mid = _insert(conn, f"case episode {i}: staging deploy trajectory",
                      kind="case", memory_type="episodic", created_by="dream")
        vec_store.upsert(conn, "mem_vec", mid, member_vec)
        members.append(conn.execute("SELECT * FROM memories WHERE id=?",
                                    (mid,)).fetchone())
    conn.commit()
    embed_text = "draft skill: staging deploy checklist and pitfalls"
    emb.registry[embed_text] = draft_vec
    cluster = {"members": members, "successes": successes,
               "failures": n - successes, "gold": False}
    draft = {"name": "staging-deploy-checklist", "description": "staging deploy",
             "embed_text": embed_text}
    return cluster, draft


def test_forge_validate_blocks_coin_flip_cluster(conn):
    """A coin-flip cluster (2/4 successes = rate 0.5) fails the statistical
    gate even though replay+probes pass -> NOT auto-promotable (review queue).
    Noise cannot ratchet a skill."""
    from brain.skillforge import forge

    emb = VecEmbedder()
    _ensure_vec(conn, emb)
    vx = _unit([(0, 1.0)])
    cluster, draft = _forge_cluster(conn, emb, n=4, successes=2,
                                    member_vec=vx, draft_vec=vx)

    validation = forge._validate(conn, {}, cluster, draft, emb)

    assert validation["passed"] is False
    assert validation["gates"]["statistical"]["passed"] is False
    assert validation["gates"]["replay"]["passed"] is True     # coherent
    assert validation["gates"]["probes"]["failed"] == 0        # healthy brain


def test_forge_validate_replay_gate_blocks_incoherent_cluster(conn):
    """An incoherent 3/3 cluster (perfect record but members OFF-TOPIC to the
    draft) fails the replay-cover gate — a skill must actually cover the cases
    it claims, not just have a good scoreboard."""
    from brain.skillforge import forge

    emb = VecEmbedder()
    _ensure_vec(conn, emb)
    cluster, draft = _forge_cluster(
        conn, emb, n=3, successes=3,
        member_vec=_unit([(1, 1.0)]), draft_vec=_unit([(0, 1.0)]))  # orthogonal

    validation = forge._validate(conn, {}, cluster, draft, emb)

    assert validation["passed"] is False
    assert validation["gates"]["replay"]["passed"] is False
    assert validation["gates"]["statistical"]["passed"] is True   # 3/3 is fine


def test_forge_validate_passes_coherent_decisive_cluster(conn):
    """Positive control: a coherent 3/3 cluster clears ALL three gates — the
    ratchet is not stuck permanently closed."""
    from brain.skillforge import forge

    emb = VecEmbedder()
    _ensure_vec(conn, emb)
    vx = _unit([(0, 1.0)])
    cluster, draft = _forge_cluster(conn, emb, n=3, successes=3,
                                    member_vec=vx, draft_vec=vx)

    validation = forge._validate(conn, {}, cluster, draft, emb)

    assert validation["passed"] is True
    assert validation["gates"]["replay"]["passed"] is True
    assert validation["gates"]["statistical"]["passed"] is True
    assert validation["gates"]["probes"]["failed"] == 0


def test_forge_validate_probe_failure_should_veto_promotion(conn):
    """A probe REGRESSION during the shift must be a hard veto on promotion
    (probes.py: 'the skill-forge treats any failure as a hard veto'). Here a
    coherent, decisive 3/3 cluster passes replay+statistical, but a real
    staleness regression (a superseded row left current) makes run_probes fail.

    DEFECT: forge._validate stores the probes gate as
    ``{"passed": report.ok(), **report.summary()}`` — ``report.summary()`` also
    carries a ``passed`` key (the INTEGER count of passing probes), which
    overwrites the boolean ``report.ok()``. ``_validate`` then computes
    ``all(g.get("passed") ...)``; the probes gate contributes a truthy integer
    (2) instead of ``False``, so the failing suite does NOT block promotion.
    Repro: run this test — ``validation["passed"]`` is True (should be False).
    A partial probe failure only vetoes when EVERY probe fails (count == 0).
    """
    from brain.skillforge import forge

    emb = VecEmbedder()
    _ensure_vec(conn, emb)
    vx = _unit([(0, 1.0)])
    cluster, draft = _forge_cluster(conn, emb, n=3, successes=3,
                                    member_vec=vx, draft_vec=vx)
    # Inject a genuine staleness regression: a superseded row still current
    # (superseded_by set, valid_to NULL) — exactly what _staleness_probes flags.
    base = _insert(conn, "old value of the config flag")
    _insert(conn, "new value of the config flag", superseded_by=base)

    validation = forge._validate(conn, {}, cluster, draft, emb)

    assert validation["gates"]["probes"]["failed"] >= 1   # the suite DID fail
    assert validation["passed"] is False                  # ...so promotion is vetoed


# ===========================================================================
# skillforge/revise.py — _harm_verdict (~161-180) + propose-only discipline
# ===========================================================================

def test_revise_harm_verdict_neutral_traffic_healthy(conn):
    """1 hurt drowned in 20 neutral is HEALTHY: the decisive-sample floor is
    over (helped+hurt), so neutral traffic can never dilute into net-harm."""
    from brain.skillforge import revise

    verdict = revise._harm_verdict(
        {"helped": 0, "hurt": 1, "neutral": 20, "total": 21})
    assert verdict["harmful"] is False


def test_revise_harm_verdict_below_floor_and_ties_healthy(conn):
    """Below the decisive floor (4 hurt, 0 helped -> only 4 decisive), and a
    dead tie (4/4), both stay HEALTHY — a handful of bad luck or an even split
    never retires a skill."""
    from brain.skillforge import revise

    assert revise._harm_verdict(
        {"helped": 0, "hurt": 4, "neutral": 0, "total": 4})["harmful"] is False
    assert revise._harm_verdict(
        {"helped": 4, "hurt": 4, "neutral": 2, "total": 10})["harmful"] is False


def test_revise_harm_verdict_flags_genuine_harm(conn):
    """A decisively net-harmful record (6 hurt vs 1 helped over 7 decisive)
    IS flagged harmful — the gate lets a truly bad skill through to a
    proposal."""
    from brain.skillforge import revise

    verdict = revise._harm_verdict(
        {"helped": 1, "hurt": 6, "neutral": 2, "total": 9})
    assert verdict["harmful"] is True
    assert verdict["harm_rate"] > 0.5 and verdict["wilson_lb"] >= revise._HARM_WILSON_MIN


def test_revise_only_proposes_pending_never_applies(conn, tmp_home):
    """The revise loop, on a genuinely-harmful brain-forged skill, writes a
    single ``pending`` skill_revision proposal and leaves the on-disk SKILL.md
    BYTE-for-byte unchanged (it proposes, the CLI applies)."""
    from brain.skillforge import revise, skilltree

    name = "staging-deploy-guard"
    md = skilltree.build_skill_md(
        name, "guard staging deploys", "## Procedure\n\n1. do the thing")
    root = skilltree.skills_root(tmp_home) / name
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(md, encoding="utf-8")
    skilltree.write_usage_record(tmp_home, name, {
        "created_by": "hermes-brain", "helped": 1, "hurt": 6, "neutral": 2})

    delta = json.dumps({"diagnosis": "trigger too broad",
                        "sections": [{"heading": "## Procedure",
                                      "new_text": "1. only run on staging"}],
                        "summary": "narrow the trigger"})
    config = {"hermes_home": str(tmp_home)}
    with fake_llm(delta):
        summary = revise.revise_once(conn, config)

    assert summary["revisions"] == 1
    row = conn.execute(
        "SELECT kind, status FROM proposals WHERE target=?", (name,)).fetchone()
    assert row is not None and row["kind"] == "skill_revision"
    assert row["status"] == "pending"                       # proposed, not applied
    assert (root / "SKILL.md").read_text(encoding="utf-8") == md  # untouched


# ===========================================================================
# recall/strategies.py — read-time deprecation (~161-172)
# ===========================================================================

def test_strategy_deprecation_drops_net_harmful_guardrail(conn):
    """A guardrail proven net-harmful (helpful=1, harmful=6, n>=5) is NEVER
    injected by retrieve_guidance — even at similarity 1.0 — while an equally
    similar net-HELPFUL guardrail still surfaces (isolates the deprecation)."""
    from brain.recall import strategies
    from brain.store import vec as vec_store

    gvec = _unit([(5, 1.0)])
    query = "should i rebase before pushing to main"
    emb = VecEmbedder({query: gvec})
    _ensure_vec(conn, emb)

    harmful_id = _insert(conn, "always rebase before pushing to main",
                         kind="guardrail", memory_type="procedural",
                         created_by="distillation", helpful_count=1, harmful_count=6)
    helpful_id = _insert(conn, "prefer squash-merge for feature branches",
                         kind="guardrail", memory_type="procedural",
                         created_by="distillation", helpful_count=6, harmful_count=1)
    for mid in (harmful_id, helpful_id):
        vec_store.upsert(conn, "mem_vec", mid, gvec)  # both at sim 1.0 to the query
    conn.commit()

    guidance = strategies.retrieve_guidance(
        conn, query, embedder=emb, trust_tier="owner", scope_user=None)
    ids = {g.id for g in guidance}

    assert harmful_id not in ids     # proven net-harmful -> read-time deprecation
    assert helpful_id in ids         # net-helpful peer still injected at same sim
