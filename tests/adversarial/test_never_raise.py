"""Phase 2 of the adversarial gauntlet: the *never-raise / degrade-into-turn*
invariant.

Everything reachable synchronously from a turn (prefetch / sync / tool-call /
worker) or a dream strategy must SWALLOW an internal exception and hand back a
safe sentinel — ``[]`` / ``None`` / ``0`` / ``{"error": ...}`` / a bounded
string — never propagate into the turn or the dream pipeline, and the caller
must still complete. Each test forces the *inner* body to blow up (a corrupt
row, a poisoned vector, a wedged write, a dead archive, an injected fault) and
then asserts BOTH the degraded sentinel AND that the caller returned normally.

No test destroys data. The load-bearing one (archive/forget) additionally
proves the opposite direction: a failed archive must LEAVE the raw content in
place (a tombstone to retry) — never null it.

Runs on the stdlib floor tier (no onnx / sqlite-vec). Seams that genuinely
need the vector tier are either driven with a throwing fake embedder (so the
degradation path still executes) or skipped with a reason.
"""

from __future__ import annotations

import json

import pytest

# brain.* subpackages import freely (they are never eagerly loaded).
from brain.store import vec as vec_store
from conftest import iso_days_ago, seed_memory
from faults import (
    break_archive_dir,
    corrupt_half_life,
    poison_mem_vec,
    raising,
    returning,
)

# NOTE: faults.sqlite_write_error is deliberately NOT used here — on this
# CPython build sqlite3.Connection.execute is a read-only C-slot and
# mock.patch.object(conn, "execute", ...) raises AttributeError, so that helper
# cannot patch the connection. We induce the equivalent "the write blew up"
# condition instead by forcing an internal callable to raise (faults.raising)
# or, where the guard only catches sqlite3.Error, with a RAISE(ABORT) trigger.


# ---------------------------------------------------------------------------
# A fake embedder whose encode_query always raises, used to drive the OPTIONAL
# vector legs on the floor tier: patch vec_available -> True so the leg runs,
# then the first embed call throws and the leg's own try/except must degrade.
# ---------------------------------------------------------------------------
class _BoomEmbedder:
    dim = 256
    name = "boom"

    def encode_query(self, query):  # noqa: D401 - test double
        raise RuntimeError("embedder exploded mid-turn")


# ===========================================================================
# recall/search.py — search() never raises (capture path)
# ===========================================================================

def test_search_swallows_corrupt_half_life_and_returns_list(conn):
    """A memory row with a non-numeric half_life makes _modulate's lifecycle
    arithmetic raise TypeError mid-scoring; the outer guard (search.py:319-321)
    must swallow it and return [] rather than propagate into the turn."""
    rid = seed_memory(conn, "deploy pipeline configuration notes")
    corrupt_half_life(conn, rid)  # half_life_days='not-a-number'

    result = search_of(conn)("deploy")

    assert isinstance(result, list)
    # fts5 is present in this env, so scoring (and thus _modulate) is reached;
    # the raise inside the hits loop degrades the whole call to [].
    assert result == []


def test_search_punctuation_only_query_short_circuits_to_empty(conn):
    """A query that tokenizes to nothing (search.py:198-199) returns [] before
    any SQL — pure punctuation/underscores must never build a MATCH expr."""
    fn = search_of(conn)
    for q in ("___", "!!!???", "   ", ""):
        out = fn(q)
        assert out == [], f"expected [] for {q!r}, got {out!r}"


def test_search_vec_leg_failure_degrades_to_fts(conn):
    """With the vec leg forced live (vec_available->True) but the embedder
    throwing, search.py:253-254 must catch and continue on FTS only — the
    keyword hit still comes back."""
    seed_memory(conn, "alpha bravo charlie delta")

    from brain.recall.search import search

    with returning(vec_store, "vec_available", True):
        hits = search(conn, "bravo", embedder=_BoomEmbedder())

    assert isinstance(hits, list)
    assert any(h.kind == "memory" and "bravo" in h.text for h in hits), \
        "FTS leg should survive a poisoned vector leg"


def test_search_with_poisoned_vector_returns_hits_or_skips(conn):
    """The faults.poison_mem_vec path: only meaningful on the vec tier. On the
    floor tier it returns False and we skip; where a vec table exists the FTS
    hits must still return."""
    rid = seed_memory(conn, "kilo lima mike november")
    if not poison_mem_vec(conn, rid):
        pytest.skip("no sqlite-vec tier in this environment")

    from brain.recall.search import search

    with returning(vec_store, "vec_available", True):
        hits = search(conn, "lima", embedder=_BoomEmbedder())
    assert isinstance(hits, list)


# ===========================================================================
# recall/search.py — log_retrieval / stamp_pending_injections no-ops
# ===========================================================================

def _mem_hit(conn, rid):
    from brain.recall.search import Hit

    uid = conn.execute("SELECT uid FROM memories WHERE id=?", (rid,)).fetchone()["uid"]
    return Hit(kind="memory", id=rid, uid=uid, text="x", summary=None,
               memory_type=None, mkind="fact", ts="2026-01-01T00:00:00.000Z",
               platform=None, score=1.0, source="fts")


def test_log_retrieval_empty_hits_is_noop(conn):
    """search.py:605 — nothing to log short-circuits before any SQL."""
    from brain.recall.search import log_retrieval

    assert log_retrieval(conn, "sess", "q", [], set()) is None
    n = conn.execute("SELECT COUNT(*) FROM retrieval_log").fetchone()[0]
    assert n == 0


def test_log_retrieval_swallows_write_failure(conn):
    """search.py:639-640 — a failed insert (here an FK violation from a
    memory_id that does not exist) is logged and degrades to a no-op; it must
    never raise into the turn."""
    from brain.recall.search import Hit, log_retrieval

    ghost = Hit(kind="memory", id=999_999, uid="01GHOSTUID0000000000000000",
                text="x", summary=None, memory_type=None, mkind="fact",
                ts="2026-01-01T00:00:00.000Z", platform=None, score=1.0,
                source="fts")
    # Must return None, not raise, even though the FK check rejects the row.
    assert log_retrieval(conn, "sess", "q", [ghost], {ghost.uid}) is None


def test_log_retrieval_swallows_internal_error(conn):
    """The same guard for a non-FK internal blow-up: force db.content_hash
    (called inside the try) to raise and assert log_retrieval still degrades to
    a no-op None."""
    from brain.recall.search import log_retrieval
    from brain.store import db as db_mod

    rid = seed_memory(conn, "loggable memory row")
    hit = _mem_hit(conn, rid)
    with raising(db_mod, "content_hash", RuntimeError("hash exploded")):
        assert log_retrieval(conn, "sess", "q", [hit], {hit.uid}) is None


def test_stamp_pending_empty_user_msg_returns_zero(conn):
    """search.py:660-661 — an empty/whitespace user_msg returns 0 before any
    SQL (nothing to hash against state.db)."""
    from brain.recall.search import stamp_pending_injections

    assert stamp_pending_injections(conn, "sess", 1, "") == 0
    assert stamp_pending_injections(conn, "sess", 1, "   \n\t ") == 0


def test_stamp_pending_swallows_write_failure(conn):
    """search.py:668-670 — a failure inside the update (here db.content_hash,
    evaluated as an UPDATE param) returns 0, never raises."""
    from brain.recall.search import stamp_pending_injections
    from brain.store import db as db_mod

    with raising(db_mod, "content_hash", RuntimeError("hash exploded")):
        assert stamp_pending_injections(conn, "sess", 1, "a real message") == 0


# ===========================================================================
# recall/strategies.py — retrieve_guidance never raises
# ===========================================================================

def test_retrieve_guidance_degrades_to_empty_on_failure(conn):
    """strategies.py:129-131 — with the vec gate forced open and the embedder
    throwing, guidance retrieval must catch and return []."""
    from brain.recall.strategies import retrieve_guidance

    with returning(vec_store, "vec_available", True):
        out = retrieve_guidance(conn, "fix the deploy", embedder=_BoomEmbedder())
    assert out == []


def test_retrieve_guidance_no_embedder_returns_empty(conn):
    """The floor-tier guard: no embedder -> [] without touching vectors."""
    from brain.recall.strategies import retrieve_guidance

    assert retrieve_guidance(conn, "anything", embedder=None) == []


# ===========================================================================
# capture/extract.py — precompress_contribution never raises, bounded output
# ===========================================================================

def test_precompress_handles_hostile_message_shapes(conn):
    """extract.py:270-272 — non-dict entries and weird content types must not
    raise; the result is always a string."""
    from brain.capture.extract import precompress_contribution

    hostile = [
        None,
        42,
        "a bare string, not a dict",
        {"role": "user", "content": {"unexpected": "dict"}},
        {"role": "assistant", "content": [{"type": "image", "url": "x"}]},
        {"role": "user"},                       # missing content
        {"role": "assistant", "content": None},
        ["a", "list"],
    ]
    out = precompress_contribution(hostile, 300)
    assert isinstance(out, str)


def test_precompress_bounds_a_multimegabyte_message(conn):
    """A multi-MB user/assistant pair must stay bounded: the pre-score clip
    (extract.py ~255) caps score_turn's input, and _clip caps each rendered
    line — so cost is bounded by message count, not transcript bytes."""
    from brain.capture.extract import precompress_contribution

    huge_u = "please REMEMBER this important preference: " + ("x" * 3_000_000)
    huge_a = "noted, decided to " + ("y" * 3_000_000)
    out = precompress_contribution(
        [{"role": "user", "content": huge_u},
         {"role": "assistant", "content": huge_a}], 300)
    assert isinstance(out, str)
    assert len(out) < 4000, f"output not bounded: {len(out)} chars"


def test_precompress_single_huge_unpaired_message(conn):
    """A single enormous message with no user->assistant pairing yields '' —
    and never raises."""
    from brain.capture.extract import precompress_contribution

    out = precompress_contribution(
        [{"role": "user", "content": "z" * 5_000_000}], 300)
    assert out == ""


# ===========================================================================
# store/entities.py — link() returns None on bad input / failure
# ===========================================================================

def test_entities_link_empty_name_returns_none(conn):
    """entities.py:42-43 — an empty/whitespace name never becomes a mention."""
    from brain.store import entities

    rid = seed_memory(conn, "a memory to mention")
    assert entities.link(conn, "", rid) is None
    assert entities.link(conn, "   ", rid) is None


def test_entities_link_none_memory_id_returns_none(conn):
    """entities.py:42-43 — memory_id=None returns None before any write."""
    from brain.store import entities

    assert entities.link(conn, "Hermes", None) is None


def test_entities_link_swallows_failure_returns_none(conn):
    """entities.py:61-63 — a genuinely failing write (FK violation from a
    dangling memory_id) is caught; link returns None, never raises into the
    dream/capture path."""
    from brain.store import entities

    # memory 999999 does not exist -> the entity_mentions FK check rejects it.
    assert entities.link(conn, "GhostEntity", 999_999) is None


# ===========================================================================
# store/archive.py + dream/forget.py — THE LOAD-BEARING non-destructive purge
# ===========================================================================

def test_archive_append_none_contract_and_roundtrip(tmp_home):
    """store/archive.py:36-63 — a working archive returns a ref and the content
    is recoverable; a broken archive returns None (the "did not persist"
    sentinel the forget caller keys off of). append_batch honors the same
    contract per-row."""
    from brain.store import archive

    ref = archive.append(tmp_home, {"uid": "01ARCHIVEOK00000000000000", "content": "keep me"})
    assert isinstance(ref, str) and ref.endswith("01ARCHIVEOK00000000000000")
    assert archive.recover_content(tmp_home, ref) == "keep me"

    batch = archive.append_batch(
        tmp_home, [{"uid": "01BATCHONE0000000000000000", "content": "b1"},
                   {"uid": "01BATCHTWO0000000000000000", "content": "b2"}])
    assert all(isinstance(r, str) for r in batch)


def test_archive_append_none_when_archive_dir_is_broken(tmp_home):
    """A dead archive dir makes append()/append_batch() return None — never
    raise. This None is what forces forget to preserve raw text."""
    from brain.store import archive

    break_archive_dir(tmp_home)
    assert archive.append(tmp_home, {"uid": "01BROKEN00000000000000000", "content": "x"}) is None
    refs = archive.append_batch(tmp_home, [{"uid": "01BROKEN00000000000000000", "content": "x"}])
    assert refs == [None]


def _seed_purgeable_tombstones(conn, n=3):
    """Rows that qualify for forget's grace purge: tombstoned, content present,
    aged well past the 30-day grace so _grace_start's recorded_at fallback
    triggers a purge."""
    seeded = []
    for i in range(n):
        content = f"stale worthless note number {i} about widget-{i}"
        rid = seed_memory(conn, content, status="tombstone")
        conn.execute("UPDATE memories SET recorded_at=? WHERE id=?",
                     (iso_days_ago(90), rid))
        uid = conn.execute("SELECT uid FROM memories WHERE id=?", (rid,)).fetchone()["uid"]
        seeded.append((rid, uid, content))
    conn.commit()
    return seeded


def _forget_shift(conn, tmp_home):
    from brain.dream import lease
    from brain.dream.shift import Shift

    holder = "test-forget-holder"
    assert lease.acquire(conn, "dream", holder)
    return Shift(shift_id="forget-shift", conn=conn,
                 config={"hermes_home": str(tmp_home), "_forced_mode": "active"},
                 holder=holder)


def test_forget_broken_archive_preserves_content(conn, tmp_home):
    """THE invariant: when archiving fails, forget's grace purge MUST NOT null
    the live content (dream/forget.py ~173-190 skips on ref=None). The rows
    stay tombstones — a retry marker — with their raw text intact. Never lose
    data."""
    from brain.dream import forget

    seeded = _seed_purgeable_tombstones(conn)
    break_archive_dir(tmp_home)  # append_batch will return all-None

    result = forget.run(_forget_shift(conn, tmp_home))

    assert isinstance(result, dict) and "error" not in result
    assert result.get("purged", 0) == 0, "nothing may be purged when archiving failed"
    for rid, _uid, content in seeded:
        row = conn.execute("SELECT content, status FROM memories WHERE id=?", (rid,)).fetchone()
        assert row["content"] == content, "raw text must be preserved on archive failure"
        assert row["status"] == "tombstone", "row stays a tombstone to retry"


def test_forget_working_archive_stubs_content_and_records_ref(conn, tmp_home):
    """The mirror: with a WORKING archive, content IS nulled to a stub, an
    'archive:<ref>' lands in source_refs, and the raw text is recoverable from
    cold storage — non-destructive, not lossy."""
    from brain.dream import forget
    from brain.store import archive

    seeded = _seed_purgeable_tombstones(conn)

    result = forget.run(_forget_shift(conn, tmp_home))

    assert isinstance(result, dict) and "error" not in result
    assert result.get("purged", 0) >= 1
    for rid, _uid, content in seeded:
        row = conn.execute(
            "SELECT content, summary, source_refs FROM memories WHERE id=?",
            (rid,)).fetchone()
        assert row["content"] is None, "content should be purged after archiving"
        refs = json.loads(row["source_refs"] or "[]")
        archive_refs = [r for r in refs if isinstance(r, str) and r.startswith("archive:")]
        assert archive_refs, "an archive:<ref> must be recorded in source_refs"
        ref = archive_refs[-1].split("archive:", 1)[1]
        assert archive.recover_content(tmp_home, ref) == content, \
            "raw text must live on in the archive (recoverable)"


# ===========================================================================
# tools.py — dispatch ALWAYS returns JSON, NEVER raises
# ===========================================================================

def _ctx(tmp_home):
    from brain.tools import ToolContext

    return ToolContext(session_id="tool-sess", trust_tier="owner",
                       hermes_home=str(tmp_home))


def test_dispatch_unknown_tool_returns_error_json(conn, tmp_home):
    """tools.py:269-271 — an unknown tool yields an errors-that-teach payload,
    not an exception."""
    from brain.tools import dispatch

    out = dispatch(conn, "brain_nonexistent", {}, ctx=_ctx(tmp_home))
    payload = json.loads(out)
    assert "error" in payload and "recovery_hint" in payload


def test_dispatch_bad_args_type_returns_error_json(conn, tmp_home):
    """dispatch guards a non-object args and teaches, never raises."""
    from brain.tools import dispatch

    out = dispatch(conn, "brain_recall", ["not", "an", "object"], ctx=_ctx(tmp_home))
    payload = json.loads(out)
    assert "error" in payload


def test_dispatch_handler_exception_returns_error_json_and_rolls_back(conn, tmp_home):
    """tools.py:272-288 — when a handler throws, dispatch rolls back and returns
    a generic error payload; the tool surface must not raise into the agent."""
    from brain import tools as tools_mod
    from brain.tools import dispatch

    with raising(tools_mod, "_remember", RuntimeError("handler exploded mid-write")):
        out = dispatch(conn, "brain_remember",
                       {"content": "a durable fact to store"}, ctx=_ctx(tmp_home))
    payload = json.loads(out)
    assert "error" in payload
    assert payload["error"].startswith("brain_remember failed")
    assert "recovery_hint" in payload
    # The failed handler left nothing behind.
    n = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE created_by='provider'").fetchone()[0]
    assert n == 0


# ===========================================================================
# dream strategy wrappers — each returns {"error": ...} and the pipeline lives
# ===========================================================================

@pytest.mark.parametrize("modname,inner", [
    ("consolidate", "_run"),
    ("contradict", "_run"),
    ("forget", "_run"),
])
def test_shift_strategy_wrapper_returns_error_dict(conn, modname, inner):
    """dream/{consolidate,contradict,forget}.run must catch an inner explosion,
    roll back, and return {'error': ...} so run.py's phase machine keeps
    going."""
    import importlib

    from brain.dream.shift import Shift

    mod = importlib.import_module(f"brain.dream.{modname}")
    shift = Shift(shift_id="s", conn=conn, config={})
    with raising(mod, inner, RuntimeError("strategy body exploded")):
        result = mod.run(shift)
    assert isinstance(result, dict)
    assert "error" in result


def test_forge_once_wrapper_returns_error_dict(conn):
    """skillforge/forge.py:105-111 — forge_once swallows an inner failure."""
    from brain.skillforge import forge

    with raising(forge, "_forge_once", RuntimeError("forge body exploded")):
        result = forge.forge_once(conn, {}, embedder=None, shift_id="s")
    assert isinstance(result, dict) and "error" in result


def test_revise_once_wrapper_returns_error_dict(conn):
    """skillforge/revise.py:95-101 — revise_once swallows an inner failure."""
    from brain.skillforge import revise

    with raising(revise, "_revise_once", RuntimeError("revise body exploded")):
        result = revise.revise_once(conn, {}, embedder=None, shift_id="s")
    assert isinstance(result, dict) and "error" in result


# ===========================================================================
# dream/run.py — run_dream never raises; the lease is always freed
# ===========================================================================

def test_run_dream_isolates_failing_strategy_and_frees_lease(conn, tmp_home):
    """dream/run.py:196-209 + 223-231 — a strategy that throws is isolated by
    _run_one (returns {'error'}), run_dream still returns a summary, and the
    lease row is released in the finally block."""
    from brain.dream import lease
    from brain.dream.run import run_dream
    from brain.recall import lane1

    # Drive just the lane1 phase and make its inner materialize() explode.
    with raising(lane1, "materialize", RuntimeError("lane1 render exploded")):
        summary = run_dream(conn, {"hermes_home": str(tmp_home)}, phase="lane1")

    assert isinstance(summary, dict)
    entry = summary.get("strategies", {}).get("lane1", {})
    assert "error" in entry, f"failing strategy should be isolated, got {entry!r}"
    # Lease freed: no live holder remains.
    assert lease.held_by(conn, "dream") is None


# ===========================================================================
# provider.py — the brain-bg worker swallows a failed job and keeps going
# ===========================================================================

def test_worker_survives_failing_job_and_keeps_processing(conn, tmp_home):
    """provider.py:597-598 — a malformed job that raises inside the per-job try
    must not kill the worker thread; a subsequent valid turn is still captured.
    """
    import brain.llm as llm
    from brain.provider import BrainProvider
    from conftest import poll_until

    # Non-empty DB -> no bootstrap job; a stub LLM -> no real aux client is ever
    # contacted if an idle sweep sneaks in.
    seed_memory(conn, "seed so the brain is not empty")
    llm.set_llm_for_tests(lambda *a, **k: "")

    provider = BrainProvider()
    try:
        provider.initialize("sess-worker", platform="cli", hermes_home=str(tmp_home))
        # A "turn" job with too few elements -> ValueError on unpack -> swallowed.
        provider._queue.put(("turn", "sess-worker"))
        # A well-formed turn afterwards must still be processed.
        provider.sync_turn("hello worker", "hi there", session_id="sess-worker")

        got = poll_until(
            lambda: conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE session_id='sess-worker'"
            ).fetchone()[0],
            timeout=6.0)
        assert got and got >= 1, "worker died on the bad job; later turn not captured"
        assert provider._worker.is_alive()
    finally:
        provider.shutdown()
        llm.set_llm_for_tests(None)


# ===========================================================================
# llm.py — _meter logs and never raises; a good response still returns
# ===========================================================================

def test_meter_failure_does_not_mask_good_response(conn):
    """llm.py:374-375 — if the ledger insert fails, _meter logs and swallows;
    call_text must still return the (good) response text."""
    import brain.llm as llm

    # _meter catches ONLY sqlite3.Error, so we need a genuine DB write failure:
    # a BEFORE INSERT trigger that RAISE(ABORT)s makes the ledger insert fail
    # while _budget_gate's SELECT (and everything else) stays healthy.
    llm.set_llm_for_tests(lambda prompt, *, system, max_tokens: "a good answer")
    conn.execute(
        "CREATE TEMP TRIGGER _block_ledger BEFORE INSERT ON llm_ledger "
        "BEGIN SELECT RAISE(ABORT, 'ledger blocked'); END")
    try:
        out = llm.call_text(conn, {"day_budget_usd": 100.0}, "hi", tier="extract")
        assert out == "a good answer"
    finally:
        conn.execute("DROP TRIGGER IF EXISTS _block_ledger")
        llm.set_llm_for_tests(None)


# ---------------------------------------------------------------------------
# small helper: bind conn into search() with floor-tier-safe defaults
# ---------------------------------------------------------------------------
def search_of(conn):
    from brain.recall.search import search

    def _run(query):
        return search(conn, query, embedder=None)

    return _run
