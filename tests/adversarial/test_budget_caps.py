"""Phase 4 — runaway & resource guards.

Every test here attacks the brain with something *expensive* (a wedged/empty
LLM, an oversized batch, an entity co-mentioned by thousands of memories, a
seed set far past SQLite's variable limit) and asserts the load-bearing
RESOURCE guard HOLDS: the brain caps, gates, and stops spending — it never
runs away, never blows the SQL binder, never processes more than one night's
worth of work in a single run.

Two headline invariants get solid coverage (the caller's priority):
  * the DAILY budget gate (`llm._budget_gate`, llm.py:256-281) — priced USD
    PLUS a token-proxy for unpriced (est_usd=0) failure rows, summed in ONE
    SQL, so a wedged provider that only produces unpriced failures still trips
    it (even after a cheap priced success);
  * the NIGHT budget gate (`dream.shift.Shift.budget_left`, shift.py:136-147) —
    a token-proxy independent of the daily gate; an LLM strategy consults it
    and increments `skipped_llm` instead of calling;
  * the graph SQL-variable cap (`recall.graph`, graph.py:32) — a hot seed/
    entity set can't raise "too many SQL variables" and silently drop the leg.

Plus a representative subset of the per-run caps (extract batch, forget
demote + wrapping cursor, consolidate lesson words, mine_state episodes).

All deterministic, stdlib-only, no network. The LLM is controlled via
`llm.set_llm_for_tests` (reset by the autouse fixture below).
"""

from __future__ import annotations

import sqlite3
import time
from unittest import mock

import pytest
from conftest import iso_days_ago, seed_memory

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_llm():
    """Never let a fake LLM leak across tests (set_llm_for_tests is a module
    global)."""
    from brain import llm

    yield
    llm.set_llm_for_tests(None)


def _seed_ledger(conn, *, tokens_in=0, tokens_out=0, est_usd=0.0,
                 strategy="extract", model="aux-default", n=1):
    """Insert n llm_ledger rows dated NOW (== today UTC, the window both gates
    read). The daily gate sums est_usd for priced rows and (tokens_in +
    tokens_out) for unpriced (est_usd=0) rows; the night gate sums
    tokens_in+tokens_out unconditionally."""
    from brain.store import db

    for _ in range(n):
        conn.execute(
            "INSERT INTO llm_ledger (strategy, model, tokens_in, tokens_out,"
            " est_usd, ts) VALUES (?,?,?,?,?,?)",
            (strategy, model, tokens_in, tokens_out, est_usd, db.iso_now()),
        )
    conn.commit()


def _ledger_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM llm_ledger").fetchone()[0]


# ===========================================================================
# DAILY budget gate — llm._budget_gate (llm.py:256-281)
# ===========================================================================

def test_daily_gate_under_budget_allows(conn):
    """Baseline: under budget, call_text runs the fake AND writes exactly one
    ledger row (metering is unconditional — llm.py:103)."""
    from brain import llm

    llm.set_llm_for_tests(lambda p, *, system=None, max_tokens=1200: "a real answer")
    out = llm.call_text(conn, {"day_budget_usd": 1.5}, "hello?")
    assert out == "a real answer"
    # metered exactly one row (the successful call).
    assert _ledger_count(conn) == 1


def test_daily_gate_trips_on_unpriced_failure_rows(conn):
    """Preload TODAY with unpriced failure rows (est_usd=0, big tokens_out)
    whose token-proxy exceeds the budget → the next call_text raises
    LLMUnavailable BEFORE the fake is ever invoked (gate precedes the call)."""
    from brain import llm

    # budget 0.005 USD => 0.005 * 400_000 = 2000 token proxy threshold.
    # one unpriced row of 5000 tokens_out => proxy 0.0125 > 0.005.
    _seed_ledger(conn, tokens_out=5000, est_usd=0.0)

    called = []
    llm.set_llm_for_tests(lambda p, *, system=None, max_tokens=1200: called.append(1) or "x")
    with pytest.raises(llm.LLMUnavailable):
        llm.call_text(conn, {"day_budget_usd": 0.005}, "please answer")
    assert called == []                       # gate fired before the fake
    assert _ledger_count(conn) == 1           # gate wrote nothing new


def test_daily_gate_priced_only_over_budget_trips(conn):
    """Pure priced spend (est_usd) over the budget trips the gate — the
    backward-compatible USD path."""
    from brain import llm

    _seed_ledger(conn, tokens_in=10, tokens_out=10, est_usd=2.0)   # priced
    with pytest.raises(llm.LLMUnavailable):
        llm._budget_gate(conn, {"day_budget_usd": 1.5})


def test_daily_gate_mixed_priced_and_unpriced_trips(conn):
    """The defended case (llm.py docstring): a cheap priced SUCCESS plus a pile
    of unpriced FAILURES still trips. The priced row alone is under budget; the
    unpriced token-proxy is what pushes the sum over."""
    from brain import llm

    budget = {"day_budget_usd": 0.01}          # 0.01 * 400_000 = 4000 tokens
    # one small priced success: 0.001 USD, well under 0.01 on its own.
    _seed_ledger(conn, tokens_in=200, tokens_out=200, est_usd=0.001)

    # Control: priced success alone does NOT trip.
    llm._budget_gate(conn, budget)             # must not raise

    # Now add unpriced failures: 5 rows * 1000 tokens_out = 5000 tokens =>
    # proxy 0.0125; total 0.0135 > 0.01.
    _seed_ledger(conn, tokens_out=1000, est_usd=0.0, n=5)
    with pytest.raises(llm.LLMUnavailable):
        llm._budget_gate(conn, budget)


def test_metering_on_failure_eventually_trips_gate(conn):
    """A perpetually-empty provider: each call METERS a proxy row (llm.py:103-
    105) then raises 'empty text'. The accumulating rows eventually make the
    gate itself trip — a wedged provider cannot be re-billed forever."""
    from brain import llm

    # Empty reply => metered, then LLMUnavailable("... empty text").
    llm.set_llm_for_tests(lambda p, *, system=None, max_tokens=1200: "")
    prompt = "x" * 4000                         # ~1000 proxy tokens_in/call
    budget = {"day_budget_usd": 0.005}          # 2000-token threshold

    empty_failures = 0
    budget_engaged = False
    for _ in range(8):
        rows_before = _ledger_count(conn)
        try:
            llm.call_text(conn, budget, prompt)
        except llm.LLMUnavailable as e:
            msg = str(e)
            if "budget" in msg:
                budget_engaged = True
                # gate precedes the meter: no new row this call.
                assert _ledger_count(conn) == rows_before
                break
            # an empty-text failure: metered exactly one row.
            empty_failures += 1
            assert _ledger_count(conn) == rows_before + 1
        else:
            pytest.fail("empty reply should have raised LLMUnavailable")

    assert empty_failures >= 1                  # failures were metered
    assert budget_engaged                       # and eventually tripped the gate
    assert _ledger_count(conn) == empty_failures


# ===========================================================================
# NIGHT budget gate — dream.shift.Shift.budget_left (shift.py:136-147)
# ===========================================================================

def _shift(conn, config):
    from brain.dream.shift import Shift

    return Shift(shift_id="test-shift", conn=conn, config=config)


def test_night_budget_left_true_when_under(conn):
    _seed_ledger(conn, tokens_out=100_000, est_usd=0.0)  # night 0.5 => 200_000
    assert _shift(conn, {"night_budget_usd": 0.5}).budget_left() is True


def test_night_budget_left_false_when_over_and_daily_independent(conn):
    """Over the night token budget => budget_left() False. The SAME rows are a
    token-proxy of only 0.75 USD, well under the 1.5 daily budget — proving the
    two gates are independent (night is token-only, ignores est_usd pricing)."""
    from brain import llm

    _seed_ledger(conn, tokens_out=300_000, est_usd=0.0)  # night 0.5 => 200_000
    assert _shift(conn, {"night_budget_usd": 0.5}).budget_left() is False
    # 300_000 / 400_000 = 0.75 USD proxy < 1.5 daily => daily NOT tripped.
    llm._budget_gate(conn, {"day_budget_usd": 1.5})       # must not raise


def test_daily_priced_trips_but_night_proxy_ok(conn):
    """The other independence direction: a single expensive PRICED row (high
    est_usd, ZERO tokens) trips the daily USD gate but leaves the night
    token-proxy untouched."""
    from brain import llm

    _seed_ledger(conn, tokens_in=0, tokens_out=0, est_usd=100.0)
    with pytest.raises(llm.LLMUnavailable):
        llm._budget_gate(conn, {"day_budget_usd": 1.5})
    assert _shift(conn, {"night_budget_usd": 0.5}).budget_left() is True


def test_contradict_skips_llm_when_night_budget_exhausted(conn):
    """Integration: with the night budget spent, the contradict strategy sees
    budget_left()==False at its pre-call gate (contradict.py:153) and
    increments skipped_llm INSTEAD of calling the LLM.

    We supply the vec plumbing contradict needs to reach that gate (the floor
    tier has no sqlite-vec, so it would otherwise early-return 'no_vec'):
    vec_available -> True and a stubbed _neighbors that returns the real
    opposing row. Everything downstream (polarity conflict, edge check, the
    budget gate) runs for real."""
    from brain import llm
    from brain.dream import contradict as contradict_mod

    # two current-truth semantic memories that polarity-conflict.
    older = seed_memory(conn, "I do not love drinking coffee anymore")
    seed_memory(conn, "I strongly love drinking coffee every morning")

    older_row = conn.execute("SELECT * FROM memories WHERE id=?", (older,)).fetchone()

    def fake_neighbors(c, cand):
        return c.execute(
            "SELECT * FROM memories WHERE id<>? AND memory_type='semantic'"
            " AND status='active' AND live=1 AND valid_to IS NULL ORDER BY id",
            (cand["id"],),
        ).fetchall()

    # spend the night budget: night 0.01 => 4000-token threshold.
    _seed_ledger(conn, tokens_out=5000, est_usd=0.0)

    llm_calls = []
    llm.set_llm_for_tests(
        lambda p, *, system=None, max_tokens=1600:
        llm_calls.append(1) or '{"contradicts": false}')

    shift = _shift(conn, {"night_budget_usd": 0.01, "_forced_mode": "active"})
    shift.embedder = object()               # non-None so the no_vec guard passes
    shift._last_renew = time.monotonic()    # keepalive short-circuits True (30s)

    assert older_row is not None
    with mock.patch.object(contradict_mod.vec_store, "vec_available", lambda c: True), \
            mock.patch.object(contradict_mod, "_neighbors", fake_neighbors):
        counts = contradict_mod._run(shift)

    assert counts.get("skipped_llm", 0) >= 1     # skipped, not called
    assert llm_calls == []                        # the LLM was never invoked


# ===========================================================================
# PER-RUN CAPS — each bounds a single night's work
# ===========================================================================

def test_extract_batch_item_cap(conn):
    """capture/extract._MAX_ITEMS_PER_BATCH=12 (applied at extract.py:520): a
    faked extractor returning 50 valid items has at most 12 CONSIDERED."""
    from brain.capture import extract as extract_mod

    assert extract_mod._MAX_ITEMS_PER_BATCH == 12

    result = [
        {"content": f"Durable fact number {i} worth remembering across sessions",
         "kind": "fact", "about_user": False, "instruction_shaped": False,
         "source_uids": []}
        for i in range(50)
    ]
    ctx = {
        "session_id": "s", "epi_by_uid8": {}, "batch_floor": "owner",
        "principal": None, "single_principal": True, "platform": "cli",
        "aids_max": 0,
    }
    counts = {"batches": 0, "items": 0, "inserted": 0, "merged": 0,
              "quarantined": 0, "skipped_llm": 0}
    # shadow=True: audit-only, no real inserts, but item/inserted counters
    # still increment for each CONSIDERED item.
    extract_mod._apply_items(conn, result, ctx, embedder=None, shadow=True,
                             actor="test", counts=counts)
    assert counts["items"] == 12
    assert counts["inserted"] == 12


def _seed_forget_qualifier(conn, **kw):
    """A memory that qualifies for demotion: no signal, ancient, short
    half-life => value score << 0.15 and age >> 2*half_life."""
    return seed_memory(conn, half_life_days=1.0, valid_from=iso_days_ago(100),
                       **kw)


def test_forget_demote_cap(conn):
    """dream/forget._MAX_DEMOTE=200 (forget.py:56): seed 250 qualifying rows,
    run forget active, assert EXACTLY 200 demoted this run — the rest wait for
    the next run."""
    from brain.dream import forget as forget_mod

    for i in range(250):
        _seed_forget_qualifier(conn, content=f"stale worthless note {i}")

    shift = _shift(conn, {"_forced_mode": "active"})
    shift._last_renew = time.monotonic()
    counts = forget_mod.run(shift)

    assert counts["demoted"] == 200
    summarized = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE status='summarized'").fetchone()[0]
    active = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE status='active'").fetchone()[0]
    assert summarized == 200
    assert active == 50
    # short batch (250 < _MAX_SCORED 5000) => cursor wraps to 0 for next run.
    assert forget_mod._get_cursor(conn) == 0


def test_forget_cursor_rotates_window(conn):
    """The wrapping id cursor (forget.py:105-115) rotates the bounded window:
    a run resumes at id > cursor and does NOT re-read the low-id prefix, then
    wraps to 0 once the tail is reached."""
    from brain.dream import forget as forget_mod

    ids = [_seed_forget_qualifier(conn, content=f"note {i}") for i in range(10)]
    ids.sort()
    cursor = ids[4]                     # pretend a prior run stopped at the 5th row
    forget_mod._set_cursor(conn, cursor)

    shift = _shift(conn, {"_forced_mode": "active"})
    shift._last_renew = time.monotonic()
    forget_mod.run(shift)

    # rows AT/BELOW the cursor were skipped this run: never scored (importance
    # still NULL, still active).
    for low in ids[:5]:
        row = conn.execute(
            "SELECT importance, status FROM memories WHERE id=?", (low,)).fetchone()
        assert row["importance"] is None
        assert row["status"] == "active"
    # rows ABOVE the cursor were scored (importance set) and demoted.
    for high in ids[5:]:
        row = conn.execute(
            "SELECT importance, status FROM memories WHERE id=?", (high,)).fetchone()
        assert row["importance"] is not None
        assert row["status"] == "summarized"
    # tail reached (short batch) => wrap to 0 so the skipped prefix is next.
    assert forget_mod._get_cursor(conn) == 0


def test_consolidate_lesson_word_cap(conn):
    """dream/consolidate._MAX_LESSON_WORDS=140 (consolidate.py:41): a lesson
    over the hard wall is rejected unpersisted; an otherwise-identical short
    lesson passes the gate."""
    from brain.dream import consolidate as consolidate_mod

    assert consolidate_mod._MAX_LESSON_WORDS == 140

    mid = seed_memory(conn, "This decision is about ProjectPhoenix and its rollout.")
    member = conn.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
    members = [member]

    def proposal(word_count):
        return {"content": " ".join(["word"] * word_count), "actionable": True,
                "cites": [member["uid"]], "entity": "ProjectPhoenix"}

    # 200 words > 140 hard wall => rejected (None).
    assert consolidate_mod._validate(conn, proposal(200), members) is None
    # 120 words <= 140 and every other gate satisfied => accepted.
    accepted = consolidate_mod._validate(conn, proposal(120), members)
    assert accepted is not None
    assert accepted["entity"] == "ProjectPhoenix"


def _build_state_db(n_verified: int):
    """A minimal read-alike of Hermes's state.db carrying n_verified terminal
    turns in ONE session (each 'verified' outcome closes its own episode)."""
    state = sqlite3.connect(":memory:")
    state.row_factory = sqlite3.Row
    state.executescript(
        """
        CREATE TABLE turn_outcomes (
            session_id TEXT, turn_id TEXT, created_at REAL, outcome TEXT,
            feedback_kind TEXT, feedback_value TEXT, api_calls INTEGER,
            tool_iterations INTEGER, retry_count INTEGER, cost_usd_delta REAL,
            skills_loaded TEXT, model TEXT);
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
            timestamp REAL, active INTEGER);
        CREATE TABLE sessions (id TEXT, source TEXT, user_id TEXT);
        """
    )
    state.execute("INSERT INTO sessions VALUES ('sess', 'cli', 'owner')")
    for i in range(n_verified):
        state.execute(
            "INSERT INTO turn_outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("sess", f"t{i}", 1000.0 + i * 60.0, "verified", None, None,
             0, 0, 0, 0.0, "[]", "m"),
        )
    state.commit()
    return state


def test_mine_state_episode_cap():
    """dream/mine_state._MAX_EPISODES=60 (mine_state.py:337): 70 terminal turns
    assemble into at most 60 episodes in a single call — one night can't grow
    an unbounded episode list."""
    from brain.dream import mine_state

    assert mine_state._MAX_EPISODES == 60
    state = _build_state_db(70)
    try:
        episodes = mine_state.assemble_episodes(state, since_epoch=0.0)
        assert len(episodes) == 60
    finally:
        state.close()


# ===========================================================================
# GRAPH SQL-variable cap — recall.graph (graph.py:32, _MAX_SQL_VARS=900)
# ===========================================================================

def _bulk_link_entity(conn, entity_name, n_memories):
    """Create n_memories real rows all co-mentioning ONE entity. Returns the
    ordered list of memory ids. Direct inserts (FK-safe: entity + memories
    before mentions)."""
    from brain.store import db

    now = db.iso_now()
    conn.execute(
        "INSERT INTO entities (canonical, display_name, entity_type,"
        " mention_count, created_at) VALUES (?,?,?,0,?)",
        (entity_name, entity_name, "concept", now))
    ent_id = conn.execute(
        "SELECT id FROM entities WHERE canonical=?", (entity_name,)).fetchone()["id"]

    mems = [
        (db.new_ulid(), "observation", "semantic", "fact", "active", 1,
         f"memory about {entity_name} number {i}",
         db.content_hash(f"{entity_name} {i}"), "", "[]", 1, "owner", "test",
         now, now)
        for i in range(n_memories)
    ]
    conn.executemany(
        "INSERT INTO memories (uid, epistemic, memory_type, kind, status, live,"
        " content, content_hash, symbols, tags, token_len, trust_tier,"
        " created_by, valid_from, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        mems)
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM memories WHERE created_by='test' ORDER BY id").fetchall()]
    conn.executemany(
        "INSERT INTO entity_mentions (entity_id, memory_id, scope_project, ts)"
        " VALUES (?,?,?,?)",
        [(ent_id, mid, None, now) for mid in ids])
    conn.commit()
    return ids


def test_graph_ppr_bounded_nonempty(conn):
    """Sanity that the leg actually FUNCTIONS and is bounded: a single seed
    over a small co-mention cluster surfaces the connected non-seed memories,
    capped at `limit`. (Proves the var-cap test below isn't vacuously empty.)"""
    from brain.recall import graph

    ids = _bulk_link_entity(conn, "sharedthing", 6)
    out = graph.ppr_leg(conn, [ids[0]], limit=24)
    assert isinstance(out, list)
    assert 0 < len(out) <= 24
    assert ids[0] not in out                     # seeds are excluded


def test_graph_var_cap_huge_seed_set_holds(conn):
    """graph._MAX_SQL_VARS=900: hand ppr_leg FAR more seeds than SQLite's
    variable limit (32766 here) AND an entity co-mentioned by thousands of
    memories. Without the //2 seed cap, `entities_of`/`_edge_neighbors` would
    bind ~44000 vars and raise 'too many SQL variables', dropping the whole
    graph leg. With the cap it must NOT raise and must stay within the node
    bound."""
    from brain.recall import graph

    assert graph._MAX_SQL_VARS == 900
    ids = _bulk_link_entity(conn, "hotentity", 2000)     # thousands of mentions
    # 2000 real ids + ~42000 fakes: dedup keeps order, _ppr truncates to 450.
    huge = ids + list(range(1_000_000, 1_042_000))
    assert len(huge) > 32766                             # would overflow uncapped

    # _ppr is the UN-wrapped path: it does NOT swallow exceptions, so a blown
    # variable binder would surface here as a raise (ppr_leg would hide it).
    out = graph._ppr(conn, huge, 24, 400, 60)
    assert isinstance(out, list)
    assert len(out) <= 24

    # the public wrapped entry is likewise bounded and non-raising.
    wrapped = graph.ppr_leg(conn, huge, limit=24)
    assert isinstance(wrapped, list)
    assert len(wrapped) <= 24
