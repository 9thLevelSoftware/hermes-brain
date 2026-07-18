"""Phase 5 (pytest half) — concurrency, lease ownership & cooperative preemption.

Asserts the load-bearing CONCURRENCY invariants HOLD (this is the "break it the
proper way" phase — the brain must NOT double-process or leak a lease):

  * exactly-once lease ownership: N threads racing ``lease.acquire`` -> EXACTLY
    one winner; the losers get no authority (their ``renew`` fails);
  * renew-authority: only the current holder can ``renew``; a holder that lost
    the lease (stolen after TTL expiry) keeps NO authority to write;
  * disjoint buffer claims: two concurrent sweepers over one brain.db never
    claim the same ``ingest_buffer`` row (WAL serializes; the loser's rows are
    already tagged) — no row is processed twice, none is lost;
  * cooperative preemption: a returning user (fresh ``activity``) latches the
    shift preempted; a lost lease makes ``keepalive``/``tick`` yield.

Seams (verified against the source, line context noted inline):
  dream/lease.py      — acquire/renew/release/held_by (atomic UPDATE mutex)
  dream/run.py:144-208 — run_dream lease acquire -> skip; renew fail -> abort
  capture/extract.py:279-306 — _claim (atomic UPDATE over LIMIT subquery)
  dream/shift.py:90-132 — Shift.preempted/keepalive/tick

In-process (threads + direct calls) counterpart to the Docker multi-PROCESS
phase (two ``hermes brain dream-now`` + SIGKILL). Fast: no real TTL waits —
timestamps are rewritten in the DB. Each raced ``fn`` opens its OWN connection
(never share a sqlite3 conn across threads — see faults.race).
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from conftest import iso_days_ago
from faults import race

# Lexically-comparable ISO sentinels (timestamps are strings throughout the
# brain). Far past => "expired/free"; far future => "still live".
PAST = "2000-01-01T00:00:00.000Z"
FUTURE = "2999-01-01T00:00:00.000Z"


def _fresh_schema(tmp_home):
    """Materialize brain.db once (schema + seeded brain_lease row) so the raced
    per-thread connections open an already-created DB, not race the CREATEs."""
    from brain.store import db

    c = db.connect(tmp_home)
    c.close()


# ===========================================================================
# 1. Lease: exactly-once ownership under a thread race (the headline invariant)
# ===========================================================================

def test_acquire_race_exactly_one_winner_two_threads(tmp_home):
    from brain.dream import lease
    from brain.store import db

    _fresh_schema(tmp_home)

    def _try(i):
        c = db.connect(tmp_home)
        try:
            return lease.acquire(c, "dream", holder=f"holder-{i}")
        finally:
            c.close()

    results = race(_try, n=2)
    assert results.count(True) == 1, results  # EXACTLY one owner
    assert results.count(False) == 1, results

    # The winner is the recorded holder; the loser owns nothing.
    winner = results.index(True)
    check = db.connect(tmp_home)
    try:
        assert lease.held_by(check, "dream") == f"holder-{winner}"
    finally:
        check.close()


def test_acquire_race_exactly_one_winner_five_threads(tmp_home):
    from brain.dream import lease
    from brain.store import db

    _fresh_schema(tmp_home)

    def _try(i):
        c = db.connect(tmp_home)
        try:
            return lease.acquire(c, "dream", holder=f"holder-{i}")
        finally:
            c.close()

    results = race(_try, n=5)
    # Five processes racing one brain.db -> still exactly one holder, four denied.
    assert results.count(True) == 1, results
    assert results.count(False) == 4, results


def test_concurrent_losers_cannot_renew_only_the_winner(tmp_home):
    """The four losers of a 5-way race hold no authority: their renew() fails,
    only the winner's succeeds (no split-brain writers)."""
    from brain.dream import lease
    from brain.store import db

    _fresh_schema(tmp_home)

    def _try(i):
        c = db.connect(tmp_home)
        try:
            return lease.acquire(c, "dream", holder=f"holder-{i}")
        finally:
            c.close()

    results = race(_try, n=5)
    winner = results.index(True)

    check = db.connect(tmp_home)
    try:
        for i in range(5):
            got = lease.renew(check, "dream", f"holder-{i}")
            assert got is (i == winner), (i, winner, got)
    finally:
        check.close()


# ===========================================================================
# 2. Lease: renew-authority, steal-after-expiry, idempotence, held_by
# ===========================================================================

def test_renew_authority_only_for_current_holder(conn):
    from brain.dream import lease

    assert lease.acquire(conn, "dream", "A") is True
    assert lease.renew(conn, "dream", "A") is True     # the holder may extend
    assert lease.renew(conn, "dream", "B") is False    # a stranger may not


def test_steal_after_expiry_revokes_prior_holder_authority(conn):
    """A held-then-expired lease can be stolen; the prior holder keeps NO
    authority (review findings #3/#6 — 'a holder that lost the lease keeps no
    authority to write')."""
    from brain.dream import lease

    assert lease.acquire(conn, "dream", "A") is True
    # Rewind A's TTL into the past -> the row is now free/expired.
    conn.execute("UPDATE brain_lease SET expires_at=? WHERE name='dream'", (PAST,))
    conn.commit()

    assert lease.acquire(conn, "dream", "B") is True   # B legitimately steals it
    assert lease.held_by(conn, "dream") == "B"
    assert lease.renew(conn, "dream", "A") is False     # A is revoked
    assert lease.renew(conn, "dream", "B") is True      # B is the true owner


def test_idempotent_reacquire_by_same_holder(conn):
    from brain.dream import lease

    assert lease.acquire(conn, "dream", "A") is True
    assert lease.acquire(conn, "dream", "A") is True    # idempotent re-acquire
    assert lease.held_by(conn, "dream") == "A"          # not a second holder


def test_acquire_denied_while_live_lease_held(conn):
    from brain.dream import lease

    assert lease.acquire(conn, "dream", "A") is True
    assert lease.acquire(conn, "dream", "B") is False   # cannot take a live lease
    assert lease.held_by(conn, "dream") == "A"


def test_held_by_free_live_and_expired(conn):
    from brain.dream import lease

    assert lease.held_by(conn, "dream") is None         # seeded free
    assert lease.acquire(conn, "dream", "A") is True
    assert lease.held_by(conn, "dream") == "A"          # live holder
    conn.execute("UPDATE brain_lease SET expires_at=? WHERE name='dream'", (PAST,))
    conn.commit()
    assert lease.held_by(conn, "dream") is None          # expired == free


def test_release_only_by_holder(conn):
    from brain.dream import lease

    assert lease.acquire(conn, "dream", "A") is True
    lease.release(conn, "dream", "B")                   # not the holder -> no-op
    assert lease.held_by(conn, "dream") == "A"
    lease.release(conn, "dream", "A")                   # holder frees it
    assert lease.held_by(conn, "dream") is None
    assert lease.acquire(conn, "dream", "B") is True     # now available


# ===========================================================================
# 3. run_dream: loser skips; a lost lease aborts the pipeline
# ===========================================================================

def test_run_dream_skips_when_lease_already_held(tmp_home):
    """A second dream over one brain.db while the lease is live must not run —
    it reports lease_held and names the holder (run.py:144-147)."""
    from brain.dream import lease, run
    from brain.store import db
    from brain import llm

    llm.set_llm_for_tests(lambda *a, **k: "[]")
    holder_conn = db.connect(tmp_home)
    main = db.connect(tmp_home)
    try:
        # A rival holds the live 'dream' lease under a holder tag run_dream
        # (which uses actor:pid@host) can never match.
        assert lease.acquire(holder_conn, "dream", "external-holder") is True
        assert lease.held_by(holder_conn, "dream") == "external-holder"

        summary = run.run_dream(main, {})
        assert summary.get("skipped") == "lease_held", summary
        assert summary.get("holder") == "external-holder", summary
        # The intruder did not disturb the real holder.
        assert lease.held_by(main, "dream") == "external-holder"
    finally:
        llm.set_llm_for_tests(None)
        holder_conn.close()
        main.close()


def test_run_dream_aborts_when_lease_lost_midpipeline(tmp_home):
    """If the lease is stolen mid-run (e.g. another process took over after a
    TTL lapse), run_dream's post-strategy renew fails and it aborts with
    lease_lost (run.py:192-195) rather than continuing to mutate memory — and
    it must NOT release the new owner's lease (release is holder-scoped)."""
    from brain.dream import run
    from brain.capture import extract
    from brain.store import db
    from brain import llm

    llm.set_llm_for_tests(lambda *a, **k: "[]")
    main = db.connect(tmp_home)
    stolen = {"done": False}

    def _thief_sweep(conn, config, **kw):
        # First strategy ('flush') runs; while it does, a rival legitimately
        # seizes the lease. Overwrite the holder so run_dream's next renew fails.
        if not stolen["done"]:
            conn.execute(
                "UPDATE brain_lease SET holder='thief', expires_at=? WHERE name='dream'",
                (FUTURE,),
            )
            conn.commit()
            stolen["done"] = True
        return {"batches": 0}

    try:
        with mock.patch.object(extract, "sweep", _thief_sweep):
            summary = run.run_dream(main, {})
        assert stolen["done"], "flush strategy never ran; test premise broken"
        assert summary.get("aborted") == "lease_lost", summary
        # run_dream must not have released the thief's lease in its finally.
        from brain.dream import lease
        assert lease.held_by(main, "dream") == "thief"
    finally:
        llm.set_llm_for_tests(None)
        main.close()


# ===========================================================================
# 4. capture/extract _claim: disjoint buffer claims (no double-processing)
# ===========================================================================

def _seed_buffer(conn, n, *, session="s1"):
    """Seed n pending (unpromoted, unclaimed) turn rows; return their ids."""
    from brain.store import db

    now = db.iso_now()
    payload = json.dumps({"user": "hello", "assistant": "hi there"})
    for _ in range(n):
        conn.execute(
            "INSERT INTO ingest_buffer (kind, session_id, episode_id, payload, ts,"
            " claimed_by, promoted_at) VALUES ('turn', ?, NULL, ?, ?, NULL, NULL)",
            (session, payload, now),
        )
    conn.commit()
    return [r["id"] for r in conn.execute(
        "SELECT id FROM ingest_buffer WHERE promoted_at IS NULL ORDER BY id"
    ).fetchall()]


def test_concurrent_claims_are_disjoint_full_drain(tmp_home):
    from brain.capture import extract
    from brain.store import db

    seed = db.connect(tmp_home)
    ids = _seed_buffer(seed, 10)
    seed.close()

    def _claim(i):
        c = db.connect(tmp_home)
        try:
            rows = extract._claim(c, f"actor-{i}", 100)
            return [r["id"] for r in rows]
        finally:
            c.close()

    a, b = race(_claim, n=2)
    # No row claimed by both actors (disjoint), none lost, none doubled:
    # WAL serializes the two UPDATEs, so one sweeper drains all pending rows and
    # the other sees them already tagged.
    assert set(a).isdisjoint(set(b)), (a, b)
    assert sorted(a + b) == sorted(ids), (a, b, ids)


def test_concurrent_claims_are_disjoint_with_limit(tmp_home):
    """With max_rows < pending, each sweeper still claims a DISJOINT slice
    (the LIMIT subquery re-checks the predicate under the write lock)."""
    from brain.capture import extract
    from brain.store import db

    seed = db.connect(tmp_home)
    ids = _seed_buffer(seed, 8)
    seed.close()

    def _claim(i):
        c = db.connect(tmp_home)
        try:
            rows = extract._claim(c, f"actor-{i}", 3)
            return [r["id"] for r in rows]
        finally:
            c.close()

    a, b = race(_claim, n=2)
    assert len(a) == 3 and len(b) == 3, (a, b)
    assert set(a).isdisjoint(set(b)), (a, b)
    assert set(a + b).issubset(set(ids))


def test_claim_reclaims_stale_row(conn):
    """A row claimed >_CLAIM_STALE_MINUTES ago (crashed worker) is reclaimable —
    the queue never wedges on a dead claim."""
    from brain.capture import extract
    from brain.store import db

    stale_ts = iso_days_ago(20 / 1440.0)  # ~20 minutes ago (window is 10)
    conn.execute(
        "INSERT INTO ingest_buffer (kind, session_id, payload, ts, claimed_by,"
        " promoted_at) VALUES ('turn', 's', ?, ?, ?, NULL)",
        (json.dumps({"user": "u", "assistant": "a"}), db.iso_now(),
         f"deadworker#abcdef@{stale_ts}"),
    )
    conn.commit()

    rows = extract._claim(conn, "fresh", 10)
    assert [r["id"] for r in rows], "stale claim was not reclaimed"


def test_claim_skips_freshly_claimed_and_promoted_rows(conn):
    """A row another actor claimed moments ago is NOT stolen, and a promoted row
    is never re-claimed (no double-processing)."""
    from brain.capture import extract
    from brain.store import db

    now = db.iso_now()
    payload = json.dumps({"user": "u", "assistant": "a"})
    # Fresh claim by someone else (not stale).
    conn.execute(
        "INSERT INTO ingest_buffer (kind, session_id, payload, ts, claimed_by,"
        " promoted_at) VALUES ('turn', 's', ?, ?, ?, NULL)",
        (payload, now, f"other#zzz@{now}"),
    )
    # Already promoted.
    conn.execute(
        "INSERT INTO ingest_buffer (kind, session_id, payload, ts, claimed_by,"
        " promoted_at) VALUES ('turn', 's', ?, ?, NULL, ?)",
        (payload, now, now),
    )
    conn.commit()

    rows = extract._claim(conn, "me", 10)
    assert rows == [], [dict(r) for r in rows]


# ===========================================================================
# 5. Shift: cooperative preemption (a returning user / lost lease yields)
# ===========================================================================

def _shift(conn, **kw):
    from brain.dream.shift import Shift

    kw.setdefault("shift_id", "s1")
    kw.setdefault("config", {})
    return Shift(conn=conn, **kw)


def test_preempted_false_when_no_new_activity(conn):
    from brain.store import db

    shift = _shift(conn, activity_baseline=FUTURE)
    db.touch_activity(conn, "provider:x")   # activity older than the future baseline
    assert shift.preempted() is False


def test_preempted_true_when_user_returns(conn):
    from brain.store import db

    shift = _shift(conn, activity_baseline=PAST)
    db.touch_activity(conn, "provider:x")   # last_seen (now) > PAST baseline
    assert shift.preempted() is True


def test_preempted_latches(conn):
    """Once a returning user preempts a shift, it stays preempted — the shift
    must not resume just because activity later looks quiet."""
    from brain.store import db

    shift = _shift(conn, activity_baseline=PAST)
    db.touch_activity(conn, "provider:x")
    assert shift.preempted() is True
    # Move the goalposts so the raw comparison would now be False; the latch wins.
    shift.activity_baseline = FUTURE
    assert shift.preempted() is True


def test_keepalive_and_tick_yield_when_lease_lost(conn):
    """keepalive/tick renew the lease while it is ours and return False the
    moment it is lost (shift.py:104-132) — the strategy must stop mutating."""
    from brain.dream import lease

    holder = "shiftA"
    assert lease.acquire(conn, "dream", holder) is True
    shift = _shift(conn, holder=holder, activity_baseline=FUTURE)

    assert shift.keepalive() is True        # first call renews; still ours
    assert shift.tick() is True             # lease held, no preemption

    # A rival seizes the lease.
    conn.execute("UPDATE brain_lease SET holder='thief', expires_at=? WHERE name='dream'",
                 (FUTURE,))
    conn.commit()
    shift._last_renew = 0.0                  # defeat the RENEW_SECONDS throttle

    assert shift.keepalive() is False       # lease lost -> yield
    assert shift.preempted() is True        # keepalive latched preemption
    assert shift.tick() is False            # and tick yields too
