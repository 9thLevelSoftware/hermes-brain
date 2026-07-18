"""Phase 6 (pytest half) — crash / corruption / migration / non-destruction.

Asserts the load-bearing RECOVERY & NON-DESTRUCTION invariants HOLD:

  * FUTURE SCHEMA: a brain.db stamped with a schema_version newer than this
    code is REFUSED (never opened blind) with a teaching message.
  * SELF-HEAL: a brain.db whose fresh-create was interrupted (the
    schema_version meta row never landed) re-heals on the next open instead
    of crashing — all DDL is IF NOT EXISTS.
  * MIGRATE-WITH-BACKUP: a v1 -> v2 upgrade runs the numbered migration,
    recreates the added table, bumps the version, AND leaves a
    `brain.pre-v2.<ts>.db` VACUUM backup — the pre-migration state stays
    recoverable (nothing destroyed).
  * SUPERSEDE-DON'T-DELETE: a confident contradiction verdict closes the
    LOSER bi-temporally (valid_to + invalidated_by) so it drops from
    current-truth recall yet stays queryable by uid — the row is never
    DELETEd. Low-confidence / 'neither' verdicts invalidate NOTHING and only
    flag needs_review.

Seams: store/db.py (connect/_ensure_schema/_migrate/_create_fresh,
FutureSchemaError, SCHEMA_VERSION) and dream/contradict.py
(_apply_verdict + the real run() pair pipeline).

The SIGKILL-mid-dream and truncated-file cases are the Docker phase's job;
here we drive the same logic in-process.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from brain.store import db
from conftest import seed_memory

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _brain_dir(tmp_home) -> Path:
    return Path(tmp_home) / "brain"


def _backups(tmp_home) -> list[Path]:
    return sorted(_brain_dir(tmp_home).glob("brain.pre-v*.db"))


def _make_v1_lookalike(tmp_home) -> None:
    """Open a fresh (v2) brain.db, then rewrite it to LOOK like a v1 database:
    stamp schema_version='1' and drop the v2-only `proposals` table. The next
    open must migrate it forward."""
    conn = db.connect(tmp_home)
    try:
        conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
        conn.execute("DROP TABLE IF EXISTS proposals")
        conn.commit()
    finally:
        conn.close()


def _table_exists(conn, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


# ===========================================================================
# 1. FUTURE SCHEMA — refuse a database newer than this code
# ===========================================================================

def test_future_schema_is_refused(tmp_home):
    """A brain.db stamped with a version this code does not understand must be
    REFUSED — opening it blind risks writing rows a newer schema forbids."""
    conn = db.connect(tmp_home)
    conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(db.FutureSchemaError):
        db.connect(tmp_home)


def test_future_schema_error_teaches_the_remedy(tmp_home):
    """The refusal is not a bare exception: it tells the operator what to do
    (update the plugin) and names both versions."""
    conn = db.connect(tmp_home)
    conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(db.FutureSchemaError) as exc:
        db.connect(tmp_home)
    msg = str(exc.value).lower()
    assert "v99" in msg
    assert f"v{db.SCHEMA_VERSION}" in msg
    assert "update the plugin" in msg


def test_future_schema_is_never_silently_downgraded(tmp_home):
    """Refusing must not mutate the file: the version stays 99, so a later
    (updated) plugin can still open it. A blind 'downgrade' would corrupt."""
    conn = db.connect(tmp_home)
    conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(db.FutureSchemaError):
        db.connect(tmp_home)

    # Re-read the raw file directly — the refused open must not have touched it.
    raw = sqlite3.connect(str(db.db_path(tmp_home)))
    try:
        row = raw.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        assert row is not None and row[0] == "99"
    finally:
        raw.close()


# ===========================================================================
# 2. SELF-HEAL — an interrupted fresh-create re-heals on next open
# ===========================================================================

def test_self_heal_restores_missing_schema_version_row(tmp_home):
    """Crash between executescript and the meta seed leaves the meta table but
    no schema_version row. Re-opening must NOT crash: all DDL is IF NOT EXISTS,
    so _create_fresh re-runs and the schema_version row is restored."""
    conn = db.connect(tmp_home)
    conn.execute("DELETE FROM meta WHERE key='schema_version'")
    conn.commit()
    conn.close()

    healed = db.connect(tmp_home)  # must not raise
    try:
        assert db.get_meta(healed, "schema_version") == str(db.SCHEMA_VERSION)
    finally:
        healed.close()


def test_self_heal_preserves_existing_rows(tmp_home):
    """Self-heal must be NON-destructive: re-running the fresh-create over a
    partially-initialized DB keeps user rows (tables are IF NOT EXISTS, never
    dropped)."""
    conn = db.connect(tmp_home)
    rid = seed_memory(conn, "a durable fact about the neptune staging host", kind="fact")
    conn.execute("DELETE FROM meta WHERE key='schema_version'")
    conn.commit()
    conn.close()

    healed = db.connect(tmp_home)
    try:
        assert db.get_meta(healed, "schema_version") == str(db.SCHEMA_VERSION)
        row = healed.execute(
            "SELECT content FROM memories WHERE id=?", (rid,)).fetchone()
        assert row is not None and "neptune" in row["content"]
    finally:
        healed.close()


# ===========================================================================
# 3. MIGRATION + VACUUM BACKUP — forward migrate, and back up first
# ===========================================================================

def test_002_proposals_migration_file_is_present(tmp_home):
    """The install-completeness guard depends on the numbered migration file
    actually shipping; a missing file would make _migrate raise 'reinstall'."""
    assert (Path(db._MIGRATIONS_DIR) / "002_proposals.sql").exists()


def test_migration_recreates_proposals_and_bumps_version(tmp_home):
    """Opening a v1-lookalike migrates it: the v2-only `proposals` table is
    recreated by migrations/002_proposals.sql and schema_version bumps to 2."""
    _make_v1_lookalike(tmp_home)

    migrated = db.connect(tmp_home)
    try:
        assert _table_exists(migrated, "proposals"), "migration must recreate proposals"
        assert db.get_meta(migrated, "schema_version") == str(db.SCHEMA_VERSION)
        # the recreated table is usable (schema, not just name)
        migrated.execute(
            "INSERT INTO proposals (uid, kind, title, created_at) VALUES (?,?,?,?)",
            (db.new_ulid(), "skill_draft", "smoke", db.iso_now()),
        )
        migrated.commit()
    finally:
        migrated.close()


def test_migration_leaves_a_vacuum_backup(tmp_home):
    """A migration must back up the pre-migration file first — a
    `brain.pre-v2.<ts>.db` snapshot lands in the brain dir."""
    assert _backups(tmp_home) == []  # nothing before
    _make_v1_lookalike(tmp_home)

    migrated = db.connect(tmp_home)
    try:
        backups = _backups(tmp_home)
        assert backups, "migration must leave a brain.pre-v2.*.db backup"
        assert all(b.stat().st_size > 0 for b in backups)
    finally:
        migrated.close()


def test_backup_captures_the_pre_migration_state(tmp_home):
    """The VACUUM backup is a faithful, recoverable snapshot of the DB BEFORE
    migration: it is a valid sqlite file, still stamped v1, still lacking the
    proposals table, and it preserves user rows. This is the non-destruction
    proof — the old state is never lost."""
    # seed a row that must survive into the backup
    conn = db.connect(tmp_home)
    seed_memory(conn, "a fact recorded under schema v1 about mercury", kind="fact")
    conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    conn.execute("DROP TABLE IF EXISTS proposals")
    conn.commit()
    conn.close()

    migrated = db.connect(tmp_home)  # runs _migrate -> VACUUM INTO backup
    migrated.close()

    backups = _backups(tmp_home)
    assert len(backups) == 1
    snap = sqlite3.connect(str(backups[0]))
    snap.row_factory = sqlite3.Row
    try:
        ver = snap.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        assert ver is not None and ver["value"] == "1", "backup must be the v1 state"
        assert snap.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='proposals'"
        ).fetchone() is None, "the pre-migration snapshot predates proposals"
        row = snap.execute(
            "SELECT content FROM memories WHERE content LIKE '%mercury%'").fetchone()
        assert row is not None, "backup must preserve pre-migration rows"
    finally:
        snap.close()


def test_no_backup_and_no_migration_on_a_current_db(tmp_home):
    """A DB already at the current version must NOT re-migrate or spew backups
    on every open — migrations are forward-only and run once."""
    db.connect(tmp_home).close()          # fresh create at SCHEMA_VERSION
    assert _backups(tmp_home) == []
    db.connect(tmp_home).close()          # reopen: current == SCHEMA_VERSION
    assert _backups(tmp_home) == [], "an up-to-date DB must not create a backup"


def test_migration_is_idempotent_second_open_is_clean(tmp_home):
    """After a v1 -> v2 migration, reopening (now v2) neither re-runs the
    migration nor creates a second backup."""
    _make_v1_lookalike(tmp_home)
    db.connect(tmp_home).close()          # migrates -> exactly one backup
    first = _backups(tmp_home)
    assert len(first) == 1

    reopened = db.connect(tmp_home)       # already v2: no-op
    try:
        assert db.get_meta(reopened, "schema_version") == str(db.SCHEMA_VERSION)
        assert _backups(tmp_home) == first, "a second open must not re-migrate"
    finally:
        reopened.close()


# ===========================================================================
# 4/5. SUPERSEDE-DON'T-DELETE — contradiction never destroys the loser
# ===========================================================================
#
# Two ways of driving the verdict are used, both exercising the REAL
# dream/contradict.py code:
#   * _apply_verdict(...) directly with a crafted verdict dict — the exact
#     invalidation SQL (bi-temporal close / needs_review flag / winner
#     coercion), independent of the vec tier. This is the deterministic
#     headline path.
#   * the full run() pipeline with a fake LLM (below) — pair discovery ->
#     polarity pre-filter -> LLM verdict -> _apply_verdict, end-to-end.

def _make_shift(conn, tmp_home, **overrides):
    from brain.dream.shift import Shift

    kw = dict(
        shift_id="s", conn=conn,
        config={"hermes_home": str(tmp_home), "_forced_mode": "active",
                "day_budget_usd": 100.0, "night_budget_usd": 100.0},
        started_at=db.iso_now(),
        activity_baseline="9999-12-31T00:00:00.000Z",  # never "preempted"
        holder="t",
    )
    kw.update(overrides)
    return Shift(**kw)


def _row(conn, rowid):
    return conn.execute("SELECT * FROM memories WHERE id=?", (rowid,)).fetchone()


def _apply(conn, tmp_home, verdict, newer_id, older_id):
    """Drive the real dream.contradict._apply_verdict with a crafted verdict."""
    from brain.dream import contradict

    shift = _make_shift(conn, tmp_home)
    counts = {"contradictions": 0, "invalidated": 0, "flagged": 0}
    contradict._apply_verdict(
        shift, verdict, _row(conn, newer_id), _row(conn, older_id),
        active=True, mode="active", counts=counts)
    return counts


def _search_uids(conn, query):
    from brain.recall.search import search

    hits = search(conn, query, limit=10, principal_id="owner", trust_tier="owner")
    return {h.uid for h in hits}


def _seed_pair(conn):
    """Two directly-contradictory current-truth semantic memories in ONE scope.
    Returns (older_id, newer_id, older_uid, newer_uid)."""
    older_id = seed_memory(conn, "the staging database is postgres 14", kind="fact")
    newer_id = seed_memory(conn, "the staging database is mysql 8", kind="fact")
    older_uid = _row(conn, older_id)["uid"]
    newer_uid = _row(conn, newer_id)["uid"]
    return older_id, newer_id, older_uid, newer_uid


def test_contradiction_supersedes_loser_it_never_deletes(conn, tmp_home):
    """HEADLINE: a confident verdict (winner=a) closes the LOSER bi-temporally
    — valid_to + invalidated_by set — so it drops from current-truth recall,
    yet the row STAYS fetchable by uid and NOTHING is deleted."""
    older_id, newer_id, older_uid, newer_uid = _seed_pair(conn)
    before_count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    assert {older_uid, newer_uid} <= _search_uids(conn, "staging database")

    # winner 'a' == the NEWER record (mysql); the older (postgres) loses.
    counts = _apply(conn, tmp_home,
                    {"contradicts": True, "winner": "a", "why": "db was migrated"},
                    newer_id, older_id)
    assert counts["invalidated"] == 1
    assert counts["flagged"] == 0

    loser = _row(conn, older_id)
    assert loser["valid_to"] is not None, "loser must be closed, not deleted"
    assert loser["invalidated_by"] == newer_id, "loser points at the winner"

    # drops from current-truth recall ...
    live = _search_uids(conn, "staging database")
    assert older_uid not in live
    assert newer_uid in live
    # ... but is STILL queryable by uid, and no row was removed.
    assert conn.execute(
        "SELECT 1 FROM memories WHERE uid=?", (older_uid,)).fetchone() is not None
    assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == before_count


def test_contradiction_winner_b_invalidates_the_newer_row(conn, tmp_home):
    """The loser is whichever the verdict rejects: winner='b' (the OLDER
    record stands) closes the NEWER one. Still supersede, never delete."""
    older_id, newer_id, older_uid, newer_uid = _seed_pair(conn)

    counts = _apply(conn, tmp_home,
                    {"contradicts": True, "winner": "b", "why": "older was right"},
                    newer_id, older_id)
    assert counts["invalidated"] == 1

    newer = _row(conn, newer_id)
    assert newer["valid_to"] is not None
    assert newer["invalidated_by"] == older_id
    # the older row remains current truth, unclosed
    assert _row(conn, older_id)["valid_to"] is None
    live = _search_uids(conn, "staging database")
    assert newer_uid not in live and older_uid in live


def test_low_confidence_neither_flags_review_and_invalidates_nothing(conn, tmp_home):
    """'neither' is the uncertainty answer: BOTH rows are flagged needs_review
    and NEITHER is invalidated — no belief is superseded on a coin-flip."""
    older_id, newer_id, older_uid, newer_uid = _seed_pair(conn)

    counts = _apply(conn, tmp_home,
                    {"contradicts": True, "winner": "neither", "why": "unsure"},
                    newer_id, older_id)
    assert counts["flagged"] == 1
    assert counts["invalidated"] == 0

    for rid in (older_id, newer_id):
        r = _row(conn, rid)
        assert r["needs_review"] == 1, "both rows flagged for review"
        assert r["valid_to"] is None, "uncertainty must not close a row"
        assert r["invalidated_by"] is None
    # both remain current truth
    assert {older_uid, newer_uid} <= _search_uids(conn, "staging database")


def test_unrecognized_winner_is_coerced_to_neither(conn, tmp_home):
    """A verdict whose winner is neither 'a' nor 'b' (a malformed/low-confidence
    label) is coerced to 'neither' (confidence gate, contradict.py ~260) — it
    flags, never invalidates. Defends against an LLM emitting garbage."""
    older_id, newer_id, _older_uid, _newer_uid = _seed_pair(conn)

    counts = _apply(conn, tmp_home,
                    {"contradicts": True, "winner": "postgres!!", "why": "??"},
                    newer_id, older_id)
    assert counts["flagged"] == 1
    assert counts["invalidated"] == 0
    assert _row(conn, older_id)["valid_to"] is None
    assert _row(conn, newer_id)["valid_to"] is None
    assert _row(conn, older_id)["needs_review"] == 1
    assert _row(conn, newer_id)["needs_review"] == 1


def test_non_contradiction_verdict_is_a_total_noop(conn, tmp_home):
    """contradicts=false: no invalidation, no review flag, no conflicts_with
    edge. A non-contradiction must leave the store exactly as it found it."""
    older_id, newer_id, older_uid, newer_uid = _seed_pair(conn)

    counts = _apply(conn, tmp_home,
                    {"contradicts": False, "winner": "a", "why": "different subjects"},
                    newer_id, older_id)
    assert counts == {"contradictions": 0, "invalidated": 0, "flagged": 0}

    for rid in (older_id, newer_id):
        r = _row(conn, rid)
        assert r["valid_to"] is None
        assert r["invalidated_by"] is None
        assert r["needs_review"] == 0
    assert conn.execute(
        "SELECT count(*) FROM edges WHERE edge_type='conflicts_with'"
    ).fetchone()[0] == 0
    assert {older_uid, newer_uid} <= _search_uids(conn, "staging database")


def test_invalidated_loser_survives_and_is_point_in_time_queryable(conn, tmp_home):
    """After invalidation the loser's CONTENT is intact (not nulled) and its
    bi-temporal envelope (valid_from .. valid_to) is well-formed — the record
    is history, fully preserved, not a tombstone."""
    older_id, newer_id, older_uid, _newer_uid = _seed_pair(conn)
    original = _row(conn, older_id)["content"]

    _apply(conn, tmp_home,
           {"contradicts": True, "winner": "a", "why": "changed"},
           newer_id, older_id)

    loser = _row(conn, older_id)
    assert loser["content"] == original, "content must be preserved, never nulled"
    assert loser["valid_from"] is not None
    assert loser["valid_from"] <= loser["valid_to"]
    # a conflicts_with edge records the relationship (audit trail)
    assert conn.execute(
        "SELECT 1 FROM edges WHERE edge_type='conflicts_with'"
        " AND src_id=? AND dst_id=?", (newer_id, older_id)).fetchone() is not None


# ---------------------------------------------------------------------------
# End-to-end through the REAL run() pipeline (fake LLM drives the verdict).
# Skips on a tier without sqlite-vec (the pair discovery needs the vec leg);
# the Docker full-tier phase covers it there too.
# ---------------------------------------------------------------------------

def test_contradict_run_end_to_end_supersedes_loser(conn, tmp_home):
    """Drive the WHOLE strategy: seed two polarity-conflicting current-truth
    memories with near-identical vectors (so the vec-KNN pair discovery finds
    them), install a fake LLM returning a confident verdict, acquire the dream
    lease, and run(). The loser must be superseded (not deleted) end-to-end."""
    from brain import llm
    from brain.dream import contradict, lease
    from brain.store import vec

    if not vec.ensure_tables(conn, 256, "probe"):
        pytest.skip("sqlite-vec unavailable on this tier — covered by the Docker phase")

    # Polarity-conflicting pair (shares >= 2 substantive tokens; exactly one
    # carries a negation) so contradict._polarity_conflict lets it reach the LLM.
    older_id = seed_memory(
        conn, "the production database accepts external network connections", kind="fact")
    newer_id = seed_memory(
        conn, "the production database does not accept external network connections",
        kind="fact")
    # near-identical unit vectors -> cosine ~1.0 >= 0.82 -> KNN neighbors
    vec.upsert(conn, "mem_vec", older_id, [0.5] * 256)
    vec.upsert(conn, "mem_vec", newer_id, [0.5] * 256)
    conn.commit()

    older_uid = _row(conn, older_id)["uid"]
    newer_uid = _row(conn, newer_id)["uid"]
    before_count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]

    # winner 'a' == the NEWER record; the older ('accepts') is the loser.
    verdict_json = '{"contradicts": true, "winner": "a", "why": "policy changed"}'
    llm.set_llm_for_tests(
        lambda prompt, *, system=None, max_tokens=1600: verdict_json)
    try:
        assert lease.acquire(conn, "dream", "t")
        shift = _make_shift(conn, tmp_home, embedder=object())
        counts = contradict.run(shift)
    finally:
        llm.set_llm_for_tests(None)

    assert "error" not in counts, counts
    assert counts.get("invalidated") == 1, counts

    loser = _row(conn, older_id)
    assert loser["valid_to"] is not None
    assert loser["invalidated_by"] == newer_id
    # still queryable by uid; nothing deleted
    assert _row(conn, older_id) is not None
    assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == before_count
    # a conflicts_with edge was recorded, newer -> older
    assert conn.execute(
        "SELECT 1 FROM edges WHERE edge_type='conflicts_with'"
        " AND src_id=? AND dst_id=?", (newer_id, older_id)).fetchone() is not None
    # drops from current-truth recall; winner remains
    live = _search_uids(conn, "production database")
    assert older_uid not in live
    assert newer_uid in live
