"""The full 10-strategy dream pipeline runs clean, in order.

A structural guard: every strategy in PIPELINE executes, none raises, the
default modes match the active-by-default contract (2026-07-17), the global
--dry-run override still neutralizes every mutation (the rollback path), and
tune never auto-applies a retrieval-weight change.
"""

from __future__ import annotations

import time

from brain.dream import run_dream
from brain.dream.shift import DEFAULT_MODES, PIPELINE
from brain.store import db

from tests.test_episode_assembly import make_state_db


def test_full_pipeline_runs_every_strategy_without_error(tmp_home):
    now = time.time()
    make_state_db(
        tmp_home,
        messages=[("s1", "user", "deploy the api to staging", now - 300),
                  ("s1", "assistant", "deployed, smoke tests green", now - 290)],
        outcomes=[("s1", "t-1", now - 280, "verified", None, None)],
    )
    conn = db.connect(tmp_home)
    try:
        summary = run_dream(conn, {"hermes_home": str(tmp_home)}, actor="e2e")
    finally:
        conn.close()

    assert "error" not in summary, summary
    strategies = summary["strategies"]
    # Every pipeline stage ran...
    assert list(strategies) == list(PIPELINE)
    # ...and none of them errored (LLMUnavailable is caught and degrades).
    for name, result in strategies.items():
        assert "error" not in result, f"{name} errored: {result}"


def test_dry_run_override_writes_no_memory(tmp_home):
    """The global --dry-run override neutralizes the active defaults: a run
    with dry_run=True mutates no live memory (the one-flag rollback path)."""
    now = time.time()
    make_state_db(
        tmp_home,
        messages=[("s1", "user", "rotate the api credentials", now - 300),
                  ("s1", "assistant", "rotated and verified", now - 290)],
        outcomes=[("s1", "t-1", now - 280, "verified", None, None)],
    )
    conn = db.connect(tmp_home)
    try:
        before = conn.execute("SELECT count(*) AS n FROM memories").fetchone()["n"]
        run_dream(conn, {"hermes_home": str(tmp_home)}, dry_run=True, actor="e2e")
        after = conn.execute("SELECT count(*) AS n FROM memories").fetchone()["n"]
        assert after == before
    finally:
        conn.close()


def test_tune_never_auto_applies(tmp_home):
    """Even under active defaults, tune only ever proposes — it never applies a
    retrieval-weight change (a hard v1 invariant)."""
    now = time.time()
    make_state_db(
        tmp_home,
        messages=[("s1", "user", "rotate the api credentials", now - 300),
                  ("s1", "assistant", "rotated and verified", now - 290)],
        outcomes=[("s1", "t-1", now - 280, "verified", None, None)],
    )
    conn = db.connect(tmp_home)
    try:
        run_dream(conn, {"hermes_home": str(tmp_home)}, actor="e2e")
        applied = conn.execute(
            "SELECT count(*) AS n FROM proposals WHERE status='applied'").fetchone()["n"]
        assert applied == 0
    finally:
        conn.close()


def test_forge_strategy_mode_gating(tmp_home):
    """The pipeline's forge step honors the shift mode: shadow/off is a no-op,
    and active degrades cleanly when the case bank has no vectors."""
    from brain.dream.run import _strategy_fn
    from brain.dream.shift import Shift

    conn = db.connect(tmp_home)
    try:
        fn = _strategy_fn("forge")

        def _shift(mode):
            return Shift(shift_id="s", conn=conn,
                         config={"_forced_mode": mode, "hermes_home": str(tmp_home)},
                         started_at=db.iso_now(),
                         activity_baseline="9999-12-31T00:00:00.000Z", holder="t")

        assert fn(_shift("shadow")) == {"skipped": "shadow"}
        # active but no embedder -> forge_once returns a clean no-vectors skip.
        assert fn(_shift("active")).get("skipped") == "no_vectors"
    finally:
        conn.close()


def test_default_modes_match_the_contract():
    # Active-by-default (user decision 2026-07-17): the mutating strategies
    # learn live on every dream run.
    for name in ("cases", "distill", "consolidate", "contradict", "forget"):
        assert DEFAULT_MODES[name] == "active", name
    assert DEFAULT_MODES["tune"] == "shadow"       # never auto-applies in v1
    assert DEFAULT_MODES["probes"] == "active"     # read-only health check
    # every pipeline stage has a declared default mode
    for name in PIPELINE:
        assert name in DEFAULT_MODES, f"{name} has no default mode"
