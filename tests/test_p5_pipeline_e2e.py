"""The full 10-strategy dream pipeline runs clean, in order, ship-inert.

A structural guard: every strategy in PIPELINE executes, none raises, the new
P5 strategies (cases/distill/tune/probes) appear with their ship-inert modes,
and nothing mutates live memory under the default (dry_run/shadow) modes.
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


def test_default_modes_are_ship_inert(tmp_home):
    """The mutating P5 strategies must not write live memory by default."""
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
        run_dream(conn, {"hermes_home": str(tmp_home)}, actor="e2e")
        after = conn.execute("SELECT count(*) AS n FROM memories").fetchone()["n"]
        # cases + distill default to dry_run -> no live memories written.
        assert after == before
        # tune ships shadow -> at most a shadow proposal, never applied.
        applied = conn.execute(
            "SELECT count(*) AS n FROM proposals WHERE status='applied'").fetchone()["n"]
        assert applied == 0
    finally:
        conn.close()


def test_ship_inert_modes_match_the_contract():
    assert DEFAULT_MODES["cases"] == "dry_run"
    assert DEFAULT_MODES["distill"] == "dry_run"
    assert DEFAULT_MODES["tune"] == "shadow"       # never active in v1
    assert DEFAULT_MODES["probes"] == "active"     # read-only health check
    # every pipeline stage has a declared default mode
    for name in PIPELINE:
        assert name in DEFAULT_MODES, f"{name} has no default mode"
