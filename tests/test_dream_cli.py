"""P4 dream CLI verbs + the strategy-mode-persistence regression.

Hermetic: fake LLM, temp HERMES_HOME, argparse-driven like the other CLI
tests. The mode-persistence test pins the bug the smoke test caught — a
bookkeeping strategy_state row must not disable the pipeline.
"""

from __future__ import annotations

import argparse

import pytest
from brain import cli as bcli
from brain import llm as brain_llm
from brain.dream.shift import DEFAULT_MODES
from brain.store import db
from conftest import seed_memory


@pytest.fixture
def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers().add_parser("brain")
    bcli.register_cli(sub)
    return p


@pytest.fixture(autouse=True)
def _home(tmp_home, monkeypatch):
    monkeypatch.setattr(bcli, "_hermes_home", lambda: tmp_home)
    return tmp_home


def test_dream_now_dry_run_mutates_nothing(parser, tmp_home, capsys):
    conn = db.connect(tmp_home)
    seed_memory(conn, "a durable fact about the deploy pipeline")
    before = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
    conn.close()

    brain_llm.set_llm_for_tests(lambda p, *, system=None, max_tokens=0: "[]")
    try:
        rc = bcli.brain_command(parser.parse_args(["brain", "dream-now", "--dry-run"]))
    finally:
        brain_llm.set_llm_for_tests(None)
    assert rc == 0
    out = capsys.readouterr().out
    assert "consolidate" in out and "forget" in out  # full pipeline reported

    conn = db.connect(tmp_home)
    after = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
    conn.close()
    assert after == before  # dry-run wrote no memories


def test_strategy_modes_survive_a_dream_run(parser, tmp_home):
    """Regression: after dream-now, an unset strategy must fall through to
    DEFAULT_MODES, not the schema default 'off' (which would silently
    disable the pipeline on the next run)."""
    brain_llm.set_llm_for_tests(lambda p, *, system=None, max_tokens=0: "[]")
    try:
        bcli.brain_command(parser.parse_args(["brain", "dream-now", "--dry-run"]))
    finally:
        brain_llm.set_llm_for_tests(None)

    conn = db.connect(tmp_home)
    try:
        from brain.dream.shift import Shift
        shift = Shift(shift_id="x", conn=conn, config={})
        for name, expected in DEFAULT_MODES.items():
            assert shift.mode(name) == expected, f"{name} collapsed to a wrong mode"
    finally:
        conn.close()


def test_dream_enable_disable(parser, tmp_home, capsys):
    assert bcli.brain_command(
        parser.parse_args(["brain", "dream", "--enable", "consolidate"])) == 0
    conn = db.connect(tmp_home)
    try:
        row = conn.execute(
            "SELECT mode FROM strategy_state WHERE strategy='consolidate'").fetchone()
        assert row["mode"] == "active"
    finally:
        conn.close()

    assert bcli.brain_command(
        parser.parse_args(["brain", "dream", "--disable", "forget"])) == 0
    conn = db.connect(tmp_home)
    try:
        row = conn.execute(
            "SELECT mode FROM strategy_state WHERE strategy='forget'").fetchone()
        assert row["mode"] == "off"
        # An unknown strategy is rejected with a teaching message.
    finally:
        conn.close()
    assert bcli.brain_command(
        parser.parse_args(["brain", "dream", "--enable", "nonsense"])) == 1


def test_dream_if_due_respects_interval(parser, tmp_home, capsys):
    # First run: nothing has ever run -> due -> runs.
    brain_llm.set_llm_for_tests(lambda p, *, system=None, max_tokens=0: "[]")
    try:
        bcli.brain_command(parser.parse_args(["brain", "dream-now", "--dry-run"]))
        capsys.readouterr()
        # Immediately after, --if-due must be a no-op (min interval not elapsed).
        rc = bcli.brain_command(parser.parse_args(["brain", "dream", "--if-due"]))
    finally:
        brain_llm.set_llm_for_tests(None)
    assert rc == 0
    assert "not due" in capsys.readouterr().out.lower()


def test_dream_now_single_phase(parser, tmp_home, capsys):
    rc = bcli.brain_command(parser.parse_args(["brain", "dream-now", "--phase", "lane1"]))
    assert rc == 0
    out = capsys.readouterr().out
    assert "lane1" in out and "consolidate" not in out  # only the one phase ran


def test_status_shows_dreams_line(parser, tmp_home, capsys):
    rc = bcli.brain_command(parser.parse_args(["brain", "status"]))
    assert rc == 0
    out = capsys.readouterr().out
    assert "last dream" in out and "strategies" in out
