"""Tests for the companion ``brain_observer`` plugin (task B3) and the brain's
work_queue drain.

Hermetic — no real Hermes host needed:
  * The observer's ``register(ctx)`` is exercised against a fake ctx that
    records registered hooks.
  * Firing a hook enqueues a ``work_queue`` row (the observer's background
    writer talks to the tmp_home brain.db via ``HERMES_HOME``).
  * The brain-side ``store.work_queue.drain_observer_signals`` marks rows done
    and writes the audit_log summary + activity heartbeat.
  * The real provider worker drains the queue end-to-end.

The observer package is loaded by path (as the host loads it standalone),
NOT as a ``brain`` submodule — it must not depend on the brain package.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from brain.store import db, work_queue
from conftest import poll_until

REPO_ROOT = Path(__file__).resolve().parent.parent

provider_mod = pytest.importorskip(
    "brain.provider",
    reason="brain.provider is written in parallel against the same contract",
)
BrainProvider = provider_mod.BrainProvider


def _load_observer():
    """Load observer/__init__.py the way the host loads a standalone plugin."""
    spec = importlib.util.spec_from_file_location(
        "brain_observer_under_test",
        REPO_ROOT / "observer" / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT / "observer")],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeCtx:
    """Minimal stand-in for hermes_cli.plugins.PluginContext."""

    def __init__(self) -> None:
        self.hooks: dict[str, list] = {}

    def register_hook(self, name, callback) -> None:
        self.hooks.setdefault(name, []).append(callback)


def _make_provider(tmp_home, session_id="sess-obs", *, agent_context="primary"):
    provider = BrainProvider()
    provider.initialize(
        session_id,
        hermes_home=str(tmp_home),
        platform="cli",
        agent_context=agent_context,
        user_id="owner",
    )
    return provider


def _pending_tool_rows(tmp_home) -> int:
    conn = db.connect(tmp_home)
    try:
        return conn.execute(
            "SELECT COUNT(*) AS n FROM work_queue "
            "WHERE done_at IS NULL AND task='observed_tool_call'"
        ).fetchone()["n"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# register(ctx)
# ---------------------------------------------------------------------------

def test_register_records_expected_hooks():
    observer = _load_observer()
    ctx = _FakeCtx()
    observer.register(ctx)
    assert "post_tool_call" in ctx.hooks
    assert "subagent_stop" in ctx.hooks
    # pre_llm_call ships OFF by default (context-injection lane gated).
    assert "pre_llm_call" not in ctx.hooks


def test_pre_llm_call_stub_returns_none_even_when_enabled():
    observer = _load_observer()
    # Directly exercise the stub — inert by design even if the flag is on.
    assert observer._on_pre_llm_call(user_message="hi", session_id="s") is None


# ---------------------------------------------------------------------------
# Hooks never raise into the host and stay observer-only
# ---------------------------------------------------------------------------

def test_hooks_never_raise_and_return_none(monkeypatch, tmp_home):
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))
    observer = _load_observer()
    # No brain.db exists yet: the signal is dropped, hook still returns None.
    assert observer._on_post_tool_call() is None
    assert observer._on_post_tool_call(tool_name="x", status="ok") is None
    assert observer._on_subagent_stop(child_role="r", child_status="success") is None
    # Weird/oversized/None kwargs must not raise.
    assert observer._on_post_tool_call(
        tool_name=None, status=None, duration_ms=None, telemetry_schema_version=1
    ) is None


def test_disable_env_silences_enqueue(monkeypatch, tmp_home):
    db.connect(tmp_home).close()  # brain.db exists
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))
    monkeypatch.setenv("BRAIN_OBSERVER_DISABLE", "1")
    observer = _load_observer()
    observer._on_post_tool_call(tool_name="read_file", status="ok", session_id="s")
    # Give any (non-existent) writer a beat; nothing should be written.
    assert not poll_until(lambda: _pending_tool_rows(tmp_home) >= 1, timeout=0.6)


# ---------------------------------------------------------------------------
# Firing a hook enqueues a work_queue row (background writer path)
# ---------------------------------------------------------------------------

def test_post_tool_call_hook_enqueues_row(monkeypatch, tmp_home):
    db.connect(tmp_home).close()  # ensure brain.db + schema exist first
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))
    monkeypatch.delenv("BRAIN_OBSERVER_DISABLE", raising=False)
    observer = _load_observer()

    observer._on_post_tool_call(
        tool_name="read_file", status="ok", session_id="s1",
        tool_call_id="tc1", turn_id="t1", duration_ms=12,
        telemetry_schema_version=1,
    )
    observer._on_post_tool_call(
        tool_name="shell_exec", status="error", error_type="tool_error",
        session_id="s1", duration_ms=99, telemetry_schema_version=1,
    )

    assert poll_until(lambda: _pending_tool_rows(tmp_home) >= 2, timeout=5.0), (
        "post_tool_call hook must enqueue a work_queue row via the background writer")

    conn = db.connect(tmp_home)
    try:
        row = conn.execute(
            "SELECT task, payload, attempts, done_at FROM work_queue "
            "WHERE task='observed_tool_call' ORDER BY id LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert row["attempts"] == 0
    assert row["done_at"] is None
    payload = json.loads(row["payload"])
    assert payload["tool_name"] == "read_file"
    assert payload["disposition"] == "ok"
    assert "args" not in payload and "result" not in payload  # metadata only


def test_subagent_stop_hook_enqueues_row(monkeypatch, tmp_home):
    db.connect(tmp_home).close()
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))
    monkeypatch.delenv("BRAIN_OBSERVER_DISABLE", raising=False)
    observer = _load_observer()

    observer._on_subagent_stop(
        parent_session_id="p", child_session_id="c", child_role="researcher",
        child_status="success", child_summary="x" * 500, duration_ms=1234,
        telemetry_schema_version=1,
    )

    def _has_subagent_row():
        conn = db.connect(tmp_home)
        try:
            return conn.execute(
                "SELECT payload FROM work_queue WHERE task='observed_subagent_stop' LIMIT 1"
            ).fetchone()
        finally:
            conn.close()

    row = poll_until(_has_subagent_row, timeout=5.0)
    assert row is not None
    payload = json.loads(row["payload"])
    assert payload["child_role"] == "researcher"
    assert payload["disposition"] == "success"
    assert len(payload["summary_preview"]) == 200  # summary truncated


# ---------------------------------------------------------------------------
# Brain-side drain (synchronous, deterministic)
# ---------------------------------------------------------------------------

def test_enqueue_row_shape(conn):
    rid = work_queue.enqueue(conn, "observed_tool_call", {"tool_name": "x"})
    conn.commit()
    row = conn.execute(
        "SELECT task, payload, created_at, attempts, done_at FROM work_queue WHERE id=?",
        (rid,),
    ).fetchone()
    assert row["task"] == "observed_tool_call"
    assert row["attempts"] == 0
    assert row["done_at"] is None
    assert row["created_at"]
    assert json.loads(row["payload"])["tool_name"] == "x"


def test_drain_marks_done_and_writes_bookkeeping(conn):
    work_queue.enqueue(conn, "observed_tool_call", {"tool_name": "read_file", "disposition": "ok"})
    work_queue.enqueue(conn, "observed_tool_call",
                       {"tool_name": "shell_exec", "disposition": "error", "error_type": "tool_error"})
    work_queue.enqueue(conn, "observed_subagent_stop", {"child_role": "researcher", "disposition": "success"})
    conn.commit()

    summary = work_queue.drain_observer_signals(conn, limit=100)
    assert summary["count"] == 3
    assert summary["claimed"] == 3
    assert summary["errors"] == 1                 # only the shell_exec tool error
    assert summary["subagents"] == 1
    assert summary["tools"].get("read_file") == 1
    assert summary["tools"].get("shell_exec") == 1

    # All observer rows marked done.
    assert work_queue.pending_count(conn) == 0
    done = conn.execute(
        "SELECT COUNT(*) AS n FROM work_queue WHERE done_at IS NOT NULL AND attempts=1"
    ).fetchone()["n"]
    assert done == 3

    # One audit summary row + an activity heartbeat.
    audit = conn.execute(
        "SELECT detail FROM audit_log WHERE actor='observer' AND action='drain'"
    ).fetchall()
    assert len(audit) == 1
    assert json.loads(audit[0]["detail"])["count"] == 3
    assert conn.execute(
        "SELECT last_seen FROM activity WHERE source='observer'"
    ).fetchone() is not None


def test_drain_empty_is_noop(conn):
    summary = work_queue.drain_observer_signals(conn)
    assert summary == {"count": 0}
    # No bookkeeping written on the idle path.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM audit_log WHERE actor='observer'"
    ).fetchone()["n"] == 0


def test_second_drain_is_noop_after_first(conn):
    work_queue.enqueue(conn, "observed_tool_call", {"tool_name": "read_file"})
    conn.commit()
    assert work_queue.drain_observer_signals(conn)["count"] == 1
    assert work_queue.drain_observer_signals(conn)["count"] == 0


def test_drain_respects_limit_and_leaves_remainder(conn):
    for i in range(5):
        work_queue.enqueue(conn, "observed_tool_call", {"tool_name": f"tool_{i}"})
    conn.commit()
    first = work_queue.drain_observer_signals(conn, limit=2)
    assert first["count"] == 2
    assert work_queue.pending_count(conn) == 3


def test_drain_ignores_non_observer_tasks(conn):
    # A non-observer task (schema's own vocabulary) must be left untouched.
    work_queue.enqueue(conn, "embed", {"memory_id": 1})
    work_queue.enqueue(conn, "observed_tool_call", {"tool_name": "read_file"})
    conn.commit()
    summary = work_queue.drain_observer_signals(conn)
    assert summary["count"] == 1
    remaining = conn.execute(
        "SELECT task FROM work_queue WHERE done_at IS NULL"
    ).fetchall()
    assert [r["task"] for r in remaining] == ["embed"]


# ---------------------------------------------------------------------------
# End-to-end: the real provider worker drains the queue
# ---------------------------------------------------------------------------

def test_provider_worker_drains_work_queue(tmp_home):
    conn = db.connect(tmp_home)
    for i in range(3):
        work_queue.enqueue(conn, "observed_tool_call",
                           {"tool_name": f"tool_{i}", "disposition": "ok", "session_id": "s"})
    conn.commit()
    conn.close()

    provider = _make_provider(tmp_home, "sess-drain")
    try:
        # Nudge the worker to iterate so the post-job drain fires (avoids the
        # 90s idle tick). on_turn_start enqueues a cheap ("touch",) job.
        provider.on_turn_start(1, "hi")
        assert poll_until(lambda: _pending_tool_rows(tmp_home) == 0, timeout=5.0), (
            "provider worker must drain observer work_queue rows")
    finally:
        provider.shutdown()

    conn = db.connect(tmp_home)
    try:
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE actor='observer'"
        ).fetchone()["n"] >= 1
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM activity WHERE source='observer'"
        ).fetchone()["n"] == 1
    finally:
        conn.close()


def test_provider_incognito_does_not_drain(tmp_home):
    from brain import config as brain_config

    brain_config.save_config(tmp_home, {"incognito": True})
    conn = db.connect(tmp_home)
    work_queue.enqueue(conn, "observed_tool_call", {"tool_name": "read_file", "disposition": "ok"})
    conn.commit()
    conn.close()

    provider = _make_provider(tmp_home, "sess-incog")
    try:
        provider.on_turn_start(1, "hi")
        # Incognito must leave the rows pending (no trace written).
        assert not poll_until(lambda: _pending_tool_rows(tmp_home) == 0, timeout=1.0)
    finally:
        provider.shutdown()
