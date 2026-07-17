"""Provider contract tests (shared P1 brief): lane-1 byte stability across a
50-turn session, queue-only sync_turn latency, worker drain on shutdown,
prefetch safety, capture gating (cron context, incognito), and tool-call
error shape.

provider.py is developed in parallel — if it is not importable yet, this
module (and only this module) skips at collection.
"""

from __future__ import annotations

import json
import statistics
import time

import pytest

provider_mod = pytest.importorskip(
    "brain.provider", reason="brain.provider is written in parallel against the same contract"
)
BrainProvider = provider_mod.BrainProvider

from brain import config as brain_config
from brain.store import db
from conftest import poll_until, seed_memory


def _make(tmp_home, session_id="sess", *, agent_context="primary", platform="cli"):
    provider = BrainProvider()
    provider.initialize(
        session_id,
        hermes_home=str(tmp_home),
        platform=platform,
        agent_context=agent_context,
        user_id="owner",
    )
    return provider


def _episode_count(tmp_home, session_id=None):
    conn = db.connect(tmp_home)
    try:
        if session_id:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM episodes WHERE session_id=?", (session_id,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()
        return row["n"]
    finally:
        conn.close()


def test_fifty_turns_lane1_stable_capture_complete_and_fast(tmp_home):
    provider = _make(tmp_home, "sess-50")
    baseline = provider.system_prompt_block()
    assert isinstance(baseline, str)

    latencies = []
    for i in range(1, 51):
        start = time.perf_counter()
        provider.sync_turn(
            f"user turn {i} poking widget_{i} in the deploy pipeline",
            f"assistant reply {i}: adjusted widget_{i}",
            session_id="sess-50",
            messages=[],
        )
        latencies.append(time.perf_counter() - start)
        assert provider.system_prompt_block() == baseline, f"lane 1 changed at turn {i}"

    start = time.perf_counter()
    provider.shutdown()
    # The worker join timeout is 5s (provider.shutdown). The real invariant is
    # that shutdown DRAINS rather than hitting that deadline and abandoning
    # work — so the bound is comfortably below 5s, and the "all 50 episodes
    # captured" assertion below is the completeness half. 4.5s tolerates CPU
    # contention on a busy CI box; a genuinely slow/abandoning worker (drain
    # ~5s, or episodes missing) still fails. Unloaded drain is ~2s.
    drain = time.perf_counter() - start
    assert drain < 4.5, f"shutdown must drain before the 5s join deadline; took {drain:.2f}s"

    assert statistics.fmean(latencies) < 0.005, (
        f"sync_turn must be queue-only; mean {statistics.fmean(latencies) * 1000:.2f}ms")

    assert poll_until(lambda: _episode_count(tmp_home, "sess-50") == 50, timeout=5.0), (
        f"expected 50 episodes after drain, got {_episode_count(tmp_home, 'sess-50')}")


def test_prefetch_empty_query_never_raises(tmp_home):
    provider = _make(tmp_home, "sess-empty")
    result = provider.prefetch("", session_id="sess-empty")
    assert isinstance(result, str)
    provider.shutdown()


def test_queue_prefetch_serves_seeded_memory(tmp_home):
    conn = db.connect(tmp_home)
    seed_memory(conn, "Warning: flux_capacitor drains the plasma coil unless vented first.",
                kind="warning", outcome="failed")
    conn.close()

    provider = _make(tmp_home, "sess-pf")
    query = "how do I vent the flux_capacitor safely"
    provider.queue_prefetch(query, session_id="sess-pf")
    result = poll_until(
        lambda: provider.prefetch(query, session_id="sess-pf") or None, timeout=3.0
    )
    provider.shutdown()

    assert result, "prefetch must serve the cached lane-2 block for a relevant seeded memory"
    assert isinstance(result, str)
    assert "flux" in result.lower()


def test_cron_context_captures_nothing(tmp_home):
    provider = _make(tmp_home, "sess-cron", agent_context="cron")
    for i in range(5):
        provider.sync_turn(f"cron user {i}", f"cron assistant {i}",
                           session_id="sess-cron", messages=[])
    provider.shutdown()
    assert _episode_count(tmp_home) == 0


def test_incognito_captures_nothing(tmp_home):
    brain_config.save_config(tmp_home, {"incognito": True})
    provider = _make(tmp_home, "sess-incog")
    for i in range(3):
        provider.sync_turn(f"secret user {i}", f"secret assistant {i}",
                           session_id="sess-incog", messages=[])
    provider.shutdown()
    assert _episode_count(tmp_home) == 0


def test_handle_tool_call_unknown_tool_returns_recovery_hint(tmp_home):
    provider = _make(tmp_home, "sess-tool")
    result = provider.handle_tool_call("totally_bogus_tool", {})
    provider.shutdown()

    assert isinstance(result, str)
    data = json.loads(result)  # must be valid JSON, never a raw traceback
    assert "recovery_hint" in json.dumps(data)
