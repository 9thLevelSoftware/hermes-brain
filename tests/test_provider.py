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
    # fts-only: what this pins is lane-1 byte stability, queue-only sync_turn
    # latency, and that shutdown DRAINS rather than abandoning work. Firing 50
    # turns with no gaps is already an artificial worst case (real turns arrive
    # minutes apart, so the queue is empty at shutdown); adding real ONNX
    # embedding of all 50 on top would measure model load time, not the drain.
    brain_config.save_config(tmp_home, {"mode": "fts-only"})
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
    brain_config.save_config(tmp_home, {"mode": "fts-only"})
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


# ---------------------------------------------------------------------------
# recall_mode (hybrid | context | tools) — the cross-provider convention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["hybrid", "context", "tools", "nonsense-typo"])
def test_recall_mode_lane1_is_byte_stable(tmp_home, mode):
    """Lane 1 is byte-stable for the session under EVERY recall_mode — the mode
    is read once at initialize(), so it can never move the prompt prefix
    mid-session (invariant #1)."""
    # fts-only: this pins lane-1 BYTE STABILITY across modes, not retrieval
    # quality — loading a real ONNX embedder here just adds seconds.
    brain_config.save_config(tmp_home, {"recall_mode": mode, "mode": "fts-only"})
    provider = _make(tmp_home, f"sess-rm-{mode}")
    baseline = provider.system_prompt_block()
    for i in range(1, 26):
        provider.sync_turn(f"user {i}", f"assistant {i}",
                           session_id=f"sess-rm-{mode}", messages=[])
        provider.queue_prefetch(f"query {i}", session_id=f"sess-rm-{mode}")
        assert provider.system_prompt_block() == baseline, f"lane 1 moved at turn {i}"
    provider.shutdown()


def test_recall_mode_context_hides_tools_but_keeps_lanes(tmp_home):
    conn = db.connect(tmp_home)
    seed_memory(conn, "Warning: flux_capacitor drains the plasma coil unless vented first.",
                kind="warning", outcome="failed")
    conn.close()

    # fts-only on purpose: this test is about lane/tool GATING, not retrieval
    # quality, and loading a real ONNX embedder here costs seconds and makes
    # the poll below flaky on machines where the [full] extra is installed.
    brain_config.save_config(tmp_home, {"recall_mode": "context", "mode": "fts-only"})
    provider = _make(tmp_home, "sess-ctx")
    assert provider.get_tool_schemas() == []

    query = "how do I vent the flux_capacitor safely"
    provider.queue_prefetch(query, session_id="sess-ctx")
    result = poll_until(
        lambda: provider.prefetch(query, session_id="sess-ctx") or None, timeout=3.0)
    provider.shutdown()
    assert result and "flux" in result.lower(), "context mode must keep lane 2"


def test_recall_mode_tools_hides_lanes_but_keeps_tools(tmp_home):
    conn = db.connect(tmp_home)
    seed_memory(conn, "Warning: flux_capacitor drains the plasma coil unless vented first.",
                kind="warning", outcome="failed")
    conn.close()

    brain_config.save_config(tmp_home, {"recall_mode": "tools", "mode": "fts-only"})
    provider = _make(tmp_home, "sess-tools")
    assert [s["function"]["name"] for s in provider.get_tool_schemas()], \
        "tools mode must still advertise the tool surface"

    query = "how do I vent the flux_capacitor safely"
    provider.queue_prefetch(query, session_id="sess-tools")
    time.sleep(0.5)  # give the worker a chance to (wrongly) fill the cache
    assert provider.prefetch(query, session_id="sess-tools") == ""
    # Lane 1 falls back to the static block — no snapshot content injected.
    assert provider.system_prompt_block() == provider_mod.lane1_static()
    provider.shutdown()


def test_unknown_recall_mode_falls_back_to_hybrid(tmp_home):
    """A typo in brain.yaml must not silently switch memory off."""
    brain_config.save_config(tmp_home, {"recall_mode": "hybird"})
    provider = _make(tmp_home, "sess-typo")
    assert provider._recall_mode == "hybrid"
    assert provider.get_tool_schemas(), "fallback must keep the tool surface"
    provider.shutdown()


# ---------------------------------------------------------------------------
# get_status_config — the `hermes memory status` hook
# ---------------------------------------------------------------------------

def test_get_status_config_reports_effective_settings(tmp_home):
    brain_config.save_config(tmp_home, {"recall_mode": "context", "lane2_tokens": 321})
    provider = _make(tmp_home, "sess-status")
    provider.sync_turn("remember the deploy pipeline", "noted", session_id="sess-status")
    provider.shutdown()

    # The host passes memory.brain from config.yaml, which is always empty for
    # us — the reported values must come from brain.yaml regardless.
    status = provider.get_status_config({})
    assert status["recall_mode"] == "context"
    assert status["lane2_tokens"] == 321
    assert "->" in status["tier"]
    assert str(db.db_path(tmp_home)) == status["db"]
    assert status["last_dream"].startswith("never")
    assert isinstance(status["episodes"], int)


def test_get_status_config_works_on_an_uninitialized_instance(tmp_home, monkeypatch):
    """`hermes memory status` builds a fresh provider purely to interrogate it
    and NEVER calls initialize() — so the profile must be resolved from the
    host, or every DB-backed field is missing on the one surface this method
    exists for."""
    monkeypatch.setattr(provider_mod, "_host_hermes_home", lambda: tmp_home)
    status = BrainProvider().get_status_config({})
    assert status["db"] == str(db.db_path(tmp_home))
    assert isinstance(status["memories"], int)


def test_get_status_config_never_raises_without_a_home(monkeypatch):
    monkeypatch.setattr(provider_mod, "_host_hermes_home", lambda: None)
    status = BrainProvider().get_status_config({})
    assert isinstance(status, dict) and status["db"].startswith("(no profile")


def test_get_status_config_prefers_the_injected_profile(tmp_home, tmp_path, monkeypatch):
    """The host fallback must never override an explicitly-passed profile —
    that would show one profile's status while another is active."""
    other = tmp_path / "other-profile"
    other.mkdir()
    monkeypatch.setattr(provider_mod, "_host_hermes_home", lambda: other)
    provider = _make(tmp_home, "sess-profile")
    try:
        assert provider.get_status_config({})["db"] == str(db.db_path(tmp_home))
    finally:
        provider.shutdown()


def test_linked_lane2_results_are_not_query_cached(tmp_home, tmp_path):
    """query_cache keys on the LOCAL mem_generation, which a write in another
    profile cannot bump and which linking/unlinking does not touch. Caching a
    merged result would make a long-running gateway serve memories from an
    unlinked profile indefinitely (PR #9 review, P2)."""
    other = tmp_path / "other_home"
    other.mkdir()
    oconn = db.connect(other)
    try:
        seed_memory(oconn, "the other profile knows about flux_capacitor venting")
    finally:
        oconn.close()

    conn = db.connect(tmp_home)
    try:
        from brain.store import links as links_mod

        links_mod.add(conn, "other", str(other))
    finally:
        conn.close()

    brain_config.save_config(tmp_home, {"mode": "fts-only", "link_lane2": True,
                                        "bootstrap_import": False})
    provider = _make(tmp_home, "sess-linkcache")
    query = "how do I vent the flux_capacitor"
    provider.queue_prefetch(query, session_id="sess-linkcache")
    result = poll_until(
        lambda: provider.prefetch(query, session_id="sess-linkcache") or None,
        timeout=5.0)
    try:
        assert result and "flux" in result.lower(), "sanity: linked lane 2 served"
        assert provider._query_cache.get(
            db.connect(tmp_home), query, kinds=None,
            scope=("sess-linkcache", "owner", "", "owner"), embedder=None) is None, \
            "a result containing linked rows must not be cached"
    finally:
        provider.shutdown()
