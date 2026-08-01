"""Context-budget and built-in-memory ownership handoff regressions."""

from __future__ import annotations

from types import SimpleNamespace

from brain import brain_setup
from brain import config as brain_config
from brain.provider import BrainProvider
from brain.recall import lane1
from brain.recall.render import guidance_block, lane2_block
from brain.store import db
from conftest import seed_memory


def test_context_defaults_use_one_800_token_budget():
    cfg = brain_config.DEFAULTS
    assert cfg["context_budget_tokens"] == 800
    assert cfg["lane1_tokens"] == 400
    assert cfg["lane2_tokens"] == 400


def test_setup_wizard_exposes_combined_budget_and_automatic_handoff():
    fields = {field["key"]: field for field in brain_setup.config_schema()}
    assert fields["context_budget_tokens"]["default"] == 800
    assert fields["lane1_tokens"]["default"] == 400
    assert fields["lane2_tokens"]["default"] == 400
    assert "combined" in fields["context_budget_tokens"]["description"].lower()
    assert "automatic ownership" in brain_setup._TRANSITION_MATRIX.lower()
    assert "memory tool stays operational" in brain_setup._TRANSITION_MATRIX.lower()


def test_effective_context_budgets_clamp_legacy_lane_settings():
    effective = brain_config.effective_context_budgets({
        "context_budget_tokens": 800,
        "lane1_tokens": 1200,
        "lane2_tokens": 600,
    })
    assert effective == {
        "context_budget_tokens": 800,
        "lane1_tokens": 800,
        "lane2_tokens": 0,
        "clamped": True,
    }


def test_lane1_only_injects_prioritized_current_items(conn):
    seed_memory(conn, "Never deploy without a rollback checkpoint", kind="warning")
    seed_memory(conn, "Open decision: choose the production queue", kind="decision")
    seed_memory(conn, "Owner prefers compact status updates", kind="preference", pinned=1)
    seed_memory(conn, "Owner profile says timezone is America/New_York",
                kind="profile", pinned=1)
    seed_memory(conn, "Low-value standing fact that should require recall", kind="fact")
    seed_memory(conn, "Unpinned preference that should require recall", kind="preference")

    lane1.materialize(conn, {})
    rendered = lane1.render(conn, 400)

    assert "rollback checkpoint" in rendered
    assert "production queue" in rendered
    assert "compact status" in rendered
    assert "America/New_York" in rendered
    assert "Low-value standing fact" not in rendered
    assert "Unpinned preference" not in rendered
    assert "memories ·" not in rendered
    assert rendered.count("deep recall") == 1
    assert db.approx_tokens(rendered) <= 400


def test_lane2_renders_memory_content_once_and_one_compact_episode():
    memory = SimpleNamespace(
        uid="01MEMORYAAAAAAAAAAAAAAAAAA",
        kind="memory",
        mkind="fact",
        summary="The deploy region is us-east",
        text="The deploy region is us-east",
        ts="2026-07-31T12:00:00.000Z",
        platform="cli",
    )
    episode_a = SimpleNamespace(
        uid="01EPISODEAAAAAAAAAAAAAAAAA",
        kind="episode",
        mkind=None,
        summary=None,
        text="User asked about the deploy. Assistant verified the rollback.",
        ts="2026-07-30T12:00:00.000Z",
        platform="cli",
    )
    episode_b = SimpleNamespace(
        uid="01EPISODEBBBBBBBBBBBBBBBBB",
        kind="episode",
        mkind=None,
        summary=None,
        text="A second raw episode must not be injected automatically.",
        ts="2026-07-29T12:00:00.000Z",
        platform="cli",
    )

    rendered = lane2_block([memory, episode_a, episode_b], 400)

    assert rendered.count("The deploy region is us-east") == 1
    assert "User asked about the deploy" in rendered
    assert "second raw episode" not in rendered


def test_guidance_never_consumes_more_than_quarter_of_lane2():
    items = [
        SimpleNamespace(
            uid=f"01GUIDE{i:02d}AAAAAAAAAAAAAAAAA",
            kind="strategy",
            title=("Use an incremental rollout with a verified rollback checkpoint " * 4),
            verdict=None,
        )
        for i in range(20)
    ]
    lane2_budget = 400
    rendered = guidance_block(items, lane2_budget, max_fraction=0.25)
    assert db.approx_tokens(rendered) <= lane2_budget // 4


def test_provider_combined_injection_respects_global_budget(tmp_home):
    brain_config.save_config(tmp_home, {
        "mode": "fts-only",
        "bootstrap_import": False,
        "context_budget_tokens": 220,
        "lane1_tokens": 200,
        "lane2_tokens": 200,
    })
    conn = db.connect(tmp_home)
    try:
        for i in range(8):
            seed_memory(conn, f"Pinned owner preference {i}: concise operational updates",
                        kind="preference", pinned=1)
        seed_memory(conn, "flux capacitor rollback vent procedure", kind="warning")
        lane1.materialize(conn, brain_config.load_config(tmp_home))
    finally:
        conn.close()

    provider = BrainProvider()
    provider.initialize("budget-session", hermes_home=str(tmp_home), platform="cli",
                        agent_context="primary", user_id="owner")
    try:
        worker_conn = provider._worker_connect()
        try:
            provider._do_retrieve(worker_conn, "budget-session",
                                  "flux capacitor rollback vent procedure")
        finally:
            worker_conn.close()
        injected = "\n".join(filter(None, [
            provider.system_prompt_block(), provider.prefetch("x", session_id="budget-session")
        ]))
        assert db.approx_tokens(injected) <= 220
        status = provider.get_status_config({})
        assert status["context_budget_tokens"] == 220
        assert status["effective_lane1_tokens"] <= 200
        assert (status["effective_lane1_tokens"]
                + status["effective_lane2_tokens"]) <= 220
        assert status["context_budget_clamped"] is True
    finally:
        provider.shutdown()


def test_provider_owns_builtins_only_after_persisted_bootstrap_marker(tmp_home):
    brain_config.save_config(tmp_home, {"bootstrap_import": False, "mode": "fts-only"})
    provider = BrainProvider()
    provider.initialize("ownership", hermes_home=str(tmp_home), platform="cli",
                        agent_context="primary", user_id="owner")
    try:
        assert provider.owns_builtin_memory() is False
        conn = db.connect(tmp_home)
        try:
            db.set_meta(conn, "builtin_import_complete_at", db.iso_now())
            conn.commit()
        finally:
            conn.close()
        assert provider.owns_builtin_memory() is True
    finally:
        provider.shutdown()


def test_provider_never_owns_builtins_when_database_open_failed(tmp_home, monkeypatch):
    import brain.provider as provider_mod

    monkeypatch.setattr(provider_mod.store_db, "connect",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("broken db")))
    provider = BrainProvider()
    provider.initialize("failed", hermes_home=str(tmp_home), platform="cli",
                        agent_context="primary", user_id="owner")
    assert provider.owns_builtin_memory() is False
    provider.shutdown()
