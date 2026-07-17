"""B2 — real LLM token/cost accounting in the llm gateway. Hermetic: the aux
client is monkeypatched or the response object is stubbed, so no network and no
real provider is contacted. Exercises _extract_usage, _meter, _budget_gate and
the _aux_call -> ledger real-usage path end to end."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from brain import llm
from brain.config import DEFAULTS
from brain.store import db


@pytest.fixture(autouse=True)
def _clear_llm_override():
    yield
    llm.set_llm_for_tests(None)


def _cfg(**overrides):
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return cfg


def _stub_response(model, prompt_tokens, completion_tokens, text="ok",
                   *, with_usage=True):
    """OpenAI-chat-completions-shaped stub: .model, .choices[0].message.content,
    .usage.prompt_tokens/.completion_tokens — exactly what call_llm returns and
    extract_content_or_reasoning / _extract_usage read."""
    usage = None
    if with_usage:
        usage = SimpleNamespace(prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens)
    msg = SimpleNamespace(content=text)
    return SimpleNamespace(model=model, choices=[SimpleNamespace(message=msg)],
                           usage=usage)


def _only_ledger(conn):
    return conn.execute("SELECT * FROM llm_ledger").fetchone()


# ---------------------------------------------------------------------------
# _provider_for_pricing — the model-prefix -> billing-provider heuristic
# ---------------------------------------------------------------------------

def test_provider_for_pricing_maps_direct_providers():
    assert llm._provider_for_pricing("claude-haiku-4-5") == "anthropic"
    assert llm._provider_for_pricing("gpt-4o-mini") == "openai"
    assert llm._provider_for_pricing("o3-mini") == "openai"
    assert llm._provider_for_pricing("gemini-2.5-flash") == "google"
    assert llm._provider_for_pricing("deepseek-chat") == "deepseek"


def test_provider_for_pricing_unknown_or_prefixed_is_none():
    assert llm._provider_for_pricing("") is None
    assert llm._provider_for_pricing("my-local-llm") is None
    # vendor-prefixed ids are left for the host router to split.
    assert llm._provider_for_pricing("anthropic/claude-haiku-4-5") is None


# ---------------------------------------------------------------------------
# _extract_usage — real tokens + priced est_usd from a response object
# ---------------------------------------------------------------------------

def test_extract_usage_prices_known_model():
    resp = _stub_response("claude-haiku-4-5", 1000, 500)
    usage = llm._extract_usage(resp)
    assert usage is not None
    assert usage.tokens_in == 1000
    assert usage.tokens_out == 500
    # anthropic claude-haiku-4-5 = $1.00/M in, $5.00/M out (host snapshot).
    assert usage.est_usd == pytest.approx(1000 * 1.00 / 1e6 + 500 * 5.00 / 1e6)


def test_extract_usage_unknown_model_keeps_tokens_drops_cost():
    resp = _stub_response("some-self-hosted-model", 320, 80)
    usage = llm._extract_usage(resp)
    assert usage is not None
    assert (usage.tokens_in, usage.tokens_out) == (320, 80)
    assert usage.est_usd is None  # no known pricing -> caller records 0.0


def test_extract_usage_none_when_no_usage_object():
    assert llm._extract_usage(_stub_response("gpt-4o-mini", 0, 0,
                                             with_usage=False)) is None


def test_extract_usage_none_when_usage_all_zero():
    # An empty usage object is indistinguishable from "no usage" for metering.
    assert llm._extract_usage(_stub_response("gpt-4o-mini", 0, 0)) is None


# ---------------------------------------------------------------------------
# _meter — real usage vs proxy fallback
# ---------------------------------------------------------------------------

def test_meter_records_real_usage(conn):
    usage = llm._Usage(tokens_in=1234, tokens_out=567, est_usd=0.0089)
    llm._meter(conn, "extract", "claude-haiku-4-5", "p", "s", "resp",
               usage=usage)
    row = _only_ledger(conn)
    assert row["tokens_in"] == 1234
    assert row["tokens_out"] == 567
    assert row["est_usd"] == pytest.approx(0.0089)
    assert row["model"] == "claude-haiku-4-5"
    assert row["strategy"] == "extract"


def test_meter_real_usage_with_unknown_cost_records_zero(conn):
    usage = llm._Usage(tokens_in=100, tokens_out=20, est_usd=None)
    llm._meter(conn, "dream", "aux-default", "p", "s", "resp", usage=usage)
    row = _only_ledger(conn)
    assert (row["tokens_in"], row["tokens_out"]) == (100, 20)
    assert row["est_usd"] == 0.0  # None -> 0.0, tokens still real


def test_meter_without_usage_falls_back_to_proxy(conn):
    llm._meter(conn, "extract", "aux-default", "p" * 40, "s" * 40, "resp text")
    row = _only_ledger(conn)
    assert row["tokens_in"] == db.approx_tokens("s" * 40 + "p" * 40)
    assert row["tokens_out"] == db.approx_tokens("resp text")
    assert row["est_usd"] == 0.0


# ---------------------------------------------------------------------------
# _budget_gate — prefer real est_usd, else the token proxy
# ---------------------------------------------------------------------------

def _stuff(conn, *, tokens_in=0, tokens_out=0, est_usd=0.0, ts=None):
    conn.execute(
        "INSERT INTO llm_ledger (strategy, model, tokens_in, tokens_out,"
        " est_usd, ts) VALUES ('extract','x',?,?,?,?)",
        (tokens_in, tokens_out, est_usd, ts or db.iso_now()))
    conn.commit()


def test_budget_gate_prefers_real_usd_when_present(conn):
    # est_usd over the 1.5 USD default budget -> raises on the USD basis.
    _stuff(conn, tokens_in=10, tokens_out=10, est_usd=2.0)
    with pytest.raises(llm.LLMUnavailable, match="budget"):
        llm._budget_gate(conn, _cfg())


def test_budget_gate_real_usd_overrides_token_proxy(conn):
    # Tokens are far over the proxy ceiling, but the real metered spend is
    # under budget -> the USD basis wins and the call is allowed.
    _stuff(conn, tokens_in=10_000_000, tokens_out=0, est_usd=0.50)
    llm._budget_gate(conn, _cfg())  # must not raise


def test_budget_gate_falls_back_to_token_proxy_without_pricing(conn):
    # No priced rows (est_usd=0.0) -> crude token proxy: 600k tokens/USD.
    _stuff(conn, tokens_in=600_001, tokens_out=0, est_usd=0.0)
    with pytest.raises(llm.LLMUnavailable, match="budget"):
        llm._budget_gate(conn, _cfg())


def test_budget_gate_token_proxy_under_ceiling_ok(conn):
    _stuff(conn, tokens_in=100, tokens_out=100, est_usd=0.0)
    llm._budget_gate(conn, _cfg())  # must not raise


def test_budget_gate_ignores_other_days_for_both_bases(conn):
    _stuff(conn, tokens_in=10, tokens_out=10, est_usd=99.0,
           ts="2001-01-01T00:00:00.000Z")
    _stuff(conn, tokens_in=9_000_000, tokens_out=0, est_usd=0.0,
           ts="2001-01-01T00:00:00.000Z")
    llm._budget_gate(conn, _cfg())  # today is clean -> allowed


# ---------------------------------------------------------------------------
# _aux_call -> ledger: the real path with a monkeypatched aux client
# ---------------------------------------------------------------------------

def _patch_aux(monkeypatch, response):
    import agent.auxiliary_client as aux

    captured = {}

    def _fake_call_llm(task, **kwargs):
        captured["task"] = task
        captured["kwargs"] = kwargs
        return response

    monkeypatch.setattr(aux, "call_llm", _fake_call_llm)
    monkeypatch.setattr(
        aux, "extract_content_or_reasoning",
        lambda r: (r.choices[0].message.content or ""))
    return captured


def test_aux_call_meters_real_tokens_and_cost(conn, monkeypatch):
    resp = _stub_response("claude-haiku-4-5", 2000, 400, text="done")
    captured = _patch_aux(monkeypatch, resp)

    out = llm.call_text(conn, _cfg(), "the prompt", tier="extract")
    assert out == "done"
    assert captured["task"] == "brain_extract"  # tier -> aux task slot

    row = _only_ledger(conn)
    assert row["strategy"] == "extract"
    assert row["model"] == "aux-default"        # empty extract_model override
    assert row["tokens_in"] == 2000             # REAL, not char/4 proxy
    assert row["tokens_out"] == 400
    assert row["est_usd"] == pytest.approx(2000 * 1.00 / 1e6 + 400 * 5.00 / 1e6)


def test_aux_call_dream_tier_uses_consolidate_slot(conn, monkeypatch):
    resp = _stub_response("gpt-4o-mini", 500, 100, text="ok")
    captured = _patch_aux(monkeypatch, resp)
    llm.call_text(conn, _cfg(), "p", tier="dream")
    assert captured["task"] == "brain_consolidate"
    row = _only_ledger(conn)
    # openai gpt-4o-mini = $0.15/M in, $0.60/M out.
    assert row["est_usd"] == pytest.approx(500 * 0.15 / 1e6 + 100 * 0.60 / 1e6)


def test_aux_call_no_usage_falls_back_to_proxy(conn, monkeypatch):
    resp = _stub_response("gpt-4o-mini", 0, 0, text="hi there", with_usage=False)
    _patch_aux(monkeypatch, resp)
    llm.call_text(conn, _cfg(), "p" * 40, system="s" * 40, tier="extract")
    row = _only_ledger(conn)
    assert row["tokens_in"] == db.approx_tokens("s" * 40 + "p" * 40)
    assert row["tokens_out"] == db.approx_tokens("hi there")
    assert row["est_usd"] == 0.0


def test_aux_call_empty_reply_meters_real_input_then_raises(conn, monkeypatch):
    # Empty completion but real input tokens were still burned (finding #14):
    # meter the usage, then signal unavailable.
    resp = _stub_response("claude-haiku-4-5", 1500, 0, text="   ")
    _patch_aux(monkeypatch, resp)
    with pytest.raises(llm.LLMUnavailable, match="empty"):
        llm.call_text(conn, _cfg(), "p", tier="extract")
    row = _only_ledger(conn)
    assert row["tokens_in"] == 1500          # real input recorded
    assert row["tokens_out"] == 0
    # 1500 input tokens priced; no output.
    assert row["est_usd"] == pytest.approx(1500 * 1.00 / 1e6)


def test_aux_call_provider_exception_meters_proxy_then_raises(conn, monkeypatch):
    import agent.auxiliary_client as aux

    def _boom(task, **kwargs):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(aux, "call_llm", _boom)
    with pytest.raises(llm.LLMUnavailable, match="failed"):
        llm.call_text(conn, _cfg(), "p" * 40, system="s" * 40, tier="extract")
    # No response object -> proxy metering on input, empty output, est_usd 0.0.
    row = _only_ledger(conn)
    assert row["tokens_in"] == db.approx_tokens("s" * 40 + "p" * 40)
    assert row["tokens_out"] == db.approx_tokens("")
    assert row["est_usd"] == 0.0


# ---------------------------------------------------------------------------
# backward-compat: the test-override (set_llm_for_tests) path is unchanged
# ---------------------------------------------------------------------------

def test_set_llm_for_tests_path_still_uses_proxy(conn):
    llm.set_llm_for_tests(lambda p, *, system=None, max_tokens=0: "reply text")
    llm.call_text(conn, _cfg(), "p" * 40, system="s" * 40, tier="extract")
    row = _only_ledger(conn)
    assert row["tokens_in"] == db.approx_tokens("s" * 40 + "p" * 40)
    assert row["tokens_out"] == db.approx_tokens("reply text")
    assert row["est_usd"] == 0.0
