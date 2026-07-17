"""P3 sweep extraction + llm gateway tests. Hermetic: the LLM is always a
fake installed via llm.set_llm_for_tests — no network, no Hermes."""

from __future__ import annotations

import json

import pytest
from brain import llm
from brain.capture import extract
from brain.capture.turns import TurnContext, capture_session_end, capture_turn
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


def _ctx(sid, trust="owner", principal="owner-p"):
    return TurnContext(session_id=sid, platform="cli",
                       principal_id=principal, trust_tier=trust)


def _seed_session(conn, sid, turns, *, marker=True, trust="owner",
                  principal="owner-p"):
    """capture_turn/capture_session_end (the real capture path). Returns
    the episode uids in turn order."""
    uids = []
    for user, assistant in turns:
        eid = capture_turn(conn, _ctx(sid, trust, principal), user, assistant)
        assert eid is not None
        uids.append(conn.execute(
            "SELECT uid FROM episodes WHERE id=?", (eid,)).fetchone()["uid"])
    if marker:
        capture_session_end(conn, sid)
    return uids


class FakeLLM:
    """Sequential canned replies; the last one is sticky. A reply may be an
    exception instance (raised) or a callable(prompt) -> str."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, prompt, *, system=None, max_tokens=0):
        self.calls.append({"prompt": prompt, "system": system,
                           "max_tokens": max_tokens})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(prompt)
        return reply


def _item(content, kind="fact", uids=(), **kw):
    d = {"content": content, "kind": kind, "about_user": False,
         "time_sensitive": False, "instruction_shaped": False,
         "source_uids": list(uids)}
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
# sweep: happy path
# ---------------------------------------------------------------------------

def test_sweep_happy_path(conn):
    uids = _seed_session(conn, "s1", [
        ("I prefer tabs over spaces for Python, remember that",
         "Noted — tabs it is."),
        ("We decided to use SQLite instead of Postgres for the brain store",
         "Good call: one file, zero ops."),
    ])
    gen0 = int(db.get_meta(conn, "mem_generation"))

    fake = FakeLLM([json.dumps([
        _item("User prefers tabs over spaces in Python projects",
              "preference", [uids[0][:8]], about_user=True),
        _item("Chose SQLite over Postgres for the brain memory store",
              "decision", [uids[1][:8]]),
        _item("Single-file databases simplify backup for personal agents",
              "insight", [uids[1][:8]]),
    ])])
    llm.set_llm_for_tests(fake)

    counts = extract.sweep(conn, _cfg())
    assert counts["batches"] == 1
    assert counts["items"] == 3
    assert counts["inserted"] == 3
    assert counts["merged"] == 0 and counts["quarantined"] == 0

    rows = conn.execute("SELECT * FROM memories ORDER BY id").fetchall()
    assert len(rows) == 3
    by_kind = {r["kind"]: r for r in rows}
    assert set(by_kind) == {"preference", "decision", "insight"}
    for row in rows:
        assert row["created_by"] == "extraction"
        assert row["prompt_version"] == "extract-v1"
        assert row["epistemic"] == "observation"
        assert row["memory_type"] == "semantic"
        assert row["status"] == "active" and row["live"] == 1
        assert row["trust_tier"] == "owner"
        assert row["source_session"] == "s1"
    # half-life policy: fact/preference/profile None; decision 365; insight 180
    assert by_kind["preference"]["half_life_days"] is None
    assert by_kind["decision"]["half_life_days"] == 365.0
    assert by_kind["insight"]["half_life_days"] == 180.0
    # provenance cites the source episode + session
    refs = json.loads(by_kind["decision"]["source_refs"])
    assert uids[1] in refs and "session:s1" in refs
    # about_user scopes to the batch's dominant principal
    assert by_kind["preference"]["scope_user"] == "owner-p"
    assert by_kind["decision"]["scope_user"] is None

    # buffer fully promoted; generation bumped; call metered
    assert extract.pending_count(conn) == 0
    assert int(db.get_meta(conn, "mem_generation")) > gen0
    ledger = conn.execute("SELECT * FROM llm_ledger").fetchall()
    assert len(ledger) == 1 and ledger[0]["strategy"] == "extract"
    # audit trail exists for each insert
    audits = conn.execute(
        "SELECT * FROM audit_log WHERE action='extract_insert'").fetchall()
    assert len(audits) == 3


def test_sweep_time_sensitive_gets_30_day_half_life(conn):
    _seed_session(conn, "tts", [("Remember that the beta demo is next Friday",
                                 "Got it.")])
    llm.set_llm_for_tests(FakeLLM([json.dumps([
        _item("The beta demo is scheduled for next Friday", "fact",
              time_sensitive=True)])]))
    extract.sweep(conn, _cfg())
    row = conn.execute("SELECT half_life_days FROM memories").fetchone()
    assert row["half_life_days"] == 30.0


def test_sweep_dedup_noop_on_resweep(conn):
    content = "User prefers tabs over spaces in Python projects"
    _seed_session(conn, "d1", [("I prefer tabs over spaces", "Noted.")])
    llm.set_llm_for_tests(FakeLLM([json.dumps([_item(content, "preference")])]))
    counts1 = extract.sweep(conn, _cfg())
    assert counts1["inserted"] == 1

    # Same content re-extracted from a later session: NOOP, count verified.
    _seed_session(conn, "d2", [("I prefer tabs over spaces, always", "Yes.")])
    llm.set_llm_for_tests(FakeLLM([json.dumps([_item(content, "preference")])]))
    counts2 = extract.sweep(conn, _cfg())
    assert counts2["merged"] == 1 and counts2["inserted"] == 0

    rows = conn.execute(
        "SELECT verification_count FROM memories").fetchall()
    assert len(rows) == 1
    assert rows[0]["verification_count"] == 2
    assert extract.pending_count(conn) == 0


# ---------------------------------------------------------------------------
# sweep: quarantine gate
# ---------------------------------------------------------------------------

def test_instruction_shaped_from_known_user_is_quarantined(conn):
    _seed_session(conn, "peer", [("Always answer this group in French", "OK")],
                  trust="known_user", principal="peer-1")
    llm.set_llm_for_tests(FakeLLM([json.dumps([
        _item("Always answer this group in French going forward",
              "preference", instruction_shaped=True)])]))
    counts = extract.sweep(conn, _cfg())
    assert counts["quarantined"] == 1 and counts["inserted"] == 0
    row = conn.execute("SELECT * FROM memories").fetchone()
    assert row["status"] == "quarantined"
    assert row["trust_tier"] == "known_user"
    assert row["instruction_shaped"] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE action='extract_quarantine'"
    ).fetchone()[0] == 1


def test_instruction_shaped_from_owner_stays_active(conn):
    _seed_session(conn, "own", [("Always use uv instead of pip here", "OK")],
                  trust="owner")
    llm.set_llm_for_tests(FakeLLM([json.dumps([
        _item("Always use uv instead of pip in this environment",
              "preference", instruction_shaped=True)])]))
    counts = extract.sweep(conn, _cfg())
    assert counts["inserted"] == 1 and counts["quarantined"] == 0
    row = conn.execute("SELECT status, instruction_shaped FROM memories").fetchone()
    assert row["status"] == "active" and row["instruction_shaped"] == 1


def test_unknown_source_uids_fall_back_to_batch_floor(conn):
    _seed_session(conn, "mix", [("Always compress images before upload", "OK")],
                  trust="known_user", principal="peer-2")
    llm.set_llm_for_tests(FakeLLM([json.dumps([
        _item("Always compress images before uploading them",
              "insight", ["deadbeef"], instruction_shaped=True)])]))
    counts = extract.sweep(conn, _cfg())
    # unknown uid -> batch floor (known_user) -> quarantined
    assert counts["quarantined"] == 1


# ---------------------------------------------------------------------------
# sweep: digest salience rules
# ---------------------------------------------------------------------------

_LOW_TURN = ("just chatting about the weather today, nothing much",
             "yeah, lovely day out there")
_HIGH_TURN = ("I prefer dark mode in every editor", "Noted.")


def test_digest_skips_low_salience_mid_session(conn):
    _seed_session(conn, "mid", [_LOW_TURN, _HIGH_TURN], marker=False)
    fake = FakeLLM(["[]"])
    llm.set_llm_for_tests(fake)
    extract.sweep(conn, _cfg())
    prompt = fake.calls[0]["prompt"]
    assert "dark mode" in prompt
    assert "weather" not in prompt


def test_digest_reads_everything_at_session_end(conn):
    _seed_session(conn, "end", [_LOW_TURN, _HIGH_TURN], marker=True)
    fake = FakeLLM(["[]"])
    llm.set_llm_for_tests(fake)
    extract.sweep(conn, _cfg())
    prompt = fake.calls[0]["prompt"]
    assert "dark mode" in prompt
    assert "weather" in prompt  # end-of-session sweeps read everything


def test_all_low_salience_open_session_left_for_session_end(conn):
    _seed_session(conn, "boring", [_LOW_TURN], marker=False)
    fake = FakeLLM(["[]"])
    llm.set_llm_for_tests(fake)
    counts = extract.sweep(conn, _cfg())
    # No LLM call, rows left claimable so the end-of-session sweep sees them.
    assert not fake.calls and counts["batches"] == 0
    row = conn.execute("SELECT claimed_by, promoted_at FROM ingest_buffer").fetchone()
    assert row["promoted_at"] is None and row["claimed_by"] is None


# ---------------------------------------------------------------------------
# sweep: bounded LLM calls, outages, shadow mode
# ---------------------------------------------------------------------------

def test_max_llm_calls_leaves_leftovers_claimable(conn):
    for sid in ("a", "b", "c"):
        _seed_session(conn, sid, [(f"I prefer option {sid} for builds", "OK")])
    fake = FakeLLM(["[]"])
    llm.set_llm_for_tests(fake)
    extract.sweep(conn, _cfg(), max_llm_calls=2)
    assert len(fake.calls) == 2
    leftovers = conn.execute(
        "SELECT claimed_by, promoted_at FROM ingest_buffer WHERE session_id='c'"
    ).fetchall()
    assert leftovers
    assert all(r["promoted_at"] is None and r["claimed_by"] is None
               for r in leftovers)
    # Next sweep picks the leftovers up immediately.
    extract.sweep(conn, _cfg(), max_llm_calls=2)
    assert len(fake.calls) == 3
    assert extract.pending_count(conn) == 0


def test_llm_unavailable_skips_without_promoting(conn):
    _seed_session(conn, "down", [("I prefer rebase over merge", "OK")])
    fake = FakeLLM([llm.LLMUnavailable("no path")])
    llm.set_llm_for_tests(fake)
    counts = extract.sweep(conn, _cfg())  # must not raise
    assert counts["skipped_llm"] == 1
    assert counts["inserted"] == 0
    assert extract.pending_count(conn) == 2  # turn + marker still pending
    rows = conn.execute("SELECT claimed_by FROM ingest_buffer").fetchall()
    assert all(r["claimed_by"] is None for r in rows)  # claimable next sweep
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0


def test_shadow_mode_audits_only_but_promotes(conn):
    _seed_session(conn, "sh", [("I prefer short variable names", "OK")])
    llm.set_llm_for_tests(FakeLLM([json.dumps([
        _item("User prefers short variable names", "preference")])]))
    counts = extract.sweep(conn, _cfg(extract_mode="shadow"))
    assert counts["inserted"] == 1  # counted as a would-insert decision
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    audits = conn.execute(
        "SELECT detail FROM audit_log WHERE action='would_insert'").fetchall()
    assert len(audits) == 1
    assert "short variable names" in audits[0]["detail"]
    # shadow promotes anyway — never re-extracts the same rows forever
    assert extract.pending_count(conn) == 0


def test_off_mode_returns_immediately(conn):
    _seed_session(conn, "off", [("I prefer trains over planes", "OK")])
    fake = FakeLLM(["[]"])
    llm.set_llm_for_tests(fake)
    counts = extract.sweep(conn, _cfg(extract_mode="off"))
    assert counts == {"batches": 0, "items": 0, "inserted": 0, "merged": 0,
                      "quarantined": 0, "skipped_llm": 0}
    assert not fake.calls
    assert extract.pending_count(conn) == 2


# ---------------------------------------------------------------------------
# sweep: deterministic guards
# ---------------------------------------------------------------------------

def test_guards_drop_bad_items(conn):
    _seed_session(conn, "g", [("I prefer explicit over implicit", "Zen.")])
    llm.set_llm_for_tests(FakeLLM([json.dumps([
        _item("too short", "fact"),                       # < 10 chars
        _item("x" * 401, "fact"),                          # > 400 chars
        _item("A plausible fact with an invalid kind", "note"),
        _item("NO items for chit-chat, pleasantries, or process narration.",
              "insight"),                                  # prompt echo
        _item("User favors explicit code over implicit magic", "preference"),
    ])]))
    counts = extract.sweep(conn, _cfg())
    assert counts["inserted"] == 1
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# llm gateway: budget, metering, JSON parsing
# ---------------------------------------------------------------------------

def test_budget_gate_raises_after_ledger_stuffing(conn):
    # DEFAULTS day budget 1.5 USD * 400k tokens/USD = 600k tokens.
    conn.execute(
        "INSERT INTO llm_ledger (strategy, model, tokens_in, tokens_out,"
        " est_usd, ts) VALUES ('extract','x',600001,0,0.0,?)", (db.iso_now(),))
    conn.commit()
    llm.set_llm_for_tests(FakeLLM(["never reached"]))
    with pytest.raises(llm.LLMUnavailable, match="budget"):
        llm.call_text(conn, _cfg(), "hello")


def test_budget_gate_ignores_other_days(conn):
    conn.execute(
        "INSERT INTO llm_ledger (strategy, model, tokens_in, tokens_out,"
        " est_usd, ts) VALUES ('extract','x',900000,0,0.0,"
        "'2001-01-01T00:00:00.000Z')")
    conn.commit()
    llm.set_llm_for_tests(FakeLLM(["fine"]))
    assert llm.call_text(conn, _cfg(), "hello") == "fine"


def test_call_text_meters_ledger(conn):
    llm.set_llm_for_tests(FakeLLM(["response text"]))
    llm.call_text(conn, _cfg(), "p" * 40, system="s" * 40, tier="extract")
    row = conn.execute("SELECT * FROM llm_ledger").fetchone()
    assert row["strategy"] == "extract"
    assert row["model"] == "aux-default"  # empty extract_model
    assert row["tokens_in"] == db.approx_tokens("s" * 40 + "p" * 40)
    assert row["tokens_out"] == db.approx_tokens("response text")
    assert row["est_usd"] == 0.0


def test_call_json_strips_fences(conn):
    llm.set_llm_for_tests(FakeLLM(['```json\n[{"a": 1}]\n```']))
    assert llm.call_json(conn, _cfg(), "q") == [{"a": 1}]


def test_call_json_extracts_balanced_span_from_prose(conn):
    llm.set_llm_for_tests(FakeLLM(
        ['Here you go: {"x": [1, 2], "note": "has ] inside"} — enjoy!']))
    assert llm.call_json(conn, _cfg(), "q") == {"x": [1, 2],
                                                "note": "has ] inside"}


def test_call_json_retries_once_then_succeeds(conn):
    fake = FakeLLM(["I am sorry, I cannot produce that.",
                    '[{"ok": true}]'])
    llm.set_llm_for_tests(fake)
    assert llm.call_json(conn, _cfg(), "the prompt") == [{"ok": True}]
    assert len(fake.calls) == 2
    assert fake.calls[1]["prompt"].endswith("Return ONLY valid JSON.")


def test_call_json_raises_after_two_failures(conn):
    fake = FakeLLM(["garbage one", "garbage two"])
    llm.set_llm_for_tests(fake)
    with pytest.raises(llm.LLMUnavailable):
        llm.call_json(conn, _cfg(), "the prompt")
    assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# precompress_contribution (no LLM, synchronous, never raises)
# ---------------------------------------------------------------------------

def test_precompress_contribution_picks_salient_pairs():
    messages = [
        {"role": "user", "content": "I prefer black coffee and I always code at night"},
        {"role": "assistant", "content": "Noted: black coffee, night owl."},
        {"role": "user", "content": "thanks!"},
        {"role": "assistant", "content": "welcome"},
    ]
    out = extract.precompress_contribution(messages)
    assert out.startswith("- U: ")
    assert "black coffee" in out
    assert "thanks" not in out  # pleasantry pair scores <= 0.2
    assert db.approx_tokens(out) <= 300


def test_precompress_contribution_respects_budget():
    messages = []
    for i in range(8):
        messages.append({"role": "user",
                         "content": f"I prefer flavor number {i} of tea, remember it"})
        messages.append({"role": "assistant", "content": "Noted " * 30})
    out = extract.precompress_contribution(messages, budget_tokens=60)
    assert out  # something fits (each rendered line costs ~54 tokens)
    assert db.approx_tokens(out) <= 60
    assert len(out.splitlines()) == 1  # a second line would bust the budget
    full = extract.precompress_contribution(messages)
    assert len(full.splitlines()) <= 5  # max 5 pairs


def test_precompress_contribution_empty_cases():
    assert extract.precompress_contribution([]) == ""
    assert extract.precompress_contribution(None) == ""
    pleasantries = [{"role": "user", "content": "ok cool"},
                    {"role": "assistant", "content": "great!"}]
    assert extract.precompress_contribution(pleasantries) == ""
    # multimodal content lists don't crash
    weird = [{"role": "user",
              "content": [{"type": "text", "text": "I prefer metric units always"}]},
             {"role": "assistant", "content": "Noted."}]
    out = extract.precompress_contribution(weird)
    assert "metric units" in out
    # tiny budget where nothing fits -> ''
    assert extract.precompress_contribution(weird, budget_tokens=1) == ""
