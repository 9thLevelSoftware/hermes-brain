"""Dream scheduling: the shared due-check, the cron offer at setup, and the
opt-in on-idle path.

Before this, `dream_schedule`/`dream_time` were collected by the setup wizard
and read by nothing — the learning system only ever ran when a human typed
`hermes brain dream-now`.
"""

from __future__ import annotations

import time

import pytest
from brain import brain_setup
from brain import config as brain_config
from brain import llm as brain_llm
from brain import provider as provider_mod
from brain.dream import lease
from brain.store import db

# ---------------------------------------------------------------------------
# dream.lease.is_due — ONE definition shared by every trigger
# ---------------------------------------------------------------------------

def _finish_a_shift(conn, finished_at):
    conn.execute(
        "INSERT INTO shift_runs (shift_id, started_at, finished_at, outcome)"
        " VALUES (?,?,?,'ok')", (f"s-{finished_at}", finished_at, finished_at))
    conn.commit()


def test_is_due_when_nothing_has_ever_run(conn):
    assert lease.is_due(conn, {})


def test_is_due_false_inside_the_interval(conn):
    _finish_a_shift(conn, db.iso_now())
    assert not lease.is_due(conn, {"dream_min_interval_hours": 6})


def test_is_due_true_once_the_interval_elapsed(conn):
    _finish_a_shift(conn, "2020-01-01T00:00:00.000Z")
    assert lease.is_due(conn, {"dream_min_interval_hours": 6})


def test_is_due_false_while_the_lease_is_held(conn):
    """A held lease reads as not-due — this is what makes a cron run and the
    on-idle path mutually exclusive rather than racing."""
    assert lease.acquire(conn, "dream", "someone-else")
    assert not lease.is_due(conn, {})


def test_due_check_treats_stored_timestamps_as_utc(conn):
    """Regression: the old helper ran the UTC timestamp through time.mktime,
    which reinterprets it as LOCAL time — skewing the interval by the machine's
    UTC offset. Pinned with an interval that only holds under correct math."""
    now = time.gmtime()
    two_hours_ago = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(
        time.mktime(now) - time.timezone - 2 * 3600)) + ".000Z"
    _finish_a_shift(conn, two_hours_ago)
    assert not lease.is_due(conn, {"dream_min_interval_hours": 6})
    assert lease.is_due(conn, {"dream_min_interval_hours": 1})


# ---------------------------------------------------------------------------
# The cron offer (brain_setup)
# ---------------------------------------------------------------------------

def test_cron_schedule_for_time_builds_an_expression_or_degrades():
    schedule = brain_setup._cron_schedule_for("03:30")
    assert schedule in ("30 3 * * *", "every 24h")


@pytest.mark.parametrize("bad", ["", "nonsense", "99:99", None])
def test_cron_schedule_for_bad_time_falls_back(bad):
    assert brain_setup._cron_schedule_for(bad) == "every 24h"


def test_cron_offer_is_skipped_for_manual_and_on_idle(tmp_home, capsys):
    brain_setup._offer_cron_job(tmp_home, {"dream_schedule": "manual"})
    assert "no job created" in capsys.readouterr().out

    brain_setup._offer_cron_job(tmp_home, {"dream_schedule": "on-idle"})
    assert "No cron job needed" in capsys.readouterr().out


def test_cron_offer_without_a_scheduler_teaches_the_alternative(tmp_home, capsys):
    """cron is gateway-resident; a CLI-only install has no `cron` package. The
    user must still be told how to get automatic learning."""
    brain_setup._offer_cron_job(tmp_home, {"dream_schedule": "auto"})
    out = capsys.readouterr().out
    assert "on-idle" in out and "dream --if-due" in out


# ---------------------------------------------------------------------------
# The opt-in on-idle dream (provider worker)
# ---------------------------------------------------------------------------

def _provider(tmp_home, session_id="sess-idle"):
    """A provider whose queue starts EMPTY.

    bootstrap_import enqueues a first-run job on an empty brain, and
    _maybe_idle_dream deliberately yields to anything queued — so leaving it on
    would make every assertion here a race against the worker draining it.
    """
    p = provider_mod.BrainProvider()
    p.initialize(session_id, hermes_home=str(tmp_home), platform="cli",
                 agent_context="primary", user_id="owner")
    return p


def test_idle_dream_does_not_run_unless_opted_in(tmp_home):
    """auto/cron/manual must never dream in-process — that is the whole point
    of the no-background-work rule."""
    brain_config.save_config(tmp_home, {"dream_schedule": "auto", "bootstrap_import": False})
    p = _provider(tmp_home)
    try:
        c = db.connect(tmp_home)
        p._maybe_idle_dream(c)
        assert c.execute("SELECT COUNT(*) AS n FROM shift_runs").fetchone()["n"] == 0
        c.close()
    finally:
        p.shutdown()


def test_idle_dream_runs_a_shift_when_opted_in(tmp_home):
    brain_config.save_config(tmp_home, {"dream_schedule": "on-idle", "bootstrap_import": False})
    p = _provider(tmp_home)
    brain_llm.set_llm_for_tests(lambda pr, *, system=None, max_tokens=0: "[]")
    try:
        c = db.connect(tmp_home)
        p._maybe_idle_dream(c)
        assert c.execute("SELECT COUNT(*) AS n FROM shift_runs").fetchone()["n"] == 1
        c.close()
    finally:
        brain_llm.set_llm_for_tests(None)
        p.shutdown()


def test_idle_dream_yields_to_a_queued_turn(tmp_home, monkeypatch):
    """A turn is waiting — a dream is never urgent.

    The queue depth is faked rather than filled: really enqueueing a job races
    the worker, which may drain it before the guard is reached. What is under
    test is the guard, not the worker's speed.
    """
    brain_config.save_config(tmp_home, {"dream_schedule": "on-idle", "bootstrap_import": False})
    p = _provider(tmp_home)
    try:
        monkeypatch.setattr(p._queue, "qsize", lambda: 1)
        c = db.connect(tmp_home)
        p._maybe_idle_dream(c)
        assert c.execute("SELECT COUNT(*) AS n FROM shift_runs").fetchone()["n"] == 0
        c.close()
    finally:
        p.shutdown()


def test_idle_dream_skipped_for_incognito_and_shutdown(tmp_home):
    brain_config.save_config(tmp_home, {"dream_schedule": "on-idle", "incognito": True,
                                        "bootstrap_import": False})
    p = _provider(tmp_home)
    try:
        c = db.connect(tmp_home)
        p._maybe_idle_dream(c)
        assert c.execute("SELECT COUNT(*) AS n FROM shift_runs").fetchone()["n"] == 0

        p._incognito = False
        p._shutting_down.set()
        p._last_idle_dream = 0.0
        p._maybe_idle_dream(c)
        assert c.execute("SELECT COUNT(*) AS n FROM shift_runs").fetchone()["n"] == 0
        c.close()
    finally:
        p.shutdown()


def test_idle_dream_respects_its_cooldown(tmp_home):
    brain_config.save_config(tmp_home, {"dream_schedule": "on-idle",
                                        "dream_min_interval_hours": 0,
                                        "bootstrap_import": False})
    p = _provider(tmp_home)
    brain_llm.set_llm_for_tests(lambda pr, *, system=None, max_tokens=0: "[]")
    try:
        c = db.connect(tmp_home)
        p._maybe_idle_dream(c)
        p._maybe_idle_dream(c)   # cooldown holds even though the interval is 0
        assert c.execute("SELECT COUNT(*) AS n FROM shift_runs").fetchone()["n"] == 1
        c.close()
    finally:
        brain_llm.set_llm_for_tests(None)
        p.shutdown()


def test_idle_dream_never_raises_into_the_worker(tmp_home, monkeypatch):
    brain_config.save_config(tmp_home, {"dream_schedule": "on-idle", "bootstrap_import": False})
    p = _provider(tmp_home)
    try:
        monkeypatch.setattr(lease, "is_due",
                            lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
        c = db.connect(tmp_home)
        p._maybe_idle_dream(c)   # must swallow
        c.close()
    finally:
        p.shutdown()


def test_worker_idle_tick_calls_the_idle_dream(tmp_home):
    """The helper must actually be wired into the worker's idle tick.

    Driving the real loop would mean waiting out its 90s queue timeout, so this
    asserts the wiring at the source level — enough to catch the failure mode
    that matters (a correct helper nothing ever calls), which every other test
    in this file, calling the helper directly, would happily pass through.
    """
    import inspect

    idle_branch = inspect.getsource(provider_mod.BrainProvider._worker_loop)
    assert "_maybe_idle_dream" in idle_branch
    # ...and specifically on the queue.Empty (idle) path, not the job path.
    before_continue = idle_branch.split("continue", 1)[0]
    assert "_maybe_idle_dream" in before_continue


# ---------------------------------------------------------------------------
# Cron job creation against a FAKE `cron.jobs` (alignment-audit.md §G4)
# ---------------------------------------------------------------------------
#
# This path has never run against a real gateway — cron is gateway-resident, so
# no test environment had it. These tests do not make it production-verified;
# they move it from "unknown" to "known against a fake", which is the honest
# distinction. What they pin is the shape of the call: no_agent=True, the right
# schedule, a script that actually exists, and idempotency.

@pytest.fixture
def fake_cron(monkeypatch):
    """Inject a stand-in `cron.jobs` module the way a gateway install provides
    the real one."""
    import sys
    import types

    calls = {"created": [], "jobs": []}

    def create_job(prompt, schedule, **kwargs):
        job = {"id": f"job-{len(calls['created']) + 1}", "prompt": prompt,
               "schedule": schedule, **kwargs}
        calls["created"].append(job)
        calls["jobs"].append(job)
        return job

    def list_jobs(include_disabled=False):
        return list(calls["jobs"])

    cron_pkg = types.ModuleType("cron")
    jobs_mod = types.ModuleType("cron.jobs")
    jobs_mod.create_job = create_job
    jobs_mod.list_jobs = list_jobs
    cron_pkg.jobs = jobs_mod
    monkeypatch.setitem(sys.modules, "cron", cron_pkg)
    monkeypatch.setitem(sys.modules, "cron.jobs", jobs_mod)
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    return calls


def test_cron_offer_creates_a_no_agent_script_job(tmp_home, fake_cron, capsys):
    brain_setup._offer_cron_job(tmp_home, {"dream_schedule": "auto",
                                           "dream_time": "03:30"})
    assert len(fake_cron["created"]) == 1
    job = fake_cron["created"][0]

    # no_agent is mandatory: an agent cron job runs skip_memory=True with a
    # 3-minute interrupt and costs tokens for work whose whole point is offline.
    assert job["no_agent"] is True
    assert job["prompt"] is None
    assert job["schedule"] in ("30 3 * * *", "every 24h")
    assert job["name"] == brain_setup._CRON_JOB_NAME

    # The script must exist where cron will look for it, and actually invoke
    # the due-gated entry point.
    script = tmp_home / "scripts" / job["script"]
    assert script.is_file()
    body = script.read_text(encoding="utf-8")
    assert "brain" in body and "dream" in body and "--if-due" in body
    assert "Scheduled" in capsys.readouterr().out


def test_cron_offer_is_idempotent(tmp_home, fake_cron, capsys):
    brain_setup._offer_cron_job(tmp_home, {"dream_schedule": "auto"})
    capsys.readouterr()
    brain_setup._offer_cron_job(tmp_home, {"dream_schedule": "auto"})
    assert len(fake_cron["created"]) == 1, "a second setup must not double-schedule"
    assert "already exists" in capsys.readouterr().out


def test_cron_offer_declined_creates_nothing(tmp_home, fake_cron, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    brain_setup._offer_cron_job(tmp_home, {"dream_schedule": "auto"})
    assert fake_cron["created"] == []
    assert "Skipped" in capsys.readouterr().out


def test_cron_creation_failure_teaches_the_manual_command(tmp_home, monkeypatch,
                                                          capsys):
    """A scheduler that rejects the job must not leave the user with nothing."""
    import sys
    import types

    jobs_mod = types.ModuleType("cron.jobs")

    def boom(*a, **kw):
        raise RuntimeError("scheduler unavailable")

    jobs_mod.create_job = boom
    jobs_mod.list_jobs = lambda include_disabled=False: []
    cron_pkg = types.ModuleType("cron")
    cron_pkg.jobs = jobs_mod
    monkeypatch.setitem(sys.modules, "cron", cron_pkg)
    monkeypatch.setitem(sys.modules, "cron.jobs", jobs_mod)
    monkeypatch.setattr("builtins.input", lambda *a: "y")

    brain_setup._offer_cron_job(tmp_home, {"dream_schedule": "auto"})
    out = capsys.readouterr().out
    assert "Could not create" in out
    assert "dream --if-due" in out, "the manual fallback must be given"
