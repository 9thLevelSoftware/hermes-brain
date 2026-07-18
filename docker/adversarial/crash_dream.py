#!/usr/bin/env python3
"""SIGKILL-mid-dream idempotent-resume harness (process level).

The `phases_done` cursor (shift_runs) is committed after each strategy, so a
dream killed mid-run and restarted with the SAME shift_id skips completed
phases and re-runs only the interrupted one (dream/run.py:8-10, 169-189). This
harness proves that across a REAL process kill:

  --run   --shift <id> --crash-after N
      pre-create the shift row, install a fake LLM, and run_dream(resume=<id>);
      monkeypatch _record_done so that immediately after the Nth phase commits
      its cursor, the process dies via os._exit(137) — a true SIGKILL analogue
      (no finally, no lease release, no _close_shift).

  --check --shift <id>
      clear the stale lease the crashed holder left behind (what `hermes brain
      doctor` / the 120s TTL would do), then run_dream(resume=<id>) again and
      assert the first N phases report `already_done` while the rest run. Prints
      RESUME_OK / RESUME_FAIL.

Runs standalone (no hermes-agent); strategies operate on an empty brain so they
complete quickly without real work. Exit 0 on success.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys


def _register_brain():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if "brain" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "brain", os.path.join(root, "__init__.py"), submodule_search_locations=[root])
    module = importlib.util.module_from_spec(spec)
    sys.modules["brain"] = module
    spec.loader.exec_module(module)


def _fake_llm():
    from brain import llm

    # extraction/JSON tasks -> empty; never a real network call.
    llm.set_llm_for_tests(lambda *a, **k: "[]")


def _opt(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else default


def cmd_run(home: str, shift_id: str, crash_after: int) -> int:
    _fake_llm()
    from brain.dream import run as dream_run
    from brain.store import db

    conn = db.connect(home)
    # Pre-create the shift row so resume_shift_id addresses a real, empty shift.
    now = db.iso_now()
    conn.execute(
        "INSERT OR IGNORE INTO shift_runs (shift_id, kind, started_at, phases_done)"
        " VALUES (?,?,?,'[]')", (shift_id, "dream", now))
    conn.commit()

    real_record = dream_run._record_done
    state = {"n": 0}

    def _record_then_maybe_die(c, sid, done):
        real_record(c, sid, done)
        state["n"] += 1
        print(f"[run] phase cursor committed #{state['n']}: {done}", flush=True)
        if state["n"] >= crash_after:
            print(f"[run] SIMULATING SIGKILL after {state['n']} phases "
                  f"(phases_done={done})", flush=True)
            os._exit(137)  # no finally, no lease release — a true crash

    dream_run._record_done = _record_then_maybe_die
    dream_run.run_dream(conn, {"mode": "auto"}, resume_shift_id=shift_id)
    # We only reach here if fewer than crash_after phases ran (unexpected).
    print("[run] pipeline finished BEFORE the crash point — not enough phases ran",
          file=sys.stderr)
    return 2


def cmd_check(home: str, shift_id: str) -> int:
    _fake_llm()
    from brain.dream import run as dream_run
    from brain.store import db

    conn = db.connect(home)
    # The crashed holder left the lease held; clear it (doctor / TTL expiry).
    conn.execute("UPDATE brain_lease SET holder=NULL, acquired_at=NULL, "
                 "expires_at=NULL WHERE name='dream'")
    conn.commit()

    before = json.loads(conn.execute(
        "SELECT phases_done FROM shift_runs WHERE shift_id=?", (shift_id,)
    ).fetchone()["phases_done"])
    print(f"[check] phases_done persisted across the crash: {before}", flush=True)

    summary = dream_run.run_dream(conn, {"mode": "auto"}, resume_shift_id=shift_id)
    strat = summary.get("strategies", {})
    resumed = [n for n, r in strat.items() if r.get("skipped") == "already_done"]
    ran = [n for n, r in strat.items() if r.get("skipped") != "already_done"]
    print(f"[check] resume skipped (already_done): {resumed}")
    print(f"[check] resume ran this pass: {ran}")

    ok = bool(before) and set(before).issubset(set(resumed))
    print("RESUME_OK" if ok else "RESUME_FAIL", flush=True)
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    _register_brain()
    home = os.environ.get("GOLDEN_HOME") or os.environ["HERMES_HOME"]
    shift_id = _opt(argv, "--shift", "crash-shift-1")
    if "--run" in argv:
        return cmd_run(home, shift_id, int(_opt(argv, "--crash-after", "2")))
    if "--check" in argv:
        return cmd_check(home, shift_id)
    print("usage: crash_dream.py --run|--check --shift <id> [--crash-after N]",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
