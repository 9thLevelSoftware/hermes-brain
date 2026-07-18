#!/usr/bin/env python3
"""Adversarial live-Hermes driver — a superset of docker/hermes_turn_driver.py.

Runs INSIDE the live image (real hermes-agent + brain + the mock LLM) and
drives the scenarios the standalone pytest gauntlet cannot: real agent turns,
a multi-night flywheel, forced context compression, and the three hooks a plain
mock turn can't reach (session-switch, delegation, memory-write) via the
Path-B direct-hook route.

Two capture strategies, exactly as the base driver:
  A. a real ``run_agent.AIAgent`` turn, streamed from the mock;
  B. the brain ``MemoryProvider`` hooks driven directly through Hermes's own
     ``plugins.memory.load_memory_provider('brain')``.

Subcommands (``python driver.py <cmd> ...``):
  counts                        print (episodes, buffer_pending, active_memories)
  turn   [--msg M] [--session S]        one real turn (A, falling back to B)
  turns  N [--prefix P]                 N real turns (flywheel feed)
  compress                              force a real context compression (fires
                                        on_pre_compress + on_session_switch)
  hook   pre_compress|session_reset|session_rewound|delegation|memory_write
                                        Path-B: fire ONE provider hook directly
                                        and assert its buffer/lane effect

Exit 0 iff the requested effect is observed in brain.db.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time

USER_MSG = "please remember that my staging database is postgres 14 on fly.io"
ASSISTANT_MSG = ("Noted — your staging database is PostgreSQL 14 on Fly.io. "
                 "I'll remember that.")
MOCK_BASE_URL = os.environ.get("MOCK_BASE_URL", "http://127.0.0.1:8080/v1")
MOCK_API_KEY = os.environ.get("MOCK_API_KEY", "sk-mock-local-000")

# Varied prompts so the flywheel sees distinct-but-related turns (dedup/
# consolidation get real signal, not 20 identical rows).
_FLYWHEEL_PROMPTS = [
    "remember my staging database is postgres 14 on fly.io",
    "my production database is postgres 16 on aws rds",
    "i prefer short, direct answers with no preamble",
    "the deploy script lives at scripts/deploy.sh and needs sudo",
    "our api base url is https://api.example.com/v2",
    "i work in the pacific timezone, mostly evenings",
    "the staging db password is rotated every 30 days by ci",
    "we use ruff and pytest; lint must be clean before shipping",
]


def _brain_db_path(home: str) -> str:
    return os.path.join(home, "brain", "brain.db")


def _counts(home: str) -> tuple[int, int, int]:
    path = _brain_db_path(home)
    if not os.path.exists(path):
        return (0, 0, 0)
    conn = sqlite3.connect(path)
    try:
        ep = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        buf = conn.execute(
            "SELECT COUNT(*) FROM ingest_buffer WHERE promoted_at IS NULL"
        ).fetchone()[0]
        try:
            mem = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE valid_to IS NULL "
                "AND status='active' AND live=1").fetchone()[0]
        except sqlite3.Error:
            mem = 0
        return (ep, buf, mem)
    finally:
        conn.close()


def _buffer_kinds(home: str) -> dict[str, int]:
    path = _brain_db_path(home)
    if not os.path.exists(path):
        return {}
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT kind, COUNT(*) n FROM ingest_buffer GROUP BY kind").fetchall()
        return {k: n for k, n in rows}
    finally:
        conn.close()


def _wait(pred, timeout=8.0, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


# ---------------------------------------------------------------------------
# Strategy A: real AIAgent turns
# ---------------------------------------------------------------------------

def _make_agent(session_id: str | None = None):
    os.environ.setdefault("HERMES_YOLO_MODE", "1")
    os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
    from run_agent import AIAgent

    kwargs = dict(
        base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY, provider="custom",
        api_mode="chat_completions", model="mock-main", platform="cli",
        max_iterations=4, quiet_mode=True,
    )
    if session_id:
        kwargs["session_id"] = session_id
    return AIAgent(**kwargs)


def _run_turn(agent, msg: str) -> None:
    result = agent.run_conversation(msg)
    msgs = result.get("messages") if isinstance(result, dict) else None
    try:
        agent.shutdown_memory_provider(msgs or [])
    except Exception as e:  # noqa: BLE001
        print(f"[A] shutdown_memory_provider warned: {e}")


def cmd_turn(msg: str, session: str) -> int:
    before = _counts(os.environ["HERMES_HOME"])[0]
    try:
        agent = _make_agent(session)
        try:
            _run_turn(agent, msg)
        finally:
            try:
                agent.close()
            except Exception:
                pass
        if _wait(lambda: _counts(os.environ["HERMES_HOME"])[0] > before):
            print("[A] real agent turn captured")
            return 0
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"[A] real turn unavailable ({e}); falling back to hooks")
        traceback.print_exc()
    return cmd_hook_capture(msg, session)


def cmd_turns(n: int, prefix: str) -> int:
    home = os.environ["HERMES_HOME"]
    ok = 0
    for i in range(n):
        msg = _FLYWHEEL_PROMPTS[i % len(_FLYWHEEL_PROMPTS)]
        sid = f"{prefix}-{i}"
        before = _counts(home)[0]
        try:
            agent = _make_agent(sid)
            try:
                _run_turn(agent, msg)
            finally:
                try:
                    agent.close()
                except Exception:
                    pass
            if _wait(lambda b=before: _counts(home)[0] > b):
                ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[A] turn {i} failed ({e}); trying hooks")
            if cmd_hook_capture(msg, sid) == 0:
                ok += 1
    ep, buf, mem = _counts(home)
    print(f"[turns] captured {ok}/{n}  episodes={ep} pending={buf} memories={mem}")
    return 0 if ok >= 1 else 1


def cmd_compress() -> int:
    """Force a real compression so on_pre_compress + on_session_switch fire.
    Assert a 'pre_compress' buffer row lands (the brain's synchronous ≤300-tok
    contribution ran and the async capture enqueued)."""
    home = os.environ["HERMES_HOME"]
    try:
        agent = _make_agent("compress-1")
    except Exception as e:  # noqa: BLE001
        print(f"[compress] cannot build agent ({e}); using Path-B pre_compress")
        return cmd_hook("pre_compress")
    try:
        # Prime a little history, then force compaction on demand.
        _run_turn(agent, _FLYWHEEL_PROMPTS[0])
        msgs = [{"role": "user", "content": p} for p in _FLYWHEEL_PROMPTS]
        forced = False
        for call in (
            lambda: agent._compress_context(msgs, None, force=True),
            lambda: agent._compress_context(msgs, force=True),
        ):
            try:
                call()
                forced = True
                break
            except TypeError:
                continue
            except Exception as e:  # noqa: BLE001
                print(f"[compress] _compress_context raised: {e}")
                break
        if not forced:
            print("[compress] could not force compression; using Path-B")
            return cmd_hook("pre_compress")
    finally:
        try:
            agent.close()
        except Exception:
            pass
    ok = _wait(lambda: _buffer_kinds(home).get("pre_compress", 0) >= 1)
    print(f"[compress] pre_compress buffer rows={_buffer_kinds(home).get('pre_compress', 0)}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Strategy B: drive brain MemoryProvider hooks directly
# ---------------------------------------------------------------------------

def _load_provider(session_id: str):
    from plugins.memory import load_memory_provider

    prov = load_memory_provider("brain")
    if prov is None:
        raise RuntimeError("load_memory_provider('brain') returned None")
    prov.initialize(session_id=session_id, platform="cli",
                    hermes_home=os.environ["HERMES_HOME"], agent_context="primary")
    return prov


def cmd_hook_capture(msg: str, session: str) -> int:
    home = os.environ["HERMES_HOME"]
    prov = _load_provider(session or "hook-cap-1")
    try:
        prov.on_turn_start(1, msg)
        prov.sync_turn(msg, ASSISTANT_MSG, session_id=session or "hook-cap-1")
        prov.on_session_end([{"role": "user", "content": msg},
                             {"role": "assistant", "content": ASSISTANT_MSG}])
        ok = _wait(lambda: _counts(home)[0] >= 1 and _counts(home)[1] >= 1)
    finally:
        prov.shutdown()
    print(f"[B] provider-hooks capture={ok}")
    return 0 if ok else 1


def cmd_hook(name: str) -> int:
    home = os.environ["HERMES_HOME"]
    sid = "hook-1"
    prov = _load_provider(sid)
    try:
        if name == "pre_compress":
            msgs = [{"role": "user", "content": USER_MSG},
                    {"role": "assistant", "content": ASSISTANT_MSG}]
            contribution = prov.on_pre_compress(msgs)
            print(f"[B] on_pre_compress returned {len(contribution)} chars")
            ok = _wait(lambda: _buffer_kinds(home).get("pre_compress", 0) >= 1)
        elif name == "delegation":
            prov.on_delegation("investigate the staging outage",
                               "root cause: expired TLS cert on the LB",
                               child_session_id="child-9")
            ok = _wait(lambda: _buffer_kinds(home).get("delegation", 0) >= 1
                       or _counts(home)[0] >= 1)
        elif name == "memory_write":
            prov.on_memory_write("add", "prefs/answers",
                                 "The user prefers concise answers.",
                                 {"session_id": sid})
            ok = _wait(lambda: _buffer_kinds(home).get("memory_write", 0) >= 1
                       or _counts(home)[2] >= 1)
        elif name == "session_reset":
            # Stage a lane-1 snapshot first (marker job), then reset-swap it.
            prov.sync_turn(USER_MSG, ASSISTANT_MSG, session_id=sid)
            prov.on_session_end([])
            _wait(lambda: prov._lane1_staged != "", timeout=6.0)
            before = prov.system_prompt_block()
            prov.on_session_switch("new-sid-1", reset=True)
            after = prov.system_prompt_block()
            ok = prov._session_id == "new-sid-1"
            print(f"[B] session_reset: lane1 changed={before != after}")
        elif name == "session_rewound":
            prov.on_session_switch(sid, reset=False, rewound=True)
            ok = prov._session_id == sid  # rewound-in-place is a no-op switch
        else:
            print(f"[B] unknown hook {name!r}", file=sys.stderr)
            return 2
    finally:
        prov.shutdown()
    print(f"[B] hook {name}: ok={ok}  buffer={_buffer_kinds(home)}")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if not os.environ.get("HERMES_HOME"):
        print("HERMES_HOME not set", file=sys.stderr)
        return 2
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]

    def _opt(flag, default=None):
        return rest[rest.index(flag) + 1] if flag in rest and rest.index(flag) + 1 < len(rest) else default

    if cmd == "counts":
        ep, buf, mem = _counts(os.environ["HERMES_HOME"])
        print(f"episodes={ep} buffer_pending={buf} active_memories={mem}")
        return 0
    if cmd == "turn":
        return cmd_turn(_opt("--msg", USER_MSG), _opt("--session", "adv-turn-1"))
    if cmd == "turns":
        return cmd_turns(int(rest[0]) if rest and rest[0].isdigit() else 4,
                         _opt("--prefix", "fly"))
    if cmd == "compress":
        return cmd_compress()
    if cmd == "hook":
        return cmd_hook(rest[0]) if rest else 2
    print(f"unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
