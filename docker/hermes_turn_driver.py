#!/usr/bin/env python3
"""Drive ONE real turn so the brain captures it — offline, against the mock LLM.

Two strategies, tried in order; the first that lands a turn in brain.db wins:

  A. REAL AGENT TURN (preferred). Construct ``run_agent.AIAgent`` directly with
     an explicit base_url/api_key pointing at the mock (agent_init builds the
     OpenAI client directly — no credential pool, no auth handshake, no network
     at startup), run ONE user message through ``run_conversation`` (the main
     model streams from the mock), then ``shutdown_memory_provider`` so the
     brain's ``sync_turn`` + ``on_session_end`` hooks fire and flush.

  B. PROVIDER HOOKS (fallback). If the full agent turn can't run offline for any
     reason, drive the REAL MemoryProvider hook sequence directly through
     Hermes's own loader — initialize(primary/owner) -> sync_turn -> on_session
     _end -> shutdown — which is the same capture contract a turn exercises.

Either way the result is a verbatim ``episodes`` row + a ``turn`` work-unit and
a ``session_end_marker`` in ``ingest_buffer``. The subsequent ``hermes brain
dream-now`` (a separate process, inside the real Hermes runtime) runs the
``flush`` extraction strategy, which calls the mock through
``agent.auxiliary_client`` and turns the buffered turn into a durable memory —
the real LLM->brain round-trip. This driver does not itself require the mock
for path B, but path A streams a real model reply from it.

Exit 0 iff a turn was captured (episodes>=1 and a pending ingest_buffer row).
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time

USER_MSG = "please remember that my staging database is postgres 14 on fly.io"
ASSISTANT_MSG = ("Noted — your staging database is PostgreSQL 14 on Fly.io. "
                 "I'll remember that.")
SESSION_ID = "p2-live-1"
MOCK_BASE_URL = os.environ.get("MOCK_BASE_URL", "http://127.0.0.1:8080/v1")
MOCK_API_KEY = os.environ.get("MOCK_API_KEY", "sk-mock-local-000")


def _brain_db_path(home: str) -> str:
    return os.path.join(home, "brain", "brain.db")


def _counts(home: str) -> tuple[int, int, int]:
    """(episodes, ingest_buffer pending, active memories) straight from brain.db."""
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
                "AND status='active' AND live=1"
            ).fetchone()[0]
        except sqlite3.Error:
            mem = 0
        return (ep, buf, mem)
    finally:
        conn.close()


def _captured(home: str) -> bool:
    ep, buf, _ = _counts(home)
    return ep >= 1 and buf >= 1


# ---------------------------------------------------------------------------
# Strategy A: a real AIAgent turn (main model streams from the mock)
# ---------------------------------------------------------------------------

def capture_via_aiagent(home: str) -> bool:
    os.environ.setdefault("HERMES_YOLO_MODE", "1")     # auto-approve, non-interactive
    os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
    from run_agent import AIAgent

    print(f"[A] constructing AIAgent -> {MOCK_BASE_URL} (model=mock-main)")
    agent = AIAgent(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        provider="custom",
        api_mode="chat_completions",
        model="mock-main",
        platform="cli",
        max_iterations=4,
        quiet_mode=True,
    )
    try:
        result = agent.run_conversation(USER_MSG)
        resp = (result.get("final_response") if isinstance(result, dict) else "") or ""
        print(f"[A] agent final_response: {resp[:160]!r}")
        msgs = result.get("messages") if isinstance(result, dict) else None
        try:
            agent.shutdown_memory_provider(msgs or [])
        except Exception as e:
            print(f"[A] shutdown_memory_provider warned: {e}")
    finally:
        try:
            agent.close()
        except Exception:
            pass
    time.sleep(0.5)
    ok = _captured(home)
    print(f"[A] real agent turn captured={ok}")
    return ok


# ---------------------------------------------------------------------------
# Strategy B: drive the MemoryProvider hooks directly (fallback)
# ---------------------------------------------------------------------------

def capture_via_hooks(home: str) -> bool:
    from plugins.memory import _get_active_memory_provider, load_memory_provider

    active = _get_active_memory_provider()
    print(f"[B] active memory provider (config): {active!r}")
    prov = load_memory_provider("brain")
    if prov is None:
        print("[B] load_memory_provider('brain') returned None", file=sys.stderr)
        return False

    prov.initialize(session_id=SESSION_ID, platform="cli", hermes_home=home,
                    agent_context="primary")
    print(f"[B] lane1 system_prompt_block: {len(prov.system_prompt_block())} chars")
    prov.on_turn_start(1, USER_MSG)
    prov.sync_turn(USER_MSG, ASSISTANT_MSG, session_id=SESSION_ID)
    prov.on_session_end([
        {"role": "user", "content": USER_MSG},
        {"role": "assistant", "content": ASSISTANT_MSG},
    ])
    prov.shutdown()
    time.sleep(0.3)
    ok = _captured(home)
    print(f"[B] provider-hooks capture={ok}")
    return ok


def main() -> int:
    home = os.environ.get("HERMES_HOME")
    if not home:
        print("HERMES_HOME not set", file=sys.stderr)
        return 2

    used = None
    try:
        if capture_via_aiagent(home):
            used = "real AIAgent turn (main model streamed from the mock)"
    except Exception as e:
        import traceback
        print(f"[A] real agent turn unavailable ({e}); falling back to provider hooks")
        traceback.print_exc()

    if used is None:
        try:
            if capture_via_hooks(home):
                used = "provider MemoryProvider hooks"
        except Exception:
            import traceback
            traceback.print_exc()

    episodes, pending, memories = _counts(home)
    print(f"post-capture: episodes={episodes} buffer_pending={pending} "
          f"active_memories={memories}  (via: {used})")

    if used is None or episodes < 1 or pending < 1:
        print("FAIL: no turn was captured into episodes + ingest_buffer",
              file=sys.stderr)
        return 1
    print(f"CAPTURE OK via {used}: run `hermes brain dream-now` to extract it "
          "through the mock LLM.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
