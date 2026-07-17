"""Replay harness — the P1 verification loop (docs/design/integration.md §6).

Drives the REAL BrainProvider hook sequence over a recorded session — a JSON
fixture or a read-only Hermes state.db — against a throwaway (or given)
hermes home, with no LLM anywhere (P1 has none). This is how every phase is
verified without burning tokens, and it doubles as the regression suite
against hermes-agent contract drift.

    python replay/run.py --fixture tests/fixtures/session_basic.json \
        [--hermes-home DIR] [--assert-lane1-stable] [--budget-check] [--turns N]
    python replay/run.py --state-db ~/.hermes/state.db --session-id SID ...

Exit codes: 0 ok; 2 lane-1 instability (--assert-lane1-stable);
3 lane-2 budget violation (--budget-check); 4 unusable --state-db.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


class StateDbError(RuntimeError):
    """--state-db is not a usable Hermes state.db (exit code 4)."""


def _load_brain():
    """Register the repo root as package 'brain', the Hermes-loader way."""
    if "brain" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "brain", REPO_ROOT / "__init__.py", submodule_search_locations=[str(REPO_ROOT)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["brain"] = module
        spec.loader.exec_module(module)
    return sys.modules["brain"]


# ---------------------------------------------------------------------------
# Turn sources
# ---------------------------------------------------------------------------

def load_fixture_turns(path: str) -> tuple[str, str, list[tuple[str, str]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    turns: list[tuple[str, str]] = []
    for turn in data.get("turns", []):
        if isinstance(turn, dict):
            turns.append((str(turn.get("user", "")), str(turn.get("assistant", ""))))
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            turns.append((str(turn[0]), str(turn[1])))
    return str(data.get("session_id", "fixture")), str(data.get("platform", "cli")), turns


def _cell_text(value) -> str:
    """Coerce a state.db content cell (plain text or JSON parts) to text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    text = str(value)
    if text[:1] in ("[", "{"):
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text
        if isinstance(parsed, list):
            parts = []
            for part in parsed:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            return "\n".join(p for p in parts if p)
        if isinstance(parsed, dict):
            return str(parsed.get("text") or parsed.get("content") or text)
    return text


def load_state_db_turns(path: str, session_id: str) -> list[tuple[str, str]]:
    """Read a real Hermes state.db READ-ONLY and pair user/assistant rows.

    The messages table has role and content columns; everything else is
    schema drift we tolerate via SELECT * + dict access. Tool/system rows
    are skipped; consecutive user/assistant rows become turns.
    """
    # Percent-encode: SQLite URIs decode %XX and treat '#'/'?' as delimiters
    # (review finding #22 — a raw path with '%'/'#' opens an EMPTY database).
    from urllib.parse import quote

    uri = "file:" + quote(str(Path(path).resolve()).replace("\\", "/")) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        try:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY id", (session_id,)
            ).fetchall()
        except sqlite3.OperationalError:
            # session column named differently (or absent): fetch all, filter soft.
            try:
                rows = conn.execute("SELECT * FROM messages ORDER BY id").fetchall()
            except sqlite3.OperationalError as e:
                # No usable messages table at all — teach, don't traceback
                # (review finding #23). Exit code 4 is documented in main().
                raise StateDbError(
                    f"no usable 'messages' table in {path} ({e}); is this really a "
                    f"Hermes state.db? (brain.db has episodes/ingest_buffer, not messages)"
                ) from e
            rows = [r for r in rows
                    if "session_id" not in r or str(r["session_id"]) == session_id]
    finally:
        conn.close()

    turns: list[tuple[str, str]] = []
    pending_user: str | None = None
    for row in rows:
        record = dict(row)
        role = str(record.get("role") or record.get("message_role") or "").lower()
        content = _cell_text(record.get("content", record.get("message")))
        if role == "user":
            pending_user = content  # a user row with no assistant reply yet is replaced
        elif role == "assistant":
            if pending_user is not None and (pending_user.strip() or content.strip()):
                turns.append((pending_user, content))
            pending_user = None
        # tool / system / anything else: skipped
    return turns


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(p / 100.0 * (len(ordered) - 1))))
    return ordered[index]


def _diff_excerpt(base: str, sample: str, context: int = 60) -> str:
    limit = min(len(base), len(sample))
    at = next((i for i in range(limit) if base[i] != sample[i]), limit)
    lo = max(0, at - context)
    return (f"  first divergence at char {at} (base {len(base)}ch, sample {len(sample)}ch)\n"
            f"  base:   {base[lo:at + context]!r}\n"
            f"  sample: {sample[lo:at + context]!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a recorded session through the real BrainProvider (no LLM).")
    parser.add_argument("--fixture", help="JSON fixture (tests/fixtures/session_basic.json)")
    parser.add_argument("--state-db", help="real Hermes state.db, opened read-only")
    parser.add_argument("--session-id", help="session to pull from --state-db")
    parser.add_argument("--hermes-home", help="target home (default: throwaway temp dir)")
    parser.add_argument("--turns", type=int, default=0, help="replay at most N turns")
    parser.add_argument("--assert-lane1-stable", action="store_true",
                        help="exit 2 unless system_prompt_block() is byte-identical throughout")
    parser.add_argument("--budget-check", action="store_true",
                        help="exit 3 if any prefetch() result exceeds config lane2_tokens")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.state_db:
        if not args.session_id:
            parser.error("--state-db requires --session-id")
        session_id = f"replay-{args.session_id}"
        try:
            turns = load_state_db_turns(args.state_db, args.session_id)
        except StateDbError as e:
            print(f"replay: {e}", file=sys.stderr)
            return 4
    elif args.fixture:
        session_id, _platform, turns = load_fixture_turns(args.fixture)
    else:
        parser.error("one of --fixture or --state-db is required")

    if args.turns:
        turns = turns[: args.turns]
    if not turns:
        print("no turns to replay")
        return 0

    _load_brain()
    from brain.config import load_config
    from brain.provider import BrainProvider
    from brain.store import db

    home = Path(args.hermes_home) if args.hermes_home else Path(
        tempfile.mkdtemp(prefix="brain-replay-"))
    lane2_budget = int(load_config(home)["lane2_tokens"])

    provider = BrainProvider()
    provider.initialize(session_id, hermes_home=str(home), platform="replay",
                        agent_context="primary")

    lane1_samples = [("before-turn-1", provider.system_prompt_block())]
    messages: list[dict] = []
    latencies_ms: list[float] = []
    budget_violations: list[tuple[int, int]] = []
    midpoint = max(1, len(turns) // 2)

    for n, (user, assistant) in enumerate(turns, start=1):
        provider.on_turn_start(n, user)
        lane2 = provider.prefetch(user, session_id=session_id) or ""
        # [simulated model turn — no LLM in P1; a real turn takes seconds, which
        # is what lets the worker populate the lane-2 cache for the next turn]
        time.sleep(0.025)
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})

        started = time.perf_counter()
        provider.sync_turn(user, assistant, session_id=session_id, messages=list(messages))
        sync_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(sync_ms)

        provider.queue_prefetch(user, session_id=session_id)

        lane2_tokens = db.approx_tokens(lane2) if lane2 else 0
        if lane2 and lane2_tokens > lane2_budget:
            budget_violations.append((n, lane2_tokens))
        print(f"turn {n:3d}  sync {sync_ms:8.2f}ms  lane2 {lane2_tokens:4d}tok "
              f"{len(lane2):5d}ch")

        if n == midpoint:
            lane1_samples.append((f"mid-run-turn-{n}", provider.system_prompt_block()))
        if n % 25 == 0:
            provider.on_pre_compress(messages[-50:])
            lane1_samples.append((f"post-compress-turn-{n}", provider.system_prompt_block()))

    provider.on_session_end(list(messages))
    lane1_samples.append(("after-session-end", provider.system_prompt_block()))
    provider.shutdown()

    conn = db.connect(home)
    try:
        episodes = conn.execute(
            "SELECT COUNT(*) AS n FROM episodes WHERE session_id=?", (session_id,)
        ).fetchone()["n"]
        buffer_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM ingest_buffer WHERE session_id=?", (session_id,)
        ).fetchone()["n"]
    finally:
        conn.close()

    print("-" * 64)
    print(f"episodes captured: {episodes} / {len(turns)} turns")
    print(f"buffer rows:       {buffer_rows}")
    print(f"sync latency:      p50 {_pct(latencies_ms, 50):.2f}ms  "
          f"p95 {_pct(latencies_ms, 95):.2f}ms")
    print(f"hermes home:       {home}")

    if args.assert_lane1_stable:
        base_label, base = lane1_samples[0]
        for label, sample in lane1_samples[1:]:
            if sample != base:
                print(f"LANE1 UNSTABLE at {label} (vs {base_label}):")
                print(_diff_excerpt(base, sample))
                return 2
        print(f"lane1 stable: {len(lane1_samples)} samples, "
              f"{len(base.encode('utf-8'))} bytes each")

    if args.budget_check:
        if budget_violations:
            for n, tokens in budget_violations:
                print(f"BUDGET VIOLATION turn {n}: lane2 {tokens}tok > {lane2_budget}tok")
            return 3
        print(f"lane2 within budget ({lane2_budget}tok) on all {len(turns)} turns")

    return 0


if __name__ == "__main__":
    sys.exit(main())
