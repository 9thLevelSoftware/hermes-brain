"""Shift: the per-run context every dream strategy receives, plus the
strategy protocol and shared helpers (mode/cooldown state, preemption,
budget, staged-write bookkeeping).

Ship-inert (docs/design/learning-system.md §3): every mutating strategy has
a mode in {off, shadow, dry_run, active}. This release defaults the
mutating strategies to `dry_run` — they compute exactly what they would do
and record it to audit_log, but write no live memory changes. `shadow` is
identical minus even the dry-run audit noise; `active` actually mutates.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..store import db
from . import lease

logger = logging.getLogger(__name__)

# Default per-strategy modes for THIS release (ship-inert): read-only
# strategies run active; anything that mutates memory ships dry_run until
# the user promotes it via `hermes brain dream --enable <strategy>`.
DEFAULT_MODES = {
    "flush": "active",          # extraction already reviewed in P3
    "mine": "active",           # only updates helpful/harmful counters + edges
    "cases": "dry_run",         # writes episodic case rows (P5)
    "distill": "dry_run",       # writes procedural strategy/guardrail items (P5)
    "consolidate": "dry_run",   # writes new semantic patterns
    "contradict": "dry_run",    # invalidates contradicted rows
    "forget": "dry_run",        # demotes/tombstones
    "tune": "shadow",           # retrieval-weight tuning — NEVER active in v1;
                                # only ever proposes (design §2: shadow-logged,
                                # reviewed before activation)
    "probes": "active",         # read-only health check; writes only review rows
    "lane1": "active",          # re-renders the index (idempotent, safe)
}

# Ordered pipeline (learning-system.md §1.2): facts -> outcome credit ->
# case bank -> strategy distillation -> semantic consolidation -> contradiction
# -> forgetting -> weight tuning -> post-shift probes -> index re-render.
# cases runs before distill so the case bank the skill-forge reads is fresh;
# probes runs last so it checks the shift's net effect; tune runs after
# forgetting so it sees the settled working set.
PIPELINE = ("flush", "mine", "cases", "distill", "consolidate", "contradict",
            "forget", "tune", "probes", "lane1")

_PREEMPT_CHECK_EVERY = 8  # work units between activity re-checks


@dataclass
class Shift:
    """Context handed to each strategy. Strategies never open their own
    connection or touch the lease — they use this."""

    shift_id: str
    conn: sqlite3.Connection
    config: dict[str, Any]
    embedder: Any = None
    started_at: str = ""
    activity_baseline: str = ""     # activity newer than this => user returned
    holder: str = ""
    kind: str = "dream"             # 'dream' | 'idle'
    counts: dict[str, int] = field(default_factory=dict)
    _work_since_check: int = 0
    _preempted: bool = False
    _last_renew: float = 0.0

    # -- preemption (Daem0n's cooperative yield, worker-thread flavored) ----

    def preempted(self) -> bool:
        """True once the user has become active since the shift began, OR the
        lease was lost. Checked between work units and before every LLM call;
        a strategy that sees it should stop cleanly and return."""
        if self._preempted:
            return True
        row = self.conn.execute(
            "SELECT MAX(last_seen) AS m FROM activity").fetchone()
        latest = (row["m"] if row else None) or ""
        if latest > self.activity_baseline:
            self._preempted = True
            logger.info("dream %s preempted: user active at %s", self.shift_id, latest)
        return self._preempted

    def keepalive(self) -> bool:
        """Time-based lease renewal — call before every LLM call and every
        few work units. Returns False if the lease was LOST (TTL lapsed and
        another process took over, or preemption): the caller MUST stop
        mutating immediately (review findings #3/#6 — a holder that lost the
        lease keeps no authority to write). Renews on a wall-clock cadence so
        a slow LLM unit can't let the 120s TTL expire mid-strategy."""
        if self._preempted:
            return False
        now = time.monotonic()
        if self._last_renew and now - self._last_renew < lease.RENEW_SECONDS:
            return True  # renewed recently; still ours
        self._last_renew = now
        if not lease.renew(self.conn, "dream", self.holder):
            logger.warning("dream %s: lease lost — yielding", self.shift_id)
            self._preempted = True
            return False
        return True

    def tick(self) -> bool:
        """Call once per work unit. Returns True to keep going, False to yield
        (lease lost or user returned)."""
        if not self.keepalive():
            return False
        self._work_since_check += 1
        if self._work_since_check >= _PREEMPT_CHECK_EVERY:
            self._work_since_check = 0
            return not self.preempted()
        return True

    # -- budget -------------------------------------------------------------

    def budget_left(self) -> bool:
        """False once the night's dollar budget (token proxy) is spent. LLM
        strategies check this before each call; llm.call_* also enforces the
        daily gate independently."""
        day = db.iso_now()[:10]
        used = self.conn.execute(
            "SELECT COALESCE(SUM(tokens_in + tokens_out), 0) FROM llm_ledger "
            "WHERE ts LIKE ?", (day + "%",),
        ).fetchone()[0]
        budget_usd = float(self.config.get("night_budget_usd", 0.5))
        # Same crude $2.50/Mtok proxy as llm.py until real pricing plumbing.
        return used < budget_usd * 400_000

    # -- mode ---------------------------------------------------------------

    def mode(self, strategy: str) -> str:
        row = self.conn.execute(
            "SELECT mode FROM strategy_state WHERE strategy=?", (strategy,)
        ).fetchone()
        if row and row["mode"]:
            return row["mode"]
        return DEFAULT_MODES.get(strategy, "dry_run")

    # -- audit --------------------------------------------------------------

    def audit(self, action: str, target: str | None, detail: dict) -> None:
        self.conn.execute(
            "INSERT INTO audit_log (actor, action, target, detail, ts)"
            " VALUES (?,?,?,?,?)",
            (f"dream:{self.shift_id}", action, target, json.dumps(detail),
             db.iso_now()),
        )


# Strategy protocol: (shift) -> counts dict. Registered in run.py.
Strategy = Callable[[Shift], dict]


def now_monotonic() -> float:
    return time.monotonic()
