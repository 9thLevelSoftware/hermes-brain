#!/usr/bin/env python3
"""Multi-PROCESS lease-race contender for the dream lease.

Each invocation opens its OWN brain.db connection and tries to take the 'dream'
lease at a shared start instant, then holds it briefly. The orchestrator runs N
of these concurrently against ONE $HERMES_HOME and asserts EXACTLY ONE prints
`WON` (exit 0) — proving WAL serialization makes the atomic acquire a true
mutual exclusion across processes, not just threads (the in-process thread race
is covered by tests/adversarial/test_concurrency.py).

Env:
  GOLDEN_HOME / HERMES_HOME  the shared brain home
  START_EPOCH                unix seconds to synchronize the acquire attempt
  HOLD_SECONDS               how long the winner holds the lease (default 2)
Exit: 0 == won the lease, 3 == lost (held by another process).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time


def _register_brain():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if "brain" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "brain", os.path.join(root, "__init__.py"), submodule_search_locations=[root])
    module = importlib.util.module_from_spec(spec)
    sys.modules["brain"] = module
    spec.loader.exec_module(module)


def main() -> int:
    _register_brain()
    from brain.dream import lease
    from brain.store import db

    home = os.environ.get("GOLDEN_HOME") or os.environ["HERMES_HOME"]
    holder = f"proc-{os.getpid()}"
    hold = float(os.environ.get("HOLD_SECONDS", "2"))

    conn = db.connect(home)
    # Synchronize the attempt so the contenders genuinely overlap.
    start = float(os.environ.get("START_EPOCH", "0"))
    while start and time.time() < start:
        time.sleep(0.002)

    won = lease.acquire(conn, "dream", holder)
    print(f"{'WON' if won else 'LOST'} {holder}", flush=True)
    if won:
        time.sleep(hold)  # hold so every concurrent contender definitely contends
        lease.release(conn, "dream", holder)
    conn.close()
    return 0 if won else 3


if __name__ == "__main__":
    raise SystemExit(main())
