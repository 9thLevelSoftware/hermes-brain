"""work_queue: brain-side drain of companion-observer host signals (task B3).

The ``work_queue`` table (store/schema.sql) is the hand-off between the
out-of-tree ``brain_observer`` plugin — which can reach host hooks the
MemoryProvider contract cannot (``post_tool_call``, ``subagent_stop``, ...) —
and the brain's background worker. The observer INSERTs lightweight signal
rows with its own short-lived connection; the single long-lived brain-bg
worker drains them here into bookkeeping (an ``audit_log`` summary + an
``activity`` heartbeat). No new tables.

Schema reality
--------------
``work_queue`` has **no** ``claimed_by`` / ``promoted_at`` columns — the
actual columns are ``id, task, payload, created_at, attempts, done_at``
(``idx_work_pending`` covers ``done_at IS NULL``). Rows are therefore
"claimed"/finished by setting ``done_at``. The finishing UPDATE re-checks
``done_at IS NULL`` so a concurrent drainer (another Hermes process's worker)
cannot double-finish a row.

This module is a store/ subpackage (not a root module), so it may import
freely at module level.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .db import iso_now

logger = logging.getLogger(__name__)

# The task names this module owns. Kept distinct from the schema's
# embed|reembed|archive|extract vocabulary so a future in-process producer that
# uses work_queue for embedding is never consumed by the observer drain.
OBSERVER_TASKS = ("observed_tool_call", "observed_subagent_stop")


def enqueue(conn, task: str, payload: dict[str, Any]) -> int | None:
    """Insert a signal row (id/attempts/done_at take their schema defaults).

    Used by tests and any in-process producer. The out-of-tree observer plugin
    writes the identical shape with its own stdlib connection (it cannot import
    brain). Caller commits.
    """
    try:
        cur = conn.execute(
            "INSERT INTO work_queue(task, payload, created_at) VALUES(?,?,?)",
            (task, json.dumps(payload, separators=(",", ":")), iso_now()),
        )
        return cur.lastrowid
    except Exception:
        logger.debug("work_queue.enqueue failed", exc_info=True)
        return None


def pending_count(conn) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM work_queue WHERE done_at IS NULL"
    ).fetchone()
    return int(row["n"]) if row else 0


def drain_observer_signals(conn, *, limit: int = 256) -> dict[str, Any]:
    """Claim up to ``limit`` pending observer rows, fold them into a single
    ``audit_log`` summary + an ``activity`` heartbeat, and mark them done.

    Returns a summary dict ``{count, claimed, tools, subagents, errors}``
    (``{"count": 0}`` when idle). Best-effort: on the empty path it does no
    writes; the single-statement ``done_at`` claim (UPDATE ... WHERE
    ``done_at IS NULL``) is the atomic hand-off. Callers wrap this — it commits
    its own batch on success.
    """
    placeholders = ",".join("?" * len(OBSERVER_TASKS))
    rows = conn.execute(
        f"SELECT id, task, payload FROM work_queue "
        f"WHERE done_at IS NULL AND task IN ({placeholders}) "
        f"ORDER BY id LIMIT ?",
        (*OBSERVER_TASKS, int(limit)),
    ).fetchall()
    if not rows:
        return {"count": 0}

    ids = [r["id"] for r in rows]
    tools: dict[str, int] = {}
    subagents = 0
    errors = 0
    _ok_dispositions = {None, "", "ok", "success", "completed", "unknown"}
    for r in rows:
        try:
            payload = json.loads(r["payload"])
        except Exception:
            payload = {}
        if r["task"] == "observed_tool_call":
            name = str(payload.get("tool_name") or "?")
            tools[name] = tools.get(name, 0) + 1
            if payload.get("disposition") not in _ok_dispositions:
                errors += 1
        elif r["task"] == "observed_subagent_stop":
            subagents += 1
            if payload.get("disposition") not in _ok_dispositions:
                errors += 1

    now = iso_now()
    # Atomic claim: only rows still pending are finished, so a concurrent
    # drainer in another process cannot double-finish them.
    id_placeholders = ",".join("?" * len(ids))
    claimed = conn.execute(
        f"UPDATE work_queue SET done_at=?, attempts=attempts+1 "
        f"WHERE id IN ({id_placeholders}) AND done_at IS NULL",
        (now, *ids),
    ).rowcount

    detail = {
        "count": len(ids),
        "claimed": int(claimed) if claimed is not None else len(ids),
        "tools": tools,
        "subagents": subagents,
        "errors": errors,
    }
    # One audit row per drain BATCH (not per signal) — bounded bookkeeping.
    conn.execute(
        "INSERT INTO audit_log(actor, action, target, detail, ts) VALUES(?,?,?,?,?)",
        ("observer", "drain", None, json.dumps(detail, separators=(",", ":")), now),
    )
    # Liveness heartbeat (one upserted row) so idle-detection / doctor can see
    # the observer feed is flowing.
    conn.execute(
        "INSERT INTO activity(source, last_seen) VALUES('observer', ?) "
        "ON CONFLICT(source) DO UPDATE SET last_seen=excluded.last_seen",
        (now,),
    )
    conn.commit()
    return detail
