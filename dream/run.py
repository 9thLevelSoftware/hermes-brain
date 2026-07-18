"""The dream phase machine: acquire the lease, run the ordered strategy
pipeline with per-strategy mode/cooldown/preemption/budget gating, record
an idempotent cursor, release.

Idempotent rerun (plan P4): shift_runs.phases_done is the cursor. A dream
killed mid-run and restarted with the SAME shift_id skips completed phases;
a fresh dream starts clean. Each strategy commits its own work, so a crash
never leaves a half-written batch (the strategies themselves are
transactional per unit).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
from typing import Any

from ..store import db
from . import lease
from .shift import PIPELINE, Shift

logger = logging.getLogger(__name__)


def _holder_tag(actor: str) -> str:
    try:
        host = socket.gethostname()[:20]
    except Exception:
        host = "?"
    return f"{actor}:{os.getpid()}@{host}"


def _strategy_fn(name: str):
    """Lazy-import the strategy callable (heavy deps stay out of module load)."""
    if name == "flush":
        from ..capture.extract import sweep

        def _run(shift: Shift) -> dict:
            # Bigger budget than the provider's idle tick — the dream is the
            # primary extraction path. Honor the forced mode (finding #5):
            # sweep gates on extract_mode, so a dream-wide dry_run/off must
            # map onto it, or --dry-run would still write memories. dry_run
            # -> shadow (compute, audit, no INSERT); off -> off.
            forced = shift.config.get("_forced_mode")
            cfg = shift.config
            if forced in ("dry_run", "shadow"):
                cfg = {**shift.config, "extract_mode": "shadow"}
            elif forced == "off":
                cfg = {**shift.config, "extract_mode": "off"}
            return sweep(shift.conn, cfg, embedder=shift.embedder,
                         actor=f"dream:{shift.shift_id}", max_rows=60,
                         max_llm_calls=6)
        return _run
    if name == "mine":
        from .mine_state import run as fn
    elif name == "cases":
        from .cases import run as fn
    elif name == "distill":
        from .distill import run as fn
    elif name == "consolidate":
        from .consolidate import run as fn
    elif name == "contradict":
        from .contradict import run as fn
    elif name == "forget":
        from .forget import run as fn
    elif name == "forge":
        def fn(shift: Shift) -> dict:
            mode = shift.config.get("_forced_mode") or shift.mode("forge")
            if mode not in ("active", "dry_run"):
                return {"skipped": mode}
            from ..skillforge import forge_once

            # dry_run drafts + validates but must NOT promote a skill into the
            # tree — force auto-approve off so the proposal stays reviewable.
            config = shift.config
            if mode == "dry_run":
                config = {**config, "skill_auto_approve": False}
            return forge_once(shift.conn, config, embedder=shift.embedder,
                              shift_id=shift.shift_id)
    elif name == "revise":
        def fn(shift: Shift) -> dict:
            mode = shift.config.get("_forced_mode") or shift.mode("revise")
            if mode not in ("active", "dry_run"):
                return {"skipped": mode}
            from ..skillforge import revise_once

            # revise ONLY ever writes reviewable proposals — it never applies a
            # revision or retirement (the CLI `hermes brain review` does). A
            # proposal is the reversible, review-gated artifact, so dry_run and
            # active behave identically here — exactly as forge's dry_run still
            # writes its draft proposal and only withholds the live tree write.
            return revise_once(shift.conn, shift.config, embedder=shift.embedder,
                               shift_id=shift.shift_id)
    elif name == "peers":
        def fn(shift: Shift) -> dict:
            # Theory-of-mind peer modeling (D3). Mode-gated like forge/revise;
            # peers additionally supports `shadow` (silent compute). 'off' is
            # already filtered upstream in run_dream before dispatch, so the
            # only mode that reaches here and isn't meaningful is a stray one.
            mode = shift.config.get("_forced_mode") or shift.mode("peers")
            if mode not in ("active", "dry_run", "shadow"):
                return {"skipped": mode}
            from .peers import run as peers_run

            return peers_run(shift)
    elif name == "tune":
        from .tune import run as fn
    elif name == "probes":
        from .probes import run as fn
    elif name == "lane1":
        def fn(shift: Shift) -> dict:
            from ..recall import lane1

            n = lane1.materialize(shift.conn, shift.config)
            shift.conn.commit()
            return {"lane1_rows": n}
    else:
        raise KeyError(name)
    return fn


def run_dream(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    *,
    embedder=None,
    phase: str | None = None,
    dry_run: bool | None = None,
    actor: str = "dream",
    resume_shift_id: str | None = None,
) -> dict:
    """Run one dream shift. Returns a summary dict (never raises — a strategy
    failure is logged and the pipeline continues; the lease is always freed).

    phase: run just one strategy (dream-now --phase X); None = full pipeline.
    dry_run: force dry-run over every strategy's configured mode (the
             CLI --dry-run flag); None = honor each strategy's mode.
    resume_shift_id: continue an interrupted shift (idempotent cursor).
    """
    holder = _holder_tag(actor)
    if not lease.acquire(conn, "dream", holder):
        held = lease.held_by(conn, "dream")
        logger.info("dream: lease held by %s; skipping", held)
        return {"skipped": "lease_held", "holder": held}

    # From here the lease is HELD — everything is inside try/finally so a
    # raise in setup (_open_shift, Shift construction) can never leak it
    # (review finding #4: the lease is a committed row; closing the conn
    # does not free it).
    shift_id: str | None = None
    summary: dict[str, Any] = {"strategies": {}}
    try:
        strategies = (phase,) if phase else PIPELINE
        if phase and phase not in PIPELINE:
            return {"error": f"unknown phase '{phase}'",
                    "recovery_hint": "phases: " + "|".join(PIPELINE)}

        shift_id, done, started = _open_shift(conn, resume_shift_id, actor)
        summary["shift_id"] = shift_id
        baseline = _activity_baseline(conn)
        shift = Shift(
            shift_id=shift_id, conn=conn, config=config, embedder=embedder,
            started_at=started, activity_baseline=baseline, holder=holder,
        )

        for name in strategies:
            if name in done:
                summary["strategies"][name] = {"skipped": "already_done"}
                continue
            if shift.preempted():
                summary["strategies"][name] = {"skipped": "preempted"}
                continue
            mode = "dry_run" if dry_run else shift.mode(name)
            if mode == "off":
                summary["strategies"][name] = {"skipped": "off"}
                continue
            if not _cooldown_ok(conn, name):
                summary["strategies"][name] = {"skipped": "cooldown"}
                continue
            summary["strategies"][name] = _run_one(shift, name, mode)
            # Only mark a strategy done if it actually completed (not
            # errored/preempted) so an idempotent resume re-runs it.
            entry = summary["strategies"][name]
            if "error" not in entry and entry.get("skipped") != "preempted":
                done.append(name)
                _record_done(conn, shift_id, done)
            # A holder that has lost the lease must stop touching memory
            # immediately (finding #6).
            if not lease.renew(conn, "dream", holder):
                logger.warning("dream %s: lease lost mid-pipeline; aborting", shift_id)
                summary["aborted"] = "lease_lost"
                break
    except Exception as e:
        # run_dream never raises (docstring): a setup error still releases the
        # lease in finally and returns a teaching error.
        logger.warning("dream: run failed: %s", e, exc_info=True)
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        summary["error"] = str(e)
    finally:
        if shift_id is not None:
            _close_shift(conn, shift_id, summary)
        lease.release(conn, "dream", holder)
    return summary


def _run_one(shift: Shift, name: str, mode: str) -> dict:
    # The strategy reads its own mode via shift.mode(); we override strategy
    # state for this run by stashing the forced mode where the strategy reads
    # it. Simplest: pass mode through config for the strategy's own gate.
    shift.config = {**shift.config, "_forced_mode": mode}
    try:
        fn = _strategy_fn(name)
        result = fn(shift)
        _mark_run(shift.conn, name)
        logger.info("dream %s: %s -> %s [%s]", shift.shift_id, name, result, mode)
        return {"mode": mode, **(result or {})}
    except Exception as e:
        logger.warning("dream %s: strategy %s failed: %s", shift.shift_id, name, e,
                       exc_info=True)
        try:
            shift.conn.rollback()
        except sqlite3.Error:
            pass
        _mark_failure(shift.conn, name)
        return {"mode": mode, "error": str(e)}


# ---------------------------------------------------------------------------
# shift_runs bookkeeping
# ---------------------------------------------------------------------------

def _open_shift(conn, resume_shift_id, actor) -> tuple:
    if resume_shift_id:
        row = conn.execute(
            "SELECT phases_done, started_at FROM shift_runs WHERE shift_id=?",
            (resume_shift_id,)).fetchone()
        if row is not None:
            done = json.loads(row["phases_done"] or "[]")
            return resume_shift_id, done, row["started_at"]
    shift_id = db.new_ulid()
    now = db.iso_now()
    conn.execute(
        "INSERT INTO shift_runs (shift_id, kind, started_at, phases_done)"
        " VALUES (?,?,?,'[]')", (shift_id, "dream", now))
    conn.commit()
    return shift_id, [], now


def _record_done(conn, shift_id, done) -> None:
    conn.execute("UPDATE shift_runs SET phases_done=? WHERE shift_id=?",
                 (json.dumps(done), shift_id))
    conn.commit()


def _close_shift(conn, shift_id, summary) -> None:
    # Derive the true outcome (finding #1): a preempted or errored shift must
    # not be stamped 'completed'.
    strategies = summary.get("strategies", {}).values()
    if summary.get("aborted"):
        outcome = "aborted"
    elif any(s.get("skipped") == "preempted" for s in strategies):
        outcome = "preempted"
    elif any("error" in s for s in strategies):
        outcome = "failed"
    else:
        outcome = "completed"
    conn.execute(
        "UPDATE shift_runs SET finished_at=?, outcome=?, notes=? WHERE shift_id=?",
        (db.iso_now(), outcome, json.dumps(summary)[:4000], shift_id))
    conn.commit()


# ---------------------------------------------------------------------------
# strategy_state (mode + cooldown + backoff)
# ---------------------------------------------------------------------------

def _cooldown_ok(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT cooldown_until FROM strategy_state WHERE strategy=?", (name,)
    ).fetchone()
    if not row or not row["cooldown_until"]:
        return True
    return row["cooldown_until"] < db.iso_now()


def _mark_run(conn, name: str) -> None:
    conn.execute(
        "INSERT INTO strategy_state (strategy, last_run_at, stats) VALUES (?,?,'{}')"
        " ON CONFLICT(strategy) DO UPDATE SET last_run_at=excluded.last_run_at",
        (name, db.iso_now()))
    conn.commit()


def _mark_failure(conn, name: str) -> None:
    # Exponential-ish backoff via a short cooldown after a failure.
    from .lease import _future_iso

    conn.execute(
        "INSERT INTO strategy_state (strategy, cooldown_until, stats)"
        " VALUES (?,?,'{}') ON CONFLICT(strategy) DO UPDATE SET"
        " cooldown_until=excluded.cooldown_until",
        (name, _future_iso(3600)))
    conn.commit()


def _activity_baseline(conn) -> str:
    row = conn.execute("SELECT MAX(last_seen) AS m FROM activity").fetchone()
    return (row["m"] if row else None) or ""
