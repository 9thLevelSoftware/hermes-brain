"""brain_observer — companion "observer" plugin for Hermes-Brain (task B3).

The brain ships as a ``MemoryProvider`` (``plugins/brain/``). That contract
cannot see the host's ``pre_tool_call``/``post_tool_call``,
``subagent_start``/``subagent_stop``, ``pre_llm_call`` or ``kanban_task_*``
hooks — only a SECOND, general-purpose plugin (a ``register(ctx)`` entry
point the host ``PluginManager`` discovers) can. This is that plugin.

What it does
------------
Registers observer-only host hooks that ENQUEUE a lightweight signal row
into the brain's ``work_queue`` table (the currently-unused hand-off lane in
``store/schema.sql``). The brain's background worker (``provider.py``:
``_maybe_drain_work_queue``) drains those rows into bookkeeping. No LLM, no
heavy work, and no host dependency at import time.

Invariants honoured
--------------------
* **Import-light module level.** Only stdlib is imported here; the host
  eagerly imports plugin ``__init__.py`` on every CLI invocation. The host
  (``hermes_constants``) is imported lazily inside a function, guarded.
* **Hooks never block a turn and never raise into the host.** Each hook only
  drops a tuple onto an in-memory queue drained by a daemon thread that owns
  the short-lived ``brain.db`` connection; every path is wrapped and degrades
  silently. Under backpressure signals are dropped, never queued unbounded.
* **This plugin never opens the brain's long-lived connection.** It writes
  with its own short-lived connection and only INSERTs — it never creates the
  DB or the schema (that is the brain's job); if ``brain.db`` does not exist
  yet the signal is dropped.
* **The ``pre_llm_call`` context-injection lane ships OFF.** The brain's
  MemoryProvider already owns lane-1/lane-2 injection; a second injector here
  would double-inject and churn the prompt cache. It is a clearly-gated stub,
  enabled only by ``BRAIN_OBSERVER_PRE_LLM_CALL=1``.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# work_queue task names this plugin produces. Deliberately namespaced with an
# ``observed_`` prefix so they never collide with the schema's own
# embed|reembed|archive|extract vocabulary, and so the brain's drainer claims
# only the observer's own rows.
_TASK_TOOL_CALL = "observed_tool_call"
_TASK_SUBAGENT_STOP = "observed_subagent_stop"

# Context-injection lane — SHIPPED OFF (see module docstring).
_ENABLE_PRE_LLM_CALL = os.environ.get(
    "BRAIN_OBSERVER_PRE_LLM_CALL", ""
).strip().lower() in {"1", "true", "yes", "on"}

# Kill switch: set to fully silence the observer without unregistering it.
_DISABLED = os.environ.get(
    "BRAIN_OBSERVER_DISABLE", ""
).strip().lower() in {"1", "true", "yes", "on"}

# Bound on the in-memory hand-off queue: signals are best-effort telemetry, so
# under sustained backpressure we drop rather than grow memory or slow a turn.
_MAX_PENDING = 2048
# Max signals drained into ONE connection + ONE commit. Batching turns a burst
# of tool-call hooks into a single fsync instead of one open/commit per signal.
_DRAIN_BATCH = 256


# ---------------------------------------------------------------------------
# Paths / time
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    """ISO-8601 UTC with millisecond precision — matches store/db.iso_now()."""
    t = time.time()
    ms = int((t % 1) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{ms:03d}Z"


def _hermes_home() -> Path | None:
    """Resolve the active Hermes home. Prefer the host's single source of
    truth (correct across profiles), falling back to the ``HERMES_HOME`` env
    var and then the conventional default. The host import is lazy + guarded
    so this module never hard-depends on hermes-agent."""
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return Path(get_hermes_home())
    except Exception:
        pass
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def _brain_db_path() -> Path | None:
    home = _hermes_home()
    if home is None:
        return None
    return Path(home) / "brain" / "brain.db"


# ---------------------------------------------------------------------------
# Background signal writer — the only place a DB connection is opened
# ---------------------------------------------------------------------------

class _SignalWriter:
    """A single daemon thread that owns the short-lived ``brain.db`` writes.

    Hooks hand rows to :meth:`submit` (a non-blocking ``put_nowait``); this
    thread does the actual INSERT off the turn path so a locked or slow DB can
    never stall the agent. Fire-and-forget: failures are logged at DEBUG and
    dropped.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(
            maxsize=_MAX_PENDING
        )
        self._thread = threading.Thread(
            target=self._run, name="brain-observer", daemon=True
        )
        self._thread.start()

    def submit(self, task: str, payload: dict[str, Any]) -> None:
        try:
            self._q.put_nowait((task, payload))
        except queue.Full:
            # Best-effort telemetry: drop under backpressure rather than block
            # the hook (which runs on the host's tool-execution thread).
            pass

    def _run(self) -> None:
        while True:
            try:
                item = self._q.get()
            except Exception:
                continue
            if item is None:
                return
            batch = [item]
            # Drain whatever else is already queued so a burst of hooks becomes
            # ONE connection + ONE commit (one fsync) rather than one per signal.
            while len(batch) < _DRAIN_BATCH:
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._flush(batch)
                    return
                batch.append(nxt)
            self._flush(batch)

    @staticmethod
    def _flush(batch: list[tuple[str, dict[str, Any]]]) -> None:
        # Honour the kill switch LIVE (re-read at write time, not just at
        # import): BRAIN_OBSERVER_DISABLE=1 silences the writer even for signals
        # already in flight. Never create the DB/schema — that is the brain's
        # job; if the brain has never initialized, drop the signals.
        if not batch or os.environ.get(
                "BRAIN_OBSERVER_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}:
            return
        db_path = _brain_db_path()
        if db_path is None or not db_path.exists():
            return
        conn = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=2.0)
            conn.execute("PRAGMA busy_timeout=2000")
            conn.executemany(
                "INSERT INTO work_queue(task, payload, created_at) VALUES(?,?,?)",
                [(task, json.dumps(payload, separators=(",", ":")), _iso_now())
                 for task, payload in batch],
            )
            conn.commit()
        except Exception:
            logger.debug("brain-observer: batch signal write failed", exc_info=True)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


_writer_lock = threading.Lock()
_writer: _SignalWriter | None = None


def _get_writer() -> _SignalWriter:
    """Lazily start the background writer on first use (keeps import light)."""
    global _writer
    w = _writer
    if w is None:
        with _writer_lock:
            if _writer is None:
                _writer = _SignalWriter()
            w = _writer
    return w


def _enqueue(task: str, payload: dict[str, Any]) -> None:
    if _DISABLED:
        return
    _get_writer().submit(task, payload)


# ---------------------------------------------------------------------------
# Hooks (observer-only: they never return a directive; every path degrades)
# ---------------------------------------------------------------------------

def _on_post_tool_call(**kwargs: Any) -> None:
    """Enqueue a lightweight signal after every tool call.

    Host kwargs (hermes_cli.plugins / model_tools._emit_post_tool_call_hook):
    ``tool_name, args, result, task_id, session_id, tool_call_id, turn_id,
    api_request_id, duration_ms, status, error_type, error_message,
    middleware_trace, telemetry_schema_version``.

    We record only cheap metadata — never ``args`` or ``result`` (they may be
    large or sensitive). ``status`` is the host's disposition ("ok" | "error").
    """
    try:
        payload: dict[str, Any] = {
            "tool_name": kwargs.get("tool_name") or "",
            "disposition": kwargs.get("status") or "ok",
            "session_id": kwargs.get("session_id") or "",
            "turn_id": kwargs.get("turn_id") or "",
            "tool_call_id": kwargs.get("tool_call_id") or "",
            "duration_ms": kwargs.get("duration_ms") or 0,
        }
        error_type = kwargs.get("error_type")
        if error_type:
            payload["error_type"] = error_type
        _enqueue(_TASK_TOOL_CALL, payload)
    except Exception:
        logger.debug("brain-observer: post_tool_call hook error", exc_info=True)
    return None


def _on_subagent_stop(**kwargs: Any) -> None:
    """Enqueue a signal when a delegated subagent finishes.

    Host kwargs (tools/delegate_tool.py): ``parent_session_id,
    parent_turn_id, child_session_id, child_role, child_summary, child_status,
    duration_ms, telemetry_schema_version``. A richer signal than the
    provider's ``on_delegation`` (which sees only task/result): role, exit
    status, and wall-clock duration.
    """
    try:
        summary = kwargs.get("child_summary") or ""
        payload: dict[str, Any] = {
            "child_session_id": kwargs.get("child_session_id") or "",
            "parent_session_id": kwargs.get("parent_session_id") or "",
            "child_role": kwargs.get("child_role") or "",
            "disposition": kwargs.get("child_status") or "unknown",
            "duration_ms": kwargs.get("duration_ms") or 0,
            # Summaries can be large; keep only a short prefix.
            "summary_preview": summary[:200] if summary else "",
        }
        _enqueue(_TASK_SUBAGENT_STOP, payload)
    except Exception:
        logger.debug("brain-observer: subagent_stop hook error", exc_info=True)
    return None


def _on_pre_llm_call(**kwargs: Any) -> dict[str, Any] | None:
    """Context-injection lane — SHIPPED OFF (see module docstring).

    When ``BRAIN_OBSERVER_PRE_LLM_CALL=1``, this could return
    ``{"context": "..."}`` to inject brain context into the user message. It
    is left inert: the brain's MemoryProvider already owns lane-1/lane-2
    injection, and a second injector here would double-inject and churn the
    prompt cache. Do NOT wire this without coordinating with the provider's
    lanes.
    """
    if not _ENABLE_PRE_LLM_CALL:
        return None
    # Intentionally inert even when the flag is on — this is the extension
    # point, not a live feature. Return None so nothing is injected.
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def register(ctx: Any) -> None:
    """Discovered by the host PluginManager. Registers observer hooks.

    Wrapped so a registration failure can never break host plugin discovery.
    """
    try:
        ctx.register_hook("post_tool_call", _on_post_tool_call)
        ctx.register_hook("subagent_stop", _on_subagent_stop)
        if _ENABLE_PRE_LLM_CALL:
            ctx.register_hook("pre_llm_call", _on_pre_llm_call)
        _register_aux_tasks(ctx)
        logger.debug(
            "brain-observer: registered hooks (pre_llm_call=%s, disabled=%s)",
            _ENABLE_PRE_LLM_CALL, _DISABLED,
        )
    except Exception:
        logger.warning("brain-observer: register() failed", exc_info=True)


def _register_aux_tasks(ctx: Any) -> None:
    """Surface the brain's sleep-time aux tasks in `hermes model → Configure
    auxiliary models`. Runtime routing already works from the config block
    brain_setup.post_setup writes; this only adds the picker entries. Guarded —
    the memory-provider ctx or an older host may lack the method, and the
    signature is best-effort."""
    reg = getattr(ctx, "register_auxiliary_task", None)
    if not callable(reg):
        return
    try:
        reg("brain_extract", "Brain: extraction",
            "Cheap/fast tier for the brain's memory-extraction sweep.",
            {"provider": "", "model": ""})
        reg("brain_consolidate", "Brain: consolidation",
            "Stronger tier for the brain's nightly consolidation/distillation.",
            {"provider": "", "model": ""})
    except Exception:
        logger.debug("brain-observer: aux-task registration skipped", exc_info=True)
