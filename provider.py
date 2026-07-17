"""BrainProvider — the Hermes MemoryProvider implementation.

Threading model (docs/design/integration.md §1.1):
  * Hook methods called by Hermes must be fast and never block a turn:
    ``prefetch``/``system_prompt_block`` read cached strings; ``sync_turn``/
    ``queue_prefetch`` enqueue and return in microseconds.
  * One owned daemon worker thread ("brain-bg") holds the only long-lived
    SQLite connection and does all real work: episode capture, retrieval,
    buffer rows. If its connection fails it retries with backoff instead of
    silently discarding a whole session (review finding #2). No LLM calls
    anywhere in P1.
  * ``initialize()`` uses a short-lived connection for setup, then hands
    everything to the worker.

Cache-safety invariants (docs/design/integration.md §2 — CI-tested):
  * Lane 1 (``system_prompt_block``) is rendered ONCE at initialize and
    byte-stable for the whole session.
  * Lane 2 (``prefetch``) is per-turn ephemeral, budget-capped, and is the
    ONLY dynamic injection channel.

Write-guards (docs/design/critique.md items 13, 14; review finding #19):
  * ``agent_context != 'primary'`` (subagent/cron/flush) => no capture
    writes AND no retrieval-log writes.
  * ``incognito: true`` in brain.yaml => same. Lane-2 *reads* still work.

Identity & trust (review findings #6, #17, #18): platform-only trust (CLI =
owner) is resolved BEFORE any DB work so a locked database can never
downgrade the owner. Identity is tracked per session — one provider
instance can serve concurrent gateway sessions, and sessions we did not
initialize default to (None, 'known_user'), never to the first session's
identity. Retrieval is scoped by the session's principal/trust.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

from ._compat import MemoryProvider
from .capture.turns import (
    TurnContext,
    capture_delegation,
    capture_memory_write,
    capture_pre_compress,
    capture_session_end,
    capture_turn,
)
from .config import DEFAULTS, load_config, save_config
from .recall.render import guidance_block, lane1_static, lane2_block
from .recall.search import log_retrieval, search, stamp_pending_injections
from .recall.strategies import retrieve_guidance
from .store import db as store_db
from .store import sysinfo
from .store import vec as vec_store
from .store.db import approx_tokens

logger = logging.getLogger(__name__)

_SENTINEL = object()

# Worker connect-retry backoff (review finding #2).
_RETRY_INITIAL_S = 5.0
_RETRY_MAX_S = 60.0

# (principal_id, trust_tier, source_author) per session.
_Identity = tuple[str | None, str, str | None]
_DEFAULT_IDENTITY: _Identity = (None, "known_user", None)


class BrainProvider(MemoryProvider):
    """Global memory brain for Hermes Agent — P1: passive capture + FTS recall."""

    def __init__(self) -> None:
        # Inert until initialize() — the plugin loader instantiates during
        # discovery and only calls is_available() on that instance.
        self._initialized = False
        self._hermes_home: Path | None = None
        self._config: dict[str, Any] = dict(DEFAULTS)
        self._session_id = ""
        self._platform = "cli"
        self._active = True          # False for subagent/cron/flush contexts
        self._incognito = False
        self._session_identity: dict[str, _Identity] = {}
        self._session_alias: dict[str, str] = {}   # old sid -> new sid (continuations)
        self._lane1 = ""
        self._lane1_staged = ""   # marker-job re-render, swapped at reset switch
        self._lane2_cache: dict[str, str] = {}
        self._turn_counts: dict[str, int] = {}
        self._queue: queue.Queue[Any] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._embedder = None  # set on the worker thread (never blocks a turn)
        self._reranker = None  # optional full-tier rerank stage (worker thread)
        self._shutting_down = threading.Event()
        self._last_sweep = 0.0   # worker-thread-only cooldown clock
        self._last_drain = 0.0   # work_queue drain cooldown (observer signals, B3)
        self._lock = threading.Lock()

    # -- identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "brain"

    def is_available(self) -> bool:
        # Floor tier is stdlib-only; the brain is always available.
        return True

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._platform = kwargs.get("platform") or "cli"
        agent_context = kwargs.get("agent_context") or "primary"
        self._active = agent_context == "primary"

        home = kwargs.get("hermes_home")
        self._hermes_home = Path(home) if home else Path.home() / ".hermes"
        self._config = load_config(self._hermes_home)
        self._incognito = bool(self._config.get("incognito"))

        # Platform-only trust FIRST — must not depend on a DB connect
        # succeeding (review finding #6).
        user_id = str(kwargs.get("user_id") or "")
        identity: _Identity
        if self._platform in ("cli", "replay"):
            identity = ("owner", "owner", None)
        else:
            identity = (None, "known_user", user_id or None)

        lane1_block = ""
        needs_bootstrap = False
        try:
            conn = store_db.connect(self._hermes_home)
            try:
                if self._platform not in ("cli", "replay") and user_id:
                    identity = self._lookup_identity(conn, user_id, identity)
                store_db.touch_activity(conn, f"provider:{session_id[:16]}")
                conn.commit()
                # Lane 1 from the materialized snapshot (deterministic —
                # critique item 17) — OWNER SESSIONS ONLY: the snapshot
                # carries the owner's profile facts, and non-owner gateway
                # sessions must never receive them (finding #15).
                if identity[1] == "owner":
                    from .recall import lane1 as lane1_mod

                    lane1_block = lane1_mod.render(
                        conn, int(self._config.get("lane1_tokens", 1200)))
                if bool(self._config.get("bootstrap_import", True)):
                    empty = conn.execute(
                        "SELECT NOT EXISTS(SELECT 1 FROM episodes) "
                        "AND NOT EXISTS(SELECT 1 FROM memories)"
                    ).fetchone()[0]
                    needs_bootstrap = bool(empty)
            finally:
                conn.close()
        except Exception:
            logger.warning("brain: initialize DB setup failed", exc_info=True)

        self._session_identity[session_id] = identity

        # Lane 1: rendered once, byte-stable for the session (invariant #1).
        self._lane1 = lane1_block or lane1_static()

        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._worker_loop, name="brain-bg", daemon=True
            )
            self._worker.start()

        self._initialized = True
        if needs_bootstrap and self._active:
            # First run on an empty brain: import MEMORY.md/USER.md + start
            # the rate-limited state.db backfill — on the worker, never
            # blocking the first turn.
            self._queue.put(("bootstrap",))
        logger.info(
            "brain: initialized session=%s platform=%s context=%s active=%s trust=%s",
            session_id[:16], self._platform, agent_context, self._active, identity[1],
        )

    def _lookup_identity(self, conn, user_id: str, fallback: _Identity) -> _Identity:
        """Upgrade a gateway user via the identities table (finding #33 root)."""
        row = conn.execute(
            "SELECT principal_id, is_owner FROM identities "
            "WHERE platform=? AND platform_user_id=?",
            (self._platform, user_id),
        ).fetchone()
        if row:
            return (row["principal_id"], "owner" if row["is_owner"] else "known_user", user_id)
        return fallback

    def _identity_for(self, session_id: str) -> _Identity:
        """Identity for a session — sessions we never initialized get the
        untrusting default, NEVER another session's identity (finding #18)."""
        return self._session_identity.get(session_id, _DEFAULT_IDENTITY)

    def shutdown(self) -> None:
        # Set BEFORE the sentinel so a marker job already in flight skips its
        # LLM sweep and does only the cheap marker insert (finding #17): the
        # host runs on_session_end() then shutdown() back-to-back, and a
        # multi-second sweep would overrun the 5s join and be killed.
        self._shutting_down.set()
        if self._worker and self._worker.is_alive():
            self._queue.put(_SENTINEL)
            # Join must outlast one busy_timeout (5s) or a single contended
            # commit abandons the session's final jobs (review finding #4).
            self._worker.join(timeout=5.0)
            if self._worker.is_alive():
                logger.warning(
                    "brain: worker did not drain at shutdown; ~%d queued jobs abandoned",
                    self._queue.qsize(),
                )
        self._initialized = False

    # -- the two lanes ---------------------------------------------------------

    def system_prompt_block(self) -> str:
        return self._lane1

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # Serve cached only — the real retrieval ran post-turn on the worker.
        return self._lane2_cache.get(session_id or self._session_id, "")

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._initialized or not query:
            return
        self._queue.put(("retrieve", session_id or self._session_id, query))

    # -- capture hooks ---------------------------------------------------------

    def _capture_allowed(self) -> bool:
        return self._initialized and self._active and not self._incognito

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self._capture_allowed():
            return
        sid = session_id or self._session_id
        with self._lock:
            self._turn_counts[sid] = self._turn_counts.get(sid, 0) + 1
            turn_no = self._turn_counts[sid]
        self._queue.put(("turn", sid, turn_no, user_content, assistant_content))

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        # Cheap liveness signal for cross-process idle detection.
        if self._initialized:
            self._queue.put(("touch",))

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        # Marker row only — extraction is out-of-band (5s drain, critique item 3).
        if self._capture_allowed():
            self._queue.put(("marker", self._session_id))

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        old = self._session_id
        self._session_id = new_session_id
        if reset:
            self._lane2_cache.pop(old, None)
            self._lane2_cache.pop(new_session_id, None)
            with self._lock:
                self._turn_counts.pop(old, None)
            self._session_identity[new_session_id] = self._session_identity.get(
                old, _DEFAULT_IDENTITY
            )
            self._session_identity.pop(old, None)
            # A reset is a real session boundary in a long-lived process:
            # honor 'takes effect next session' promises (finding #25) and
            # swap in a lane-1 snapshot re-rendered by the marker job
            # (finding #27) — byte-stability holds WITHIN a session only.
            self._config = load_config(self._hermes_home)
            self._incognito = bool(self._config.get("incognito"))
            with self._lock:
                if self._lane1_staged and self._session_identity.get(
                        new_session_id, _DEFAULT_IDENTITY)[1] == "owner":
                    self._lane1 = self._lane1_staged
                self._lane1_staged = ""
        elif old and old != new_session_id:
            # Logical continuation (/resume, /branch, compression): MOVE all
            # per-session state — copying leaked one entry per rotation in
            # long-lived gateway processes (review finding #5).
            if old in self._lane2_cache:
                self._lane2_cache[new_session_id] = self._lane2_cache.pop(old)
            with self._lock:
                if old in self._turn_counts:
                    self._turn_counts[new_session_id] = self._turn_counts.pop(old)
            if old in self._session_identity:
                self._session_identity[new_session_id] = self._session_identity.pop(old)
            self._session_alias[old] = new_session_id

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        if not self._capture_allowed():
            return ""
        self._queue.put(("precompress", self._session_id, messages))
        # Synchronous, LLM-free, ≤300 tokens: the compressor preserves the
        # salient pairs from the span being discarded (ABC return contract).
        try:
            from .capture.extract import precompress_contribution

            return precompress_contribution(messages, 300)
        except Exception:
            logger.warning("brain: precompress contribution failed", exc_info=True)
            return ""

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._capture_allowed():
            return
        # Attribute to the session the HOST says the write came from,
        # captured at enqueue time (review finding #3).
        md = dict(metadata or {})
        sid = str(md.get("session_id") or self._session_id)
        self._queue.put(("memwrite", sid, action, target, content, md))

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        if self._capture_allowed():
            self._queue.put(("delegation", self._session_id, task, result, child_session_id))

    # -- tools -------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        if not self._initialized:
            return []
        from . import tools

        return tools.get_schemas()

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        """Runs on the host's tool-execution thread: short-lived connection
        per call (tool calls are user-paced; never share the worker's conn
        across threads)."""
        from . import tools

        sid = str(kwargs.get("session_id") or self._session_id)
        principal_id, trust_tier, source_author = self._identity_for(sid)
        try:
            conn = store_db.connect(self._hermes_home)
        except Exception as e:
            return json.dumps({
                "error": f"brain storage unavailable: {e}",
                "recovery_hint": "run 'hermes brain doctor' to diagnose",
            })
        try:
            return tools.dispatch(conn, tool_name, args, ctx=tools.ToolContext(
                session_id=sid, principal_id=principal_id, trust_tier=trust_tier,
                source_author=source_author, platform=self._platform,
                embedder=self._embedder, config=self._config,
                hermes_home=str(self._hermes_home),
            ))
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # -- config / setup ----------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        from .brain_setup import config_schema

        return config_schema()

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        save_config(hermes_home, values)

    def post_setup(self, hermes_home: str, config: dict[str, Any]) -> None:
        # hermes memory setup delegates here when the attribute exists
        # (verified: hermes_cli/memory_setup.py).
        from .brain_setup import post_setup

        post_setup(hermes_home, config)

    def backup_paths(self) -> list[str]:
        return []  # everything lives under HERMES_HOME/brain/

    # -- worker -------------------------------------------------------------------

    def _turn_context(self, session_id: str, turn_no: int | None) -> TurnContext:
        principal_id, trust_tier, source_author = self._identity_for(session_id)
        return TurnContext(
            session_id=session_id,
            turn_no=turn_no,
            platform=self._platform,
            source_author=source_author,
            principal_id=principal_id,
            trust_tier=trust_tier,
        )

    def _worker_connect(self) -> Any | None:
        """Connect with drain-and-retry backoff: a transient lock (checkpoint,
        migration, AV scan) must not silently disable capture for the whole
        session (review finding #2). Jobs arriving during an outage are
        discarded (producers never block); recovery is logged."""
        delay = _RETRY_INITIAL_S
        attempt = 0
        while True:
            try:
                conn = store_db.connect(self._hermes_home)
                if attempt:
                    logger.info("brain: worker DB connection recovered (attempt %d)", attempt + 1)
                return conn
            except Exception as e:
                attempt += 1
                if attempt == 1:
                    logger.warning("brain: worker cannot open brain.db (%s) — retrying "
                                   "with backoff; capture is paused", e)
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline:
                    try:
                        job = self._queue.get(timeout=max(0.1, deadline - time.monotonic()))
                    except queue.Empty:
                        break
                    if job is _SENTINEL:
                        return None
                delay = min(_RETRY_MAX_S, delay * 2)

    def _worker_setup_retrieval(self, conn) -> None:
        """Embedder + vector tables, on the worker thread so a slow ONNX init
        (or a missing model) never touches turn latency. Model files are
        NEVER downloaded here (embed.py download=False path) — that is
        setup/doctor's job; until then the vector legs are simply absent."""
        self._embedder = None
        self._reranker = None
        mode = sysinfo.resolve_mode(str(self._config.get("mode", "auto")))
        try:
            from .recall.embed import get_embedder

            embedder = get_embedder(self._config, mode, allow_download=False)
            if embedder is not None and vec_store.ensure_tables(
                conn, embedder.dim, embedder.name, allow_rebuild=False
            ):
                self._embedder = embedder
                logger.info("brain: retrieval tier '%s' via %s", mode, embedder.name)
            elif embedder is not None:
                # Either sqlite-vec is absent or the index belongs to a
                # different embedder (findings #4/#26): never destroy the
                # live index — FTS-only until 'hermes brain reindex'.
                logger.info("brain: vector legs disabled (no vec index for %s); "
                            "FTS-only — 'hermes brain reindex' rebuilds", embedder.name)
        except Exception:
            logger.warning("brain: retrieval setup failed; FTS-only", exc_info=True)
        # Rerank stage is independent of the vector legs (it reorders the fused
        # FTS/vector candidates) and never downloads mid-turn. Absent on lite/
        # fts-only, or when models/deps are missing — search() then no-ops it.
        try:
            from .recall.rerank import get_reranker

            self._reranker = get_reranker(self._config, mode, allow_download=False)
            if self._reranker is not None:
                logger.info("brain: rerank stage via %s", self._reranker.name)
        except Exception:
            logger.warning("brain: rerank setup failed; skipping stage", exc_info=True)

    def _embed_row(self, conn, table: str, row_id: int | None, text: str) -> None:
        if self._embedder is None or row_id is None or not text.strip():
            return
        try:
            vector = self._embedder.encode_documents([text[:8000]])[0]
            vec_store.upsert(conn, table, row_id, vector)
            if table == "mem_vec":
                conn.execute("UPDATE memories SET embedded_with=? WHERE id=?",
                             (self._embedder.name, row_id))
            conn.commit()
        except Exception as e:
            logger.warning("brain: embed for %s:%s failed: %s", table, row_id, e)

    def _worker_loop(self) -> None:
        conn = self._worker_connect()
        if conn is None:  # sentinel arrived while retrying
            return
        self._worker_setup_retrieval(conn)

        last_touch = 0.0
        while True:
            try:
                job = self._queue.get(timeout=90.0)
            except queue.Empty:
                # Idle tick: opportunistic bounded sweep (docs/design plan —
                # extraction is out-of-band; LLMUnavailable just re-queues).
                # _maybe_sweep owns the cooldown clock, so a genuinely idle
                # worker doesn't sweep every 90s (findings #6/#16).
                self._maybe_sweep(conn)
                self._maybe_drain_work_queue(conn)
                continue
            if job is _SENTINEL:
                break
            try:
                kind = job[0]
                if kind == "turn":
                    _, sid, turn_no, user, asst = job
                    episode_id = capture_turn(conn, self._turn_context(sid, turn_no), user, asst)
                    # This turn consumed the block cached by the PREVIOUS
                    # turn's retrieve job; `user` is the raw text state.db
                    # also stores, so this is the only place the injection
                    # can be attributed to a resolvable turn. Ordering is
                    # guaranteed: the host serializes provider calls on a
                    # single-worker executor (memory_manager.py:376-379).
                    stamped = stamp_pending_injections(conn, sid, turn_no, user)
                    if stamped:
                        conn.commit()
                    self._embed_row(conn, "epi_vec", episode_id, f"{user}\n{asst}")
                elif kind == "retrieve":
                    _, sid, query_text = job
                    self._do_retrieve(conn, sid, query_text)
                elif kind == "marker":
                    capture_session_end(conn, job[1])
                    # End-of-session extraction FIRST (the Honcho-pattern
                    # boundary; bounded, best-effort, on the worker — never
                    # in the 5s drain window), so the lane-1 snapshot built
                    # right after includes what this session taught us. But
                    # NOT while shutting down (finding #17) — a multi-second
                    # LLM sweep would overrun the join and be killed.
                    if not self._shutting_down.is_set():
                        self._maybe_sweep(conn, force=True)
                    # Session end is the lane-1 refresh point until the P4
                    # dream owns scheduling. The re-render is STAGED and only
                    # swapped in at a reset session boundary (finding #27) —
                    # a fresh process picks it up via initialize() anyway.
                    try:
                        from .recall import lane1 as lane1_mod

                        lane1_mod.materialize(conn, self._config)
                        staged = lane1_mod.render(
                            conn, int(self._config.get("lane1_tokens", 1200)))
                        with self._lock:
                            self._lane1_staged = staged
                    except Exception:
                        logger.warning("brain: lane1 materialize failed", exc_info=True)
                elif kind == "bootstrap":
                    from . import bootstrap as bootstrap_mod

                    counts = bootstrap_mod.run_bootstrap(
                        conn, self._hermes_home, self._config, embedder=self._embedder
                    )
                    logger.info("brain: first-run bootstrap: %s", counts)
                elif kind == "precompress":
                    capture_pre_compress(conn, job[1], job[2])
                elif kind == "memwrite":
                    _, sid, action, target, content, metadata = job
                    mem_id = capture_memory_write(
                        conn, self._turn_context(sid, None),
                        action, target, content, metadata,
                    )
                    if action in ("add", "replace"):
                        self._embed_row(conn, "mem_vec", mem_id, content or "")
                    # Rows leaving current truth take their vectors with them
                    # (finding #3: dead vectors waste KNN top-k slots forever).
                    if mem_id is not None and self._embedder is not None:
                        try:
                            if action == "replace":
                                old = conn.execute(
                                    "SELECT id FROM memories WHERE superseded_by=?",
                                    (mem_id,)).fetchone()
                                if old:
                                    vec_store.delete(conn, "mem_vec", old["id"])
                            elif action == "remove":
                                vec_store.delete(conn, "mem_vec", mem_id)
                            conn.commit()
                        except Exception:
                            logger.debug("brain: stale-vector cleanup failed", exc_info=True)
                elif kind == "delegation":
                    _, sid, task, result, child = job
                    capture_delegation(conn, self._turn_context(sid, None), task, result, child)
                # "touch" falls through to the heartbeat below.

                now = time.monotonic()
                if now - last_touch > 30:
                    store_db.touch_activity(conn, f"provider:{self._session_id[:16]}")
                    conn.commit()
                    last_touch = now
                # Drain companion-observer signals (B3): cheap, bounded, yields
                # to any queued turn, never runs an LLM or blocks a turn.
                self._maybe_drain_work_queue(conn)
            except Exception:
                logger.warning("brain: worker job %s failed", job[0] if job else "?", exc_info=True)

        try:
            conn.close()
        except Exception:
            pass

    _SWEEP_COOLDOWN_S = 600.0

    def _maybe_sweep(self, conn, *, force: bool = False) -> None:
        """Bounded opportunistic extraction on the worker thread.

        Guards: capture allowed (primary + not incognito), extract_mode,
        self-owned cooldown (findings #6/#16 — reset ONLY when a sweep
        actually runs, so a continuously idle worker sweeps at most once per
        cooldown, not every 90s tick), and real work pending. Defers when
        the queue is non-empty (finding #18) — a user turn is waiting, don't
        spend seconds in an LLM call ahead of it. ``force`` (end-of-session
        marker) bypasses the cooldown but still yields to a queued turn.
        LLMUnavailable leaves buffer rows queued — honest degradation.
        """
        if not self._capture_allowed():
            return
        if str(self._config.get("extract_mode", "active")) == "off":
            return
        if not force and self._last_sweep and \
                time.monotonic() - self._last_sweep < self._SWEEP_COOLDOWN_S:
            return
        if self._queue.qsize() > 0:
            return  # a turn/marker is waiting — extraction is never urgent
        try:
            from .capture import extract

            if extract.pending_count(conn) == 0:
                self._last_sweep = time.monotonic()
                return
            counts = extract.sweep(conn, self._config, embedder=self._embedder,
                                   actor="provider-sweep", max_llm_calls=1)
            self._last_sweep = time.monotonic()
            if counts.get("inserted") or counts.get("merged"):
                logger.info("brain: sweep %s", counts)
        except Exception:
            logger.warning("brain: sweep failed", exc_info=True)

    _DRAIN_COOLDOWN_S = 20.0
    _DRAIN_BATCH = 256

    def _maybe_drain_work_queue(self, conn) -> None:
        """Drain companion-observer signal rows from work_queue (task B3).

        The out-of-tree ``brain_observer`` plugin reaches host hooks the
        MemoryProvider contract cannot (``post_tool_call``, ``subagent_stop``,
        ...) and enqueues lightweight rows into the shared ``work_queue`` table
        with its own short-lived connection. This worker is the ONLY drainer,
        and it runs on the single long-lived brain-bg connection — the
        single-connection invariant holds.

        Bounded and non-urgent: yields to any queued provider job (a turn is
        waiting), self-throttles on a cooldown, and processes at most
        ``_DRAIN_BATCH`` rows per pass into bookkeeping only (an audit_log
        summary + an activity heartbeat — no new tables). Skipped for incognito
        sessions so an incognito brain leaves no trace. Never raises into the
        worker loop.
        """
        if not self._initialized or self._incognito:
            return
        if self._queue.qsize() > 0:
            return  # a turn/marker is waiting — bookkeeping is never urgent
        now = time.monotonic()
        if self._last_drain and now - self._last_drain < self._DRAIN_COOLDOWN_S:
            return
        self._last_drain = now
        try:
            from .store import work_queue as wq

            summary = wq.drain_observer_signals(conn, limit=self._DRAIN_BATCH)
            if summary.get("count"):
                logger.info(
                    "brain: drained %d observer signal(s) tools=%s subagents=%d errors=%d",
                    summary["count"], summary.get("tools"),
                    summary.get("subagents", 0), summary.get("errors", 0),
                )
        except Exception:
            logger.warning("brain: work_queue drain failed", exc_info=True)

    def _do_retrieve(self, conn, session_id: str, query_text: str) -> None:
        # Follow continuation renames; drop results for sessions that were
        # reset away while this job sat in the queue (review finding #5).
        session_id = self._session_alias.get(session_id, session_id)
        with self._lock:
            known = session_id in self._turn_counts
        if session_id != self._session_id and not known:
            return

        budget = int(self._config.get("lane2_tokens", 600))
        if budget <= 0:
            self._lane2_cache[session_id] = ""
            return

        principal_id, trust_tier, source_author = self._identity_for(session_id)

        # Lane 2 has two subsections within one budget: learned guidance
        # (strategy/guardrail items + cases, retrieved by similarity × proven
        # usefulness) on top, then recalled facts. Guidance takes at most half
        # so it can never crowd out the facts a turn actually asked for.
        guidance = retrieve_guidance(
            conn, query_text, embedder=self._embedder,
            scope_user=principal_id, trust_tier=trust_tier)
        gblock = guidance_block(guidance, budget // 2)
        remaining = max(0, budget - approx_tokens(gblock)) if gblock else budget

        hits = search(
            conn, query_text, limit=8,
            exclude_session=session_id,
            exclude_kinds=("strategy", "guardrail", "case"),
            principal_id=principal_id,
            source_author=source_author,
            trust_tier=trust_tier,
            embedder=self._embedder,
            reranker=self._reranker,
        )
        facts_block = lane2_block(hits, remaining)
        block = "\n".join(p for p in (gblock, facts_block) if p)
        self._lane2_cache[session_id] = block

        # Retrieval-log writes are capture-class writes: skip them for
        # non-primary contexts and incognito (review finding #19) so cron
        # traffic can't skew recall_count and incognito leaves no trace.
        if not self._capture_allowed():
            return
        injected = {h.uid for h in hits if h.uid[:8] in block}
        guidance_injected = [(g.id, g.uid) for g in guidance if g.uid[:8] in block]
        # Rows land pending; the NEXT sync_turn stamps them with the turn
        # that actually consumed this block (see log_retrieval's docstring).
        log_retrieval(conn, session_id, query_text, hits, injected,
                      guidance=guidance_injected)
        conn.commit()
