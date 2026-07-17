"""Strategy 'mine' AND the single state.db reader for the whole plugin
(critique item 9: "state.db mining appears three times — build one module").

Two things live here:

  1. The 'mine' strategy: turn_outcomes mining — close the retrieval loop
     (docs/design/learning-system.md §1.2d step 5, the edge Daem0n never had).
  2. The shared read-only state.db surface every other strategy consumes:
     ``open_state_ro`` / ``has_table`` / ``iso_to_epoch`` / ``decode_content``
     and ``assemble_episodes`` — the task-episode assembler that P5's
     ReasoningBank distillation (dream/distill.py) and Memento case bank
     (dream/cases.py) both read. Neither opens its own connection.

Column contract verified against hermes-agent (hermes_state.py:815-838,
agent/turn_ledger.py:28-51, agent/turn_outcome.py:8-27, SCHEMA_VERSION=22):
turn_outcomes(session_id, turn_id, created_at REAL, outcome, outcome_reason,
turn_exit_reason, api_calls, tool_iterations, retry_count, guardrail_halt,
cost_usd_delta, input_tokens_delta, output_tokens_delta,
cache_read_tokens_delta, skills_loaded, model, feedback_kind, feedback_value,
feedback_source, feedback_at, feedback_event_id). `skills_loaded` is a JSON
list (hermes_state.py:2413). `outcome` is one of 8 values and is NOT
DB-constrained, so unknown values must degrade, never raise.

Joins injected retrieval_log rows to their eventual turn outcomes: a
memory injected into a turn that ended 'verified' (or drew positive
feedback) earns helpful_count++; one injected into a 'failed'/'blocked'
turn (or a thumbs-down) earns harmful_count++. Attribution is noisy
per-turn but unbiased in aggregate; the counters gate ranking, never
existence.

Resolution (critique item 4): retrieval_log carries (session_id,
user_msg_hash, ts). state.db messages has (session_id, role, content,
timestamp) and turn_outcomes has (session_id, turn_id, created_at,
outcome, feedback_*) — messages carry no turn_id, so we find the user
message whose content hash matches within a time window, then take the
first turn_outcomes row finalized after it (hermes_state resolves
platform messages to turns by the same timestamp-proximity trick).
Unresolvable rows are tolerated — the signal is aggregate.

state.db is opened strictly READ-ONLY (percent-encoded mode=ro URI, the
store/db.py read_only pattern). No LLM anywhere in this strategy; it is
pure SQL + hashing and ships 'active' by default. Preemption-aware
(shift.tick()) and capped per run.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from calendar import timegm
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..store import db
from .shift import Shift

logger = logging.getLogger(__name__)

_MAX_ROWS = 500                     # unresolved rows examined per run
_MSG_WINDOW_SECS = 3600.0           # user msg must sit within ±1h of the log ts
_OUTCOME_WINDOW_SECS = 6 * 3600.0   # turn must finalize within 6h of its user msg
_FRESH_GRACE_SECS = 48 * 3600.0     # young unresolved rows block the watermark
_WATERMARK_KEY = "mine:watermark"

# Feedback vocab: hermes-agent reflection_triggers._NEGATIVE_REACTIONS plus
# the obvious positive mirrors; turn_outcome.py's 8-value outcome enum.
_NEGATIVE_FEEDBACK = {"\U0001f44e", "thumbs_down", "thumbsdown", "no", "dislike", "-1"}
_POSITIVE_FEEDBACK = {"\U0001f44d", "thumbs_up", "thumbsup", "yes", "like", "+1"}
_HARMFUL_OUTCOMES = {"failed", "blocked"}

# hermes_state.py sentinel for JSON-encoded structured (multimodal) content.
_CONTENT_JSON_PREFIX = "\x00json:"


def run(shift: Shift) -> dict:
    """Never raises — a mining failure must not sink the pipeline."""
    try:
        return _run(shift)
    except Exception as e:  # noqa: BLE001 — strategy contract
        logger.warning("mine: failed: %s", e, exc_info=True)
        try:
            shift.conn.rollback()
        except sqlite3.Error:
            pass
        return {"error": str(e)}


def _run(shift: Shift) -> dict:
    mode = shift.config.get("_forced_mode") or shift.mode("mine")
    active = mode == "active"

    home = shift.config.get("hermes_home")
    if not home:
        return {"skipped": "no_hermes_home"}
    path = Path(home) / "state.db"
    if not path.exists():
        return {"skipped": "no_state_db"}

    state = open_state_ro(path)
    try:
        if not has_table(state, "turn_outcomes") or not has_table(state, "messages"):
            return {"skipped": "no_state_db"}

        watermark = _get_watermark(shift.conn)
        rows = shift.conn.execute(
            "SELECT id, session_id, user_msg_hash, ts, memory_id FROM retrieval_log"
            " WHERE injected=1 AND resolved_turn_id IS NULL AND id>?"
            " ORDER BY id LIMIT ?",
            (watermark, _MAX_ROWS),
        ).fetchall()

        counts = {"resolved": 0, "helpful": 0, "harmful": 0, "unresolved": 0}
        deltas: dict[int, list[int]] = {}          # memory_id -> [helpful, harmful]
        resolutions: list[tuple[str, int]] = []    # (turn_id, retrieval_log.id)
        fresh_cutoff = time.time() - _FRESH_GRACE_SECS
        new_watermark, blocked = watermark, False

        for row in rows:
            if not shift.tick():
                blocked = True  # never advance past unexamined rows
                break
            verdict = _resolve_row(state, row)
            if verdict is None:
                counts["unresolved"] += 1
                ts_epoch = iso_to_epoch(row["ts"] or "") or 0.0
                if ts_epoch > fresh_cutoff:
                    # Outcome may simply not have landed yet — retry next run.
                    blocked = True
            else:
                # `cred`, not `credit`: the module-level credit() is a
                # function now that this is the shared state.db surface.
                turn_id, cred = verdict
                counts["resolved"] += 1
                resolutions.append((turn_id, row["id"]))
                if cred:
                    counts[cred] += 1
                    d = deltas.setdefault(row["memory_id"], [0, 0])
                    d[0 if cred == "helpful" else 1] += 1
            if not blocked:
                new_watermark = row["id"]

        detail: dict[str, Any] = {
            **counts,
            "credits": {str(k): v for k, v in list(deltas.items())[:50]},
        }
        if active:
            for turn_id, rlog_id in resolutions:
                shift.conn.execute(
                    "UPDATE retrieval_log SET resolved_turn_id=? WHERE id=?",
                    (turn_id, rlog_id),
                )
            for mem_id, (helpful, harmful) in deltas.items():
                shift.conn.execute(
                    "UPDATE memories SET helpful_count=helpful_count+?,"
                    " harmful_count=harmful_count+? WHERE id=?",
                    (helpful, harmful, mem_id),
                )
            if deltas:
                db.bump_generation(shift.conn)
            if resolutions or deltas:
                shift.audit("mine_credit", None, detail)
            if new_watermark != watermark:
                _set_watermark(shift.conn, new_watermark)
        elif mode == "dry_run" and (resolutions or deltas):  # shadow silent (#8)
            shift.audit("would_credit", None, {**detail, "mode": mode})
        shift.conn.commit()
        return counts
    finally:
        state.close()


# ---------------------------------------------------------------------------
# state.db access (READ-ONLY) — the shared surface (critique item 9)
# ---------------------------------------------------------------------------

def state_db_path(config: dict[str, Any]) -> Path | None:
    """<hermes_home>/state.db if configured and present, else None."""
    home = config.get("hermes_home")
    if not home:
        return None
    path = Path(home) / "state.db"
    return path if path.exists() else None


def open_state_ro(path: Path) -> sqlite3.Connection:
    """Read-only open, percent-encoded URI (store/db.py read_only pattern).

    state.db belongs to Hermes; the brain is a guest and NEVER writes to it.
    mode=ro makes that structural rather than a promise.
    """
    uri = "file:" + quote(str(path).replace("\\", "/")) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=5000")
    except sqlite3.Error:
        pass
    return conn


def has_table(state: sqlite3.Connection, name: str) -> bool:
    try:
        return state.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None
    except sqlite3.Error:
        return False


def has_column(state: sqlite3.Connection, table: str, column: str) -> bool:
    """state.db SCHEMA_VERSION is pinned at 22 but WILL drift (critique item
    34) — a strategy that wants an optional column asks first."""
    try:
        return any(r["name"] == column
                   for r in state.execute(f"PRAGMA table_info({table})"))
    except sqlite3.Error:
        return False




def _resolve_row(state: sqlite3.Connection, row: sqlite3.Row) -> tuple | None:
    """(turn_id, credit) for one injected retrieval_log row, else None.

    credit is 'helpful' | 'harmful' | None (resolved but neutral — e.g. a
    'partial' or 'completed_unverified' turn with no feedback).
    """
    want_hash = row["user_msg_hash"]
    ts_epoch = iso_to_epoch(row["ts"] or "")
    if not want_hash or ts_epoch is None:
        return None
    try:
        msgs = state.execute(
            "SELECT content, timestamp FROM messages"
            " WHERE session_id=? AND role='user' AND timestamp BETWEEN ? AND ?"
            " ORDER BY timestamp",
            (row["session_id"], ts_epoch - _MSG_WINDOW_SECS,
             ts_epoch + _MSG_WINDOW_SECS),
        ).fetchall()
    except sqlite3.Error:
        return None

    msg_ts: float | None = None
    for msg in msgs:
        raw = msg["content"] if isinstance(msg["content"], str) else ""
        if not raw:
            continue
        if db.content_hash(decode_content(raw)) == want_hash \
                or db.content_hash(raw) == want_hash:
            msg_ts = float(msg["timestamp"])
            break
    if msg_ts is None:
        return None

    try:
        # Lower bound is msg_ts, not msg_ts-1.0 (review finding #11): a turn's
        # outcome always finalizes AFTER the message it answers, so any row
        # with created_at < msg_ts belongs to a PRIOR turn and would miscredit
        # the wrong memory.
        out = state.execute(
            "SELECT turn_id, outcome, feedback_kind, feedback_value"
            " FROM turn_outcomes WHERE session_id=? AND created_at>=? AND created_at<=?"
            " ORDER BY created_at LIMIT 1",
            (row["session_id"], msg_ts, msg_ts + _OUTCOME_WINDOW_SECS),
        ).fetchone()
    except sqlite3.Error:
        return None
    if out is None:
        return None
    return str(out["turn_id"]), credit(out)


def credit(out: sqlite3.Row) -> str | None:
    """Map an outcome row to 'helpful' | 'harmful' | None. Explicit negative
    feedback overrides the outcome label (learning-system.md §1.2d).

    Deliberately stricter than Hermes's own skill attribution buckets
    (tools/skill_usage.py:512-513 counts 'completed_unverified' as helped and
    'unresolved' as hurt): a turn nobody verified is not evidence a memory
    helped, and an unresolved turn is not evidence it hurt. Neutral rows
    resolve (so the watermark advances) but move no counter.
    """
    fb = str(out["feedback_value"] or "").strip().lower()
    kind = str(out["feedback_kind"] or "").strip().lower()
    if fb in _NEGATIVE_FEEDBACK or kind == "thumbs_down":
        return "harmful"
    outcome = str(out["outcome"] or "")
    if outcome in _HARMFUL_OUTCOMES:
        return "harmful"
    if outcome == "verified" or fb in _POSITIVE_FEEDBACK:
        return "helpful"
    return None


def decode_content(raw: str) -> str:
    """Flatten hermes_state structured content to text (bootstrap pattern)."""
    if not raw.startswith(_CONTENT_JSON_PREFIX):
        return raw
    stripped = raw[len(_CONTENT_JSON_PREFIX):]
    try:
        parts = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return stripped[:4000]
    if not isinstance(parts, list):
        parts = [parts]
    pieces: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            pieces.append(str(part.get("text", "") or "") if part.get("type") == "text"
                          else "[image]")
        elif isinstance(part, str):
            pieces.append(part)
    return "\n".join(p for p in pieces if p).strip()


# ---------------------------------------------------------------------------
# Task-episode assembly (learning-system.md §1.2d step 1) — no LLM.
#
# The shared input to ReasoningBank distillation and the Memento case bank.
# An episode is a run of turns in one session that closes on a TERMINAL
# outcome, or at session end. Two practical guards the design doc leaves
# implicit, both to stop one "episode" swallowing a week of unrelated work:
#   * a gap longer than _EPISODE_GAP_SECS between consecutive turns starts a
#     new episode (a session left open overnight is not one task);
#   * an episode never exceeds _MAX_EPISODE_TURNS.
# Both close the episode as-is; nothing is discarded.
# ---------------------------------------------------------------------------

_TERMINAL_OUTCOMES = {"verified", "failed", "blocked"}
_FAILURE_OUTCOMES = {"failed", "blocked"}
_EPISODE_GAP_SECS = 2 * 3600.0
_MAX_EPISODE_TURNS = 20
_MAX_EPISODES = 60                  # per assemble call (bounded night)
_MAX_MSG_CHARS = 2000               # per message, before assembly
_MAX_TRANSCRIPT_CHARS = 12000       # per episode (prompt-budget guard)
# A turn's outcome finalizes AFTER the user message that triggered it, so the
# transcript window must reach back before the first turn's outcome time to
# catch that opening message. Bounded by the previous episode's end (below)
# so one episode's transcript never bleeds into the next.
_TURN_LOOKBACK_SECS = 3600.0


@dataclass(frozen=True)
class TaskEpisode:
    """One assembled task episode. Times are epoch seconds (state.db native)."""

    session_id: str
    turn_ids: tuple[str, ...]
    closing_turn_id: str
    started_at: float
    ended_at: float
    outcome: str                      # the CLOSING turn's raw outcome label
    verdict: str                      # success | failure | ambiguous
    feedback_kind: str | None = None
    feedback_value: str | None = None
    api_calls: int = 0
    tool_iterations: int = 0
    retry_count: int = 0
    cost_usd: float = 0.0
    model: str | None = None
    platform: str | None = None       # sessions.source
    source_user_id: str | None = None  # sessions.user_id (for trust resolution)
    skills_loaded: tuple[str, ...] = ()
    # (role, text) in order — user/assistant only, truncated.
    transcript: tuple[tuple[str, str], ...] = field(default=())

    @property
    def user_goal(self) -> str:
        """The first user message — what the task was asked to do."""
        for role, text in self.transcript:
            if role == "user":
                return text
        return ""

    def transcript_text(self) -> str:
        return "\n".join(f"{role}: {text}" for role, text in self.transcript)


_WM_EPSILON = 0.001


def advance_watermark(conn, episodes: list, resolved_ids: set,
                      current: float, setter) -> float:
    """Advance a per-strategy episode watermark WITHOUT skipping an unresolved
    episode (review: watermark over-advance).

    resolved_ids = the closing_turn_ids we reached a terminal decision on this
    run. The watermark filters individual turn created_at, and an episode
    spans several turns, so the watermark may never pass the STARTED_at of an
    unresolved episode — doing so would drop it (created_at <= watermark) or,
    worse, fragment it (only its later turns re-assemble). So: advance to just
    before the earliest unresolved episode's start; only when EVERY assembled
    episode is resolved may we jump to the batch's max end. Monotonic — never
    moves backward. Returns the new watermark (== current if unchanged).
    setter(conn, value) persists it under the strategy's own key.
    """
    unresolved = [e for e in episodes if e.closing_turn_id not in resolved_ids]
    if unresolved:
        new = min(e.started_at for e in unresolved) - _WM_EPSILON
    elif episodes:
        new = max(e.ended_at for e in episodes)
    else:
        return current
    if new > current:
        setter(conn, new)
        return new
    return current


def episode_verdict(outcome: str, feedback_kind: Any, feedback_value: Any) -> str:
    """success | failure | ambiguous for one closing turn.

    Explicit human feedback OVERRIDES the machine label in both directions
    (learning-system.md §1.2d step 1) — a thumbs_down on a
    'completed_unverified' turn is a failure the agent should learn from.
    Anything else ambiguous is left for the judge; 'ambiguous' NEVER
    distills (never guess — step 2).
    """
    fb = str(feedback_value or "").strip().lower()
    kind = str(feedback_kind or "").strip().lower()
    if fb in _NEGATIVE_FEEDBACK or kind == "thumbs_down":
        return "failure"
    if fb in _POSITIVE_FEEDBACK or kind == "thumbs_up":
        return "success"
    if outcome == "verified":
        return "success"
    if outcome in _FAILURE_OUTCOMES:
        return "failure"
    return "ambiguous"


def _messages_between(state: sqlite3.Connection, session_id: str,
                      start: float, end: float) -> tuple[tuple[str, str], ...]:
    """The user/assistant transcript slice for an episode's time span.

    `start` is the first turn's OUTCOME time; the message that opened that
    turn precedes it, so the lower bound is `start` (already floored by the
    caller against the previous episode's end to prevent bleed).
    """
    try:
        rows = state.execute(
            "SELECT role, content, timestamp FROM messages"
            " WHERE session_id=? AND timestamp BETWEEN ? AND ?"
            " AND role IN ('user','assistant') AND active=1"
            " ORDER BY timestamp, id",
            (session_id, start, end + 1.0),
        ).fetchall()
    except sqlite3.Error:
        return ()
    out: list[tuple[str, str]] = []
    total = 0
    for row in rows:
        text = decode_content(row["content"] if isinstance(row["content"], str) else "")
        text = text.strip()[:_MAX_MSG_CHARS]
        if not text:
            continue
        total += len(text)
        if total > _MAX_TRANSCRIPT_CHARS:
            break
        out.append((str(row["role"]), text))
    return tuple(out)


def _parse_skills(raw: Any) -> tuple[str, ...]:
    """skills_loaded is a JSON list (hermes_state.py:2413) — tolerate junk."""
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(s) for s in parsed if isinstance(s, (str, int)))


def _session_meta(state: sqlite3.Connection) -> dict[str, tuple[str | None, str | None]]:
    """session_id -> (source, user_id) for platform + trust resolution."""
    try:
        return {r["id"]: (r["source"], r["user_id"]) for r in
                state.execute("SELECT id, source, user_id FROM sessions")}
    except sqlite3.Error:
        return {}


def assemble_episodes(
    state: sqlite3.Connection,
    *,
    since_epoch: float = 0.0,
    limit: int = _MAX_EPISODES,
    include_open: bool = False,
) -> list[TaskEpisode]:
    """Group turn_outcomes into task episodes. Read-only, no LLM. Never raises.

    since_epoch: only turns finalized after this (the caller's watermark).
    include_open: also return the trailing, unterminated run of each session
                  (useful for the case bank's "what is in flight"; distillation
                  leaves them alone until they close).
    """
    if not has_table(state, "turn_outcomes") or not has_table(state, "messages"):
        return []
    try:
        rows = state.execute(
            "SELECT * FROM turn_outcomes WHERE created_at > ?"
            " ORDER BY session_id, created_at LIMIT ?",
            (since_epoch, limit * _MAX_EPISODE_TURNS),
        ).fetchall()
    except sqlite3.Error as e:
        logger.warning("assemble_episodes: turn_outcomes unreadable: %s", e)
        return []

    sessions = _session_meta(state)
    episodes: list[TaskEpisode] = []
    run: list[sqlite3.Row] = []
    last_ended: dict[str, float] = {}   # per-session previous episode end

    def close(rows_in_run: list[sqlite3.Row], *, terminated: bool) -> None:
        if not rows_in_run or len(episodes) >= limit:
            return
        if not terminated and not include_open:
            return
        last = rows_in_run[-1]
        sid = str(last["session_id"])
        started = float(rows_in_run[0]["created_at"])
        ended = float(last["created_at"])
        # Reach back for the opening user message, but never past the previous
        # episode's end in this session (no cross-episode transcript bleed).
        window_start = max(started - _TURN_LOOKBACK_SECS, last_ended.get(sid, 0.0))
        last_ended[sid] = ended
        outcome = str(last["outcome"] or "")
        skills: list[str] = []
        for r in rows_in_run:
            for name in _parse_skills(r["skills_loaded"]):
                if name not in skills:
                    skills.append(name)
        episodes.append(TaskEpisode(
            session_id=sid,
            turn_ids=tuple(str(r["turn_id"]) for r in rows_in_run),
            closing_turn_id=str(last["turn_id"]),
            started_at=started,
            ended_at=ended,
            outcome=outcome,
            verdict=episode_verdict(outcome, last["feedback_kind"],
                                    last["feedback_value"]),
            feedback_kind=last["feedback_kind"],
            feedback_value=last["feedback_value"],
            api_calls=sum(int(r["api_calls"] or 0) for r in rows_in_run),
            tool_iterations=sum(int(r["tool_iterations"] or 0) for r in rows_in_run),
            retry_count=sum(int(r["retry_count"] or 0) for r in rows_in_run),
            cost_usd=sum(float(r["cost_usd_delta"] or 0.0) for r in rows_in_run),
            model=last["model"],
            platform=(sessions.get(sid) or (None, None))[0],
            source_user_id=(sessions.get(sid) or (None, None))[1],
            skills_loaded=tuple(skills),
            # The transcript spans the whole run: an episode's lesson lives in
            # how it started as much as how it ended.
            transcript=_messages_between(state, sid, window_start, ended),
        ))

    for row in rows:
        if run:
            same_session = row["session_id"] == run[-1]["session_id"]
            gap = float(row["created_at"]) - float(run[-1]["created_at"])
            if not same_session:
                # A different session appeared, so the previous session is
                # definitively OVER — Hermes gives continuations a new
                # session id, so nothing more can land in it. Its trailing
                # run is a CLOSED episode (terminated=True), not an open one;
                # dropping it would silently lose a feedback-confirmed
                # session-end success/failure that never hit a terminal
                # outcome label (review: session-boundary episode drop).
                close(run, terminated=True)
                run = []
            elif gap > _EPISODE_GAP_SECS or len(run) >= _MAX_EPISODE_TURNS:
                close(run, terminated=True)   # boundary guard: keep what we have
                run = []
        run.append(row)
        if str(row["outcome"] or "") in _TERMINAL_OUTCOMES:
            close(run, terminated=True)
            run = []
        if len(episodes) >= limit:
            return episodes
    close(run, terminated=False)   # trailing open run
    return episodes


# ---------------------------------------------------------------------------
# watermark + time helpers
# ---------------------------------------------------------------------------

def _get_watermark(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT watermark FROM sweep_state WHERE key=?", (_WATERMARK_KEY,)
    ).fetchone()
    if not row:
        return 0
    try:
        return int(json.loads(row["watermark"]).get("id", 0))
    except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
        return 0


def _set_watermark(conn: sqlite3.Connection, rlog_id: int) -> None:
    conn.execute(
        "INSERT INTO sweep_state (key, watermark, updated_at) VALUES (?,?,?)"
        " ON CONFLICT(key) DO UPDATE SET watermark=excluded.watermark,"
        " updated_at=excluded.updated_at",
        (_WATERMARK_KEY, json.dumps({"id": rlog_id}), db.iso_now()),
    )


def epoch_to_iso(t: float) -> str:
    """Epoch seconds -> brain-style ISO-8601 UTC (state.db times are REAL)."""
    ms = int((t % 1) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{ms:03d}Z"


def iso_to_epoch(ts: str) -> float | None:
    """ISO-8601 UTC ('2026-07-16T21:04:05.123Z') -> epoch seconds."""
    try:
        base = timegm(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return None
    ms = 0.0
    if len(ts) > 20 and ts[19] == ".":
        try:
            ms = float("0." + ts[20:23])
        except ValueError:
            ms = 0.0
    return base + ms
