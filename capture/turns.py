"""Turn capture: episodes + ingest_buffer writes on the provider worker thread.

The episodic lane (docs/design/memory-engine.md; critique item 22): every
non-incognito primary-context turn lands verbatim in `episodes` and is
FTS-indexed by the schema triggers at capture time, so "remember everything"
holds even on the no-LLM floor tier. Each write also drops an extraction
work unit into `ingest_buffer` — the sweep/dream processes those out-of-band
(critique item 3: the 5s-drain rule; nothing heavy happens in-turn).

Rule 5 (capture must never break the agent): every public function here
wraps its body in try/except, logs, rolls back, and returns None on failure.
Callers (provider worker, replay) treat None as "skipped".
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass

from ..store import db
from .salience import score_turn
from .symbols import symbols_field

logger = logging.getLogger(__name__)

# pre_compress payload bounds: keep the buffer row a work unit, not a dump.
_PRE_COMPRESS_MAX_MESSAGES = 200
_PRE_COMPRESS_MAX_CHARS = 400_000

# delegation payload bounds (task/result are summaries, not transcripts).
_DELEGATION_TASK_CHARS = 4_000
_DELEGATION_RESULT_CHARS = 8_000


@dataclass
class TurnContext:
    """Per-session capture context resolved once by provider.initialize()."""

    session_id: str
    turn_no: int | None = None
    platform: str | None = None
    source_channel: str | None = None
    source_author: str | None = None
    principal_id: str | None = None
    trust_tier: str = "known_user"  # 'owner'|'agent'|'known_user'|'tool'|'untrusted'


def capture_turn(
    conn: sqlite3.Connection,
    ctx: TurnContext,
    user_content: str,
    assistant_content: str,
    ts: str | None = None,
) -> int | None:
    """Insert one verbatim episode + its 'turn' buffer row. Returns episode id.

    ``ts`` overrides episodes.ts for historical replays (state.db backfill:
    the episode keeps the original conversation time). ingest_buffer.ts stays
    now() either way — it is a queue arrival time, not an event time.

    FTS indexing is automatic via the episodes_ai trigger — episode_fts is
    never touched directly.
    """
    try:
        user = user_content or ""
        assistant = assistant_content or ""
        if not user.strip() and not assistant.strip():
            return None

        now = db.iso_now()
        episode_ts = ts or now
        salience = score_turn(user, assistant)
        cur = conn.execute(
            "INSERT INTO episodes (uid, session_id, turn_no, platform, source_channel,"
            " source_author, principal_id, trust_tier, user_content, assistant_content,"
            " symbols, token_len, salience, ts)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                db.new_ulid(),
                ctx.session_id,
                ctx.turn_no,
                ctx.platform,
                ctx.source_channel,
                ctx.source_author,
                ctx.principal_id,
                ctx.trust_tier,
                user,
                assistant,
                symbols_field(user, assistant),
                db.approx_tokens(user + assistant),
                salience,
                episode_ts,
            ),
        )
        episode_id = cur.lastrowid
        conn.execute(
            "INSERT INTO ingest_buffer (kind, session_id, episode_id, payload, ts)"
            " VALUES ('turn',?,?,?,?)",
            (ctx.session_id, episode_id, json.dumps({"salience": salience}), now),
        )
        conn.commit()
        return episode_id
    except Exception as e:
        logger.warning("capture_turn failed (session=%s): %s", ctx.session_id, e)
        _rollback(conn)
        return None


def capture_session_end(conn: sqlite3.Connection, session_id: str) -> None:
    """Drop a session-end marker into the buffer.

    5s-drain rule (docs/design/critique.md item 3): only a marker at session
    end — extraction over the session's turns happens out-of-band in the
    sweep/dream, never in the provider's shutdown window.
    """
    try:
        conn.execute(
            "INSERT INTO ingest_buffer (kind, session_id, payload, ts)"
            " VALUES ('session_end_marker',?,'{}',?)",
            (session_id, db.iso_now()),
        )
        conn.commit()
    except Exception as e:
        logger.warning("capture_session_end failed (session=%s): %s", session_id, e)
        _rollback(conn)


def capture_pre_compress(conn: sqlite3.Connection, session_id: str, messages: list) -> None:
    """Snapshot the about-to-be-compacted tail so context loss is recoverable.

    Payload is bounded (last 200 messages, 400k chars total) — the buffer is
    a work queue, not an archive; the sweep extracts and discards.
    """
    try:
        tail = list(messages or [])[-_PRE_COMPRESS_MAX_MESSAGES:]
        # Trim oldest-first until the serialized payload fits the char budget.
        while tail and len(json.dumps(tail, default=str)) > _PRE_COMPRESS_MAX_CHARS:
            tail = tail[len(tail) // 4 or 1:]
        payload = json.dumps(
            {"n_messages": len(messages or []), "messages": tail}, default=str
        )
        conn.execute(
            "INSERT INTO ingest_buffer (kind, session_id, payload, ts)"
            " VALUES ('pre_compress',?,?,?)",
            (session_id, payload, db.iso_now()),
        )
        conn.commit()
    except Exception as e:
        logger.warning("capture_pre_compress failed (session=%s): %s", session_id, e)
        _rollback(conn)


def capture_memory_write(
    conn: sqlite3.Connection,
    ctx: TurnContext,
    action: str,
    target: str,
    content: str,
    metadata: dict | None,
) -> int | None:
    """Mirror a built-in memory-tool write into `memories` (transition period).

    Versions-are-rows discipline (schema.sql header; review finding #8):
    'replace' closes the old row (valid_to + superseded_by) and inserts the
    successor with supersedes_id and a bumped version — Hermes forwards the
    replaced text as metadata['old_text']. 'remove' tombstones (supersede-
    don't-delete). Exact duplicates on 'add'/'replace' bump
    verification_count, but ONLY on rows that are genuinely live —
    quarantined/staged rows must not absorb writes (review finding #12).
    Every write is audited and bumps the mem generation counter.
    """
    try:
        text = (content or "").strip()
        if not text:
            return None
        now = db.iso_now()
        chash = db.content_hash(text)
        current = _current_by_hash(conn, chash)

        if action in ("add", "replace"):
            if current is not None and current["status"] == "active" and current["live"] == 1:
                # Exact-dup NOOP path: same normalized content already live.
                conn.execute(
                    "UPDATE memories SET verification_count = verification_count + 1"
                    " WHERE id=?",
                    (current["id"],),
                )
                db.bump_generation(conn, "mem")
                _audit(conn, f"memory_write_noop:{action}", current["uid"], target, now)
                conn.commit()
                return current["id"]

            # 'replace': find the row being replaced via metadata['old_text']
            # so we can close it and chain the version properly.
            old_row = None
            if action == "replace":
                old_text = ((metadata or {}).get("old_text") or "").strip()
                if old_text:
                    old_row = _current_by_hash(conn, db.content_hash(old_text))

            uid = db.new_ulid()
            cur = conn.execute(
                "INSERT INTO memories (uid, epistemic, memory_type, kind, status, live,"
                " content, content_hash, symbols, tags, token_len,"
                " source_platform, source_channel, source_author, source_session,"
                " trust_tier, created_by, scope_user, version, supersedes_id,"
                " valid_from, recorded_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    uid,
                    "observation",
                    "semantic",
                    "profile" if target == "user" else "fact",
                    "active",
                    1,
                    text,
                    chash,
                    symbols_field(text),
                    json.dumps(["builtin-mirror", target]),
                    db.approx_tokens(text),
                    ctx.platform,
                    ctx.source_channel,
                    ctx.source_author,
                    ctx.session_id,
                    "agent",
                    "memory_tool",
                    ctx.principal_id if target == "user" else None,
                    (old_row["version"] + 1) if old_row else 1,
                    old_row["id"] if old_row else None,
                    now,
                    now,
                ),
            )
            new_id = cur.lastrowid
            audit_action = f"memory_write:{action}"
            if old_row is not None:
                conn.execute(
                    "UPDATE memories SET valid_to=?, superseded_by=? WHERE id=?",
                    (now, new_id, old_row["id"]),
                )
            elif action == "replace":
                # Old text unknown or predates the brain — record the gap so
                # the sweep can reconcile instead of silently forking truth.
                audit_action = "memory_write:replace_unmatched"
            db.bump_generation(conn, "mem")
            _audit(conn, audit_action, uid, target, now)
            conn.commit()
            return new_id

        if action == "remove":
            if current is None:
                logger.debug("capture_memory_write: remove target not found (%s)", target)
                return None
            conn.execute(
                "UPDATE memories SET status='tombstone', valid_to=? WHERE id=?",
                (now, current["id"]),
            )
            db.bump_generation(conn, "mem")
            _audit(conn, "memory_write:remove", current["uid"], target, now)
            conn.commit()
            return current["id"]

        logger.debug("capture_memory_write: unknown action %r", action)
        return None
    except Exception as e:
        logger.warning("capture_memory_write failed (action=%s): %s", action, e)
        _rollback(conn)
        return None


def capture_delegation(
    conn: sqlite3.Connection,
    ctx: TurnContext,
    task: str,
    result: str,
    child_session_id: str,
) -> None:
    """Buffer a subagent delegation (task + result digest) for extraction."""
    try:
        payload = json.dumps(
            {
                "task": (task or "")[:_DELEGATION_TASK_CHARS],
                "result": (result or "")[:_DELEGATION_RESULT_CHARS],
                "child_session_id": child_session_id,
            }
        )
        conn.execute(
            "INSERT INTO ingest_buffer (kind, session_id, payload, ts)"
            " VALUES ('delegation',?,?,?)",
            (ctx.session_id, payload, db.iso_now()),
        )
        conn.commit()
    except Exception as e:
        logger.warning("capture_delegation failed (session=%s): %s", ctx.session_id, e)
        _rollback(conn)


def _current_by_hash(conn: sqlite3.Connection, chash: str):
    return conn.execute(
        "SELECT id, uid, status, live, version FROM memories"
        " WHERE content_hash=? AND valid_to IS NULL",
        (chash,),
    ).fetchone()


def _audit(conn: sqlite3.Connection, action: str, uid: str, target: str, now: str) -> None:
    conn.execute(
        "INSERT INTO audit_log (actor, action, target, detail, ts)"
        " VALUES ('provider',?,?,?,?)",
        (action, uid, json.dumps({"tool_target": target}), now),
    )


def _rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.rollback()
    except sqlite3.Error:
        pass
