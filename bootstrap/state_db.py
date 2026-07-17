"""state.db backfill: replay Hermes session history into the episodic lane.

Reads ``<hermes_home>/state.db`` (hermes-agent hermes_state.py: sessions has
id/source/user_id/started_at/ended_at/..., messages has session_id/role/
content/timestamp/active) strictly READ-ONLY via a percent-encoded
``mode=ro`` URI — the same pattern as store/db.connect's read_only branch —
so the backfill can never corrupt the agent's live state. Schema drift is
tolerated by projected SELECTs with a ``SELECT *`` fallback plus dict.get()
access, so older/newer files without the projected columns still import.

Turns are written through capture.turns.capture_turn (one code path for
episodes + buffer rows: backfilled history is then just normal sweep work)
with the ORIGINAL message timestamp — historical episodes must not decay as
if they happened today. Idempotency is per-session watermarks in sweep_state
('bootstrap:<id>'), set AFTER a session's turns land, so an interrupted run
resumes cleanly. Sessions that are still running (ended_at present but NULL)
are skipped WITHOUT a watermark — stamping one now would freeze their
partial transcript forever. ``max_sessions`` caps one run (default 20) — a
2-year history must not stall the first initialize(); the next run continues
where this stopped.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..capture.turns import TurnContext, capture_turn
from ..store import db, vec

logger = logging.getLogger(__name__)

_EMBED_MAX_CHARS = 8000  # match provider._embed_row
_EMBED_CHUNK = 32        # embed as we go — never accumulate a whole session

# hermes_state.py sentinel for JSON-encoded structured (multimodal) content.
_CONTENT_JSON_PREFIX = "\x00json:"
# Fallback bound when the JSON after the sentinel is unparseable: keep the
# episode useful without letting a multi-MB blob through.
_DECODE_FALLBACK_CHARS = 4000


def _open_state_ro(path: Path) -> sqlite3.Connection:
    """Read-only open, percent-encoded URI (store/db.py read_only pattern)."""
    uri = "file:" + quote(str(path).replace("\\", "/")) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=5000")
    except sqlite3.Error:
        pass
    return conn


def _watermark_exists(conn: sqlite3.Connection, session_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sweep_state WHERE key=?", (f"bootstrap:{session_id}",)
    ).fetchone() is not None


def _set_watermark(conn: sqlite3.Connection, session_id: str, turns: int) -> None:
    conn.execute(
        "INSERT INTO sweep_state (key, watermark, updated_at) VALUES (?,?,?)"
        " ON CONFLICT(key) DO UPDATE SET watermark=excluded.watermark,"
        " updated_at=excluded.updated_at",
        (f"bootstrap:{session_id}", json.dumps({"turns": turns}), db.iso_now()),
    )
    conn.commit()


def _decode_content(raw: Any) -> str:
    """Flatten hermes_state's message content to plain text.

    Structured (multimodal) content is stored as '\\x00json:' + json.dumps
    (hermes_state.py _CONTENT_JSON_PREFIX / _encode_content): a list of parts
    like {'type': 'text', 'text': ...} plus image parts carrying base64
    payloads. Text parts are joined; anything image-ish becomes '[image]' —
    multi-MB base64 blobs must never land in episodes. Unparseable JSON falls
    back to the prefix-stripped text, truncated.
    """
    text = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
    if not text.startswith(_CONTENT_JSON_PREFIX):
        return text
    stripped = text[len(_CONTENT_JSON_PREFIX):]
    try:
        parts = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return stripped[:_DECODE_FALLBACK_CHARS]
    if not isinstance(parts, list):
        parts = [parts]
    pieces: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            if part.get("type") == "text":
                pieces.append(str(part.get("text", "") or ""))
            else:
                pieces.append("[image]")  # image_url/image/... — drop the payload
        elif isinstance(part, str):
            pieces.append(part)
    return "\n".join(p for p in pieces if p).strip()


def _epoch_to_iso(value: Any) -> str | None:
    """REAL epoch seconds -> the exact iso format db.iso_now() writes (UTC)."""
    try:
        t = float(value)
    except (TypeError, ValueError):
        return None
    ms = int((t % 1) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{ms:03d}Z"


def _session_trust(
    conn: sqlite3.Connection, session: dict[str, Any]
) -> tuple[str | None, str, str | None]:
    """(principal_id, trust_tier, source_author) for one sessions row.

    Local surfaces (cli/replay/tui) ARE the owner at the keyboard. Gateway
    sessions are only owner-trusted when their platform identity is enrolled
    with is_owner (critique item 33 — the trust root); enrolled non-owners
    are known_user with their stable principal, unknown users stay
    known_user with no principal.
    """
    source = str(session.get("source") or "")
    if source in ("cli", "replay", "tui"):
        return ("owner", "owner", None)
    user_id = str(session.get("user_id") or "")
    if not user_id:
        return (None, "known_user", None)
    try:
        row = conn.execute(
            "SELECT principal_id, is_owner FROM identities"
            " WHERE platform=? AND platform_user_id=?",
            (source, user_id),
        ).fetchone()
    except sqlite3.Error as e:
        logger.warning("bootstrap: identity lookup failed for %s/%s: %s", source, user_id, e)
        row = None
    if row is not None:
        return (
            row["principal_id"],
            "owner" if row["is_owner"] else "known_user",
            user_id,
        )
    return (None, "known_user", user_id)


def _iter_messages(state: sqlite3.Connection, session_id: str) -> Iterator[dict[str, Any]]:
    """Stream one session's messages oldest-first (projected; drift fallback)."""
    try:
        cur = state.execute(
            "SELECT session_id, role, content, timestamp, active FROM messages"
            " WHERE session_id=? ORDER BY rowid",
            (session_id,),
        )
    except sqlite3.OperationalError:
        try:  # ancient schema without some projected column
            cur = state.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY rowid",
                (session_id,),
            )
        except sqlite3.Error as e:
            logger.warning("bootstrap: cannot read messages for %s: %s", session_id, e)
            return
    except sqlite3.Error as e:
        logger.warning("bootstrap: cannot read messages for %s: %s", session_id, e)
        return
    for row in cur:
        yield dict(row)


def _pair_turns(
    messages: Iterable[dict[str, Any]],
) -> Iterator[tuple[str, str, str | None]]:
    """Pair user/assistant messages into (user, assistant, iso_ts) turns.

    tool/system/anything-else rows are skipped, as are rows compacted out of
    the live context (active=0 — missing/None counts as active); an assistant
    reply closes the pending user message (later assistant chunks in the same
    tool loop are dropped — backfill wants the conversational skeleton, not
    the tool trace). The turn timestamp is the user message's, falling back
    to the assistant's.
    """
    pending_user: str | None = None
    pending_ts: str | None = None
    for msg in messages:
        if msg.get("active") == 0:
            continue
        role = str(msg.get("role") or "")
        if role == "user":
            pending_user = _decode_content(msg.get("content"))
            pending_ts = _epoch_to_iso(msg.get("timestamp"))
        elif role == "assistant" and pending_user is not None:
            content = _decode_content(msg.get("content"))
            if content.strip():
                yield pending_user, content, pending_ts or _epoch_to_iso(msg.get("timestamp"))
                pending_user = None
                pending_ts = None


def _flush_embeddings(
    conn: sqlite3.Connection,
    embedder,
    session_id: str,
    batch: list[tuple[int, str]],
) -> None:
    """Embed + upsert one chunk of turns; failures degrade to FTS-only."""
    try:
        vectors = embedder.encode_documents([t for _, t in batch])
        for (episode_id, _), vector in zip(batch, vectors, strict=False):
            vec.upsert(conn, "epi_vec", episode_id, vector)
        conn.commit()
    except Exception as e:
        logger.warning("bootstrap: embedding session %s failed: %s", session_id, e)


def backfill_sessions(
    conn: sqlite3.Connection,
    hermes_home: str | Path,
    *,
    max_sessions: int = 20,
    embedder=None,
) -> dict[str, Any]:
    """Import up to ``max_sessions`` un-watermarked sessions, oldest first.

    Returns {'sessions': imported, 'turns': written, 'skipped': already
    watermarked or still running} (+ 'note' when state.db is absent or not
    actually a state.db). Sessions and messages are streamed via SQL-ordered
    cursors — a multi-year history must not be materialized in memory.
    """
    counts: dict[str, Any] = {"sessions": 0, "turns": 0, "skipped": 0}
    path = Path(hermes_home) / "state.db"
    if not path.exists():
        counts["note"] = f"no state.db at {path}"
        return counts

    state = _open_state_ro(path)
    try:
        try:
            state.execute("SELECT count(*) FROM sessions").fetchone()
        except sqlite3.Error as e:
            counts["note"] = f"not a state.db ({e})"
            return counts

        try:
            session_rows = state.execute(
                "SELECT id, source, started_at, user_id, ended_at FROM sessions"
                " ORDER BY COALESCE(started_at, 0)"
            )
        except sqlite3.OperationalError:
            # Ancient schema missing a projected column: take everything,
            # id order (the best stable proxy without started_at).
            session_rows = state.execute("SELECT * FROM sessions ORDER BY id")

        vec_ok = False
        if embedder is not None:
            try:
                vec_ok = vec.ensure_tables(
                    conn, embedder.dim, getattr(embedder, "name", "") or ""
                )
            except Exception as e:
                logger.warning("bootstrap: vector tables unavailable (%s)", e)

        for session_row in session_rows:
            session = dict(session_row)
            session_id = str(session.get("id") or "")
            if not session_id:
                continue
            if "ended_at" in session and session["ended_at"] is None:
                # Still running: a watermark now would freeze the partial
                # transcript forever. No watermark — next run retries.
                counts["skipped"] += 1
                continue
            if _watermark_exists(conn, session_id):
                counts["skipped"] += 1
                continue
            if counts["sessions"] >= max_sessions:
                break

            principal_id, trust_tier, source_author = _session_trust(conn, session)
            session_ts = _epoch_to_iso(session.get("started_at"))
            ctx = TurnContext(
                session_id=session_id,
                platform=str(session["source"]) if session.get("source") else None,
                source_author=source_author,
                principal_id=principal_id,
                trust_tier=trust_tier,
            )
            embed_batch: list[tuple[int, str]] = []
            written = 0
            for turn_no, (user, assistant, turn_ts) in enumerate(
                _pair_turns(_iter_messages(state, session_id)), start=1
            ):
                ctx.turn_no = turn_no
                episode_id = capture_turn(
                    conn, ctx, user, assistant, ts=turn_ts or session_ts
                )
                if episode_id is None:
                    continue
                written += 1
                if vec_ok:
                    embed_batch.append(
                        (episode_id, f"{user}\n{assistant}"[:_EMBED_MAX_CHARS])
                    )
                    if len(embed_batch) >= _EMBED_CHUNK:
                        _flush_embeddings(conn, embedder, session_id, embed_batch)
                        embed_batch = []

            if embed_batch:
                _flush_embeddings(conn, embedder, session_id, embed_batch)

            _set_watermark(conn, session_id, written)
            counts["sessions"] += 1
            counts["turns"] += written
    finally:
        state.close()

    logger.info(
        "bootstrap: backfilled %d session(s), %d turn(s) (%d already done or live)",
        counts["sessions"], counts["turns"], counts["skipped"],
    )
    return counts
