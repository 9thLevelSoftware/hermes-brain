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
    *,
    include_compacted: bool = True,
) -> Iterator[tuple[str, str, str | None]]:
    """Pair user/assistant messages into (user, assistant, iso_ts) turns.

    tool/system/anything-else rows are skipped. An assistant reply with text
    closes the pending user message; blank assistant rows (the tool-call
    scaffolding of an agentic loop — on a real install those are the MAJORITY,
    1006 of 1383 in the sample that motivated this) correctly do NOT close it,
    so a user turn still pairs with the reply that eventually arrives after the
    tool trace. The turn timestamp is the user message's, falling back to the
    assistant's.

    Two corrections over the original (see docs/design/alignment-audit.md §F1),
    both found by running this against a real state.db rather than a fixture:

    * **Consecutive user messages are JOINED, not dropped.** A second user
      message used to overwrite ``pending_user``, silently discarding the
      first. Real transcripts are full of follow-ups sent before the assistant
      replies, and every one of them took its predecessor with it. The eventual
      assistant reply answers all of them, so joining is the faithful reading.
    * **Compacted rows are included by default.** ``active=0`` means "compressed
      out of the live context", and skipping those is right for live capture but
      backwards for bootstrap: compacted history is precisely what an external
      memory system exists to preserve. ``include_compacted=False`` restores the
      old behavior.
    """
    pending: list[str] = []
    pending_ts: str | None = None
    for msg in messages:
        if not include_compacted and msg.get("active") == 0:
            continue
        role = str(msg.get("role") or "")
        if role == "user":
            text = _decode_content(msg.get("content"))
            if text.strip():
                pending.append(text)
                if pending_ts is None:
                    pending_ts = _epoch_to_iso(msg.get("timestamp"))
        elif role == "assistant" and pending:
            content = _decode_content(msg.get("content"))
            if content.strip():
                yield ("\n\n".join(pending), content,
                       pending_ts or _epoch_to_iso(msg.get("timestamp")))
                pending = []
                pending_ts = None
    if pending:
        # Trailing user messages with no reply — the session ended, crashed, or
        # never produced assistant text after them. On the install that
        # motivated this, 156 of 290 user messages were in this state and were
        # discarded wholesale. They are often the most useful rows in the file
        # (the last thing the user asked for), so emit them with an empty
        # assistant side rather than losing them.
        yield "\n\n".join(pending), "", pending_ts


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


def _last_message_epoch(state: sqlite3.Connection, session_id: str) -> float | None:
    """Newest message timestamp in a session, or None when it has none."""
    try:
        row = state.execute(
            "SELECT MAX(timestamp) AS t FROM messages WHERE session_id=?",
            (session_id,),
        ).fetchone()
    except sqlite3.Error:
        return None
    try:
        return float(row["t"]) if row and row["t"] is not None else None
    except (TypeError, ValueError):
        return None


def _session_is_abandoned(state: sqlite3.Connection, session_id: str,
                          stale_days: float, now: float) -> bool:
    """True when an ``ended_at IS NULL`` session is stale enough to import.

    Hermes only stamps ``ended_at`` on a clean close, so on a real install most
    sessions never get one — 22 of 49 in the sample that motivated this, holding
    210 of 290 user messages. The original skip is right for a session that is
    genuinely live (watermarking it would freeze a partial transcript forever)
    but there was no reaper, so "still running" also meant "abandoned three
    months ago", permanently.

    A session whose NEWEST message is older than ``stale_days`` is not running.
    A session with no messages at all has nothing to freeze, so it is safe too.
    """
    last = _last_message_epoch(state, session_id)
    if last is None:
        return True
    return (now - last) > stale_days * 86400.0


def backfill_sessions(
    conn: sqlite3.Connection,
    hermes_home: str | Path,
    *,
    max_sessions: int = 20,
    embedder=None,
    stale_days: float = 7.0,
    include_compacted: bool = True,
) -> dict[str, Any]:
    """Import up to ``max_sessions`` un-watermarked sessions, oldest first.

    Returns counts: ``sessions``/``turns`` imported, ``skipped`` (already
    watermarked), ``skipped_live`` (open AND recently active — the only
    sessions still deliberately withheld), and ``reaped`` (open but stale, so
    imported anyway). ``note`` is set when state.db is absent or unreadable.
    Sessions and messages are streamed via SQL-ordered cursors — a multi-year
    history must not be materialized in memory.
    """
    counts: dict[str, Any] = {"sessions": 0, "turns": 0, "skipped": 0,
                              "skipped_live": 0, "reaped": 0}
    path = Path(hermes_home) / "state.db"
    if not path.exists():
        counts["note"] = f"no state.db at {path}"
        return counts

    now_epoch = time.time()
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
            reaped = False
            if "ended_at" in session and session["ended_at"] is None:
                if not _session_is_abandoned(state, session_id, stale_days, now_epoch):
                    # Genuinely live: a watermark now would freeze the partial
                    # transcript forever. No watermark — next run retries.
                    counts["skipped_live"] += 1
                    continue
                reaped = True
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
                _pair_turns(_iter_messages(state, session_id),
                            include_compacted=include_compacted), start=1
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
            if reaped:
                counts["reaped"] += 1
    finally:
        state.close()

    logger.info(
        "bootstrap: backfilled %d session(s), %d turn(s) "
        "(%d reaped-stale, %d already done, %d still live)",
        counts["sessions"], counts["turns"], counts["reaped"],
        counts["skipped"], counts["skipped_live"],
    )
    return counts
