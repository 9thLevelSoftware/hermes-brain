"""Model-facing tool surface: brain_recall / brain_remember / brain_outcome /
brain_manage (docs/design/integration.md §3.1).

Four small tools with per-action schemas (<=6 params each) — never a
Daem0n-style 28-param union. ``dispatch`` ALWAYS returns a JSON string and
NEVER raises: every failure is an errors-that-teach payload
``{"error": ..., "recovery_hint": <complete corrective call>}``.

Loader note: the Hermes plugin loader eagerly imports every root ``*.py``,
so module level stays stdlib-only; sibling imports are deferred into the
handler bodies (same discipline as cli.py).

Trust: the MODEL speaks at 'agent' tier at most — writes are capped there,
and a lower session tier (known_user/tool/untrusted) caps lower, never
higher. Non-owner sessions can only see/touch unscoped memories or ones
scoped to their own principal (mirrors recall.search._scope_memories).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Model-surface kind vocabulary (integration.md §3.1) — the internal kinds
# strategy|case|profile are dream/CLI-owned and deliberately absent.
_KINDS = ("fact", "decision", "preference", "warning", "insight")

# Model-facing outcome vocabulary -> schema.sql CHECK values ('mixed' is an
# alias: the schema stores 'partial').
_OUTCOMES = {"worked": "worked", "failed": "failed", "mixed": "partial"}

_ACTIONS = ("forget", "pin", "unpin", "incognito_on", "incognito_off")

_LIMIT_DEFAULT = 8
_LIMIT_MAX = 25
_DEEP_FULL_TEXT_TOP = 3
_UID_MIN_PREFIX = 6
_NOTE_MAX_CHARS = 2000

# schema.sql CHECK order; lower index = more trusted.
_TRUST_ORDER = ("owner", "agent", "known_user", "tool", "untrusted")

_KIND_HINT = "valid kinds: " + "|".join(_KINDS) + \
    ' — e.g. brain_remember(content="...", kind="fact")'


@dataclass
class ToolContext:
    """Per-call context the provider builds from the session identity."""

    session_id: str = ""
    principal_id: str | None = None
    trust_tier: str = "known_user"
    source_author: str | None = None
    platform: str | None = None
    embedder: Any = None
    config: dict[str, Any] = field(default_factory=dict)
    # brain_manage incognito_* needs the config file location
    # (config.save_config writes <hermes_home>/brain/brain.yaml).
    # None => the incognito actions return an error that teaches the CLI.
    hermes_home: str | None = None


class _ToolError(Exception):
    """Internal: carries an errors-that-teach payload up to dispatch."""

    def __init__(self, error: str, recovery_hint: str) -> None:
        super().__init__(error)
        self.payload = {"error": error, "recovery_hint": recovery_hint}


# ---------------------------------------------------------------------------
# Schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

def get_schemas() -> list[dict]:
    """The four model tools, per-action schemas, <=6 params each."""
    return [
        {
            "type": "function",
            "function": {
                "name": "brain_recall",
                "description": (
                    "Search long-term memory (distilled memories + past "
                    "conversation episodes). Pass query for ranked search, OR "
                    "id (a uid prefix from a previous result, >=6 chars) to "
                    "drill into one memory in full — never both."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "search words; required unless id "
                                           "is given (mutually exclusive with id)",
                        },
                        "id": {
                            "type": "string",
                            "description": "uid prefix (>=6 chars) of one memory "
                                           "to fetch in full, with envelope + "
                                           "content; mutually exclusive with query",
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["quick", "deep"],
                            "description": "quick (default) = one index line per "
                                           "hit; deep = index lines + full text "
                                           "for the top 3 hits",
                        },
                        "kind": {
                            "type": "string",
                            "enum": list(_KINDS),
                            "description": "restrict to one memory kind",
                        },
                        "project": {
                            "type": "string",
                            "description": "restrict to one project scope",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _LIMIT_MAX,
                            "default": _LIMIT_DEFAULT,
                            "description": "max results (default 8, max 25)",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "brain_remember",
                "description": (
                    "Save one durable memory. Exact duplicates merge silently "
                    "and report what they merged with."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "the memory text (one self-contained fact)",
                        },
                        "kind": {
                            "type": "string",
                            "enum": list(_KINDS),
                            "description": "memory kind (default fact)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "topic tags for later filtering",
                        },
                        "project": {
                            "type": "string",
                            "description": "project scope (omit for global)",
                        },
                        "ttl_days": {
                            "type": "number",
                            "description": "expire after N days (known-transient "
                                           "facts); also sets decay half-life, "
                                           "capped at 30 days",
                        },
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "brain_outcome",
                "description": (
                    "Record how a remembered decision/approach turned out. "
                    "Closes open loops and feeds learning."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "uid prefix (>=6 chars) of the memory",
                        },
                        "outcome": {
                            "type": "string",
                            "enum": ["worked", "failed", "mixed"],
                            "description": "how it turned out",
                        },
                        "note": {
                            "type": "string",
                            "description": "short why/what-happened note",
                        },
                    },
                    "required": ["id", "outcome"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "brain_manage",
                "description": (
                    "Manage memory: forget (soft, reversible), pin/unpin "
                    "(recall boost), incognito_on/incognito_off (pause/resume "
                    "capture for future sessions)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": list(_ACTIONS),
                            "description": "what to do",
                        },
                        "id": {
                            "type": "string",
                            "description": "uid prefix (>=6 chars); required "
                                           "for forget/pin/unpin",
                        },
                        "reason": {
                            "type": "string",
                            "description": "why (stored as provenance)",
                        },
                    },
                    "required": ["action"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch(conn: sqlite3.Connection, tool_name: str, args: dict | None,
             *, ctx: ToolContext) -> str:
    """Route one tool call. ALWAYS returns a JSON string; NEVER raises."""
    handlers = {
        "brain_recall": _recall,
        "brain_remember": _remember,
        "brain_outcome": _outcome,
        "brain_manage": _manage,
    }
    try:
        handler = handlers.get(tool_name)
        if handler is None:
            raise _ToolError(
                f"unknown tool '{tool_name}'",
                "available tools: brain_recall, brain_remember, brain_outcome, "
                'brain_manage — e.g. brain_recall(query="deploy pipeline")',
            )
        if args is not None and not isinstance(args, dict):
            raise _ToolError(
                f"{tool_name} args must be an object, got {type(args).__name__}",
                f"pass named parameters as an object — e.g. "
                f'{tool_name}({{"...": "..."}})',
            )
        return json.dumps(handler(conn, dict(args or {}), ctx))
    except _ToolError as e:
        _rollback(conn)
        return json.dumps(e.payload)
    except Exception as e:  # the tool surface must never raise into the agent
        logger.warning("brain tool %s failed", tool_name, exc_info=True)
        _rollback(conn)
        msg = " ".join(str(e).split())[:200] or type(e).__name__
        return json.dumps({
            "error": f"{tool_name} failed: {msg}",
            "recovery_hint": "internal error, not a bad call — retry once; if "
                             "it persists, tell the user to run "
                             "'hermes brain doctor'",
        })


def _rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.rollback()
    except sqlite3.Error:
        pass


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def resolve_uid(conn: sqlite3.Connection, prefix: Any, *,
                current_only: bool = True,
                ctx: ToolContext | None = None) -> sqlite3.Row:
    """uid-prefix -> exactly one memories row, or a _ToolError that teaches.

    Same approach as cli._resolve_uid, reimplemented here because importing
    cli into the tool path is forbidden (import weight); cli.py may migrate
    to this helper later.
    """
    if not isinstance(prefix, str):
        raise _ToolError(
            f"id must be a string uid prefix, got {type(prefix).__name__}",
            'pass the uid shown in brain_recall results — e.g. '
            'brain_recall(id="01ABC234")',
        )
    prefix = prefix.strip().upper()
    if len(prefix) < _UID_MIN_PREFIX:
        raise _ToolError(
            f"id '{prefix}' is too short — at least {_UID_MIN_PREFIX} "
            "characters of the uid are required",
            'copy the leading id from a brain_recall result line, e.g. '
            'brain_recall(id="01ABC234")',
        )
    like = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    sql = "SELECT * FROM memories WHERE uid LIKE ? ESCAPE '\\'"
    params: list = [like + "%"]
    if current_only:
        sql += " AND valid_to IS NULL"
    # Scope-filter INSIDE the resolver (finding #2): a non-owner must never
    # see another principal's uids surface in the ambiguity listing, and an
    # out-of-scope collision must be indistinguishable from a miss.
    if ctx is not None and ctx.trust_tier != "owner":
        sql += " AND (scope_user IS NULL OR scope_user = ?)"
        params.append(ctx.principal_id or "")
    rows = conn.execute(sql + " LIMIT 5", params).fetchall()
    if not rows:
        raise _not_found(prefix)
    if len(rows) > 1:
        listing = ", ".join(r["uid"][:12] for r in rows)
        raise _ToolError(
            f"id '{prefix}' is ambiguous ({listing})",
            f'pass more characters of the uid — e.g. '
            f'brain_recall(id="{rows[0]["uid"][:12]}")',
        )
    return rows[0]


def _not_found(prefix: str) -> _ToolError:
    return _ToolError(
        f"no current memory matches id '{prefix}'",
        'find the uid with brain_recall(query="<words>") first — ids are the '
        "leading token of each result line",
    )


def _check_visible(row: sqlite3.Row, ctx: ToolContext, prefix: str) -> None:
    """Non-owner sessions only see/touch unscoped rows or their own
    (mirrors recall.search._scope_memories). Raises the SAME error as a
    miss so existence is not leaked."""
    if ctx.trust_tier == "owner":
        return
    if row["scope_user"] is not None and row["scope_user"] != (ctx.principal_id or ""):
        raise _not_found(prefix)


def _effective_trust(tier: str) -> str:
    """Cap the model's writes at 'agent'; a lower session tier caps lower."""
    if tier not in _TRUST_ORDER:
        return "untrusted"
    return tier if _TRUST_ORDER.index(tier) > _TRUST_ORDER.index("agent") else "agent"


# Deterministic instruction-shape heuristic for the write path (the
# extraction path gets this flag from its LLM; there is no shared detector,
# so brain_remember uses this conservative regex — false positives only cost
# a low-trust write a review, never data). Finding #1/#11.
_INSTRUCTION_RE = re.compile(
    r"\b(ignore|disregard|override|bypass)\b.{0,40}\b"
    r"(instruction|instructions|rule|rules|policy|policies|prompt|guardrail)"
    r"|\bfrom now on\b|\byou (must|should|will) (always|never)\b"
    r"|\bsystem prompt\b|\balways (approve|allow|run|execute|trust)\b",
    re.IGNORECASE | re.DOTALL,
)


def _looks_instruction_shaped(text: str) -> bool:
    return bool(_INSTRUCTION_RE.search(text or ""))


def _require_str(args: dict, key: str, example: str) -> str | None:
    val = args.get(key)
    if val is not None and not isinstance(val, str):
        raise _ToolError(
            f"{key} must be a string, got {type(val).__name__}",
            f"e.g. {example}",
        )
    return val


def _audit(conn: sqlite3.Connection, action: str, target: str | None,
           detail: dict, now: str) -> None:
    conn.execute(
        "INSERT INTO audit_log (actor, action, target, detail, ts)"
        " VALUES ('provider',?,?,?,?)",
        (action, target, json.dumps(detail), now),
    )


def _iso_in_days(days: float) -> str:
    t = time.time() + days * 86400.0
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".000Z"


# ---------------------------------------------------------------------------
# brain_recall
# ---------------------------------------------------------------------------

def _recall(conn: sqlite3.Connection, args: dict, ctx: ToolContext) -> dict:
    query = _require_str(args, "query", 'brain_recall(query="deploy pipeline")')
    mem_id = args.get("id")
    if query and mem_id:
        raise _ToolError(
            "query and id are mutually exclusive",
            'search with brain_recall(query="...") OR drill down with '
            'brain_recall(id="01ABC234") — not both',
        )
    if not query and mem_id is None:
        raise _ToolError(
            "brain_recall needs query or id",
            'e.g. brain_recall(query="deploy pipeline") to search, or '
            'brain_recall(id="01ABC234") to fetch one memory',
        )

    if mem_id is not None:
        return _recall_by_id(conn, mem_id, ctx)

    depth = args.get("depth") or "quick"
    if depth not in ("quick", "deep"):
        raise _ToolError(
            f"unknown depth '{depth}'",
            'depth is "quick" (index lines) or "deep" (index lines + full '
            'text for top 3) — e.g. brain_recall(query="...", depth="deep")',
        )
    kind = _require_str(args, "kind", 'brain_recall(query="...", kind="warning")')
    if kind is not None and kind not in _KINDS:
        raise _ToolError(f"unknown kind '{kind}'",
                         "valid kinds: " + "|".join(_KINDS) +
                         ' — e.g. brain_recall(query="...", kind="warning")')
    project = _require_str(args, "project",
                           'brain_recall(query="...", project="hermes")')
    try:
        limit = int(args.get("limit", _LIMIT_DEFAULT))
    except (TypeError, ValueError):
        raise _ToolError(
            f"limit must be an integer, got {args.get('limit')!r}",
            'e.g. brain_recall(query="...", limit=10) — default 8, max 25',
        )
    limit = max(1, min(limit, _LIMIT_MAX))

    from .recall.render import index_line
    from .recall.search import search

    hits = search(
        conn, query,
        limit=limit,
        kinds=[kind] if kind else None,
        scope_project=project,
        exclude_session=ctx.session_id or None,
        principal_id=ctx.principal_id,
        source_author=ctx.source_author,
        trust_tier=ctx.trust_tier,
        embedder=ctx.embedder,
    )

    if depth == "deep":
        from .recall.search import _memories_by_ids
        from .store import entities

        results: list = []
        seed_ids: list[int] = []
        for i, hit in enumerate(hits):
            entry: dict[str, Any] = {"line": index_line(hit)}
            if i < _DEEP_FULL_TEXT_TOP and hit.text:
                entry["text"] = hit.text
            results.append(entry)
            if hit.kind == "memory":
                seed_ids.append(hit.id)

        # Graph traversal: memories sharing an entity with the results, scoped
        # exactly like the search legs (co_mentioned returns ids; the row fetch
        # applies the caller's access filters).
        seen = {h.uid for h in hits}
        neighbors: list[dict[str, Any]] = []
        nbr_ids = entities.co_mentioned(conn, seed_ids, limit=6)
        nbr_rows = _memories_by_ids(conn, nbr_ids, [kind] if kind else None,
                                    project, ctx.principal_id, ctx.trust_tier, ())
        for mid in nbr_ids:
            row = nbr_rows.get(mid)
            if row is None or row["uid"] in seen:
                continue
            neighbors.append({
                "id": row["uid"][:8],
                "kind": row["kind"],
                "text": (row["content"] or row["summary"] or "")[:200],
            })

        payload: dict[str, Any] = {
            "results": results,
            "total": len(hits),
            "note": "depth=deep shows full text for the top 3 and graph "
                    "neighbors (memories sharing an entity with the results)",
        }
        if neighbors:
            payload["neighbors"] = neighbors
    else:
        payload = {"results": [index_line(h) for h in hits], "total": len(hits)}

    if hits:
        payload["hint"] = (
            f'drill down with brain_recall(id="{hits[0].uid[:8]}") / refine '
            "with kind=" + "|".join(_KINDS)
        )
    else:
        payload["hint"] = ("no matches — try fewer or broader words, or drop "
                           "the kind/project filter")
    return payload


def _recall_by_id(conn: sqlite3.Connection, mem_id: Any, ctx: ToolContext) -> dict:
    row = resolve_uid(conn, mem_id, current_only=True, ctx=ctx)
    _check_visible(row, ctx, str(mem_id))
    try:
        tags = json.loads(row["tags"] or "[]")
    except json.JSONDecodeError:
        tags = []
    envelope = {
        "id": row["uid"][:8],
        "uid": row["uid"],
        "kind": row["kind"],
        "memory_type": row["memory_type"],
        "epistemic": row["epistemic"],
        "status": row["status"],
        "trust_tier": row["trust_tier"],
        "created_by": row["created_by"],
        "pinned": bool(row["pinned"]),
        "tags": tags,
        "project": row["scope_project"],
        "valid_from": row["valid_from"],
        "recorded_at": row["recorded_at"],
        "outcome": row["outcome"],
        "outcome_note": row["outcome_note"],
        "counts": {
            "recall": row["recall_count"],
            "helpful": row["helpful_count"],
            "harmful": row["harmful_count"],
            "verification": row["verification_count"],
        },
        "summary": row["summary"],
        "content": row["content"],
    }
    return {
        "results": [envelope],
        "total": 1,
        "hint": 'record how it turned out with brain_outcome(id="'
                + row["uid"][:8] + '", outcome="worked"|"failed"|"mixed")',
    }


# ---------------------------------------------------------------------------
# brain_remember
# ---------------------------------------------------------------------------

def _remember(conn: sqlite3.Connection, args: dict, ctx: ToolContext) -> dict:
    content = args.get("content")
    if not isinstance(content, str) or not content.strip():
        raise _ToolError(
            "content is required and must be a non-empty string",
            'e.g. brain_remember(content="the staging DB lives on host X", '
            'kind="fact")',
        )
    text = content.strip()

    kind = args.get("kind") or "fact"
    if kind not in _KINDS:
        raise _ToolError(f"unknown kind '{kind}'", _KIND_HINT)

    tags = args.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise _ToolError(
            "tags must be an array of strings",
            'e.g. brain_remember(content="...", tags=["deploy", "ci"])',
        )
    project = _require_str(args, "project",
                           'brain_remember(content="...", project="hermes")')

    ttl_days = args.get("ttl_days")
    if ttl_days is not None and (
        isinstance(ttl_days, bool) or not isinstance(ttl_days, (int, float))
        or ttl_days <= 0
    ):
        raise _ToolError(
            f"ttl_days must be a positive number, got {ttl_days!r}",
            'e.g. brain_remember(content="...", ttl_days=14)',
        )

    from .store import db

    trust = _effective_trust(ctx.trust_tier)
    now = db.iso_now()
    chash = db.content_hash(text)

    # A non-owner/agent write is scoped to the caller's principal, so it can
    # NEVER surface in the owner's (or any peer's) recall as a global fact
    # (finding #1/#11). Instruction-shaped content from a low-trust session
    # is quarantined out of the active-recall lanes exactly as the
    # extraction path quarantines it.
    scoped = trust not in ("owner", "agent")
    # A scoped write MUST NOT be global. If the caller's principal is
    # unresolved, fall back to a non-null sentinel so the row can never match
    # another principal's recall (scope_user=NULL would make a non-owner write
    # a GLOBAL fact — the very leak lines 606-608 promise cannot happen).
    scope_user = (ctx.principal_id or f"unresolved:{ctx.session_id or 'anon'}") \
        if scoped else None
    quarantine = scoped and _looks_instruction_shaped(text)
    status = "quarantined" if quarantine else "active"

    # Dedup: report the merge instead of creating a duplicate — but only
    # against a row in the SAME scope (a peer must not learn of, or bump,
    # the owner's memories).
    scope_pred = "scope_user IS ?" if scope_user is None else "scope_user = ?"
    existing = conn.execute(
        f"SELECT id, uid FROM memories WHERE content_hash=? AND valid_to IS NULL"
        f" AND status='active' AND live=1 AND {scope_pred}",
        (chash, scope_user),
    ).fetchone()
    if existing is not None and not quarantine:
        conn.execute(
            "UPDATE memories SET verification_count = verification_count + 1"
            " WHERE id=?", (existing["id"],))
        _audit(conn, "brain_remember_noop", existing["uid"],
               {"reason": "content_hash"}, now)
        conn.commit()
        return {
            "id": existing["uid"][:8],
            "deduped_against": existing["uid"][:8],
            "note": "identical memory already existed — its verification "
                    "count was bumped; kind/tags/ttl were not changed",
        }

    # Single-transaction insert with the full envelope (finding #3: no
    # write-then-restamp two-step that a crash could leave inconsistent).
    ttl_at = _iso_in_days(float(ttl_days)) if ttl_days is not None else None
    half_life = min(float(ttl_days), 30.0) if ttl_days is not None else None
    uid = db.new_ulid()
    from .capture.symbols import symbols_field

    try:
        cur = conn.execute(
            "INSERT INTO memories (uid, epistemic, memory_type, kind, status, live,"
            " content, content_hash, symbols, tags, token_len, source_platform,"
            " source_author, source_session, source_refs, trust_tier, created_by,"
            " instruction_shaped, scope_user, scope_project, ttl_at,"
            " half_life_days, valid_from, recorded_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                uid, "observation", "semantic", kind, status, 1,
                text, chash, symbols_field(text), json.dumps(tags),
                db.approx_tokens(text), ctx.platform, ctx.source_author,
                ctx.session_id, json.dumps([f"session:{ctx.session_id}"]),
                trust, "memory_tool", 1 if quarantine else 0,
                scope_user, project, ttl_at, half_life, now, now,
            ),
        )
        row_id = cur.lastrowid
        db.bump_generation(conn, "mem")
        _audit(conn, "brain_remember", uid,
               {"kind": kind, "trust_tier": trust, "status": status}, now)
        conn.commit()
    except sqlite3.Error as e:
        _rollback(conn)
        raise _ToolError(
            f"memory write failed ({e})",
            "retry once; if it persists, tell the user to run "
            "'hermes brain doctor'",
        )

    if quarantine:
        return {
            "id": uid[:8], "deduped_against": None,
            "note": "saved but QUARANTINED: instruction-shaped content from a "
                    "non-owner session is held for review and not auto-recalled",
        }

    # Best-effort embedding (mirrors provider._embed_row): a missing vector
    # never fails the write — the reindex backfill catches it later. Never
    # embed a quarantined row (handled above by the early return).
    if ctx.embedder is not None:
        try:
            from .store import vec as vec_store

            if vec_store.vec_available(conn):
                vector = ctx.embedder.encode_documents([text[:8000]])[0]
                vec_store.upsert(conn, "mem_vec", row_id, vector)
                conn.execute("UPDATE memories SET embedded_with=? WHERE id=?",
                             (ctx.embedder.name, row_id))
                conn.commit()
        except Exception as e:
            logger.warning("brain_remember: embed failed for row %s: %s", row_id, e)
            _rollback(conn)

    note = f"remembered (kind={kind}"
    if ttl_days is not None:
        note += f", expires {ttl_at}"
    note += ")"
    return {"id": uid[:8], "deduped_against": None, "note": note}


# ---------------------------------------------------------------------------
# brain_outcome
# ---------------------------------------------------------------------------

def _outcome(conn: sqlite3.Connection, args: dict, ctx: ToolContext) -> dict:
    if "id" not in args:
        raise _ToolError(
            "id is required",
            'e.g. brain_outcome(id="01ABC234", outcome="worked") — ids come '
            "from brain_recall results",
        )
    outcome = args.get("outcome")
    stored = _OUTCOMES.get(outcome) if isinstance(outcome, str) else None
    if stored is None:
        raise _ToolError(
            f"unknown outcome {outcome!r}",
            'valid outcomes: worked|failed|mixed — e.g. '
            'brain_outcome(id="01ABC234", outcome="mixed", note="partially")',
        )
    note = _require_str(args, "note",
                        'brain_outcome(id="01ABC234", outcome="failed", '
                        'note="broke in prod")')
    if note:
        note = note.strip()[:_NOTE_MAX_CHARS] or None

    row = resolve_uid(conn, args.get("id"), current_only=True, ctx=ctx)
    _check_visible(row, ctx, str(args.get("id")))

    from .store import db

    now = db.iso_now()
    # Versions-are-rows does NOT apply here: outcome is a learning-signal
    # column, not content — stamp the CURRENT row in place (design:
    # learning-system.md §1.2; no new version for outcome stamps).
    conn.execute(
        "UPDATE memories SET outcome=?, outcome_confidence=NULL,"
        " outcome_note=? WHERE id=?",
        (stored, note, row["id"]),
    )
    _audit(conn, "brain_outcome", row["uid"],
           {"outcome": stored, "note": note, "session": ctx.session_id}, now)
    db.bump_generation(conn, "mem")
    conn.commit()
    return {"id": row["uid"][:8], "outcome": stored, "note_saved": bool(note)}


# ---------------------------------------------------------------------------
# brain_manage
# ---------------------------------------------------------------------------

def _manage(conn: sqlite3.Connection, args: dict, ctx: ToolContext) -> dict:
    action = args.get("action")
    if action not in _ACTIONS:
        raise _ToolError(
            f"unknown action {action!r}",
            "valid actions: " + "|".join(_ACTIONS) + ' — e.g. '
            'brain_manage(action="forget", id="01ABC234", reason="outdated")',
        )
    reason = _require_str(args, "reason",
                          'brain_manage(action="forget", id="01ABC234", '
                          'reason="user asked")')

    if action in ("incognito_on", "incognito_off"):
        return _manage_incognito(conn, action, reason, ctx)

    if "id" not in args:
        raise _ToolError(
            f"action '{action}' requires id",
            f'e.g. brain_manage(action="{action}", id="01ABC234") — ids come '
            "from brain_recall results",
        )
    row = resolve_uid(conn, args.get("id"), current_only=True, ctx=ctx)
    _check_visible(row, ctx, str(args.get("id")))

    from .store import db

    now = db.iso_now()
    uid8 = row["uid"][:8]
    detail = {"reason": reason, "session": ctx.session_id}

    if action == "forget":
        # SOFT tombstone only (hard purge is CLI-only): close the row, pull
        # it out of lane 1 immediately, drop its vector best-effort. The
        # dream's forgetting pass purges after the grace period.
        conn.execute(
            "UPDATE memories SET status='tombstone', valid_to=? WHERE id=?",
            (now, row["id"]),
        )
        conn.execute("DELETE FROM lane1_snapshot WHERE memory_id=?", (row["id"],))
        try:
            from .store import vec as vec_store

            if vec_store.vec_available(conn):
                vec_store.delete(conn, "mem_vec", row["id"])
        except Exception as e:
            logger.warning("brain_manage forget: vec delete failed: %s", e)
        _audit(conn, "brain_forget", row["uid"], detail, now)
        db.bump_generation(conn, "mem")
        conn.commit()
        grace = ctx.config.get("forget_grace_days", 30)
        return {
            "id": uid8,
            "action": "forget",
            "note": f"tombstoned (soft) — excluded from recall now, purged "
                    f"after {grace} days; the user can undo via CLI until then",
        }

    pinned = 1 if action == "pin" else 0
    conn.execute("UPDATE memories SET pinned=? WHERE id=?", (pinned, row["id"]))
    _audit(conn, f"brain_{action}", row["uid"], detail, now)
    db.bump_generation(conn, "mem")
    conn.commit()
    return {"id": uid8, "action": action, "pinned": bool(pinned)}


def _manage_incognito(conn: sqlite3.Connection, action: str,
                      reason: str | None, ctx: ToolContext) -> dict:
    turning_on = action == "incognito_on"
    if not ctx.hermes_home:
        raise _ToolError(
            "incognito toggle unavailable: the provider did not supply "
            "hermes_home",
            "ask the user to run: hermes brain incognito "
            + ("on" if turning_on else "off"),
        )
    from . import config as config_mod
    from .store import db

    # Tools must not mutate provider state they don't own: we flip the CONFIG
    # FILE only. The provider reads incognito at initialize/reset (it may
    # additionally honor a live flag — optional wiring, provider-side).
    config_mod.save_config(ctx.hermes_home, {"incognito": turning_on})
    try:
        _audit(conn, f"brain_{action}", None,
               {"reason": reason, "session": ctx.session_id}, db.iso_now())
        conn.commit()
    except Exception as e:  # config write already succeeded; audit best-effort
        logger.warning("brain_manage incognito: audit failed: %s", e)
        _rollback(conn)
    verb = "paused" if turning_on else "resumed"
    return {
        "action": action,
        "incognito": turning_on,
        "note": f"capture {verb} for FUTURE sessions; this session's setting "
                "applies at next initialize",
    }
