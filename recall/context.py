"""Token-budgeted context assembly for the host's compression path (Phase E).

A single ``assemble()`` entry point renders a compact, access-scoped context
block whose ``db.approx_tokens`` never exceeds the caller's budget. It is called
SYNCHRONOUSLY on the host's compression path, so — like
``capture.extract.precompress_contribution`` and ``recall.search.search`` — it
must be cheap, bounded, and NEVER raise: it logs and returns ``''`` on any
failure or when nothing qualifies.

Budget discipline (a concept re-implemented from scratch to a spec):

  * FIXED contributions are subtracted from the budget FIRST, in priority
    order — (1) stable identity (`memory_type='core'`) then (2) a single
    relevant peer card (owner-only). They earn their tokens before anything
    dynamic competes for the remainder.
  * The REMAINING budget is split ``summary_ratio`` (default 0.40) to a
    distilled SUMMARY of recent semantic memories and the rest to RECENT
    EXTRACTS mined from the live transcript. Summary budget it does not spend
    rolls forward to extracts.
  * Each section is bounded by BOTH a token cap and a derived word cap
    (``tokens * 0.75``) — word caps pack more predictably than tokens alone.
  * A final hard wall re-measures the joined block and trims trailing lines
    until ``approx_tokens(result) <= budget_tokens`` holds unconditionally.

Access scoping is the same invariant search enforces (review finding #17): a
non-owner caller sees only unscoped or their-own-principal rows and NEVER a
peer_card (the owner's private theory-of-mind of a person, which must not leak
to the very peer it describes). Every query is read-only, current-truth
(``valid_to IS NULL AND status='active' AND live=1``), and LIMIT-bounded.
"""

from __future__ import annotations

import logging
import re

from ..capture.salience import score_turn
from ..store import db

logger = logging.getLogger(__name__)

# Current-truth predicate (store/schema.sql is law): supersede-don't-delete
# means "live now" = not superseded, active, and not demoted.
_CURRENT = "valid_to IS NULL AND status = 'active' AND live = 1"

# Internal, non-fact kinds never belong in a generic summary block (they are
# planning guidance or the owner's private peer models) — same exclusion set
# recall/blend.py and recall/ask.py apply to generic recall.
_EXCLUDE_KINDS = ("strategy", "guardrail", "case", "peer_card")

# Per-query row caps and per-line character clips — bound the synchronous cost.
_IDENTITY_LIMIT = 8
_SUMMARY_LIMIT = 12
_EXTRACT_CANDIDATES = 20
_IDENTITY_CLIP = 200
_SUMMARY_CLIP = 200
_PEER_CLIP = 400
_EXTRACT_CLIP = 160

_WS = re.compile(r"\s+")

_HDR_IDENTITY = "## Identity"
_HDR_SUMMARY = "## Summary"
_HDR_RECENT = "## Recent"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assemble(conn, messages, budget_tokens, *, principal_id="owner",
             trust_tier="owner", embedder=None, config=None,
             summary_ratio=0.4) -> str:
    """A token-budgeted, access-scoped context block. Never raises — ``''`` on
    any failure or when nothing qualifies. See the module docstring for the
    budget scheme; the hard invariant is ``approx_tokens(result) <= budget``."""
    try:
        budget = int(budget_tokens or 0)
        if budget <= 0:
            return ""
        ratio = summary_ratio if 0.0 <= summary_ratio <= 1.0 else 0.4
        is_owner = trust_tier == "owner"

        out: list[str] = []
        used = 0

        # -- FIXED #1: identity (current-truth core rows, scoped) --
        block, cost = _fit(_HDR_IDENTITY,
                           _identity_lines(conn, principal_id, trust_tier),
                           budget - used)
        out += block
        used += cost

        # -- FIXED #2: peer card (owner-only, one, scoped to the interlocutor) --
        if is_owner and principal_id and principal_id != "owner":
            peer = _peer_card_lines(conn, principal_id)
            if peer:
                block, cost = _fit(f"## About {_clip(principal_id, 60)}",
                                   peer, budget - used)
                out += block
                used += cost

        # -- DYNAMIC: split the remainder summary_ratio / (1 - summary_ratio) --
        remaining = budget - used
        summary_budget = int(remaining * ratio)

        block, cost = _fit(_HDR_SUMMARY,
                           _summary_lines(conn, principal_id, trust_tier),
                           summary_budget)
        out += block
        used += cost

        # Unspent summary budget rolls forward to the extracts.
        extract_budget = budget - used
        block, cost = _fit(_HDR_RECENT, _extract_lines(messages), extract_budget)
        out += block
        used += cost

        # -- HARD WALL: measure the real joined string and trim to fit. --
        result = "\n".join(out)
        while out and db.approx_tokens(result) > budget:
            out.pop()
            while out and out[-1].startswith("#"):
                out.pop()  # never leave a dangling header with no body
            result = "\n".join(out)
        return result
    except Exception as e:  # host compression path — never raise
        logger.warning("context.assemble failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Section packing
# ---------------------------------------------------------------------------

def _fit(header: str, lines: list[str], token_budget: int) -> tuple[list[str], int]:
    """Pack ``lines`` under a header into ``token_budget`` (accounting for the
    header) and a derived word budget (``token_budget * 0.75``). Returns
    ``([], 0)`` — header omitted — when nothing fits, so empty sections vanish.
    """
    if token_budget <= 0 or not lines:
        return [], 0
    header_t = db.approx_tokens(header)
    if header_t >= token_budget:
        return [], 0
    word_budget = int(token_budget * 0.75)
    used_t = header_t
    used_w = 0
    kept: list[str] = []
    for line in lines:
        t = db.approx_tokens(line)
        w = len(line.split())
        if used_t + t > token_budget or used_w + w > word_budget:
            break
        kept.append(line)
        used_t += t
        used_w += w
    if not kept:
        return [], 0
    return [header, *kept], used_t


# ---------------------------------------------------------------------------
# Fixed contributions
# ---------------------------------------------------------------------------

def _identity_lines(conn, principal_id: str | None, trust_tier: str) -> list[str]:
    """Stable identity: current-truth ``memory_type='core'`` rows, scoped."""
    try:
        sql = (f"SELECT content FROM memories WHERE {_CURRENT} "
               "AND memory_type = 'core'")
        params: list = []
        sql = _scope(sql, params, principal_id, trust_tier)
        sql += " ORDER BY pinned DESC, valid_from DESC LIMIT ?"
        params.append(_IDENTITY_LIMIT)
        return _content_lines(conn, sql, params, _IDENTITY_CLIP)
    except Exception as e:
        logger.warning("context identity query failed: %s", e)
        return []


def _peer_card_lines(conn, principal_id: str) -> list[str]:
    """The owner's current-truth peer card for ``principal_id`` (one row).

    Caller has already gated on ``trust_tier == 'owner'``; the ``scope_user``
    match keys it to the observed interlocutor. A non-owner never reaches here.
    """
    try:
        sql = (f"SELECT content FROM memories WHERE {_CURRENT} "
               "AND kind = 'peer_card' AND scope_user = ? "
               "ORDER BY valid_from DESC LIMIT 1")
        return _content_lines(conn, sql, [principal_id], _PEER_CLIP)
    except Exception as e:
        logger.warning("context peer_card query failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Dynamic contributions
# ---------------------------------------------------------------------------

def _summary_lines(conn, principal_id: str | None, trust_tier: str) -> list[str]:
    """Recent distilled memories: current-truth ``memory_type='semantic'``,
    internal kinds excluded, scoped, most-recent first."""
    try:
        sql = (f"SELECT content FROM memories WHERE {_CURRENT} "
               "AND memory_type = 'semantic' "
               f"AND (kind IS NULL OR kind NOT IN ({','.join('?' * len(_EXCLUDE_KINDS))}))")
        params: list = list(_EXCLUDE_KINDS)
        sql = _scope(sql, params, principal_id, trust_tier)
        sql += " ORDER BY valid_from DESC LIMIT ?"
        params.append(_SUMMARY_LIMIT)
        return _content_lines(conn, sql, params, _SUMMARY_CLIP)
    except Exception as e:
        logger.warning("context summary query failed: %s", e)
        return []


def _extract_lines(messages) -> list[str]:
    """Top-salience user/assistant pairs from the live transcript, rendered
    ``- U: … / A: …`` in conversation order (reuses ``score_turn``, the same
    no-LLM heuristic ``precompress_contribution`` uses). Content is clipped
    BEFORE scoring so cost scales with message count, not transcript bytes."""
    try:
        pairs: list[tuple[str, str]] = []
        pending: str | None = None
        for msg in messages or []:
            role = (msg.get("role") if isinstance(msg, dict) else None) or ""
            text = _msg_text(msg)
            if role == "user":
                pending = text
            elif role == "assistant" and pending is not None:
                pairs.append((pending, text))
                pending = None
        scored = [(score_turn(u[:2000], a[:2000]), i, u, a)
                  for i, (u, a) in enumerate(pairs)]
        top = sorted((s for s in scored if s[0] > 0.2),
                     key=lambda s: s[0], reverse=True)[:_EXTRACT_CANDIDATES]
        top.sort(key=lambda s: s[1])  # render in conversation order
        return [f"- U: {_clip(u, _EXTRACT_CLIP)} / A: {_clip(a, _EXTRACT_CLIP)}"
                for _s, _i, u, a in top]
    except Exception as e:
        logger.warning("context extract mining failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scope(sql: str, params: list, principal_id: str | None, trust_tier: str) -> str:
    """Non-owner callers see only unscoped or their-own-principal rows, and
    NEVER a peer_card (finding #17 — belt-and-braces, mirrors
    ``recall.search._scope_memories``)."""
    if trust_tier == "owner":
        return sql
    sql += (" AND (scope_user IS NULL OR scope_user = ?)"
            " AND (kind IS NULL OR kind != 'peer_card')")
    params.append(principal_id or "")
    return sql


def _content_lines(conn, sql: str, params: list, clip: int) -> list[str]:
    lines: list[str] = []
    for row in conn.execute(sql, params).fetchall():
        text = _clip(row["content"], clip)
        if text:
            lines.append(f"- {text}")
    return lines


def _msg_text(msg) -> str:
    if not isinstance(msg, dict):
        return str(msg or "")
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal: join the text parts
        return " ".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")


def _clip(text: str | None, limit: int) -> str:
    return _WS.sub(" ", text or "").strip()[:limit]
