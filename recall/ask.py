"""Phase D — the dialectic "ask the brain" agent.

A bounded, tool-using natural-language question answerer over the brain's
memory. It seeds context with a dual prefetch (explicit observations vs.
derived inference/belief, kept in separate legs so derived rows never dilute
explicit recall), then runs a short JSON action loop through ``llm.call_json``:
the model picks ONE action per turn — search memory, grep past episodes, walk
a reasoning chain, search a date window, or answer — until it answers or the
iteration cap is hit.

Every action re-applies the caller's access scope through ``recall.search``
(and the scoped uid resolver here), so a non-owner can never reach a
``peer_card`` or another principal's rows through ANY tool. The whole body is
wrapped: no LLM path / spent budget / unexpected error degrades to a
recall-only result with the dual-prefetch citations — ``ask`` NEVER raises.

Reasoning levels map onto the EXISTING llm tiers (no new tiers):
  * ``fast`` -> tier ``extract`` (cheap), cap ``min(2, max_iterations)``
  * ``deep`` -> tier ``consolidate`` (strong), cap ``max_iterations`` (6)

Module level is stdlib-only; the sibling recall/store/llm modules and the
optional embedder/reranker are imported/threaded inside the call path.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Internal kinds are guidance/theory-of-mind, never generic facts recall (and
# a peer_card must never surface here — belt-and-braces over search's own
# non-owner peer_card block).
_EXCLUDE_KINDS = ("strategy", "guardrail", "case", "peer_card")

_PREFETCH_LIMIT = 6
_TOOL_HIT_LIMIT = 6
_SNIPPET_CHARS = 220
_UID_LEN = 8

_ABSTAIN_MARKERS = (
    "i don't know", "i do not know", "dont know", "don't have",
    "cannot determine", "can't determine", "could not determine",
    "no evidence", "not enough", "insufficient", "unable to",
    "no information", "no record", "nothing in memory", "unknown",
)

_SYSTEM_PROMPT = (
    "You are the brain's question-answering agent. You answer strictly from "
    "the owner's stored memory using the tools given — you never invent facts "
    "the evidence does not support.\n"
    "\n"
    "Work one step at a time. Each turn, reply with a SINGLE JSON object that "
    "is one action (see the action list in the user message). Gather evidence "
    "with the search/grep/chain tools, then finish with the `answer` action. "
    "Cite the memories that back your answer by their short uid.\n"
    "\n"
    "Protocols:\n"
    "1. Counting / listing ('how many', 'list every'): do NOT trust a single "
    "search. Grep the episodes first, then run at least three different "
    "searches with varied wording, collect every distinct uid, remove "
    "duplicates, and only then state a count. An undercount is a wrong "
    "answer.\n"
    "2. Conflicting evidence: when two memories disagree, do NOT silently "
    "pick one. Report BOTH claims and attach each one's timestamp so the "
    "reader can judge which is current.\n"
    "3. 'What changed' / 'current' questions: re-search with words like "
    "'changed', 'updated', 'now', 'no longer', and let the most recent "
    "memory win — a newer memory supersedes an older one that contradicts "
    "it.\n"
    "4. Abstain when the evidence is not there. A clear, confident 'I don't "
    "know' is ALWAYS an acceptable answer and is far better than a guess. "
    "Prefer abstaining to fabricating."
)

_ACTIONS_HELP = (
    "ACTIONS (choose exactly one; reply with one JSON object):\n"
    '  {"action":"search_memory","query":"<words>","epistemic":"observation"'
    '|"inference"|"belief" (optional)} — ranked memory search.\n'
    '  {"action":"grep_episodes","pattern":"<words>"} — keyword search over '
    "past conversation turns.\n"
    '  {"action":"get_reasoning_chain","uid":"<uid>"} — provenance/version '
    "walk for one memory.\n"
    '  {"action":"date_range_search","query":"<words>","from":"YYYY-MM-DD",'
    '"to":"YYYY-MM-DD"} — search restricted to a time window.\n'
    '  {"action":"answer","text":"<your answer>","citations":["<uid>",...],'
    '"abstain":false} — TERMINAL. Set abstain=true (or say you do not know) '
    "when the evidence is absent."
)


@dataclass
class AskResult:
    answer: str | None
    answered: bool
    citations: list[dict]
    iterations: int
    level: str
    tools_used: list[str] = field(default_factory=list)
    degraded: bool = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ask(
    conn: sqlite3.Connection,
    question: str,
    *,
    level: str = "deep",
    principal_id: str | None = None,
    source_author: str | None = None,
    trust_tier: str = "owner",
    scope_project: str | None = None,
    embedder=None,
    reranker=None,
    config=None,
    max_iterations: int = 6,
) -> AskResult:
    """Answer ``question`` from stored memory with a bounded tool loop.

    Never raises: any ``LLMUnavailable`` (no path / spent budget) or
    unexpected failure returns a degraded, recall-only ``AskResult`` carrying
    the dual-prefetch hits as citations.
    """
    from .. import llm
    from ..store import facts as facts_mod
    from . import search as search_mod

    level = "fast" if level == "fast" else "deep"
    tier = "extract" if level == "fast" else "consolidate"
    cap = min(2, max_iterations) if level == "fast" else max_iterations
    cap = max(1, int(cap))
    cfg = _resolve_config(config)

    registry: dict[str, str] = {}          # uid8 -> snippet, for citations
    tools_used: list[str] = []
    iterations = 0

    # -- dual prefetch (no LLM): explicit observations vs derived, scoped --
    prefetch_hits = _dual_prefetch(
        conn, search_mod, question, scope_project, principal_id,
        source_author, trust_tier, embedder, reranker,
    )
    for h in prefetch_hits:
        registry[h.uid[:_UID_LEN]] = _snip(h.text or h.summary or "")
    prefetch_citations = [
        {"uid": h.uid[:_UID_LEN], "snippet": registry[h.uid[:_UID_LEN]]}
        for h in prefetch_hits
    ]

    def _degraded() -> AskResult:
        return AskResult(
            answer=None, answered=False, citations=list(prefetch_citations),
            iterations=iterations, level=level, tools_used=tools_used,
            degraded=True,
        )

    observations: list[dict] = []
    try:
        while iterations < cap:
            prompt = _build_prompt(
                question, prefetch_citations, observations, cap, iterations)
            action = llm.call_json(
                conn, cfg, prompt, system=_SYSTEM_PROMPT, tier=tier,
                max_tokens=1200)
            iterations += 1
            name, params = _parse_action(action)
            if name == "answer":
                return _finalize_answer(
                    params, registry, prefetch_citations, iterations, level,
                    tools_used)
            result, tool = _dispatch(
                conn, search_mod, facts_mod, name, params, registry,
                scope_project, principal_id, source_author, trust_tier,
                embedder, reranker)
            if tool:
                tools_used.append(tool)
            observations.append({"action": name or "?", "result": result})
        # Cap reached with no `answer`: abstain rather than fabricate or loop.
        return AskResult(
            answer=None, answered=False, citations=list(prefetch_citations),
            iterations=iterations, level=level, tools_used=tools_used,
            degraded=False,
        )
    except llm.LLMUnavailable:
        return _degraded()
    except Exception:  # pragma: no cover - defensive; ask never raises
        logger.warning("brain ask failed for %r", question, exc_info=True)
        return _degraded()


# ---------------------------------------------------------------------------
# Dual prefetch
# ---------------------------------------------------------------------------

def _dual_prefetch(conn, search_mod, question, scope_project, principal_id,
                   source_author, trust_tier, embedder, reranker):
    """Two scoped searches: explicit observations, then derived (inference/
    belief), so derived rows don't dilute explicit recall. Deduped, explicit
    first. Never raises (search() is capture-path safe)."""
    explicit = _scoped_search(
        search_mod, conn, question, ("observation",), scope_project,
        principal_id, source_author, trust_tier, embedder, reranker,
        _PREFETCH_LIMIT)
    derived = _scoped_search(
        search_mod, conn, question, ("inference", "belief"), scope_project,
        principal_id, source_author, trust_tier, embedder, reranker,
        _PREFETCH_LIMIT)
    seen: set[str] = set()
    out = []
    for h in [*explicit, *derived]:
        if h.uid in seen:
            continue
        seen.add(h.uid)
        out.append(h)
    return out


def _scoped_search(search_mod, conn, query, epistemic, scope_project,
                   principal_id, source_author, trust_tier, embedder,
                   reranker, limit):
    """Single scope-enforced search() call. Scope (principal/trust) is applied
    inside search — a non-owner never gets a peer_card or a foreign row."""
    return search_mod.search(
        conn, query, limit=limit, exclude_kinds=_EXCLUDE_KINDS,
        scope_project=scope_project, principal_id=principal_id,
        source_author=source_author, trust_tier=trust_tier,
        epistemic=epistemic, embedder=embedder, reranker=reranker,
    )


# ---------------------------------------------------------------------------
# Action dispatch (every branch re-applies caller scope)
# ---------------------------------------------------------------------------

def _dispatch(conn, search_mod, facts_mod, name, params, registry,
              scope_project, principal_id, source_author, trust_tier,
              embedder, reranker):
    """Run one tool action. Returns (result_dict, tool_name_or_None).
    Never raises — a bad/failed action becomes an error observation."""
    try:
        if name == "search_memory":
            hits = _scoped_search(
                search_mod, conn, str(params.get("query") or ""),
                _norm_epistemic(params.get("epistemic")), scope_project,
                principal_id, source_author, trust_tier, embedder, reranker,
                _TOOL_HIT_LIMIT)
            return {"hits": _summarize(hits, registry)}, "search_memory"

        if name == "grep_episodes":
            return _grep_episodes(
                conn, search_mod, params, registry, principal_id, source_author,
                trust_tier), "grep_episodes"

        if name == "get_reasoning_chain":
            return _reasoning_chain(
                conn, facts_mod, params, registry, principal_id,
                trust_tier), "get_reasoning_chain"

        if name == "date_range_search":
            return _date_range_search(
                search_mod, conn, params, registry, scope_project,
                principal_id, source_author, trust_tier, embedder,
                reranker), "date_range_search"

        return {
            "error": f"unknown action {name!r}",
            "hint": "use search_memory | grep_episodes | get_reasoning_chain "
                    "| date_range_search | answer",
        }, None
    except Exception as e:  # a tool bug must never break the ask loop
        logger.warning("brain ask action %r failed: %s", name, e)
        return {"error": "action failed", "action": name}, (name or None)


def _grep_episodes(conn, search_mod, params, registry, principal_id,
                   source_author, trust_tier):
    """Scoped FTS over episode_fts. Reuses search.py's episode scoping — a
    non-owner only ever sees episodes attributable to them. Registers each hit
    in `registry` so the model can CITE grep evidence (finalization keeps only
    citations backed by the registry, so an unregistered uid is dropped)."""
    pattern = str(params.get("pattern") or params.get("query") or "")
    match = search_mod._match_expr(pattern)
    if not match:
        return {"episodes": []}
    try:
        rows = search_mod._episodes_rows(
            conn, match, _TOOL_HIT_LIMIT, None, principal_id, source_author,
            trust_tier)
    except Exception as e:  # e.g. no fts5 on the floor tier
        logger.warning("brain ask grep_episodes failed: %s", e)
        rows = []
    episodes = []
    for r in rows:
        uid8 = r["uid"][:_UID_LEN]
        snippet = _snip(f"{r['user_content']} / {r['assistant_content']}")
        registry[uid8] = snippet
        episodes.append({"uid": uid8, "ts": r["ts"], "snippet": snippet})
    return {"episodes": episodes}


def _reasoning_chain(conn, facts_mod, params, registry, principal_id, trust_tier):
    """uid -> memory id (scope-verified: the caller must be able to SEE that
    memory first) -> provenance chain. Out-of-scope uids read as not found.
    Registers each visible node so the model can cite the chain."""
    row = _resolve_memory_scoped(
        conn, params.get("uid") or params.get("id"), principal_id, trust_tier)
    if row is None:
        return {"error": "no memory visible for that uid"}
    chain = facts_mod.reasoning_chain(conn, row["id"])
    # SECURITY: reasoning_chain walks the supersedes_id version chain and hands
    # back each predecessor's content. A knowledge-update supersedes ACROSS
    # memories, so the chain can cross scopes — the HEAD being visible does NOT
    # make its predecessors visible. Re-apply the caller's scope to EVERY node
    # (scope predicate only, NOT current-truth: predecessors are superseded).
    if trust_tier != "owner":
        chain = _scope_filter_chain(conn, chain, principal_id)
    out = []
    for c in chain:
        uid8 = (c.get("uid") or "")[:_UID_LEN]
        snippet = _snip(c.get("content") or c.get("summary") or "")
        if uid8:
            registry[uid8] = snippet
        out.append({"uid": uid8, "kind": c.get("kind"), "snippet": snippet})
    return {"chain": out}


def _scope_filter_chain(conn, chain, principal_id):
    """Keep only chain nodes a non-owner may see: unscoped or their-own
    principal, and never a peer_card — mirroring _resolve_memory_scoped's
    non-owner clause but without the current-truth filter."""
    ids = [c.get("id") for c in chain if c.get("id") is not None]
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    visible: dict = {}
    try:
        for r in conn.execute(
            f"SELECT id, scope_user, kind FROM memories WHERE id IN ({placeholders})",
            ids).fetchall():
            visible[r["id"]] = (
                (r["scope_user"] is None or r["scope_user"] == (principal_id or ""))
                and (r["kind"] is None or r["kind"] != "peer_card"))
    except Exception:  # capture-path safety — on any error, reveal nothing
        return []
    return [c for c in chain if visible.get(c.get("id"), False)]


def _date_range_search(search_mod, conn, params, registry, scope_project,
                       principal_id, source_author, trust_tier, embedder,
                       reranker):
    """Scoped search filtered to a [from, to] valid_from window. Scope is
    enforced by search(); the window is a post-filter on the hit timestamp."""
    hits = _scoped_search(
        search_mod, conn, str(params.get("query") or ""), None, scope_project,
        principal_id, source_author, trust_tier, embedder, reranker,
        _TOOL_HIT_LIMIT * 2)
    frm = params.get("from")
    to = params.get("to")
    kept = [h for h in hits if _in_window(h.ts, frm, to)][:_TOOL_HIT_LIMIT]
    return {"hits": _summarize(kept, registry)}


def _resolve_memory_scoped(conn, uid_prefix, principal_id, trust_tier):
    """Current memory row for a uid prefix, applying the caller's scope: a
    non-owner never resolves a foreign-principal row or a peer_card. Returns
    None (indistinguishable from a miss) when out of scope or absent."""
    if not isinstance(uid_prefix, str) or len(uid_prefix.strip()) < 6:
        return None
    like = uid_prefix.strip().upper().replace("\\", "\\\\") \
        .replace("%", "\\%").replace("_", "\\_")
    sql = (
        "SELECT id, uid FROM memories WHERE uid LIKE ? ESCAPE '\\' "
        "AND valid_to IS NULL AND status='active' AND live=1"
    )
    params: list = [like + "%"]
    if trust_tier != "owner":
        sql += (" AND (scope_user IS NULL OR scope_user = ?)"
                " AND (kind IS NULL OR kind != 'peer_card')")
        params.append(principal_id or "")
    return conn.execute(sql + " LIMIT 1", params).fetchone()


# ---------------------------------------------------------------------------
# Answer finalization
# ---------------------------------------------------------------------------

def _finalize_answer(params, registry, prefetch_citations, iterations, level,
                     tools_used):
    text = str(params.get("text") or params.get("answer") or "").strip()
    abstain = bool(params.get("abstain")) or _is_abstention(text)
    citations = _norm_citations(params.get("citations"), registry)
    answered = bool(text) and not abstain
    if answered and not citations:
        citations = list(prefetch_citations)  # back the answer with the seed
    if abstain:
        citations = citations or list(prefetch_citations)
    return AskResult(
        answer=text if answered else None,
        answered=answered,
        citations=citations,
        iterations=iterations,
        level=level,
        tools_used=tools_used,
        degraded=False,
    )


def _is_abstention(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _ABSTAIN_MARKERS)


def _norm_citations(raw, registry) -> list[dict]:
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for c in raw:
        if isinstance(c, str):
            uid = c.strip().upper()[:_UID_LEN]
        elif isinstance(c, dict):
            uid = str(c.get("uid") or c.get("id") or "").strip().upper()[:_UID_LEN]
        else:
            continue
        # Only cite evidence the agent ACTUALLY gathered (the registry holds
        # every scoped hit it saw). This drops hallucinated or out-of-scope
        # uids the model may name, and forces the snippet to come from what was
        # retrieved — never model-supplied text. A model can't cite (or leak) a
        # row it was never allowed to see.
        if not uid or uid in seen or uid not in registry:
            continue
        seen.add(uid)
        out.append({"uid": uid, "snippet": registry[uid]})
    return out


# ---------------------------------------------------------------------------
# Prompt + parsing helpers
# ---------------------------------------------------------------------------

def _build_prompt(question, prefetch, observations, cap, done) -> str:
    lines = [f"QUESTION: {question}", ""]
    lines.append("SEED CONTEXT (an initial recall; may be partial or empty):")
    if prefetch:
        for c in prefetch[:8]:
            lines.append(f"  - {c['uid']}: {c['snippet']}")
    else:
        lines.append("  (nothing surfaced yet)")
    lines.append("")
    if observations:
        lines.append("TOOL RESULTS SO FAR:")
        for i, o in enumerate(observations, 1):
            blob = json.dumps(o["result"])[:700]
            lines.append(f"  [{i}] {o['action']} -> {blob}")
        lines.append("")
    remaining = cap - done
    lines.append(
        f"You have {remaining} tool step(s) left before you MUST answer "
        "(use the `answer` action).")
    lines.append("")
    lines.append(_ACTIONS_HELP)
    lines.append("")
    lines.append("Reply with exactly ONE JSON object for your next action.")
    return "\n".join(lines)


def _parse_action(action):
    """Normalize a parsed action into (name, params). Tolerates a single-item
    list wrapper and an 'args'/'input' params envelope."""
    if isinstance(action, list):
        action = next((a for a in action if isinstance(a, dict)), {})
    if not isinstance(action, dict):
        return "", {}
    name = action.get("action") or action.get("tool") or action.get("name") or ""
    params = action.get("args")
    if not isinstance(params, dict):
        params = action.get("input")
    if not isinstance(params, dict):
        params = action
    return str(name), params


def _norm_epistemic(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return None
    allowed = {"observation", "inference", "belief"}
    picked = tuple(str(v) for v in value if str(v) in allowed)
    return picked or None


def _summarize(hits, registry) -> list[dict]:
    out = []
    for h in hits[:_TOOL_HIT_LIMIT]:
        uid8 = h.uid[:_UID_LEN]
        snippet = _snip(h.text or h.summary or "")
        registry[uid8] = snippet
        out.append({"uid": uid8, "kind": h.mkind or h.kind, "snippet": snippet})
    return out


def _snip(text) -> str:
    return " ".join((text or "").split())[:_SNIPPET_CHARS]


def _in_window(ts, frm, to) -> bool:
    """Prefix-compare an ISO timestamp against date/datetime bounds. ISO-8601
    sorts lexicographically, so a same-length prefix comparison gives an
    inclusive, day- or instant-granular window."""
    if not ts:
        return False
    if frm:
        f = str(frm)
        if ts[:len(f)] < f:
            return False
    if to:
        t = str(to)
        if ts[:len(t)] > t:
            return False
    return True


def _resolve_config(config) -> dict:
    if config:
        return dict(config)
    try:
        from ..config import DEFAULTS
        return dict(DEFAULTS)
    except Exception:
        return {"day_budget_usd": 1.5}
