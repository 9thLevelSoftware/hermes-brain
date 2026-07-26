"""Generate paraphrase retrieval queries from the live brain.

One aux LLM call per BATCH of source items (not per item): 150 queries is ~19
calls instead of 150, which is the difference between a cent and a dollar. A
failed or unparseable batch is skipped rather than aborting the run — partial
query sets are useful, and `LLMUnavailable` (no aux client, budget spent) must
degrade like every other brain-initiated call.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

BATCH = 8
_MAX_SOURCE_CHARS = 1200

_SYSTEM = """You write retrieval benchmark queries.

For each numbered source text, write ONE natural question that a person would \
plausibly ask, whose answer is in that source.

Hard rules:
- PARAPHRASE. Do not reuse distinctive words from the source — no shared \
identifiers, file paths, product names, or rare terms. If the source says \
"the Hetzner CX22 box", ask about "the server we rented", not "the CX22".
- The question must be answerable from that source alone.
- Ask about the substance, not the conversation ("what database do we use?", \
never "what did the assistant say about databases?").
- If a source is too thin or generic to ask anything specific about, return \
null for it.

Return ONLY a JSON array, one entry per source, in order:
[{"n": 1, "q": "the question"}, {"n": 2, "q": null}]"""


def _sample_sources(conn: sqlite3.Connection, limit: int) -> list[dict[str, Any]]:
    """Memories first (they are what recall is FOR), then episodes to fill.

    Ordered by length so thin rows — which produce unanswerable questions —
    are not sampled first, and randomized within the pool so a query set is
    not just the newest N items.
    """
    rows: list[dict[str, Any]] = []
    try:
        for r in conn.execute(
            "SELECT uid, content FROM memories"
            " WHERE valid_to IS NULL AND status='active' AND live=1"
            "   AND kind NOT IN ('peer_card','strategy','guardrail','case')"
            "   AND length(content) > 80"
            " ORDER BY RANDOM() LIMIT ?", (limit,)
        ):
            rows.append({"uid": r["uid"], "text": r["content"], "kind": "memory"})
    except sqlite3.Error as e:
        logger.warning("eval: memory sampling failed (%s)", e)

    if len(rows) < limit:
        try:
            for r in conn.execute(
                "SELECT uid, user_content, assistant_content FROM episodes"
                " WHERE length(user_content) + length(assistant_content) > 200"
                " ORDER BY RANDOM() LIMIT ?", (limit - len(rows),)
            ):
                rows.append({
                    "uid": r["uid"],
                    "text": f"{r['user_content']}\n{r['assistant_content']}",
                    "kind": "episode",
                })
        except sqlite3.Error as e:
            logger.warning("eval: episode sampling failed (%s)", e)
    return rows


def _ask_batch(conn, config, batch: list[dict[str, Any]]) -> list[str | None]:
    """One LLM call -> one question per source (None where it declined)."""
    from .. import llm

    parts = []
    for i, item in enumerate(batch, start=1):
        text = " ".join((item["text"] or "").split())[:_MAX_SOURCE_CHARS]
        parts.append(f"--- SOURCE {i} ---\n{text}")
    prompt = "\n\n".join(parts)

    parsed = llm.call_json(conn, config, prompt, system=_SYSTEM,
                           tier="extract", max_tokens=900)
    out: list[str | None] = [None] * len(batch)
    if not isinstance(parsed, list):
        return out
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("n", 0)) - 1
        except (TypeError, ValueError):
            continue
        question = entry.get("q")
        if 0 <= idx < len(batch) and isinstance(question, str) and question.strip():
            out[idx] = question.strip()
    return out


def _too_similar(question: str, source: str) -> bool:
    """Reject a 'paraphrase' that simply lifted the source's rare words.

    Cheap and deliberately strict: high overlap on LONG tokens is the signature
    of a copied identifier, and an identifier in the query is exactly what
    makes a benchmark measure BM25's ability to find itself.
    """
    def longish(text: str) -> set[str]:
        return {w.strip(".,:;!?\"'()[]").lower() for w in text.split()
                if len(w.strip(".,:;!?\"'()[]")) >= 7}

    q, s = longish(question), longish(source)
    if not q:
        return False
    return len(q & s) / len(q) > 0.5


def generate_queryset(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    *,
    limit: int = 150,
    progress=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a paraphrase query set from the live brain.

    Returns ``(queries, meta)``. Each query is
    ``{"query": str, "gold": [uid], "source_kind": ..., "source_uid": ...}``.
    Stops early and returns what it has on ``LLMUnavailable`` (no aux client,
    or the daily budget is spent) — a partial query set still measures
    something; an exception halfway through measures nothing.
    """
    from ..llm import LLMUnavailable

    sources = _sample_sources(conn, limit)
    queries: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"sampled": len(sources), "batches": 0,
                            "rejected_verbatim": 0, "declined": 0}
    if not sources:
        meta["note"] = "no source material — the brain is empty"
        return queries, meta

    for start in range(0, len(sources), BATCH):
        batch = sources[start:start + BATCH]
        try:
            questions = _ask_batch(conn, config, batch)
        except LLMUnavailable as e:
            meta["stopped_early"] = str(e)
            break
        except Exception as e:  # a bad batch must not kill the run
            logger.warning("eval: batch failed (%s)", e)
            meta["declined"] += len(batch)
            continue
        meta["batches"] += 1
        for item, question in zip(batch, questions, strict=False):
            if not question:
                meta["declined"] += 1
                continue
            if _too_similar(question, item["text"]):
                meta["rejected_verbatim"] += 1
                continue
            queries.append({
                "query": question,
                "gold": [item["uid"]],
                "source_kind": item["kind"],
                "source_uid": item["uid"],
            })
        if progress is not None:
            progress(len(queries), len(sources))
    return queries, meta
