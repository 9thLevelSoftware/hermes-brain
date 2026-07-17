"""The injection grammar: lane 1 (static), lane 2 (recalled context), CLI.

Lane 1 in P1 is a fixed, byte-stable instructions block rendered once at
provider initialize (docs/design/integration.md §2.1) — zero dynamic
content so the CI byte-stability check and prompt caching both hold. The
dream-materialized lane1_snapshot renderer replaces it in P4.

Lane 2 is a budget-packed block of index lines + verbatim memory text.
The format is stable: downstream parsers and the CI budget check rely on
it, so change it only with a version bump of the grammar.
"""

from __future__ import annotations

import logging
import re
import textwrap

from ..store import db

logger = logging.getLogger(__name__)

_LANE1 = (
    "## BRAIN (hermes-brain memory)\n"
    "Passive memory capture is active: every turn is recorded and distilled "
    "into long-term memory automatically.\n"
    "To drill down into past context, search with: hermes brain search <query>"
)

_WS = re.compile(r"\s+")
_SNIPPET_CHARS = 120

LANE2_HEADER = "## Recalled context (hermes-brain)"
GUIDANCE_HEADER = "## Learned guidance (hermes-brain)"

_GUIDANCE_GLYPH = {"strategy": "→ strategy", "guardrail": "⚑ guardrail"}
_GUIDANCE_SNIPPET = 200


def lane1_static() -> str:
    """P1's byte-stable lane 1 block. MUST stay free of dynamic content."""
    return _LANE1


def index_line(hit) -> str:
    """One-line index entry: [uid8 · kind · date · platform] snippet."""
    snippet = _WS.sub(" ", (hit.summary or hit.text or "")).strip()[:_SNIPPET_CHARS]
    label = hit.mkind or hit.kind
    date = (hit.ts or "")[:10]
    return f"[{hit.uid[:8]} · {label} · {date} · {hit.platform or '-'}] {snippet}"


def lane2_block(hits: list, budget_tokens: int) -> str:
    """Budget-packed lane 2. '' when nothing to inject; otherwise header +
    per-hit index line (+ verbatim text for memory hits), packed top-down
    with db.approx_tokens, always keeping at least the first index line.
    """
    if not hits or budget_tokens <= 0:
        return ""
    parts = [LANE2_HEADER]
    used = db.approx_tokens(LANE2_HEADER)
    for i, hit in enumerate(hits):
        line = index_line(hit)
        chunk = f"{line}\n{hit.text}" if (hit.kind == "memory" and hit.text) else line
        cost = db.approx_tokens(chunk)
        if used + cost > budget_tokens:
            if i == 0:
                parts.append(line)  # guarantee at least the first index line
            break
        parts.append(chunk)
        used += cost
    return "\n".join(parts)


def guidance_block(items: list, budget_tokens: int) -> str:
    """Lane-2 learned-guidance subsection (strategy/guardrail items + cases).

    Rendered ABOVE the recalled-context section within the shared lane-2
    budget. '' when empty or out of budget. Each line is compact and
    parse-stable; a guardrail/strategy leads with a glyph, a case is tagged
    with its verdict so a FAILED past task reads as a warning.
    """
    if not items or budget_tokens <= 0:
        return ""
    parts = [GUIDANCE_HEADER]
    used = db.approx_tokens(GUIDANCE_HEADER)
    for i, g in enumerate(items):
        title = _WS.sub(" ", (g.title or "")).strip()[:_GUIDANCE_SNIPPET]
        if not title:
            continue
        if g.kind == "case":
            tag = "FAILED" if g.verdict == "failure" else "past task"
            line = f"↺ similar {tag}: {title}"
        else:
            line = f"{_GUIDANCE_GLYPH.get(g.kind, g.kind)}: {title}"
        line = f"[{g.uid[:8]}] {line}"
        cost = db.approx_tokens(line)
        if used + cost > budget_tokens:
            if i == 0:
                parts.append(line)
            break
        parts.append(line)
        used += cost
    return "\n".join(parts) if len(parts) > 1 else ""


def render_hits_text(hits: list) -> str:
    """Human rendering for `hermes brain search`: index line, score, and a
    2-line wrapped snippet per hit.
    """
    if not hits:
        return "No matches."
    out: list[str] = []
    for hit in hits:
        out.append(f"{index_line(hit)}  (score {hit.score:.3f}, {hit.source})")
        body = _WS.sub(" ", hit.text or "").strip()
        if body:
            wrapped = textwrap.wrap(body, width=96, max_lines=2, placeholder=" ...")
            out.extend(f"    {line}" for line in wrapped)
    return "\n".join(out)
