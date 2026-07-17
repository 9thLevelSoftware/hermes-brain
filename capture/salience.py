"""Capture-time salience scoring — pure heuristics, no LLM.

Every episode gets a 0..1 salience score at capture so the sweep/dream can
prioritize extraction work without re-reading everything (docs/design/
memory-engine.md, episodic lane). The score is additive over independent
signal families and clamped; it is deliberately crude — its only consumers
are extraction ordering and the very-short-pleasantry floor, so precision
beyond "boring / interesting / must-keep" would be dead weight.

Design constraints:
  * user-side-only families (corrections, preferences) match ONLY the user
    text — the assistant echoing "I prefer..." back must not inflate score.
  * shared families (errors, decisions, remember-requests, code) match the
    concatenation of both sides, case-insensitively.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Error/failure markers: failed turns are the highest-value episodes for the
# warnings/anti-pattern lane (Daem0n's one proven win). +0.25
_ERROR_RE = re.compile(r"traceback|error:|\bfailed\b|\bexception\b", re.I)

# Decision language: "decided", "let's go with", "instead of", "switched to"
# mark choice points the dream distills into decision memories. +0.2
_DECISION_RE = re.compile(
    r"\bdecided\b|let'?s go with|\binstead of\b|switched to", re.I
)

# User corrections (user text only): a correction is a high-signal negative
# example — the assistant just got something wrong. +0.2
_CORRECTION_RE = re.compile(r"\bno,|\bactually\b|that'?s wrong|\bdon'?t\b", re.I)

# Preferences / identity (user text only): durable profile facts, the bread
# and butter of the user_profile core block. +0.3
_PREFERENCE_RE = re.compile(
    r"i prefer|i always|i never|my name is|i work|call me", re.I
)

# Explicit remember requests: the user is TELLING us this matters — highest
# single boost. +0.35
_REMEMBER_RE = re.compile(r"\bremember\b|note that|don'?t forget", re.I)

# Code presence: fenced blocks or a handful of extracted identifiers means
# the turn is technical substance, not chit-chat. +0.1
_CODE_FENCE_RE = re.compile(r"```")

# Very short exchanges with none of the above are pleasantries ("thanks!",
# "ok cool") — capped at 0.05 so they never win an extraction slot. The cap
# only bites because every turn carries a small base salience (a boring but
# substantive turn still beats "thanks!").
_PLEASANTRY_MAX_CHARS = 40
_BASE = 0.1
_PLEASANTRY_CAP = 0.05


def score_turn(user_content: str, assistant_content: str) -> float:
    """Heuristic 0..1 salience for one (user, assistant) turn."""
    user = user_content or ""
    assistant = assistant_content or ""
    both = f"{user}\n{assistant}"

    boost = 0.0
    if _ERROR_RE.search(both):
        boost += 0.25
    if _DECISION_RE.search(both):
        boost += 0.2
    if _CORRECTION_RE.search(user):
        boost += 0.2
    if _PREFERENCE_RE.search(user):
        boost += 0.3
    if _REMEMBER_RE.search(both):
        boost += 0.35
    if _CODE_FENCE_RE.search(both) or _has_code_symbols(both):
        boost += 0.1

    # Pleasantry cap: tiny turn, no signal families fired -> near-zero.
    if boost == 0.0 and len(both.strip()) <= _PLEASANTRY_MAX_CHARS:
        return _PLEASANTRY_CAP

    return min(1.0, _BASE + boost)


def _has_code_symbols(text: str) -> bool:
    """>= 3 extracted code identifiers counts as code presence."""
    from .symbols import extract_code_symbols

    return len(extract_code_symbols(text)) >= 3
