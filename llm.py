"""The ONLY gateway for brain-initiated LLM calls (locked decision #2).

Resolution order:
  1. Test override installed via ``set_llm_for_tests`` (hermetic tests).
  2. The Hermes auxiliary client (``agent.auxiliary_client``) — the brain
     runs in-process inside Hermes, so brain calls resolve through the
     active profile's provider config exactly like background_review and
     the curator do. Asymmetric tiers: 'extract' (cheap/fast, JSON mode)
     vs 'dream'/'consolidate' (strong model). Model overrides come from
     brain.yaml: ``extract_model`` / ``dream_model``; empty string means
     "use the auxiliary default".
  3. No path: raise ``LLMUnavailable`` — standalone (outside Hermes) has
     no LLM in P3; callers treat this as "skip and retry next run".

Metering: every successful call inserts one ``llm_ledger`` row
(strategy=tier, tokens via db.approx_tokens, est_usd=0.0 — real pricing
plumbing arrives with the P4 dream).

Budget gate: before every call, the ledger's token total for the current
UTC day is compared against ``day_budget_usd * 400_000`` tokens — a
deliberately crude $2.50/Mtok blended-price proxy that stands in until
real usage/pricing plumbing exists. Over budget => LLMUnavailable.

Module level is stdlib-only (plus our own stdlib-only store.db); the
auxiliary client is imported lazily inside the call path.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Callable
from typing import Any

from .store import db

logger = logging.getLogger(__name__)

# Crude blended-price proxy: $2.50 per 1M tokens => 400k tokens per USD.
# Documented stand-in until real usage plumbing exists (see module docs).
_TOKENS_PER_USD = 400_000

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)```", re.S)
_RETRY_SUFFIX = "\n\nReturn ONLY valid JSON."
_PARSE_FAILED = object()

# tier -> brain.yaml model-override key ('' = auxiliary default).
_TIER_MODEL_KEY = {
    "extract": "extract_model",
    "dream": "dream_model",
    "consolidate": "dream_model",
}
# tier -> auxiliary task slot (design: learning-system.md §4).
_TIER_TASK = {
    "extract": "brain_extract",
    "dream": "brain_consolidate",
    "consolidate": "brain_consolidate",
}


class LLMUnavailable(RuntimeError):
    """No LLM path / budget exhausted / unusable response — callers skip."""


_test_llm: Callable[..., str] | None = None


def set_llm_for_tests(fn: Callable[..., str] | None) -> None:
    """Install (or clear, with None) a fake: fn(prompt, *, system, max_tokens) -> str."""
    global _test_llm
    _test_llm = fn


# ---------------------------------------------------------------------------
# call_text
# ---------------------------------------------------------------------------

def call_text(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    prompt: str,
    *,
    system: str | None = None,
    tier: str = "extract",
    max_tokens: int = 1200,
) -> str:
    """One brain LLM call -> response text. Raises LLMUnavailable when there
    is no path, the daily budget is spent, or the call itself fails."""
    _budget_gate(conn, config)
    model = str(config.get(_TIER_MODEL_KEY.get(tier, "extract_model")) or "")
    if _test_llm is not None:
        text = _test_llm(prompt, system=system, max_tokens=max_tokens)
        # Meter regardless (finding #14: even an unusable reply burned
        # tokens), then treat empty as unavailable — same contract as the
        # aux path so tests exercise the real behavior.
        _meter(conn, tier, model or "aux-default", prompt, system, text)
        if not (text or "").strip():
            raise LLMUnavailable("test LLM returned empty text")
        return text
    return _aux_call(conn, prompt, system, tier, model, max_tokens)


def _aux_call(conn: sqlite3.Connection, prompt: str, system: str | None,
              tier: str, model: str, max_tokens: int) -> str:
    try:
        # Lazy: agent.* exists only inside a Hermes process.
        from agent import auxiliary_client as aux
    except ImportError:
        # No provider was contacted — nothing to meter.
        raise LLMUnavailable(
            "no LLM path: inside Hermes the auxiliary client serves brain "
            "calls; standalone requires none in P3"
        )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": 0.0,  # extraction tiers are deterministic by design
    }
    if model:  # '' = auxiliary default — never pass a speculative override
        kwargs["model"] = model
    label = model or "aux-default"
    # Meter every call that REACHES the provider, success or not (finding
    # #14): a provider that times out or returns empty still burns input
    # tokens, and if those calls weren't recorded the daily budget gate
    # could never trip on a wedged provider — it would be re-billed forever.
    try:
        response = aux.call_llm(_TIER_TASK.get(tier, "brain_extract"), **kwargs)
        text = aux.extract_content_or_reasoning(response)
    except Exception as e:
        _meter(conn, tier, label, prompt, system, "")
        raise LLMUnavailable(
            f"auxiliary LLM call failed ({e}); the buffer is durable — "
            "the next sweep retries"
        ) from e
    if not (text or "").strip():
        _meter(conn, tier, label, prompt, system, "")
        raise LLMUnavailable(
            "auxiliary LLM returned empty text; the next sweep retries")
    _meter(conn, tier, label, prompt, system, text)
    return text


# ---------------------------------------------------------------------------
# call_json
# ---------------------------------------------------------------------------

def call_json(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    prompt: str,
    *,
    system: str | None = None,
    tier: str = "extract",
    max_tokens: int = 1600,
) -> object:
    """call_text + JSON parsing (fence stripping, first balanced [..]/{..}
    span). ONE retry appending 'Return ONLY valid JSON.'; LLMUnavailable on
    the second parse failure (callers treat it as skip)."""
    attempt_prompt = prompt
    last = ""
    for _ in range(2):
        text = call_text(conn, config, attempt_prompt, system=system,
                         tier=tier, max_tokens=max_tokens)
        parsed = _parse_json(text)
        if parsed is not _PARSE_FAILED:
            return parsed
        last = text
        attempt_prompt = prompt + _RETRY_SUFFIX
    raise LLMUnavailable(
        f"LLM returned unparseable JSON twice (last reply began {last[:80]!r}); "
        "skipping this unit — the next sweep retries"
    )


def _parse_json(text: str):
    if not text:
        return _PARSE_FAILED
    candidates = [m.strip() for m in _FENCE_RE.findall(text) if m.strip()]
    candidates.append(text)
    scalar: Any = _PARSE_FAILED
    for cand in candidates:
        # Try EVERY balanced span in order, not just the first (finding #13):
        # a bracketed token in prose preamble ('[1]', '{note}') must not be
        # mistaken for the payload. Prefer a list/dict; keep a scalar only as
        # a last resort so a stray '[1]' never masks a real array further on.
        for span in _balanced_spans(cand):
            try:
                parsed = json.loads(span)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, (list, dict)):
                return parsed
            if scalar is _PARSE_FAILED:
                scalar = parsed
    return scalar


def _balanced_spans(text: str):
    """Yield each balanced [...] or {...} span in order, string-literal aware."""
    i, n = 0, len(text)
    while i < n:
        if text[i] not in "[{":
            i += 1
            continue
        span = _span_from(text, i)
        if span is None:
            i += 1
            continue
        yield span
        i += len(span)


def _span_from(text: str, start: int) -> str | None:
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


# ---------------------------------------------------------------------------
# Metering + budget
# ---------------------------------------------------------------------------

def _budget_gate(conn: sqlite3.Connection, config: dict[str, Any]) -> None:
    day = db.iso_now()[:10]
    used = conn.execute(
        "SELECT COALESCE(SUM(tokens_in + tokens_out), 0) FROM llm_ledger "
        "WHERE ts LIKE ?",
        (day + "%",),
    ).fetchone()[0]
    budget_usd = float(config.get("day_budget_usd", 1.5))
    if used > budget_usd * _TOKENS_PER_USD:
        raise LLMUnavailable(
            f"daily brain LLM budget reached ({used} tokens today > "
            f"${budget_usd:.2f} * {_TOKENS_PER_USD} tokens/USD proxy); calls "
            "resume tomorrow UTC — or raise day_budget_usd in brain.yaml"
        )


def _meter(conn: sqlite3.Connection, tier: str, model: str, prompt: str,
           system: str | None, response: str) -> None:
    try:
        conn.execute(
            "INSERT INTO llm_ledger (strategy, model, tokens_in, tokens_out,"
            " est_usd, ts) VALUES (?,?,?,?,0.0,?)",
            (
                tier,
                model,
                db.approx_tokens((system or "") + (prompt or "")),
                db.approx_tokens(response or ""),
                db.iso_now(),
            ),
        )
        conn.commit()
    except sqlite3.Error as e:  # metering must never mask a good response
        logger.warning("llm_ledger insert failed: %s", e)
