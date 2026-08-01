"""Deterministic retention policy for extracted memory candidates.

The model may describe its intent, but high-precision local rules own the
final decision.  This keeps routine work narration out of durable memory and
makes every temporary row carry an enforceable absolute expiry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


class RetentionError(ValueError):
    """The extractor supplied an invalid retention contract."""


@dataclass(frozen=True)
class RetentionPolicy:
    retention: str
    ttl_days: int | None = None
    ttl_at: str | None = None
    expiry_source: str | None = None


_RETENTIONS = frozenset({"episode_only", "temporary", "durable"})
_DURABLE_KINDS = frozenset({"preference", "warning", "profile", "decision"})

# Intentionally high precision.  False negatives remain ordinary episodes and
# can be revisited by dream; false positives would discard a durable memory.
_PROCESS_NARRATION_RE = re.compile(
    r"\b(?:created|made|pushed|amended|rebased|squashed)\s+(?:the\s+)?commit\b"
    r"|\bcommit\s+[0-9a-f]{7,40}\b"
    r"|\b(?:ran|running|executed)\s+(?:the\s+)?(?:tests?|pytest|ruff|lint|build)\b"
    r"|\b(?:command|process)\s+(?:completed|exited|returned)\b",
    re.IGNORECASE,
)
_OPERATIONAL_RE = re.compile(
    r"\b(?:branch|deployment|deploy(?:ment|ed|ing|s)?|pull request|pr\s*#?\d+|"
    r"ci|currently working on|working on)\b",
    re.IGNORECASE,
)
_END_DATE_RE = re.compile(
    r"\b(?:until|through|ending|ends|expires|expiry|end date(?: is)?)\s+"
    r"(\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


def _iso(value: datetime) -> str:
    value = value.astimezone(UTC)
    return value.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _validate_days(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 365:
        raise RetentionError("ttl_days must be an integer from 1 through 365")
    return value


def classify_retention(
    content: str,
    kind: str,
    *,
    requested: str | None = None,
    ttl_days=None,
    now: datetime | None = None,
) -> RetentionPolicy:
    """Return the authoritative policy for one candidate.

    ``requested`` is the extractor's v3 nomination.  Deterministic process
    narration always stays episode-only.  Explicit temporary requests are
    honored (including for normally durable kinds) only with a valid bounded
    TTL.  Otherwise durable identity/preferences/warnings/decisions never get
    an inferred expiry.
    """
    text = " ".join((content or "").split())
    clock = now or datetime.now(UTC)

    if requested is not None and requested not in _RETENTIONS:
        raise RetentionError(
            "retention must be episode_only, temporary, or durable"
        )

    if _PROCESS_NARRATION_RE.search(text):
        return RetentionPolicy("episode_only")

    if requested == "episode_only":
        if ttl_days is not None:
            raise RetentionError("episode_only items cannot specify ttl_days")
        return RetentionPolicy("episode_only")

    if requested == "temporary":
        days = _validate_days(ttl_days if ttl_days is not None else 7)
        return RetentionPolicy(
            "temporary",
            ttl_days=days,
            ttl_at=_iso(clock + timedelta(days=days)),
            expiry_source="extractor_ttl",
        )

    if ttl_days is not None:
        raise RetentionError("ttl_days is only valid for temporary retention")

    dated = _END_DATE_RE.search(text)
    if dated:
        try:
            end = datetime.strptime(dated.group(1), "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError as exc:  # defensive: regex admits calendar-invalid dates
            raise RetentionError("explicit end date is not a valid calendar date") from exc
        expiry = end + timedelta(days=1)
        days = max(1, min(365, (expiry.date() - clock.date()).days))
        return RetentionPolicy(
            "temporary",
            ttl_days=days,
            ttl_at=_iso(expiry),
            expiry_source="explicit_end_date",
        )

    # These kinds resist merely inferred operational TTLs, but a source that
    # states its own end date above is explicitly temporary.
    if kind in _DURABLE_KINDS and requested != "temporary":
        return RetentionPolicy("durable")

    if _OPERATIONAL_RE.search(text):
        days = 7
        return RetentionPolicy(
            "temporary",
            ttl_days=days,
            ttl_at=_iso(clock + timedelta(days=days)),
            expiry_source="operational_default",
        )

    return RetentionPolicy("durable")


def lifecycle_meta(raw: str | None, policy: RetentionPolicy) -> str:
    """Merge retention fields into a row's JSON escape-hatch metadata."""
    import json

    try:
        meta = json.loads(raw or "{}")
    except (TypeError, ValueError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    meta["retention"] = policy.retention
    if policy.expiry_source:
        meta["expiry_source"] = policy.expiry_source
    else:
        meta.pop("expiry_source", None)
    return json.dumps(meta, sort_keys=True)
