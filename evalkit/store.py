"""Query-set persistence. User data, never the repo.

The query set is derived from the owner's real memories and quotes them, so it
lives beside brain.db under HERMES_HOME (already covered by `hermes backup`)
and never in the plugin tree — a queryset committed to git would leak the
owner's conversations into a public repo.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def queryset_path(hermes_home: str | Path) -> Path:
    return Path(hermes_home) / "brain" / "eval" / "queryset.json"


def save_queryset(hermes_home: str | Path, queries: list[dict[str, Any]],
                  *, meta: dict[str, Any] | None = None) -> Path:
    """Atomic write, so an interrupted generate never leaves a half-file that
    the next --compare would silently score against."""
    path = queryset_path(hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA_VERSION,
        "meta": dict(meta or {}),
        "queries": queries,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return path


def load_queryset(hermes_home: str | Path) -> dict[str, Any] | None:
    """Parsed query set, or None when absent/unreadable (never raises — the
    caller prints a remedy, it does not crash a CLI verb)."""
    path = queryset_path(hermes_home)
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("eval: queryset unreadable (%s)", e)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        logger.warning("eval: queryset at %s is malformed", path)
        return None
    return data
