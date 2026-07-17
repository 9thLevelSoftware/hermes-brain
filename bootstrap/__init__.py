"""First-run bootstrap: make a fresh brain instantly useful from what exists.

Order matters (docs/design/integration.md §5.4): MEMORY.md/USER.md first
(curated facts seed the profile before anything noisier), then the state.db
episodic backfill (rate-limited per run), then the optional Daem0n import.
Every stage is idempotent (content_hash dedup / sweep_state watermarks) and
reads its source strictly READ-ONLY, so ``run_bootstrap`` is safe to call on
every initialize() with an empty DB and re-runnable from the CLI forever.

Never raises: a broken stage records {'error': ...} beside the partial
counts already earned — bootstrap failure must not block the provider.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from .daemon_import import import_daemon_db
from .memory_md import import_memory_files
from .state_db import backfill_sessions

logger = logging.getLogger(__name__)

__all__ = ["run_bootstrap", "import_memory_files", "backfill_sessions", "import_daemon_db"]


def run_bootstrap(
    conn: sqlite3.Connection,
    hermes_home: str | Path,
    config: dict[str, Any],
    *,
    embedder=None,
    daemon_db: str | Path | None = None,
    max_sessions: int | None = None,
) -> dict[str, Any]:
    """Run all bootstrap stages; return merged counts (never raises)."""
    counts: dict[str, Any] = {
        "memory_md": 0, "user_md": 0, "memory_skipped": 0,
        "sessions": 0, "turns": 0, "sessions_skipped": 0,
    }
    if not config.get("bootstrap_import", True):
        counts["disabled"] = True
        return counts

    try:
        mem = import_memory_files(conn, hermes_home)
        counts["memory_md"] = mem["memory"]
        counts["user_md"] = mem["user"]
        counts["memory_skipped"] = mem["skipped"]
    except Exception as e:
        logger.warning("bootstrap: memory file import failed: %s", e, exc_info=True)
        counts["error"] = f"memory files: {e}"
        return counts

    try:
        # Forward max_sessions only when the caller set one — state_db's
        # default (20/run) stays the single source of truth for the cap.
        backfill_kwargs: dict[str, Any] = {"embedder": embedder}
        if max_sessions is not None:
            backfill_kwargs["max_sessions"] = max_sessions
        sessions = backfill_sessions(conn, hermes_home, **backfill_kwargs)
        counts["sessions"] = sessions["sessions"]
        counts["turns"] = sessions["turns"]
        counts["sessions_skipped"] = sessions["skipped"]
    except Exception as e:
        logger.warning("bootstrap: state.db backfill failed: %s", e, exc_info=True)
        counts["error"] = f"state.db backfill: {e}"
        return counts

    if daemon_db is not None:
        try:
            daemon = import_daemon_db(conn, daemon_db)
            counts["daemon_imported"] = daemon["imported"]
            counts["daemon_skipped"] = daemon["skipped"]
            if "error" in daemon:
                counts["error"] = daemon["error"]
        except Exception as e:
            logger.warning("bootstrap: daemon import failed: %s", e, exc_info=True)
            counts["error"] = f"daemon import: {e}"

    logger.info(
        "bootstrap: %d MEMORY.md + %d USER.md entries, %d session(s)/%d turn(s)"
        " backfilled, %d daemon row(s)%s",
        counts["memory_md"], counts["user_md"], counts["sessions"], counts["turns"],
        counts.get("daemon_imported", 0),
        f" — ERROR: {counts['error']}" if "error" in counts else "",
    )
    return counts
