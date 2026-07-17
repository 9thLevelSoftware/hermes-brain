"""Lane 1 materializer + renderer: the dream-maintained session index.

Two halves, strictly separated because of the byte-stability invariant
(docs/design/integration.md §2.1, critique item 17):

* ``materialize(conn, config)`` reads the LIVE memory tables and rebuilds
  the ``lane1_snapshot`` table in one transaction. It runs at dream/sweep
  time and at session boundaries — never per turn.
* ``render(conn, lane1_tokens)`` reads ONLY ``lane1_snapshot``. Because it
  never touches live tables, two renders between materializations are
  byte-identical no matter what capture wrote in the meantime — the
  provider can cache the string and prompt caching holds.

Lines are index lines (never full bodies), produced by
``recall.render.index_line`` so lane 1 and lane 2 speak one grammar.
Section framing (headers) is added at render time, not stored, so header
wording can evolve without a re-materialize.

Budget: ``render`` hard-truncates to ``lane1_tokens`` (db.approx_tokens)
by dropping whole lines from the END of sections in the order
facts -> open_loops -> stats -> warnings. Warnings are sacred: they are
sacrificed last, and only because the budget is a guarantee, not a hope.
An emptied section loses its header too.

``render`` returns '' when the snapshot is empty; the caller falls back
to ``render.lane1_static()`` (the P1 fixed block).
"""

from __future__ import annotations

import logging
import sqlite3
from types import SimpleNamespace
from typing import Any

from .. import __version__
from ..store import db
from .render import index_line

logger = logging.getLogger(__name__)

# Live current-truth predicate — the only rows lane 1 may ever index.
_CURRENT = "valid_to IS NULL AND status = 'active' AND live = 1"

_CAP_WARNINGS = 8
_CAP_OPEN_LOOPS = 5
_CAP_FACTS = 12

_HEADER = "## Brain (persistent memory) — session index"
_SECTION_HEADERS: dict[str, str] = {
    "warnings": "### ⚠ Failures & warnings (avoid repeating)",
    "open_loops": "### ◔ Open loops — outcomes unknown",
    "facts": "### ● Standing facts & preferences",
    "stats": "",  # stats lines are self-framing (counts + drill-down hint)
}
_SECTION_ORDER = ("warnings", "open_loops", "facts", "stats")
# Truncation sacrifice order: facts first, warnings last (sacred).
_DROP_ORDER = ("facts", "open_loops", "stats", "warnings")

_STATS_HINT = "deep recall: ask, or hermes brain search <query>"


# ---------------------------------------------------------------------------
# Half 1: materialize (live tables -> lane1_snapshot)
# ---------------------------------------------------------------------------

def _as_hit(row: sqlite3.Row) -> SimpleNamespace:
    """Adapt a memories row to the attribute surface index_line expects."""
    return SimpleNamespace(
        uid=row["uid"],
        summary=row["summary"],
        text=row["content"] or "",
        mkind=row["kind"],
        kind="memory",
        ts=row["valid_from"],
        platform=row["source_platform"],
    )


def _section_rows(conn: sqlite3.Connection) -> list[tuple[str, int, int, str]]:
    """(section, rank, memory_id, line) tuples for the three memory sections."""
    out: list[tuple[str, int, int, str]] = []

    warnings_sql = (
        f"SELECT * FROM memories WHERE {_CURRENT} "
        "AND (outcome = 'failed' OR kind = 'warning') "
        "ORDER BY harmful_count DESC, valid_from DESC LIMIT ?"
    )
    open_loops_sql = (
        f"SELECT * FROM memories WHERE {_CURRENT} "
        "AND kind = 'decision' AND outcome IS NULL "
        "ORDER BY valid_from DESC LIMIT ?"
    )
    facts_sql = (
        f"SELECT * FROM memories WHERE {_CURRENT} "
        "AND memory_type IN ('core','semantic') "
        "AND kind IN ('fact','preference','profile') "
        "ORDER BY pinned DESC, recall_count + verification_count DESC, "
        "valid_from DESC LIMIT ?"
    )
    for section, sql, cap in (
        ("warnings", warnings_sql, _CAP_WARNINGS),
        ("open_loops", open_loops_sql, _CAP_OPEN_LOOPS),
        ("facts", facts_sql, _CAP_FACTS),
    ):
        for rank, row in enumerate(conn.execute(sql, (cap,)).fetchall()):
            out.append((section, rank, row["id"], index_line(_as_hit(row))))
    return out


def materialize(conn: sqlite3.Connection, config: dict[str, Any]) -> int:
    """Rebuild lane1_snapshot from live tables. One transaction (DELETE all
    + INSERT); returns rows written. Dream/CLI-side: exceptions propagate to
    the caller — this is never on the capture path.
    """
    rows = _section_rows(conn)

    mem_count = conn.execute(
        f"SELECT COUNT(*) AS n FROM memories WHERE {_CURRENT}"
    ).fetchone()["n"]
    epi_count = conn.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()["n"]
    rows.append((
        "stats", 0, None,
        f"{mem_count} memories · {epi_count} episodes · brain v{__version__}",
    ))
    rows.append(("stats", 1, None, _STATS_HINT))

    now = db.iso_now()
    with conn:  # one transaction: readers see old snapshot or new, never half
        conn.execute("DELETE FROM lane1_snapshot")
        conn.executemany(
            "INSERT INTO lane1_snapshot (section, rank, memory_id, line, rendered_at) "
            "VALUES (?,?,?,?,?)",
            [(s, r, mid, line, now) for s, r, mid, line in rows],
        )
    logger.info("lane1_snapshot materialized: %d rows", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# Half 2: render (lane1_snapshot -> string; never touches live tables)
# ---------------------------------------------------------------------------

def _compose(lines_by_section: dict[str, list[str]]) -> str:
    parts: list[str] = [_HEADER]
    for section in _SECTION_ORDER:
        lines = lines_by_section.get(section)
        if not lines:
            continue
        parts.append("")  # blank line between blocks
        header = _SECTION_HEADERS[section]
        if header:
            parts.append(header)
        parts.extend(lines)
    return "\n".join(parts)


def render(conn: sqlite3.Connection, lane1_tokens: int) -> str:
    """Deterministic lane 1 from the snapshot ONLY. '' when the snapshot is
    empty (caller falls back to render.lane1_static()). Hard-truncated to
    ``lane1_tokens`` by dropping trailing lines, facts first, warnings last.
    """
    rows = conn.execute(
        "SELECT section, line FROM lane1_snapshot ORDER BY section, rank"
    ).fetchall()
    if not rows:
        return ""

    lines_by_section: dict[str, list[str]] = {s: [] for s in _SECTION_ORDER}
    for row in rows:
        lines_by_section.setdefault(row["section"], []).append(row["line"])

    text = _compose(lines_by_section)
    while db.approx_tokens(text) > lane1_tokens:
        for section in _DROP_ORDER:
            if lines_by_section.get(section):
                lines_by_section[section].pop()  # drop from the END (lowest rank)
                break
        else:
            # Even the bare header exceeds the budget: nothing sane to emit.
            return ""
        text = _compose(lines_by_section)
    return text
