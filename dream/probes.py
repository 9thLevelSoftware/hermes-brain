"""Capability-regression probe suite (learning-system.md §3).

Deterministic, no LLM, meant to finish in well under a second on a normal
brain. Runs as the last dream strategy (checks the shift's net effect) and
is ALSO called directly by the skill-forge's validation gate — one suite,
two callers.

Four probe families, all derived from the brain's own state (no hand-authored
fixtures to rot):

  retrieval  — a high-value memory (pinned / helpful / important) must still
               retrieve itself in the top-K by its own words. Regression here
               means the index or ranking broke.
  staleness  — a superseded version must be closed (valid_to set) so it can
               never outrank its successor; a tombstone must never be current.
  injection  — a quarantined instruction-shaped row must never appear in the
               lane-1 snapshot or a lane-2 render of its own content.
  latency    — a cold search must return under the budget.

A failing probe never rolls anything back here (the dream commits per
strategy, not shift-wide); it writes an audit row and a `proposal`-free
review signal, and returns the failures so the caller decides. The
skill-forge treats any failure as a hard veto on promotion.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field

from ..recall.search import search
from .shift import Shift

logger = logging.getLogger(__name__)

_RETRIEVAL_TOPK = 5
_MAX_RETRIEVAL_PROBES = 12
_LATENCY_BUDGET_S = 2.0


@dataclass
class ProbeResult:
    name: str
    family: str
    passed: bool
    detail: str = ""


@dataclass
class ProbeReport:
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def failed(self) -> list[ProbeResult]:
        return [r for r in self.results if not r.passed]

    @property
    def ran(self) -> int:
        return len(self.results)

    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> dict:
        return {"ran": self.ran, "passed": self.ran - len(self.failed),
                "failed": len(self.failed),
                "failures": [f"{r.family}:{r.name}" for r in self.failed][:20]}


def run_probes(conn: sqlite3.Connection, config: dict, *, embedder=None) -> ProbeReport:
    """Run the whole suite. Never raises — a probe that errors is a FAILURE
    (a health check that can't run is not a pass)."""
    report = ProbeReport()
    for family, fn in (("retrieval", _retrieval_probes),
                       ("staleness", _staleness_probes),
                       ("injection", _injection_probes),
                       ("latency", _latency_probes)):
        try:
            fn(conn, config, embedder, report)
        except Exception as e:
            logger.warning("probes: family %s errored: %s", family, e, exc_info=True)
            report.results.append(ProbeResult(
                name="suite_error", family=family, passed=False, detail=str(e)))
    return report


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------

def _retrieval_probes(conn, config, embedder, report) -> None:
    """A high-value memory must retrieve itself by its own words."""
    rows = conn.execute(
        "SELECT id, uid, content, summary FROM memories"
        " WHERE valid_to IS NULL AND status='active' AND live=1"
        " AND (pinned=1 OR helpful_count>0 OR importance>=0.6)"
        " AND content IS NOT NULL"
        " ORDER BY pinned DESC, helpful_count DESC, importance DESC"
        " LIMIT ?", (_MAX_RETRIEVAL_PROBES,)).fetchall()
    for row in rows:
        query = _probe_query(row["summary"] or row["content"])
        if not query:
            continue
        hits = search(conn, query, limit=_RETRIEVAL_TOPK, trust_tier="owner",
                      include_episodes=False, embedder=embedder)
        found = any(h.kind == "memory" and h.id == row["id"] for h in hits)
        report.results.append(ProbeResult(
            name=row["uid"][:8], family="retrieval", passed=found,
            detail="" if found else f"'{query}' did not surface {row['uid'][:8]} "
                   f"in top-{_RETRIEVAL_TOPK}"))


def _staleness_probes(conn, config, embedder, report) -> None:
    """Superseded/tombstoned rows must be closed — never current truth."""
    live_superseded = conn.execute(
        "SELECT count(*) AS n FROM memories"
        " WHERE superseded_by IS NOT NULL AND valid_to IS NULL").fetchone()["n"]
    report.results.append(ProbeResult(
        name="superseded_closed", family="staleness", passed=live_superseded == 0,
        detail="" if live_superseded == 0
        else f"{live_superseded} superseded rows still current (valid_to NULL)"))

    live_tombstone = conn.execute(
        "SELECT count(*) AS n FROM memories"
        " WHERE status='tombstone' AND valid_to IS NULL").fetchone()["n"]
    report.results.append(ProbeResult(
        name="tombstone_closed", family="staleness", passed=live_tombstone == 0,
        detail="" if live_tombstone == 0
        else f"{live_tombstone} tombstones still current"))


def _injection_probes(conn, config, embedder, report) -> None:
    """Quarantined instruction-shaped rows must never reach the lanes."""
    quarantined = conn.execute(
        "SELECT id, uid, content FROM memories WHERE status='quarantined'"
        " AND content IS NOT NULL LIMIT 20").fetchall()
    if not quarantined:
        return  # nothing to check — not a failure

    # Lane 1: a quarantined memory_id must never be in the snapshot.
    snap_ids = {r["memory_id"] for r in conn.execute(
        "SELECT memory_id FROM lane1_snapshot WHERE memory_id IS NOT NULL")}
    for row in quarantined:
        report.results.append(ProbeResult(
            name=f"lane1_{row['uid'][:8]}", family="injection",
            passed=row["id"] not in snap_ids,
            detail="" if row["id"] not in snap_ids
            else f"quarantined {row['uid'][:8]} is in the lane-1 snapshot"))

    # Lane 2: searching a quarantined row's OWN content must not return it.
    canary = quarantined[0]
    query = _probe_query(canary["content"])
    if query:
        hits = search(conn, query, limit=8, trust_tier="owner",
                      include_episodes=False, embedder=embedder)
        leaked = any(h.kind == "memory" and h.id == canary["id"] for h in hits)
        report.results.append(ProbeResult(
            name=f"lane2_{canary['uid'][:8]}", family="injection", passed=not leaked,
            detail="" if not leaked
            else f"quarantined {canary['uid'][:8]} surfaced in recall"))


def _latency_probes(conn, config, embedder, report) -> None:
    start = time.monotonic()
    try:
        search(conn, "deploy staging database migration", limit=8,
               trust_tier="owner", embedder=embedder)
    except Exception as e:
        report.results.append(ProbeResult(
            name="cold_search", family="latency", passed=False, detail=str(e)))
        return
    elapsed = time.monotonic() - start
    report.results.append(ProbeResult(
        name="cold_search", family="latency", passed=elapsed < _LATENCY_BUDGET_S,
        detail="" if elapsed < _LATENCY_BUDGET_S
        else f"cold search took {elapsed:.2f}s (> {_LATENCY_BUDGET_S}s)"))


def _probe_query(text: str) -> str:
    """First few content words — a self-retrieval query with real signal."""
    words = (text or "").split()
    return " ".join(words[:8])


# ---------------------------------------------------------------------------
# Strategy protocol wrapper
# ---------------------------------------------------------------------------

def run(shift: Shift) -> dict:
    """Post-shift health check. Read-only over memories; on failure it writes
    ONE audit row (a review-queue signal) regardless of mode — a regression is
    a fact the operator must see whether the shift ran active or dry_run, and
    the audit row is not a memory mutation. There is deliberately no
    mode-gating here."""
    try:
        report = run_probes(shift.conn, shift.config, embedder=shift.embedder)
    except Exception as e:
        logger.warning("probes: failed: %s", e, exc_info=True)
        return {"error": str(e)}
    result = report.summary()
    if report.failed:
        # A regression the operator should see — surfaces in `hermes brain
        # review` and status. Never rolls memory back (per-strategy commits).
        shift.audit("probe_failure", None, result)
        shift.conn.commit()
        logger.warning("probes: %d/%d FAILED: %s", result["failed"],
                       result["ran"], result["failures"])
    return result
