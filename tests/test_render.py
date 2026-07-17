"""Render tests: lane-1 byte stability (the cache-safety contract — one
render at initialize, identical forever), lane-2 budget enforcement
(integration.md: the budget is a guarantee, not a hope), and index-line
grammar stability.
"""

from __future__ import annotations

import re

from brain.config import load_config
from brain.recall.render import index_line, lane1_static, lane2_block
from brain.recall.search import search
from brain.store import db
from conftest import seed_memory

# Anything date-shaped would break byte-stability the moment the clock ticks.
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\b20\d{2}\b|\d{1,2}/\d{1,2}/\d{2,4}")


def test_lane1_static_byte_stable_across_100_calls():
    renders = {lane1_static().encode("utf-8") for _ in range(100)}
    assert len(renders) == 1
    assert renders.pop()  # non-empty


def test_lane1_static_contains_no_dates():
    match = _DATE_RE.search(lane1_static())
    assert match is None, f"date-like digits in lane 1 break byte-stability: {match.group()!r}"


def _hits_for(conn, keyword, count=20):
    filler = ("the ingress controller for this namespace needs its sidecar restarted "
              "whenever the config map rotates, and the autoscaler misreads memory "
              "pressure during rolling deploys of the payment workers")
    for i in range(count):
        seed_memory(conn, f"{keyword} cluster note {i}: {filler} (case {i}).", kind="fact")
    hits = search(conn, keyword, limit=50)
    assert len(hits) >= 8, "seeded rows must be retrievable to exercise the budget"
    return hits


def test_lane2_block_respects_budget(conn, tmp_home):
    hits = _hits_for(conn, "kubernetes")
    budget = int(load_config(tmp_home)["lane2_tokens"])
    block = lane2_block(hits, budget)
    assert isinstance(block, str)
    assert block, "relevant hits must render something"
    slack = 64  # header/ellipsis allowance
    assert db.approx_tokens(block) <= budget + slack, (
        f"lane 2 blew its budget: {db.approx_tokens(block)} > {budget} + {slack}")


def test_lane2_block_empty_hits_is_empty_string(tmp_home):
    budget = int(load_config(tmp_home)["lane2_tokens"])
    assert lane2_block([], budget) == ""


def test_index_line_format_stable(conn):
    seed_memory(conn, "Never deploy on Fridays without the rollback script staged.",
                kind="warning", outcome="failed")
    hits = search(conn, "rollback script", limit=5)
    assert hits
    line = index_line(hits[0])
    assert isinstance(line, str)
    assert "\n" not in line, "index_line must be a single line"
    # Grammar: "[uid8 · kind · date · platform] snippet" — bracketed header, then content.
    assert re.match(r"^\[[^\]\n]+\]\s+\S", line), f"unexpected index line shape: {line!r}"
