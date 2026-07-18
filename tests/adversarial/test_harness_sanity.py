"""Sanity checks for the adversarial harness itself — proves the fault toolkit
imports under pytest (``from faults import ...``) and that each helper induces
its condition against a real brain connection. If this file is red, every
downstream adversarial phase is untrustworthy, so it runs first (alphabetically
``test_harness_*``) and cheap."""

from __future__ import annotations

from conftest import seed_memory
from faults import (
    break_archive_dir,
    corrupt_half_life,
    returning,
    simulate_no_fts5,
)


def _search(conn, query):
    from brain.recall.search import search

    return search(conn, query, limit=5, principal_id="owner", trust_tier="owner")


def test_baseline_recall_then_corrupt_row_degrades_not_raises(conn):
    rowid = seed_memory(conn, "the staging database is postgres 14 on fly.io", kind="fact")
    hits = _search(conn, "staging database")
    assert isinstance(hits, list) and len(hits) >= 1  # baseline works

    # A corrupt lifecycle field must never propagate into the (turn) caller.
    corrupt_half_life(conn, rowid)
    hits = _search(conn, "staging database")
    assert isinstance(hits, list)  # degraded to a safe list, no exception


def test_simulate_no_fts5_routes_to_like_and_still_recalls(conn):
    seed_memory(conn, "deploy script lives at scripts/deploy.sh", kind="fact")
    simulate_no_fts5(conn)
    from brain.store import db

    assert db.capabilities(conn).get("fts5") is False
    hits = _search(conn, "deploy script")
    assert isinstance(hits, list) and len(hits) >= 1  # LIKE fallback found it


def test_break_archive_dir_makes_append_return_none(tmp_home):
    from brain.store import archive, db

    break_archive_dir(tmp_home)
    ref = archive.append(tmp_home, {"uid": db.new_ulid(), "content": "raw text"})
    assert ref is None  # the load-bearing "archiving failed" sentinel


def test_returning_patch_forces_archive_none(tmp_home):
    from brain.store import archive, db

    # even with a healthy dir, force the None sentinel via the patch helper
    with returning(archive, "append", None):
        assert archive.append(tmp_home, {"uid": db.new_ulid(), "content": "x"}) is None
    # patch lifted: a real append now succeeds
    ref = archive.append(tmp_home, {"uid": db.new_ulid(), "content": "x"})
    assert ref and ref.endswith(":" + ref.split(":")[-1])
