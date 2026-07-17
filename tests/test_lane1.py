"""Lane 1 materialize/render: snapshot round trip, byte-stability (the
render half must depend on lane1_snapshot ONLY), budget sacrifice order
(facts before open loops before warnings), and lifecycle filtering
(quarantined/superseded never indexed; open loops close on outcome).
"""

from __future__ import annotations

from brain.recall import lane1
from brain.store import db
from conftest import seed_memory

BIG = 100_000  # a budget nothing here can exceed


def _mat(conn) -> int:
    return lane1.materialize(conn, {})


def test_materialize_render_round_trip(conn):
    seed_memory(conn, "pip install inside Termux kills the gateway", kind="warning")
    seed_memory(conn, "chose LanceDB fallback threshold", kind="decision",
                outcome="failed")
    seed_memory(conn, "migrate kanban to FTS5 triggers", kind="decision")  # open loop
    seed_memory(conn, "user prefers terse answers", kind="preference")
    seed_memory(conn, "the VPS is Hetzner CX22 with 2GB RAM", kind="fact")
    seed_memory(conn, "user timezone is America/Chicago", kind="fact", pinned=1)

    written = _mat(conn)
    # 2 warnings + 1 open loop + 3 facts + 2 stats lines
    assert written == 8

    out = lane1.render(conn, BIG)
    assert out.startswith("## Brain (persistent memory) — session index")
    assert "### ⚠ Failures & warnings (avoid repeating)" in out
    assert "### ◔ Open loops — outcomes unknown" in out
    assert "### ● Standing facts & preferences" in out
    assert "pip install inside Termux" in out
    assert "chose LanceDB fallback" in out            # failed outcome -> warnings
    assert "migrate kanban to FTS5" in out            # open loop
    assert "6 memories · 0 episodes · brain v" in out  # stats counts line
    assert "hermes brain search" in out                # drill-down hint

    # pinned fact ranks first inside the facts section
    facts_block = out.split("### ● Standing facts & preferences")[1]
    assert facts_block.index("America/Chicago") < facts_block.index("Hetzner CX22")
    assert facts_block.index("America/Chicago") < facts_block.index("terse answers")


def test_render_is_byte_stable_until_rematerialize(conn):
    seed_memory(conn, "never rebase the shared branch", kind="warning")
    _mat(conn)

    first = lane1.render(conn, BIG)
    assert first == lane1.render(conn, BIG)

    # Live-table writes must NOT leak into render output...
    seed_memory(conn, "a brand new fact about ducks", kind="fact")
    assert lane1.render(conn, BIG) == first
    # ...until the snapshot is rebuilt. That's the point.
    _mat(conn)
    after = lane1.render(conn, BIG)
    assert after != first
    assert "ducks" in after


def test_budget_drops_facts_then_open_loops_before_warnings(conn):
    seed_memory(conn, "warning about the fragile deploy script " + "w" * 80,
                kind="warning")
    seed_memory(conn, "open decision on the caching layer " + "o" * 80,
                kind="decision")
    for i in range(6):
        seed_memory(conn, f"long standing fact number {i} " + "f" * 80, kind="fact")
    _mat(conn)

    full = lane1.render(conn, BIG)
    assert db.approx_tokens(full) > 90  # sanity: truncation will have to bite

    tight = lane1.render(conn, 90)
    assert db.approx_tokens(tight) <= 90
    assert "fragile deploy script" in tight       # warnings survive
    assert "long standing fact number 5" not in tight  # facts sacrificed first

    tighter = lane1.render(conn, 70)
    assert db.approx_tokens(tighter) <= 70
    assert "fragile deploy script" in tighter     # warnings sacred
    assert "caching layer" not in tighter          # open loops dropped before warnings
    assert "long standing fact" not in tighter


def test_empty_snapshot_renders_empty_string(conn):
    assert lane1.render(conn, BIG) == ""
    # materialize on an empty brain still writes the two stats lines,
    # so render is non-empty afterwards
    assert _mat(conn) == 2
    out = lane1.render(conn, BIG)
    assert "0 memories · 0 episodes" in out


def test_quarantined_and_superseded_rows_never_appear(conn):
    seed_memory(conn, "quarantined poison warning", kind="warning",
                status="quarantined")
    sid = seed_memory(conn, "superseded stale fact", kind="fact")
    conn.execute("UPDATE memories SET valid_to = ? WHERE id = ?",
                 (db.iso_now(), sid))
    conn.commit()
    seed_memory(conn, "the one living fact", kind="fact")

    _mat(conn)
    out = lane1.render(conn, BIG)
    assert "poison" not in out
    assert "stale fact" not in out
    assert "one living fact" in out
    assert "1 memories" in out  # counts respect the same current-truth filter


def test_open_loop_closes_when_outcome_recorded(conn):
    mid = seed_memory(conn, "decided to shard the queue table", kind="decision")
    _mat(conn)
    assert "shard the queue" in lane1.render(conn, BIG)

    conn.execute("UPDATE memories SET outcome = 'worked' WHERE id = ?", (mid,))
    conn.commit()
    # snapshot unchanged until re-materialize (byte-stability)
    assert "shard the queue" in lane1.render(conn, BIG)
    _mat(conn)
    assert "shard the queue" not in lane1.render(conn, BIG)
