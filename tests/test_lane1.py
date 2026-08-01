"""Lane 1 materialize/render: compact priorities and byte stability."""

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
    seed_memory(conn, "user prefers terse answers", kind="preference", pinned=1)
    seed_memory(conn, "the VPS is Hetzner CX22 with 2GB RAM", kind="fact")
    seed_memory(conn, "user timezone is America/Chicago", kind="profile", pinned=1)

    written = _mat(conn)
    # 2 warnings + 1 open loop + 2 pinned identity items + one hint.
    assert written == 6

    out = lane1.render(conn, BIG)
    assert out.startswith("## Brain (persistent memory) — session index")
    assert "### ⚠ Failures & warnings (avoid repeating)" in out
    assert "### ◔ Open loops — outcomes unknown" in out
    assert "### ● Pinned profile & preferences" in out
    assert "pip install inside Termux" in out
    assert "chose LanceDB fallback" in out            # failed outcome -> warnings
    assert "migrate kanban to FTS5" in out            # open loop
    assert "hermes brain search" in out                # drill-down hint
    assert "Hetzner CX22" not in out                    # low-value standing fact

    pinned_block = out.split("### ● Pinned profile & preferences")[1]
    assert "America/Chicago" in pinned_block
    assert "terse answers" in pinned_block


def test_render_is_byte_stable_until_rematerialize(conn):
    seed_memory(conn, "never rebase the shared branch", kind="warning")
    _mat(conn)

    first = lane1.render(conn, BIG)
    assert first == lane1.render(conn, BIG)

    # Live-table writes must NOT leak into render output...
    seed_memory(conn, "user prefers duck examples", kind="preference", pinned=1)
    assert lane1.render(conn, BIG) == first
    # ...until the snapshot is rebuilt. That's the point.
    _mat(conn)
    after = lane1.render(conn, BIG)
    assert after != first
    assert "duck examples" in after


def test_budget_drops_open_loops_and_pinned_items_before_warnings(conn):
    seed_memory(conn, "warning about the fragile deploy script " + "w" * 80,
                kind="warning")
    seed_memory(conn, "open decision on the caching layer " + "o" * 80,
                kind="decision")
    for i in range(6):
        seed_memory(
            conn,
            f"long pinned preference number {i} " + "p" * 80,
            kind="preference",
            pinned=1,
        )
    _mat(conn)

    full = lane1.render(conn, BIG)
    assert db.approx_tokens(full) > 90  # sanity: truncation will have to bite

    tight = lane1.render(conn, 90)
    assert db.approx_tokens(tight) <= 90
    assert "fragile deploy script" in tight       # warnings survive
    assert "long pinned preference number 5" not in tight

    assert "caching layer" not in tight


def test_empty_snapshot_renders_empty_string(conn):
    assert lane1.render(conn, BIG) == ""
    # Materializing an empty brain still writes exactly one drill-down hint.
    assert _mat(conn) == 1
    out = lane1.render(conn, BIG)
    assert "hermes brain search" in out
    assert "memories ·" not in out


def test_quarantined_and_superseded_rows_never_appear(conn):
    seed_memory(conn, "quarantined poison warning", kind="warning",
                status="quarantined")
    sid = seed_memory(conn, "superseded stale fact", kind="fact")
    conn.execute("UPDATE memories SET valid_to = ? WHERE id = ?",
                 (db.iso_now(), sid))
    conn.commit()
    seed_memory(conn, "the one living warning", kind="warning")

    _mat(conn)
    out = lane1.render(conn, BIG)
    assert "poison" not in out
    assert "stale fact" not in out
    assert "one living warning" in out
    assert "memories ·" not in out


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
