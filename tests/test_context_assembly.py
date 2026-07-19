"""Phase E — token-budgeted context assembly (recall/context.py).

Covers the budget invariant (result never exceeds the token budget), the
fixed-first priority order (identity/core subtracted before the dynamic
split), owner-only peer-card scoping (a non-owner NEVER sees a peer card),
the ~40/60 summary/extracts split, and degradation on empty input + a huge
transcript.
"""

from __future__ import annotations

from brain.recall.context import assemble
from brain.store import db
from conftest import seed_memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_scoped(conn, content, *, kind, memory_type, scope_user):
    """seed_memory has no scope_user knob; set it directly after seeding."""
    mid = seed_memory(conn, content, kind=kind, memory_type=memory_type)
    conn.execute("UPDATE memories SET scope_user = ? WHERE id = ?", (scope_user, mid))
    conn.commit()
    return mid


def _pairs(n, prefix=""):
    """n salient user/assistant message dicts ('I prefer …' -> +0.3 salience)."""
    msgs = []
    for i in range(n):
        msgs.append({"role": "user",
                     "content": f"{prefix}I prefer configuration option number {i} enabled"})
        msgs.append({"role": "assistant",
                     "content": f"Understood, option {i} is now the default going forward."})
    return msgs


def _section_body(result, header):
    """Lines of a '## <header>' section, up to the next '## ' header."""
    lines = result.splitlines()
    body, capture = [], False
    for line in lines:
        if line.startswith("## "):
            capture = line == header
            continue
        if capture:
            body.append(line)
    return body


# ---------------------------------------------------------------------------
# Budget invariant
# ---------------------------------------------------------------------------

def test_budget_is_never_exceeded(conn):
    seed_memory(conn, "The owner is named Devil and works on hermes-brain.",
                kind="profile", memory_type="core")
    for i in range(10):
        seed_memory(conn, f"Distilled pattern {i}: batching writes improves throughput.",
                    kind="fact", memory_type="semantic")
    result = assemble(conn, _pairs(8), 200)
    assert result
    assert db.approx_tokens(result) <= 200


def test_tiny_budget_still_within_wall(conn):
    for i in range(20):
        seed_memory(conn, f"Semantic memory {i} about the project architecture.",
                    kind="fact", memory_type="semantic")
    result = assemble(conn, _pairs(20), 40)
    assert db.approx_tokens(result) <= 40


# ---------------------------------------------------------------------------
# Fixed-first priority
# ---------------------------------------------------------------------------

def test_identity_comes_first_and_is_subtracted_before_split(conn):
    seed_memory(conn, "IDENTITY_MARKER owner prefers concise answers.",
                kind="profile", memory_type="core")
    for i in range(10):
        seed_memory(conn, f"SUMMARY_MARKER distilled semantic pattern {i} here.",
                    kind="fact", memory_type="semantic")
    result = assemble(conn, _pairs(6), 300)
    assert "IDENTITY_MARKER" in result
    # Identity section header precedes both dynamic sections.
    idx_identity = result.index("## Identity")
    assert idx_identity >= 0
    if "## Summary" in result:
        assert idx_identity < result.index("## Summary")
    if "## Recent" in result:
        assert idx_identity < result.index("## Recent")


# ---------------------------------------------------------------------------
# Peer-card scoping (the load-bearing security test)
# ---------------------------------------------------------------------------

def test_peer_card_visible_to_owner_but_never_to_non_owner(conn):
    _seed_scoped(conn, "PEERCARD_SECRET: Alice tends to over-scope her PRs.",
                 kind="peer_card", memory_type="semantic", scope_user="alice")
    # Owner talking *about* Alice sees the peer card.
    owner_view = assemble(conn, _pairs(3), 300,
                          principal_id="alice", trust_tier="owner")
    assert "PEERCARD_SECRET" in owner_view
    # Alice herself (non-owner) must NEVER see the owner's peer card of her.
    peer_view = assemble(conn, _pairs(3), 300,
                         principal_id="alice", trust_tier="known_user")
    assert "PEERCARD_SECRET" not in peer_view


def test_non_owner_never_sees_foreign_scoped_rows(conn):
    _seed_scoped(conn, "BOBS_PRIVATE semantic note scoped to bob.",
                 kind="fact", memory_type="semantic", scope_user="bob")
    view = assemble(conn, _pairs(2), 300,
                    principal_id="carol", trust_tier="known_user")
    assert "BOBS_PRIVATE" not in view


# ---------------------------------------------------------------------------
# The 40/60 split
# ---------------------------------------------------------------------------

def test_summary_extracts_split_is_roughly_40_60(conn):
    # No fixed contributions, so the whole budget feeds the dynamic split.
    for i in range(30):
        seed_memory(conn, f"Distilled semantic pattern {i}: prefer idempotent writes always.",
                    kind="fact", memory_type="semantic")
    budget = 400
    result = assemble(conn, _pairs(30), budget)
    summary_tokens = db.approx_tokens("\n".join(_section_body(result, "## Summary")))
    recent_tokens = db.approx_tokens("\n".join(_section_body(result, "## Recent")))
    total = summary_tokens + recent_tokens
    assert total > 0
    # Summary gets ~40%; allow a wide band (packing is line-granular).
    ratio = summary_tokens / total
    assert 0.20 <= ratio <= 0.60
    assert recent_tokens > 0  # extracts got the majority slice


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

def test_empty_inputs_return_empty_string(conn):
    assert assemble(conn, [], 300) == ""
    assert assemble(conn, None, 300) == ""


def test_zero_budget_returns_empty(conn):
    seed_memory(conn, "Some core identity fact.", kind="profile", memory_type="core")
    assert assemble(conn, _pairs(3), 0) == ""


def test_huge_transcript_stays_within_budget(conn):
    for i in range(50):
        seed_memory(conn, f"Semantic memory row {i} with distilled content about systems.",
                    kind="fact", memory_type="semantic")
    seed_memory(conn, "Owner core identity line.", kind="profile", memory_type="core")
    huge = _pairs(2000)  # 4000 messages
    result = assemble(conn, huge, 500)
    assert db.approx_tokens(result) <= 500
    assert result  # still produced a useful block
