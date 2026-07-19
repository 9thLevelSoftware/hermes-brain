"""Info-content dedup contest (config `dedup_contest`).

When the near-dup VECTOR merge path finds an existing memory at cosine >= 0.95,
the incoming text and the stored text are scored by information content
(token_count + 10 * novel-token bonus). If the NEW text wins, the older row is
superseded by a richer new version (versions-are-rows) that carries the
learning counters forward; otherwise the older row is reinforced as before.

Hermetic: the LLM is a fake installed via ``llm.set_llm_for_tests``; the stub
embedder + sqlite-vec give real cosine behavior with no downloads (mirrors
tests/test_write_rewrite.py). Skipped wholesale if sqlite-vec is not importable.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("sqlite_vec")

from brain import llm  # noqa: E402
from brain.capture import extract  # noqa: E402
from brain.capture.turns import (  # noqa: E402
    TurnContext,
    capture_session_end,
    capture_turn,
)
from brain.config import DEFAULTS  # noqa: E402
from brain.recall.embed import StubEmbedder  # noqa: E402
from brain.store import vec as vec_store  # noqa: E402

# A long shared base keeps the near-dup cosine well above 0.95 for the small
# edits below (verified: rich=0.986, subset=0.974 under StubEmbedder).
_OLD = ("The engineering team decided to migrate the primary customer account "
        "database from a hosted Postgres cluster to a single local SQLite file "
        "for the personal agent long term memory store")
# +1 novel token -> strictly higher info content -> the NEW text wins.
_NEW_RICHER = _OLD + " permanently"
# drops "long term" -> the OLD text keeps two novel tokens and wins/ties.
_NEW_POORER = ("The engineering team decided to migrate the primary customer "
               "account database from a hosted Postgres cluster to a single "
               "local SQLite file for the personal agent memory store")


@pytest.fixture(autouse=True)
def _clear_llm_override():
    yield
    llm.set_llm_for_tests(None)


class _FakeLLM:
    """One canned reply; ignores system/tier (the contest only cares about the
    single item the reply carries)."""

    def __init__(self, reply: str):
        self.reply = reply

    def __call__(self, prompt, *, system=None, max_tokens=0):
        return self.reply


def _cfg(**overrides):
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return cfg


def _ctx(sid):
    return TurnContext(session_id=sid, platform="cli",
                       principal_id="owner-p", trust_tier="owner")


def _item(content, *, instruction_shaped=False):
    return {"content": content, "kind": "fact", "about_user": False,
            "time_sensitive": False, "instruction_shaped": instruction_shaped,
            "source_uids": [], "search_aids": []}


def _ingest(conn, content, *, sid, embedder=None, config=None,
            instruction_shaped=False):
    """Drive the REAL capture -> sweep path once with a single canned item.
    Returns the sweep counts."""
    capture_turn(conn, _ctx(sid),
                 "We reorganized the storage backend this quarter.",
                 "Logged it, thanks.")
    capture_session_end(conn, sid)
    llm.set_llm_for_tests(_FakeLLM(json.dumps(
        [_item(content, instruction_shaped=instruction_shaped)])))
    return extract.sweep(conn, config or _cfg(), embedder=embedder)


def _current(conn):
    """Live (current-truth) memory rows."""
    return conn.execute(
        "SELECT * FROM memories WHERE valid_to IS NULL AND status='active'"
        " AND live=1 ORDER BY id").fetchall()


def _all(conn):
    return conn.execute("SELECT * FROM memories ORDER BY id").fetchall()


# ---------------------------------------------------------------------------
# (1) new text wins -> old superseded, counters carried forward + reinforced
# ---------------------------------------------------------------------------

def test_new_text_wins_supersedes_and_carries_counters(conn):
    embedder = StubEmbedder()
    assert vec_store.ensure_tables(conn, embedder.dim)

    c1 = _ingest(conn, _OLD, sid="s-old", embedder=embedder)
    assert c1["inserted"] == 1
    old = _all(conn)[0]
    # give the old row some accumulated learning history to carry forward
    conn.execute(
        "UPDATE memories SET verification_count=4, helpful_count=5,"
        " recall_count=3 WHERE id=?", (old["id"],))
    conn.commit()

    c2 = _ingest(conn, _NEW_RICHER, sid="s-new", embedder=embedder)
    # counted as a merge (not a fresh insert)
    assert c2["merged"] == 1 and c2["inserted"] == 0

    rows = _all(conn)
    assert len(rows) == 2  # versions-are-rows: old kept, new version added
    old_row = conn.execute("SELECT * FROM memories WHERE id=?",
                           (old["id"],)).fetchone()
    new_row = conn.execute(
        "SELECT * FROM memories WHERE valid_to IS NULL").fetchone()

    # old row closed and chained
    assert old_row["valid_to"] is not None
    assert old_row["superseded_by"] == new_row["id"]

    # new version carries the richer text + the version chain
    assert new_row["content"] == _NEW_RICHER
    assert new_row["status"] == "active" and new_row["live"] == 1
    assert new_row["version"] == old_row["version"] + 1
    assert new_row["supersedes_id"] == old_row["id"]
    assert new_row["trust_tier"] == "owner"

    # counters carried forward; verification bumped +1 for the reinforcement
    assert new_row["verification_count"] == 5   # 4 + 1
    assert new_row["helpful_count"] == 5
    assert new_row["recall_count"] == 3

    # audit trail for the supersede
    assert conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE action='extract_contest_supersede'"
    ).fetchone()[0] == 1

    # the stale vector was moved onto the new row: a re-run of the SAME rich
    # text now finds the NEW version (exact hash) and merges it, not a 3rd row.
    c3 = _ingest(conn, _NEW_RICHER, sid="s-again", embedder=embedder)
    assert c3["merged"] == 1
    assert len(_all(conn)) == 2


# ---------------------------------------------------------------------------
# (2) old text wins (or ties) -> no new version, existing row reinforced
# ---------------------------------------------------------------------------

def test_old_text_wins_reinforces_without_new_version(conn):
    embedder = StubEmbedder()
    assert vec_store.ensure_tables(conn, embedder.dim)

    _ingest(conn, _OLD, sid="s-old", embedder=embedder)
    c2 = _ingest(conn, _NEW_POORER, sid="s-new", embedder=embedder)
    assert c2["merged"] == 1 and c2["inserted"] == 0

    rows = _all(conn)
    assert len(rows) == 1                       # no version chain created
    row = rows[0]
    assert row["valid_to"] is None
    assert row["superseded_by"] is None
    assert row["content"] == _OLD               # the richer OLD text is kept
    assert row["verification_count"] == 2       # legacy reinforce: 1 + 1


# ---------------------------------------------------------------------------
# (3) an instruction-shaped near-dup NEVER contests (falls back to reinforce)
# ---------------------------------------------------------------------------

def test_instruction_shaped_near_dup_never_contests(conn):
    embedder = StubEmbedder()
    assert vec_store.ensure_tables(conn, embedder.dim)

    # owner + instruction_shaped stays ACTIVE (not quarantined) and embedded.
    c1 = _ingest(conn, _OLD, sid="s-old", embedder=embedder,
                 instruction_shaped=True)
    assert c1["inserted"] == 1
    seed = _all(conn)[0]
    assert seed["instruction_shaped"] == 1 and seed["status"] == "active"

    # A strictly richer near-dup would WIN the contest — but must be skipped
    # because the existing row is instruction-shaped.
    c2 = _ingest(conn, _NEW_RICHER, sid="s-new", embedder=embedder,
                 instruction_shaped=True)
    assert c2["merged"] == 1 and c2["inserted"] == 0

    rows = _all(conn)
    assert len(rows) == 1                        # never rewritten to a version
    assert rows[0]["content"] == _OLD
    assert rows[0]["verification_count"] == 2    # reinforced, not superseded
    assert conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE action='extract_contest_supersede'"
    ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# (4) dedup_contest=False preserves the byte-for-byte legacy merge behavior
# ---------------------------------------------------------------------------

def test_contest_disabled_uses_legacy_reinforce(conn):
    embedder = StubEmbedder()
    assert vec_store.ensure_tables(conn, embedder.dim)
    cfg = _cfg(dedup_contest=False)

    _ingest(conn, _OLD, sid="s-old", embedder=embedder, config=cfg)
    # NEW text would win the contest if it were enabled — with it off, the
    # older row is simply reinforced and the richer text discarded.
    c2 = _ingest(conn, _NEW_RICHER, sid="s-new", embedder=embedder, config=cfg)
    assert c2["merged"] == 1 and c2["inserted"] == 0

    rows = _all(conn)
    assert len(rows) == 1
    assert rows[0]["content"] == _OLD
    assert rows[0]["valid_to"] is None
    assert rows[0]["verification_count"] == 2


# ---------------------------------------------------------------------------
# (5) no embedder -> vector path (and thus the contest) never runs
# ---------------------------------------------------------------------------

def test_no_embedder_leaves_exact_hash_path_intact(conn):
    # A near-dup with NO embedder cannot reach the vector path: it is a genuine
    # novel insert (two live rows), and no version chain is formed.
    _ingest(conn, _OLD, sid="s-old", embedder=None)
    c2 = _ingest(conn, _NEW_RICHER, sid="s-new", embedder=None)
    assert c2["inserted"] == 1 and c2["merged"] == 0

    rows = _all(conn)
    assert len(rows) == 2
    assert all(r["valid_to"] is None for r in rows)
    assert all(r["supersedes_id"] is None for r in rows)

    # An EXACT-hash dup with no embedder still merges (legacy path unchanged).
    c3 = _ingest(conn, _OLD, sid="s-dup", embedder=None)
    assert c3["merged"] == 1 and c3["inserted"] == 0
    assert len(_all(conn)) == 2
