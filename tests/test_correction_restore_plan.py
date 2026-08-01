"""Version-chain contract for explicit correction, restoration, and fact updates."""

from __future__ import annotations

import json

from brain import tools
from brain.config import DEFAULTS
from brain.store import db, facts
from brain.store.lifecycle import expire_due
from conftest import seed_memory


def _ctx(tmp_home, trust="owner"):
    return tools.ToolContext(
        session_id="correction-test",
        principal_id="owner" if trust == "owner" else "peer-1",
        trust_tier=trust,
        platform="cli",
        config=dict(DEFAULTS),
        hermes_home=str(tmp_home),
    )


def _call(conn, tmp_home, args, *, trust="owner"):
    return json.loads(tools.dispatch(
        conn, "brain_manage", args, ctx=_ctx(tmp_home, trust),
    ))


def test_manage_schema_exposes_correct_restore_and_content():
    schema = next(
        s["function"] for s in tools.get_schemas()
        if s["function"]["name"] == "brain_manage"
    )
    props = schema["parameters"]["properties"]
    assert {"correct", "restore"} <= set(props["action"]["enum"])
    assert props["content"]["type"] == "string"


def test_correct_creates_successor_and_retires_every_active_index(
    conn, tmp_home, monkeypatch,
):
    old = seed_memory(
        conn, "The deploy target is alpha", kind="warning", pinned=1,
        trust_tier="owner", tags=["deploy"],
    )
    peer = seed_memory(conn, "Deploy owner is Sam", kind="profile")
    old_row = conn.execute("SELECT * FROM memories WHERE id=?", (old,)).fetchone()
    facts.add_fact(conn, "deploy target", "is", "alpha", memory_id=old)
    now = db.iso_now()
    conn.execute(
        "INSERT INTO edges (src_id,dst_id,edge_type,created_by,valid_from,recorded_at)"
        " VALUES (?,?,?,?,?,?)",
        (old, peer, "related_to", "test", now, now),
    )
    conn.execute(
        "INSERT INTO lane1_snapshot (section,rank,memory_id,line,rendered_at)"
        " VALUES ('warnings',0,?,'old line',?)",
        (old, now),
    )
    conn.commit()

    removed: list[int] = []
    monkeypatch.setattr("brain.store.vec.vec_available", lambda _conn: True)
    monkeypatch.setattr(
        "brain.store.vec.delete",
        lambda _conn, _table, row_id: removed.append(row_id),
    )

    out = _call(conn, tmp_home, {
        "action": "correct",
        "id": old_row["uid"],
        "content": "The deploy target is beta",
        "reason": "production moved",
    })

    assert out["action"] == "correct"
    assert out["old_id"] == old_row["uid"]
    assert out["new_id"] and out["new_id"] != out["old_id"]
    assert out["restore_call"] == (
        f'brain_manage(action="restore", id="{old_row["uid"]}", '
        'reason="undo correction")'
    )

    retired = conn.execute("SELECT * FROM memories WHERE id=?", (old,)).fetchone()
    successor = conn.execute(
        "SELECT * FROM memories WHERE uid LIKE ?", (out["new_id"] + "%",),
    ).fetchone()
    assert retired["valid_to"] is not None
    assert retired["superseded_by"] == successor["id"]
    assert successor["supersedes_id"] == old and successor["version"] == 2
    assert successor["content"] == "The deploy target is beta"
    for inherited in ("kind", "memory_type", "epistemic", "trust_tier", "pinned"):
        assert successor[inherited] == old_row[inherited]
    assert successor["tags"] == old_row["tags"]

    assert conn.execute(
        "SELECT valid_until FROM facts WHERE memory_id=?", (old,),
    ).fetchone()[0] is not None
    assert conn.execute(
        "SELECT count(*) FROM edges WHERE src_id=? AND valid_to IS NULL",
        (old,),
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT count(*) FROM edges WHERE src_id=? AND dst_id=? "
        "AND edge_type='supersedes' AND valid_to IS NULL",
        (successor["id"], old),
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT count(*) FROM lane1_snapshot WHERE memory_id=?", (old,),
    ).fetchone()[0] == 0
    assert removed == [old]

    audit = conn.execute(
        "SELECT detail FROM audit_log WHERE action='memory_corrected'",
    ).fetchone()
    detail = json.loads(audit["detail"])
    assert detail["reason"] == "production moved"
    assert detail["evidence"] == "explicit_manage"
    assert detail["restore_lineage"]["selected_uid"] == old_row["uid"]


def test_restore_appends_version_and_never_reopens_history(conn, tmp_home):
    original = seed_memory(conn, "Service endpoint is alpha", kind="fact")
    original_uid = conn.execute(
        "SELECT uid FROM memories WHERE id=?", (original,),
    ).fetchone()[0]
    corrected = _call(conn, tmp_home, {
        "action": "correct", "id": original_uid,
        "content": "Service endpoint is beta",
    })

    restored = _call(conn, tmp_home, {
        "action": "restore", "id": original_uid,
        "reason": "rollback",
    })

    assert restored["action"] == "restore"
    assert restored["restored_from"] == original_uid
    rows = conn.execute(
        "SELECT * FROM memories ORDER BY version, id",
    ).fetchall()
    assert [r["version"] for r in rows] == [1, 2, 3]
    assert rows[0]["valid_to"] is not None and rows[1]["valid_to"] is not None
    assert rows[2]["valid_to"] is None and rows[2]["status"] == "active"
    assert rows[2]["content"] == "Service endpoint is alpha"
    assert rows[2]["supersedes_id"] == rows[1]["id"]
    assert rows[1]["uid"].startswith(corrected["new_id"])
    assert json.loads(rows[2]["meta"])["restored_from_uid"] == original_uid


def test_expired_history_can_be_recalled_and_restored_current(conn, tmp_home):
    mid = seed_memory(conn, "Temporary but worth restoring", kind="fact")
    row = conn.execute("SELECT uid FROM memories WHERE id=?", (mid,)).fetchone()
    conn.execute(
        "UPDATE memories SET ttl_at='2000-01-01T00:00:00.000Z', "
        "meta=? WHERE id=?",
        (json.dumps({"retention": "temporary", "expiry_source": "test"}), mid),
    )
    conn.commit()
    expire_due(conn)

    recalled = json.loads(tools.dispatch(
        conn, "brain_recall", {"id": row["uid"][:8]}, ctx=_ctx(tmp_home),
    ))
    assert recalled["results"][0]["status"] == "expired"
    assert recalled["results"][0]["retention"] == "temporary"
    assert recalled["results"][0]["expiry_source"] == "test"

    restored = _call(conn, tmp_home, {
        "action": "restore", "id": row["uid"][:8], "reason": "still useful",
    })
    current = conn.execute(
        "SELECT * FROM memories WHERE uid LIKE ?", (restored["new_id"] + "%",),
    ).fetchone()
    assert current["status"] == "active" and current["valid_to"] is None
    assert current["ttl_at"] is None
    assert json.loads(current["meta"])["retention"] == "durable"


def test_lower_trust_cannot_correct_pinned_owner_memory(conn, tmp_home):
    mid = seed_memory(
        conn, "Pinned owner warning", kind="warning", pinned=1,
        trust_tier="owner",
    )
    uid = conn.execute("SELECT uid FROM memories WHERE id=?", (mid,)).fetchone()[0]

    rejected = _call(conn, tmp_home, {
        "action": "correct", "id": uid[:8], "content": "Peer replacement",
    }, trust="known_user")

    assert "owner evidence" in rejected["error"]
    assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1


def test_lower_trust_cannot_restore_owner_history(conn, tmp_home):
    original = seed_memory(
        conn, "Owner-selected service endpoint is alpha",
        kind="fact", trust_tier="owner",
    )
    original_uid = conn.execute(
        "SELECT uid FROM memories WHERE id=?", (original,),
    ).fetchone()[0]
    _call(conn, tmp_home, {
        "action": "correct", "id": original_uid,
        "content": "Owner-selected service endpoint is beta",
    })

    rejected = _call(conn, tmp_home, {
        "action": "restore", "id": original_uid,
        "reason": "peer requested rollback",
    }, trust="known_user")

    assert "lower trust" in rejected["error"]
    rows = conn.execute("SELECT * FROM memories ORDER BY version").fetchall()
    assert len(rows) == 2
    assert rows[-1]["content"].endswith("beta") and rows[-1]["valid_to"] is None


def test_fact_update_uses_same_version_chain_service(conn):
    old = seed_memory(conn, "Ada works at OldCorp", kind="fact")
    facts.add_fact(conn, "Ada", "works_at", "OldCorp", memory_id=old)
    conn.commit()
    new = seed_memory(conn, "Ada works at NewCorp", kind="fact")

    facts.add_fact(conn, "Ada", "works_at", "NewCorp", memory_id=new)
    conn.commit()

    old_row = conn.execute("SELECT * FROM memories WHERE id=?", (old,)).fetchone()
    new_row = conn.execute("SELECT * FROM memories WHERE id=?", (new,)).fetchone()
    assert old_row["valid_to"] is not None and old_row["superseded_by"] == new
    assert new_row["supersedes_id"] == old and new_row["version"] == 2
    assert conn.execute(
        "SELECT count(*) FROM edges WHERE src_id=? AND dst_id=? "
        "AND edge_type='supersedes' AND valid_to IS NULL",
        (new, old),
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT count(*) FROM audit_log WHERE action='memory_superseded' "
        "AND target=?",
        (new_row["uid"],),
    ).fetchone()[0] == 1
