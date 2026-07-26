"""`export --full` / `import` round trip (alignment-audit.md §G3).

`export` wrote only current-truth memories while integration.md §5.3 called it
"lossless, re-importable". It was neither. These tests pin the claim to the
behavior.
"""

from __future__ import annotations

import argparse
import json

import pytest
from brain import cli
from brain.store import db
from conftest import seed_episode, seed_memory


def _run(argv):
    parser = argparse.ArgumentParser(prog="brain")
    cli.register_cli(parser)
    return cli.brain_command(parser.parse_args(argv))


@pytest.fixture
def home(tmp_home, monkeypatch):
    monkeypatch.setattr(cli, "_hermes_home", lambda: tmp_home)
    return tmp_home


def _populate(home):
    """A brain with history worth losing: a superseded version, a tombstone,
    episodes, a fact triple and an entity."""
    conn = db.connect(home)
    try:
        live = seed_memory(conn, "the VPS is a Hetzner CX32")
        old = seed_memory(conn, "the VPS is a Hetzner CX22")
        # Supersession is versions-are-rows: valid_to closes the old version;
        # `status` stays 'active' (the schema CHECK has no 'superseded' value).
        # Current truth is valid_to IS NULL AND status='active' AND live=1.
        conn.execute("UPDATE memories SET valid_to=?, superseded_by=? WHERE id=?",
                     (db.iso_now(), live, old))
        dead = seed_memory(conn, "a tombstoned thought")
        conn.execute("UPDATE memories SET status='tombstone', valid_to=? WHERE id=?",
                     (db.iso_now(), dead))
        seed_episode(conn, "how do I deploy", "use buildkite", session_id="s1")
        seed_episode(conn, "and rollback?", "revert the pipeline", session_id="s1",
                     turn_no=2)
        from brain.store import entities

        entities.link(conn, "Hetzner", live)
        conn.commit()
        return {
            "memories": conn.execute("SELECT count(*) FROM memories").fetchone()[0],
            "episodes": conn.execute("SELECT count(*) FROM episodes").fetchone()[0],
        }
    finally:
        conn.close()


def test_default_export_is_current_truth_only_and_says_so(home, capsys):
    _populate(home)
    out_dir = home / "exp"
    assert _run(["export", "--out", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "current-truth memories only" in out
    assert "--full" in out, "the default must point at the complete mode"

    recs = [json.loads(ln) for ln in (out_dir / "memories.jsonl").read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    assert recs, "sanity: something was exported"
    assert all(r["status"] == "active" and r["valid_to"] is None for r in recs), \
        "the default export is current truth only"
    assert not (out_dir / "manifest.json").exists()


def test_full_export_writes_a_manifest_and_every_table(home, capsys):
    _populate(home)
    out_dir = home / "full"
    assert _run(["export", "--out", str(out_dir), "--full"]) == 0
    assert "full export" in capsys.readouterr().out

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "hermes-brain-full-export"
    assert manifest["tables"]["memories"] >= 3
    assert manifest["tables"]["episodes"] == 2
    for table in ("memories", "episodes", "entities"):
        assert (out_dir / f"{table}.jsonl").exists()


def test_full_export_keeps_superseded_and_tombstoned_rows(home):
    """versions-are-rows is the storage model — dropping them discards exactly
    the history `hermes brain why` reads back."""
    _populate(home)
    out_dir = home / "full"
    _run(["export", "--out", str(out_dir), "--full"])

    recs = [json.loads(ln) for ln in
            (out_dir / "memories.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    assert any(r["valid_to"] is not None and r["superseded_by"] for r in recs), \
        "the closed previous version must survive"
    assert any(r["status"] == "tombstone" for r in recs), \
        "tombstones must survive — forget is reversible until the dream purges"


def test_full_roundtrip_into_an_empty_brain(home, tmp_path, monkeypatch, capsys):
    before = _populate(home)
    out_dir = home / "full"
    _run(["export", "--out", str(out_dir), "--full"])
    capsys.readouterr()

    target = tmp_path / "restored"
    target.mkdir()
    monkeypatch.setattr(cli, "_hermes_home", lambda: target)
    assert _run(["import", str(out_dir / "manifest.json")]) == 0
    assert "restored full export" in capsys.readouterr().out

    conn = db.connect(target)
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == \
            before["memories"]
        assert conn.execute("SELECT count(*) FROM episodes").fetchone()[0] == \
            before["episodes"]
        # uids are preserved, so provenance survives the trip.
        src = db.connect(home)
        try:
            src_uids = {r["uid"] for r in src.execute("SELECT uid FROM memories")}
        finally:
            src.close()
        dst_uids = {r["uid"] for r in conn.execute("SELECT uid FROM memories")}
        assert src_uids == dst_uids
    finally:
        conn.close()


def test_import_finds_the_manifest_beside_a_named_file(home, tmp_path, monkeypatch):
    _populate(home)
    out_dir = home / "full"
    _run(["export", "--out", str(out_dir), "--full"])

    target = tmp_path / "restored2"
    target.mkdir()
    monkeypatch.setattr(cli, "_hermes_home", lambda: target)
    # Passing memories.jsonl from a full export should still restore everything.
    assert _run(["import", str(out_dir / "memories.jsonl")]) == 0
    conn = db.connect(target)
    try:
        assert conn.execute("SELECT count(*) FROM episodes").fetchone()[0] == 2
    finally:
        conn.close()


def test_full_import_is_idempotent(home, tmp_path, monkeypatch, capsys):
    _populate(home)
    out_dir = home / "full"
    _run(["export", "--out", str(out_dir), "--full"])

    target = tmp_path / "restored3"
    target.mkdir()
    monkeypatch.setattr(cli, "_hermes_home", lambda: target)
    _run(["import", str(out_dir / "manifest.json")])
    capsys.readouterr()
    assert _run(["import", str(out_dir / "manifest.json")]) == 0

    conn = db.connect(target)
    try:
        # Second pass added nothing — uid collisions are skipped, not duplicated.
        assert conn.execute("SELECT count(*) FROM episodes").fetchone()[0] == 2
    finally:
        conn.close()


def test_a_bad_manifest_is_refused_with_a_remedy(home, tmp_path, capsys):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.json").write_text('{"format": "something-else"}', encoding="utf-8")
    assert _run(["import", str(bad / "manifest.json")]) == 1
    assert "not a full-export manifest" in capsys.readouterr().err


def test_memories_only_import_still_works(home, tmp_path, monkeypatch, capsys):
    """The plain path must not regress: a bare memories.jsonl with no manifest
    beside it goes through the original content-hash-dedup importer."""
    _populate(home)
    plain = tmp_path / "plain"
    plain.mkdir()
    _run(["export", "--out", str(plain)])
    capsys.readouterr()

    target = tmp_path / "restored4"
    target.mkdir()
    monkeypatch.setattr(cli, "_hermes_home", lambda: target)
    assert _run(["import", str(plain / "memories.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "restored full export" not in out
    conn = db.connect(target)
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] >= 1
    finally:
        conn.close()
