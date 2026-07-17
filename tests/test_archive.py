"""A3 episodic archive: the cold-storage that makes `forget`'s purge
non-destructive. Unit round-trips plus the correctness gate — an active
purge archives the raw text BEFORE nulling content, and is skipped entirely
when there is nowhere to archive (raw text is never lost).
"""

from __future__ import annotations

import json

from brain.dream.forget import run as forget_run
from brain.store import archive
from conftest import iso_days_ago, seed_memory

from tests.test_dream_forget import make_shift, mem_row, stamp_audit


def _archive_ref(source_refs):
    refs = json.loads(source_refs or "[]")
    return next((r[len("archive:"):] for r in refs if r.startswith("archive:")), None)


# ---------------------------------------------------------------------------
# archive unit round-trips
# ---------------------------------------------------------------------------

def test_append_and_read_roundtrip(tmp_path):
    ref = archive.append(tmp_path, {"uid": "ULID0001", "kind": "memory",
                                    "content": "the full raw text"})
    assert ref and ref.endswith(":ULID0001")
    rec = archive.read(tmp_path, ref)
    assert rec is not None and rec["content"] == "the full raw text"
    assert rec["archived_at"]  # stamped on append
    assert archive.recover_content(tmp_path, ref) == "the full raw text"


def test_read_supersede_last_wins(tmp_path):
    archive.append(tmp_path, {"uid": "ULID0002", "content": "first"})
    ref = archive.append(tmp_path, {"uid": "ULID0002", "content": "second"})
    assert archive.recover_content(tmp_path, ref) == "second"


def test_append_rejects_bad_uid(tmp_path):
    assert archive.append(tmp_path, {"content": "no uid"}) is None
    assert archive.append(tmp_path, {"uid": "bad/uid", "content": "x"}) is None


def test_read_malformed_ref(tmp_path):
    assert archive.read(tmp_path, "") is None
    assert archive.read(tmp_path, "no-colon") is None
    assert archive.read(tmp_path, "../escape.jsonl.gz:ULID") is None
    assert archive.read(tmp_path, "2020-01.jsonl.gz:MISSING") is None  # file absent


def test_purge_uid_scrubs(tmp_path):
    ref_a = archive.append(tmp_path, {"uid": "ULIDAAAA", "content": "keep me"})
    ref_b = archive.append(tmp_path, {"uid": "ULIDBBBB", "content": "scrub me"})
    removed = archive.purge_uid(tmp_path, "ULIDBBBB")
    assert removed == 1
    assert archive.read(tmp_path, ref_b) is None       # gone
    assert archive.recover_content(tmp_path, ref_a) == "keep me"  # neighbor intact


# ---------------------------------------------------------------------------
# forget correctness gate
# ---------------------------------------------------------------------------

def test_forget_active_archives_before_purge(conn, tmp_home):
    raw = "raw secret text that must survive the purge"
    mem = seed_memory(conn, raw, status="tombstone")
    uid = mem_row(conn, mem)["uid"]
    stamp_audit(conn, "forget_tombstone", uid, iso_days_ago(400))  # past grace

    result = forget_run(make_shift(conn, {"_forced_mode": "active",
                                          "hermes_home": str(tmp_home)}))
    assert result["purged"] == 1, result
    row = mem_row(conn, mem)
    assert row["content"] is None          # live content nulled
    ref = _archive_ref(row["source_refs"])  # ref recorded in source_refs
    assert ref
    assert archive.recover_content(str(tmp_home), ref) == raw


def test_forget_purge_skipped_without_archive_target(conn):
    # No hermes_home -> nowhere to archive -> the purge must be skipped so the
    # raw text is preserved (it stays a tombstone and retries next run).
    mem = seed_memory(conn, "content with no archive target", status="tombstone")
    uid = mem_row(conn, mem)["uid"]
    stamp_audit(conn, "forget_tombstone", uid, iso_days_ago(400))

    result = forget_run(make_shift(conn, {"_forced_mode": "active"}))
    assert result["purged"] == 0
    assert mem_row(conn, mem)["content"] is not None  # raw text preserved
