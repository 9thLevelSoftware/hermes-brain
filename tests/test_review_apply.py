"""Team follow-ups: the CLI review APPLY path for skill_revision/skill_retire
proposals (#18) + its skilltree helpers, and the observer plugin registering
the brain's aux tasks for picker visibility (#19).
"""

from __future__ import annotations

import argparse
import json

from brain import cli
from brain.skillforge import skilltree
from brain.store import db


def _run(handler, capsys, **kw):
    rc = handler(argparse.Namespace(**kw))
    return rc, capsys.readouterr().out


def _write_skill(home, name, body):
    d = skilltree.skills_root(home) / name
    d.mkdir(parents=True, exist_ok=True)
    md = d / "SKILL.md"
    md.write_text(skilltree.build_skill_md(name, "a test skill", body), encoding="utf-8")
    return md


# ---------------------------------------------------------------------------
# skilltree apply helpers (#18)
# ---------------------------------------------------------------------------

def test_apply_revision_replaces_named_section(tmp_home):
    md = _write_skill(tmp_home, "deploy-helper",
                      "## When and why\nUse for deploys.\n\n## Procedure\nOld steps.\n")
    assert skilltree.apply_revision(
        md, [{"heading": "## Procedure", "new_text": "New corrected steps."}])
    text = md.read_text(encoding="utf-8")
    assert "New corrected steps." in text
    assert "Old steps." not in text
    assert "## When and why" in text          # sibling section preserved
    assert "name: deploy-helper" in text      # frontmatter untouched


def test_apply_revision_appends_absent_section(tmp_home):
    md = _write_skill(tmp_home, "s2", "## Procedure\nsteps\n")
    assert skilltree.apply_revision(md, [{"heading": "## Gotchas", "new_text": "watch out"}])
    assert "## Gotchas" in md.read_text(encoding="utf-8")


def test_mark_stale_sets_state(tmp_home):
    _write_skill(tmp_home, "s3", "## Procedure\nx\n")
    skilltree.mark_stale(tmp_home, "s3")
    rec = skilltree.read_usage(tmp_home).get("s3", {})
    assert rec.get("state") == "stale" and rec.get("archived_at")


# ---------------------------------------------------------------------------
# CLI review apply path (#18)
# ---------------------------------------------------------------------------

def test_review_approve_applies_revision(tmp_home, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))
    md = _write_skill(tmp_home, "deploy-helper", "## Procedure\nOld.\n")
    conn = db.connect(tmp_home)
    uid = db.new_ulid()
    payload = {"name": "deploy-helper", "path": str(md),
               "revision": {"sections": [{"heading": "## Procedure", "new_text": "Revised."}]}}
    conn.execute(
        "INSERT INTO proposals (uid, kind, target, title, payload, status, created_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (uid, "skill_revision", "deploy-helper", "revise skill", json.dumps(payload),
         "pending", db.iso_now()))
    conn.commit()
    conn.close()

    rc, out = _run(cli.cmd_review, capsys, approve=uid[:8], reject=None)
    assert rc == 0 and "revised" in out
    assert "Revised." in md.read_text(encoding="utf-8")
    conn = db.connect(tmp_home)
    assert conn.execute("SELECT status FROM proposals WHERE uid=?",
                        (uid,)).fetchone()["status"] == "applied"
    conn.close()


def test_review_approve_applies_retire(tmp_home, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))
    _write_skill(tmp_home, "bad-skill", "## Procedure\nx\n")
    conn = db.connect(tmp_home)
    uid = db.new_ulid()
    conn.execute(
        "INSERT INTO proposals (uid, kind, target, title, payload, status, created_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (uid, "skill_retire", "bad-skill", "retire skill",
         json.dumps({"name": "bad-skill", "action": "mark_stale"}), "pending", db.iso_now()))
    conn.commit()
    conn.close()

    rc, out = _run(cli.cmd_review, capsys, approve=uid[:8], reject=None)
    assert rc == 0 and "retired" in out
    assert skilltree.read_usage(tmp_home).get("bad-skill", {}).get("state") == "stale"
    conn = db.connect(tmp_home)
    assert conn.execute("SELECT status FROM proposals WHERE uid=?",
                        (uid,)).fetchone()["status"] == "applied"
    conn.close()


# ---------------------------------------------------------------------------
# observer registers the brain aux tasks (#19)
# ---------------------------------------------------------------------------

def test_observer_registers_aux_tasks():
    from brain import observer

    class _Ctx:
        def __init__(self):
            self.hooks = []
            self.aux = []

        def register_hook(self, name, cb):
            self.hooks.append(name)

        def register_auxiliary_task(self, key, *rest):
            self.aux.append(key)

    ctx = _Ctx()
    observer.register(ctx)
    assert "post_tool_call" in ctx.hooks
    assert "brain_extract" in ctx.aux and "brain_consolidate" in ctx.aux


def test_observer_register_survives_ctx_without_aux():
    from brain import observer

    class _Ctx:  # older host: no register_auxiliary_task
        def register_hook(self, name, cb):
            pass

    observer.register(_Ctx())  # must not raise
