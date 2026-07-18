#!/usr/bin/env python3
"""Tier-agnostic behavioral golden — capture -> recall -> lane-1, standalone.

No hermes-agent, no LLM. Proves that on WHATEVER tier this interpreter resolves
to (floor fts-only / lite / full onnx+vec), the brain: opens brain.db (capability
probe), captures a memory, RECALLS it (via FTS, LIKE, or FTS+vec), and renders a
byte-stable lane-1 block — degrading, never crashing. Used by the adversarial
suite as an explicit proof at each tier (Phase 1) and under a hard memory cap
(Phase 4: `docker run --memory=256m`). Exit 0 == the tier works.

Runs from the plugin repo root (the floor/full images COPY the repo to /plugin):
    python docker/adversarial/golden.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile


def _register_brain():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if "brain" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "brain", os.path.join(root, "__init__.py"), submodule_search_locations=[root])
    module = importlib.util.module_from_spec(spec)
    sys.modules["brain"] = module
    spec.loader.exec_module(module)


def _seed_memory(conn, content, *, kind="fact", memory_type="semantic",
                 scope_user=None):
    from brain.capture.symbols import symbols_field
    from brain.store import db

    now = db.iso_now()
    conn.execute(
        "INSERT INTO memories (uid, epistemic, memory_type, kind, status, live,"
        " content, content_hash, symbols, tags, token_len, trust_tier, created_by,"
        " scope_user, valid_from, recorded_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (db.new_ulid(), "observation", memory_type, kind, "active", 1, content,
         db.content_hash(content), symbols_field(content), "[]",
         db.approx_tokens(content), "owner", "golden", scope_user, now, now))
    conn.commit()


def main() -> int:
    _register_brain()
    from brain.recall import lane1
    from brain.recall.search import search
    from brain.store import db, sysinfo

    home = os.environ.get("GOLDEN_HOME") or tempfile.mkdtemp()
    conn = db.connect(home)
    caps = db.capabilities(conn)
    mode = sysinfo.resolve_mode("auto")
    print(f"[golden] home={home} tier={mode} caps={caps}")

    fails = []

    # 1. capture
    _seed_memory(conn, "the staging database is postgres 14 running on fly.io")
    _seed_memory(conn, "the deploy runbook lives in docs/deploy.md")
    n = conn.execute("SELECT count(*) FROM memories WHERE valid_to IS NULL "
                     "AND status='active' AND live=1").fetchone()[0]
    if n != 2:
        fails.append(f"expected 2 active memories, got {n}")

    # 2. recall (whatever leg the tier supports)
    hits = search(conn, "staging database", limit=5, principal_id="owner",
                  trust_tier="owner")
    if not any("staging" in (h.text or "").lower() for h in hits):
        fails.append(f"recall did not surface the staging fact: {[h.text for h in hits]}")
    else:
        print(f"[golden] recall OK via leg(s): {sorted({h.source for h in hits})}")

    # 3. lane-1 render is byte-stable (invariant #1) across repeated renders
    lane1.materialize(conn, {})
    r1 = lane1.render(conn, 1200)
    r2 = lane1.render(conn, 1200)
    if r1 != r2:
        fails.append("lane-1 render is not byte-stable across two calls")
    if "brain v" not in r1:
        fails.append("lane-1 render missing the version stamp")

    conn.close()
    if fails:
        print("GOLDEN FAILURES:")
        for f in fails:
            print(" -", f)
        return 1
    print(f"GOLDEN OK ({mode} tier): capture + recall + byte-stable lane-1")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
