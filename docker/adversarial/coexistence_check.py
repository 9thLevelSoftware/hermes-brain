#!/usr/bin/env python3
"""Coexistence proof — runs INSIDE the live hermes image (real hermes-agent
checkout + the real brain installed at $HERMES_HOME/plugins/brain).

Proves two things the brain install promises about the host:

  A. PROVIDER isolation — installing brain is additive and a legitimate
     alternative. With brain installed and active, the bundled Honcho provider
     is still discoverable, brain is selectable, a bogus provider load is
     isolated (returns None, never raises), and Honcho resolves to the BUNDLED
     package, not the user tree (bundled-wins).

  B. CURATOR coexistence — a brain-forged skill is NOT garbage-collected by
     Hermes's skill curator. A real, no-LLM curator pass
     (agent.curator.apply_automatic_transitions) is run over a temp skills tree
     holding (1) a brain-forged skill written with the brain's own artifacts
     (created_by: hermes-brain + a matching .usage.json record) and (2) an
     aged agent-authored control skill (created_by: agent, >90d idle). The
     brain skill must survive untouched; the control proves the curator is
     actually curating (it's a candidate, and gets archived).

Why here and not pytest: `import plugins.memory` / `import agent` on a dev box
resolves to whatever hermes-agent is pip-installed, which can lag the shipped
checkout. This image installs the exact checkout that ships, so it is the
authoritative environment. Exit 0 == both invariants held.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

HOME = os.environ.get("HERMES_HOME", "/hermes-home")
fails: list[str] = []


def check(cond, label):
    print(("  OK  " if cond else " FAIL ") + label)
    if not cond:
        fails.append(label)


def _register_brain():
    root = Path(HOME) / "plugins" / "brain"
    if "brain" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "brain", str(root / "__init__.py"), submodule_search_locations=[str(root)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["brain"] = mod
    spec.loader.exec_module(mod)


# ---------------------------------------------------------------------------
# A. provider isolation / coexistence
# ---------------------------------------------------------------------------

def part_a_providers():
    print("=== A. memory-provider coexistence (brain is additive, honcho intact) ===")
    from plugins.memory import (
        discover_memory_providers,
        find_provider_dir,
        load_memory_provider,
    )

    names = {n for n, _desc, _avail in discover_memory_providers()}
    check("honcho" in names, "bundled honcho still discoverable with brain installed")
    check("brain" in names, "brain discoverable as a user provider")

    brain = load_memory_provider("brain")
    check(brain is not None and getattr(brain, "name", None) == "brain",
          "brain loads and is selectable (memory.provider=brain)")

    # isolation: a bogus provider never raises, just returns None
    isolated = True
    try:
        check(load_memory_provider("does_not_exist_xyz") is None,
              "unknown provider -> None (isolated, no crash)")
    except Exception as e:  # noqa: BLE001
        isolated = False
        check(False, f"unknown provider raised instead of None: {e}")

    # honcho load must not raise (may be None without runtime ctx / SDK — the
    # point is discovery + selection are unaffected and failures are contained)
    try:
        load_memory_provider("honcho")
        check(True, "honcho load path does not raise (SDK-optional, isolated)")
    except Exception as e:  # noqa: BLE001
        check(False, f"honcho load raised: {e}")

    # bundled-wins: honcho resolves to the installed package, not $HERMES_HOME
    hd = find_provider_dir("honcho")
    check(hd is not None and "plugins" in str(hd) and str(Path(HOME)) not in str(hd),
          "honcho resolves to the BUNDLED package, not the user tree")
    _ = isolated


# ---------------------------------------------------------------------------
# B. curator coexistence
# ---------------------------------------------------------------------------

def part_b_curator():
    print("=== B. skill-curator coexistence (brain-forged skill survives) ===")
    _register_brain()
    from brain.skillforge import skilltree

    skills = Path(HOME) / "skills"
    skills.mkdir(parents=True, exist_ok=True)

    # 1. a real brain-forged skill (the brain's own artifacts)
    bname = "adv-brain-forged-skill"
    bdir = skills / bname
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "SKILL.md").write_text(
        skilltree.build_skill_md(
            bname, "a brain-forged skill used to prove curator coexistence",
            "## Steps\n\nDo the load-bearing thing, then verify."),
        encoding="utf-8")
    skilltree.write_usage_record(
        HOME, bname,
        skilltree.brain_skill_record(evidence_count=3, success_rate=0.8,
                                     shift_id="coexist", draft_uid="coexist"))

    # 2. an aged, agent-authored control skill the curator SHOULD act on
    cname = "adv-agent-authored-skill"
    cdir = skills / cname
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "SKILL.md").write_text(
        f"---\nname: {cname}\ndescription: an aged agent skill (curation control)\n"
        "---\n\nold body\n", encoding="utf-8")
    aged = (datetime.now(UTC) - timedelta(days=95)).strftime(
        "%Y-%m-%dT%H:%M:%S") + "Z"
    usage_path = skills / ".usage.json"
    usage = {}
    if usage_path.exists():
        try:
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            usage = {}
    usage[cname] = {"created_by": "agent", "use_count": 0, "created_at": aged,
                    "last_used_at": None, "state": "active", "pinned": False}
    usage_path.write_text(json.dumps(usage, indent=2), encoding="utf-8")

    # candidacy: the provenance gate excludes brain, includes the agent skill
    os.environ["HERMES_HOME"] = HOME  # curator keys off this
    from tools import skill_usage

    candidates = set(skill_usage.list_agent_created_skill_names())
    check(bname not in candidates,
          "brain skill EXCLUDED from the curator candidate set (created_by != agent)")
    check(cname in candidates,
          "agent-authored control IS a candidate (curator actually curates)")

    # run the real, no-LLM curator pass
    from agent.curator import apply_automatic_transitions

    counts = apply_automatic_transitions()
    print(f"  curator transitions: {counts}")

    # brain skill untouched
    check((bdir / "SKILL.md").exists(),
          "brain SKILL.md still present after a real curator pass")
    check(not (skills / ".archive" / bname).exists(),
          "brain skill NOT archived by the curator")
    # control acted upon (archived after 90d idle) — proves the pass did work
    archived = (skills / ".archive" / cname).exists() or not (cdir / "SKILL.md").exists()
    check(archived, "aged agent-authored control was curated (archived)")


def main() -> int:
    for part in (part_a_providers, part_b_curator):
        try:
            part()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            fails.append(f"{part.__name__} raised: {e}")
    print("=== coexistence result ===")
    if fails:
        print("COEXISTENCE FAILED:")
        for f in fails:
            print(" -", f)
        return 1
    print("COEXISTENCE OK: brain install is additive (honcho intact + selectable) "
          "and brain-forged skills survive the curator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
