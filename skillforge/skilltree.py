"""The Hermes skills-tree filesystem contract (verified against hermes-agent).

The brain DRAFTS skills and, on validation, promotes them into Hermes's own
skills tree — where Hermes's loader, not the brain, is the runtime authority
(locked decision #4). This module is the thin, careful adapter to that tree.
It imports nothing from hermes-agent (standalone + tests must work); every
constant below is pinned to a verified source line.

Curator safety — the decisive facts (tools/skill_usage.py, agent/curator.py):
  * The skills root is ``get_hermes_home()/"skills"`` — %LOCALAPPDATA%\\hermes
    on Windows, ~/.hermes on POSIX (hermes_constants.py:46-73).
  * Usage telemetry is ONE shared file ``<skills>/.usage.json`` keyed by the
    skill's frontmatter ``name`` (tools/skill_usage.py:85-86), guarded by a
    ``.usage.json.lock`` sidecar, written atomically, unknown keys preserved
    (:497-499, :540-561).
  * The curator's AUTOMATIC archival walk skips any record that is not
    "curator-managed" — i.e. whose ``created_by != 'agent'`` and
    ``agent_created`` is not true (tools/skill_usage.py:378-381, :473-477).

Design decision (deviates from learning-system.md §2's mark_agent_created,
and says why): brain-forged skills are written with ``created_by:
"hermes-brain"`` — HONEST provenance, NOT impersonating Hermes's own
background-review fork (the only legitimate caller of mark_agent_created,
tools/skill_manager_tool.py:1404-1406). Consequence, and the reason this is
safer: the curator's auto-walk skips them, so a skill forged for a
once-a-month task can NEVER be archived out from under it as "stale"
(critique item 26, solved structurally). The skill still LOADS normally — the
loader reads any ``<skills>/<name>/SKILL.md`` regardless of telemetry — so it
feeds Hermes's loop exactly as intended; only its RETIREMENT stays the
brain's job, via its own degradation logic. Belt-and-braces: we also write
``pinned: true`` so a future host that changes the gate still protects it.

The one collision trap (tools/skill_usage.py:371-377): matching a name in
``<skills>/.bundled_manifest`` bypasses the record gate and makes a skill
archivable. So ``forge`` refuses any name already present there or on disk.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# tools/skill_manager_tool.py:475 — VALID_NAME_RE.
VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
# tools/skill_manager_tool.py:171 — MAX_DESCRIPTION_LENGTH is 1024, but the
# skill-forge holds itself to the tighter agentskills.io bar (§2).
DESCRIPTION_HARD_MAX = 1024
DESCRIPTION_QUALITY_MAX = 60
USAGE_FILE = ".usage.json"
USAGE_LOCK = ".usage.json.lock"
BUNDLED_MANIFEST = ".bundled_manifest"


def skills_root(hermes_home: str | Path) -> Path:
    """<hermes_home>/skills. The caller resolves hermes_home the same way the
    CLI does (hermes_constants when inside Hermes, else env/~/.hermes), so on
    Windows this is already the LOCALAPPDATA path."""
    return Path(hermes_home) / "skills"


def drafts_root(hermes_home: str | Path) -> Path:
    """Drafts live OUTSIDE the skills tree (critique item 26) so the loader
    never sees an unvalidated skill and the curator never touches it."""
    return Path(hermes_home) / "brain" / "drafts"


# ---------------------------------------------------------------------------
# name availability (the collision trap)
# ---------------------------------------------------------------------------

def name_is_valid(name: str) -> bool:
    return bool(name) and len(name) <= 64 and VALID_NAME_RE.match(name) is not None


def bundled_names(hermes_home: str | Path) -> set[str]:
    """Names in <skills>/.bundled_manifest ('name:hash' per line). Matching one
    is the single way a brain skill becomes curator-archivable — avoid it."""
    path = skills_root(hermes_home) / BUNDLED_MANIFEST
    names: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and ":" in line:
                names.add(line.split(":", 1)[0].strip())
    except (OSError, UnicodeDecodeError):
        pass
    return names


def existing_skill_names(hermes_home: str | Path) -> set[str]:
    """Every skill name already on disk (dir name at depth 1 or 2 with a
    SKILL.md), plus bundled names. `forge` must not collide with any."""
    root = skills_root(hermes_home)
    names = set(bundled_names(hermes_home))
    if not root.exists():
        return names
    try:
        for skill_md in root.rglob("SKILL.md"):
            if ".archive" in skill_md.parts or ".hub" in skill_md.parts:
                continue
            names.add(skill_md.parent.name)
    except OSError:
        pass
    return names


def name_available(hermes_home: str | Path, name: str) -> bool:
    return name_is_valid(name) and name not in existing_skill_names(hermes_home)


# ---------------------------------------------------------------------------
# SKILL.md
# ---------------------------------------------------------------------------

def build_skill_md(name: str, description: str, body: str,
                   frontmatter_extra: dict | None = None) -> str:
    """A minimal, valid SKILL.md (agent/skill_utils.py:123-157 parses it).

    Required keys only — name, description — plus brain provenance. We keep
    the frontmatter a flat scalar map so BOTH the strict validator and the
    lenient runtime loader accept it.
    """
    lines = ["---", f"name: {name}", f"description: {_yaml_scalar(description)}"]
    extra = {"created_by": "hermes-brain", **(frontmatter_extra or {})}
    for key, val in extra.items():
        lines.append(f"{key}: {_yaml_scalar(val)}")
    lines.append("---")
    lines.append("")
    lines.append(body.strip())
    lines.append("")
    return "\n".join(lines)


def _yaml_scalar(val) -> str:
    """Render a scalar safely for the flat frontmatter (quote if risky)."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val)
    if s == "" or re.search(r'[:#\[\]{}",&*!|>%@`]', s) or s.strip() != s:
        return json.dumps(s, ensure_ascii=False)
    return s


def validate_frontmatter(name: str, description: str, body: str) -> str | None:
    """The host's create-time checks, mirrored (skill_manager_tool.py:524-575).
    Returns an error string, or None if the SKILL.md would be accepted."""
    if not name_is_valid(name):
        return f"invalid skill name {name!r} (must match {VALID_NAME_RE.pattern})"
    if not description.strip():
        return "description is required"
    if len(description) > DESCRIPTION_HARD_MAX:
        return f"description exceeds {DESCRIPTION_HARD_MAX} chars"
    if not body.strip():
        return "skill body is empty"
    return None


# ---------------------------------------------------------------------------
# .usage.json (the shared, name-keyed telemetry file)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    # state.db/usage use plain ISO; match without importing store.db here.
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _empty_record() -> dict:
    """tools/skill_usage.py:484-506, verbatim shape (forward-compat: any keys
    a newer host adds are preserved on read-modify-write)."""
    now = _now_iso()
    return {
        "created_by": None, "use_count": 0, "view_count": 0,
        "last_used_at": None, "last_viewed_at": None, "patch_count": 0,
        "last_patched_at": None, "created_at": now, "state": "active",
        "pinned": False, "outcome_counts": {}, "helped": 0, "hurt": 0,
        "neutral": 0, "outcome_cost_usd": 0.0, "last_outcome_at": None,
        "archived_at": None,
    }


def read_usage(hermes_home: str | Path) -> dict:
    path = skills_root(hermes_home) / USAGE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def write_usage_record(hermes_home: str | Path, name: str, record: dict) -> None:
    """Merge one skill's record into <skills>/.usage.json, atomically, under
    the same lock sidecar the host uses. Unknown keys of an existing record
    are preserved (forward-compat contract)."""
    root = skills_root(hermes_home)
    root.mkdir(parents=True, exist_ok=True)
    path = root / USAGE_FILE
    with _usage_lock(root):
        current = read_usage(hermes_home)
        merged = {**current.get(name, {}), **record}
        current[name] = merged
        # Per-process tmp name: if the lock is bypassed on its degradation
        # path (timeout / stale-takeover), two writers must not share one tmp
        # path and race os.replace into a FileNotFoundError. os.replace stays
        # atomic; the residual (whole-file last-writer-wins) matches the
        # host's own best-effort telemetry contract.
        tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(current, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, path)


class _usage_lock:
    """Cross-process advisory lock via an O_EXCL sidecar (mirrors the host's
    _usage_file_lock). Best-effort with a short spin + stale takeover; the
    write itself is atomic (os.replace), so the worst case on the degradation
    path is whole-file last-writer-wins — a concurrent writer's record can be
    lost, but the file is never corrupted. This matches Hermes's own
    best-effort telemetry contract; the file is advisory, not a source of
    truth (the SKILL.md on disk is)."""

    def __init__(self, root: Path, timeout: float = 3.0) -> None:
        self._lock_path = root / USAGE_LOCK
        self._timeout = timeout
        self._held = False

    def __enter__(self):
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fd = os.open(str(self._lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self._held = True
                return self
            except FileExistsError:
                # Stale lock (older than the timeout) -> take it over.
                try:
                    if time.time() - self._lock_path.stat().st_mtime > self._timeout:
                        self._lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    logger.warning("usage lock contended; proceeding without it")
                    return self
                time.sleep(0.05)

    def __exit__(self, *exc):
        if self._held:
            try:
                self._lock_path.unlink(missing_ok=True)
            except OSError:
                pass
        return False


def brain_skill_record(*, evidence_count: int, success_rate: float,
                       shift_id: str, draft_uid: str) -> dict:
    """A .usage.json record for a freshly promoted brain skill.

    created_by='hermes-brain' keeps it out of the curator's auto-walk
    (curator-safe, see the module docstring); pinned=True is belt-and-braces;
    the timestamps give it a fresh grace anchor either way. The extra brain_*
    keys are ignored by the host and read back by `hermes brain skills`.
    """
    now = _now_iso()
    rec = _empty_record()
    rec.update({
        "created_by": "hermes-brain", "created_at": now,
        "last_patched_at": now, "patch_count": 1, "pinned": True,
        "brain_evidence_count": evidence_count,
        "brain_success_rate_at_creation": round(success_rate, 3),
        "brain_shift_id": shift_id, "brain_draft_uid": draft_uid,
    })
    return rec
