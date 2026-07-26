"""Cross-profile links (docs/design/alignment-audit.md §G2).

Each Hermes profile has its own `HERMES_HOME` and therefore its own brain.db —
correct isolation, and a real cost: `hermes profile create coder` gives you a
second, empty brain, and nothing you told the first one follows.

A link registers a sibling profile whose memories this profile may READ.

**Two rules make this safe, and both are enforced elsewhere but stated here
because this module is where a reader will come looking:**

1. **Links are traversed only for an owner-trust caller** (`recall/linked.py`).
   The operator chose full owner access across links, which means a linked
   profile is searched as its owner — including `peer_card` rows. That is
   defensible when it is YOU reading YOUR other profile. It would not be
   defensible for a gateway peer or an MCP `tool`-trust session, so those never
   traverse a link at all. A link can therefore never become a
   privilege-escalation path; it widens what the owner sees, nothing else.
2. **Links are read-only.** Mutations (`forget`, `pin`, `brain_outcome`, ...)
   refuse on a uid that lives in a linked profile and name the profile to run
   it in. Writing through a link would make provenance a lie: the row would
   record a session and platform that never touched that database.

Stored as one JSON `meta` row, the same pattern as `recall/weights.py`, so this
needs no migration.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from . import db

logger = logging.getLogger(__name__)

META_KEY = "profile_links"
MAX_LINKS = 8   # fusion cost is linear in links; a sane ceiling beats a footgun


def _brain_db_path(hermes_home: str | Path) -> Path:
    return db.db_path(Path(hermes_home))


def load(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Registered links. Never raises — callers are on the capture path."""
    try:
        raw = db.get_meta(conn, META_KEY)
    except Exception:
        return []
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("profile links unparseable; treating as none")
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if isinstance(item, dict) and item.get("name") and item.get("hermes_home"):
            out.append({
                "name": str(item["name"]),
                "hermes_home": str(item["hermes_home"]),
                "enabled": bool(item.get("enabled", True)),
            })
    return out


def enabled(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [link for link in load(conn) if link["enabled"]]


def _save(conn: sqlite3.Connection, links: list[dict[str, Any]]) -> None:
    db.set_meta(conn, META_KEY, json.dumps(links, sort_keys=True))
    conn.commit()


def validate_target(conn: sqlite3.Connection, hermes_home: str | Path) -> str:
    """Resolve and check a link target. Raises ValueError with a remedy.

    Refuses a self-link outright: it would double every local hit and make the
    RRF merge score rows against themselves, which looks like a relevance win
    and is an artifact.
    """
    target = _brain_db_path(hermes_home)
    if not target.is_file():
        raise ValueError(
            f"no brain.db at {target} — is that a HERMES_HOME with the brain "
            f"installed? (run 'hermes brain status' there first)")

    # Self-link check against the connection's OWN main database file.
    own = _main_db_file(conn)
    if own is not None and _same_file(own, target):
        raise ValueError("refusing to link a profile to itself — it would double "
                         "every local hit and score rows against themselves")

    # Reject a DB written by a NEWER plugin: its rows may use columns this code
    # does not know, and silently reading a subset is worse than refusing.
    probe = open_link({"name": "(probe)", "hermes_home": str(hermes_home)})
    if probe is None:
        raise ValueError(f"cannot open {target} read-only")
    try:
        row = probe.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.Error as e:
        raise ValueError(f"{target} is not a brain.db ({e})")
    finally:
        probe.close()
    version = int(row[0]) if row and row[0] is not None else 0
    if version > db.SCHEMA_VERSION:
        raise ValueError(
            f"{target} is schema v{version}; this plugin understands up to "
            f"v{db.SCHEMA_VERSION}. Update the plugin before linking.")
    return str(Path(hermes_home))


def _main_db_file(conn: sqlite3.Connection) -> Path | None:
    """Path backing this connection's `main` database, if it has one."""
    try:
        for _seq, name, file in conn.execute("PRAGMA database_list").fetchall():
            if name == "main" and file:
                return Path(file)
    except sqlite3.Error:
        pass
    return None


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return str(a) == str(b)


def add(conn: sqlite3.Connection, name: str, hermes_home: str | Path) -> dict[str, Any]:
    name = (name or "").strip()
    if not name:
        raise ValueError("a link needs a name — e.g. 'hermes brain link coder --home ...'")
    links = load(conn)
    if any(link["name"] == name for link in links):
        raise ValueError(f"a link named {name!r} already exists — 'unlink' it first")
    if len(links) >= MAX_LINKS:
        raise ValueError(f"at most {MAX_LINKS} links (fusion cost grows with each)")
    resolved = validate_target(conn, hermes_home)
    entry = {"name": name, "hermes_home": resolved, "enabled": True}
    links.append(entry)
    _save(conn, links)
    return entry


def remove(conn: sqlite3.Connection, name: str) -> bool:
    links = load(conn)
    kept = [link for link in links if link["name"] != name]
    if len(kept) == len(links):
        return False
    _save(conn, kept)
    return True


def set_enabled(conn: sqlite3.Connection, name: str, value: bool) -> bool:
    links = load(conn)
    found = False
    for link in links:
        if link["name"] == name:
            link["enabled"] = bool(value)
            found = True
    if found:
        _save(conn, links)
    return found


def open_link(link: dict[str, Any]) -> sqlite3.Connection | None:
    """Read-only connection to a linked brain.db, or None.

    Read-only at the SQLite level, not merely by convention: a link must not be
    able to write to another profile even through a bug.
    """
    target = _brain_db_path(link["hermes_home"])
    if not target.is_file():
        logger.warning("link %s: no brain.db at %s", link["name"], target)
        return None
    uri = "file:" + target.as_posix().replace(" ", "%20") + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    except sqlite3.Error as e:
        logger.warning("link %s: cannot open %s (%s)", link["name"], target, e)
        return None
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=2000")
    except sqlite3.Error:
        pass
    return conn
