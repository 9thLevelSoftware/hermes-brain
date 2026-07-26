"""The `memories` tool — an Anthropic-memory-tool-shaped file interface over
brain storage (docs/design/integration.md §3.1, tool #5).

Why a file interface at all: Claude models are trained on the memory tool's
command grammar and `/memories` path convention, so this inherits useful
behavior nearly for free. On other models it is a harmless secondary surface —
everything here is also reachable through `brain_recall`/`brain_remember`.

**These are VIRTUAL views, not files.** Nothing is written to disk; every
command maps onto the same memories table the rest of the brain uses:

    /memories/                  directory listing
    /memories/profile.md        standing facts & preferences (editable)
    /memories/index.md          the rendered lane-1 index (READ-ONLY)
    /memories/topics/           the tag namespace
    /memories/topics/<tag>.md   memories carrying <tag> (editable)

Named `memories` because `memory` is a reserved core tool name (F6).

Invariants (each pinned by a test in tests/adversarial/test_memfs_scope.py):

* a path must resolve under /memories — no traversal, no absolute escape
* every read re-applies the caller's scope, exactly like recall.search
* `peer_card` (and the other dream-owned internal kinds) are unreachable
* writes are capped at the caller's trust tier and quarantined when
  instruction-shaped — the same rules as brain_remember
* `delete` is a SOFT tombstone, never a purge; hard deletion stays CLI-only

Placement: this is a subpackage, not a root module, because the Hermes loader
eagerly imports every root `*.py` on every CLI invocation. Only the schema
lives in tools.py; the implementation loads lazily on first call.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

ROOT = "/memories"
INDEX_TOKENS = 1200

# Kinds a file view may ever show. `peer_card` is the owner's private
# theory-of-mind of a person and must never reach the peer it describes;
# strategy/guardrail/case are dream-owned planning items, not user memory.
_VISIBLE_KINDS = ("fact", "decision", "preference", "warning", "insight")

_COMMANDS = ("view", "create", "str_replace", "insert", "delete", "rename")

_PATH_HINT = (
    'paths live under /memories — e.g. memories(command="view", '
    'path="/memories/profile.md")'
)


# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------

def parse_path(raw: Any) -> tuple[str, str]:
    """``/memories/topics/deploy.md`` -> ``("topic", "deploy")``.

    Returns ``(kind, arg)`` where kind is root|profile|index|topics|topic.
    Raises the tool's errors-that-teach on anything outside /memories — this
    is the traversal gate, so it rejects rather than normalizes.
    """
    from ..tools import _ToolError

    if not isinstance(raw, str) or not raw.strip():
        raise _ToolError("path is required", _PATH_HINT)

    # Normalize separators and collapse redundant segments WITHOUT resolving
    # '..' — a '..' anywhere is rejected outright rather than folded away, so
    # there is no arithmetic for an attacker to get wrong on our behalf.
    text = raw.strip().replace("\\", "/")
    if ".." in text.split("/"):
        raise _ToolError(f"path {raw!r} escapes /memories",
                         "relative traversal is not allowed — " + _PATH_HINT)
    parts = [p for p in text.split("/") if p not in ("", ".")]
    if not parts or parts[0] != "memories":
        raise _ToolError(f"path {raw!r} is outside /memories", _PATH_HINT)
    rest = parts[1:]

    if not rest:
        return ("root", "")
    if len(rest) == 1:
        if rest[0] == "profile.md":
            return ("profile", "")
        if rest[0] == "index.md":
            return ("index", "")
        if rest[0] == "topics":
            return ("topics", "")
        raise _ToolError(
            f"no such path: {raw}",
            'valid: /memories/profile.md, /memories/index.md, '
            '/memories/topics/<tag>.md — list them with '
            'memories(command="view", path="/memories")',
        )
    if len(rest) == 2 and rest[0] == "topics":
        name = rest[1]
        if not name.endswith(".md"):
            raise _ToolError(
                f"topic files end in .md: {raw}",
                'e.g. memories(command="view", path="/memories/topics/deploy.md")',
            )
        tag = name[:-3].strip().lower()
        if not tag:
            raise _ToolError("topic name is empty", _PATH_HINT)
        return ("topic", tag)
    raise _ToolError(f"no such path: {raw}", _PATH_HINT)


# ---------------------------------------------------------------------------
# Scoped reads
# ---------------------------------------------------------------------------

def _scope_sql(ctx) -> tuple[str, list]:
    """Scope predicate mirroring recall.search._scope_memories.

    An owner sees everything current; anyone else sees only unscoped rows or
    rows scoped to their own principal. Quarantined and non-live rows are
    excluded for EVERY caller — the lanes and this surface agree.
    """
    sql = (" AND valid_to IS NULL AND status='active' AND live=1"
           " AND kind IN (" + ",".join("?" * len(_VISIBLE_KINDS)) + ")")
    params: list = list(_VISIBLE_KINDS)
    if ctx.trust_tier != "owner":
        sql += " AND (scope_user IS NULL OR scope_user = ?)"
        params.append(ctx.principal_id or "")
    return sql, params


def _rows_for(conn: sqlite3.Connection, view: str, tag: str, ctx) -> list[sqlite3.Row]:
    scope, params = _scope_sql(ctx)
    if view == "profile":
        # Standing facts & preferences — the lane-1 profile section's source.
        #
        # profile.md is the CALLER'S OWN profile, so the owner sees the global
        # (unscoped) rows only. recall.search lets an owner read everything,
        # which is right for search but wrong here: a peer's private,
        # peer-scoped preference is not one of the owner's standing facts, and
        # rendering it into a file called "profile" both misattributes it and
        # shows the owner something the peer wrote about themselves.
        sql = ("SELECT id, uid, content, kind, tags, pinned FROM memories"
               " WHERE kind IN ('fact','preference')" + scope)
        if ctx.trust_tier == "owner":
            sql += " AND scope_user IS NULL"
    else:
        # tags is a JSON array; LIKE on the serialized form is the floor-tier
        # way to filter it (no JSON1 dependency), then verified exactly below.
        sql = ("SELECT id, uid, content, kind, tags, pinned FROM memories"
               " WHERE tags LIKE ?" + scope)
        params = [f'%"{tag}"%', *params]
    rows = conn.execute(sql + " ORDER BY pinned DESC, id DESC LIMIT 200",
                        params).fetchall()
    if view == "topic":
        rows = [r for r in rows if tag in _tags_of(r)]
    return rows


def _tags_of(row: sqlite3.Row) -> list[str]:
    try:
        tags = json.loads(row["tags"] or "[]")
    except (ValueError, TypeError):
        return []
    return [str(t).lower() for t in tags] if isinstance(tags, list) else []


def _render(rows: list[sqlite3.Row], title: str) -> str:
    if not rows:
        return f"# {title}\n\n(empty)\n"
    lines = [f"# {title}", ""]
    for row in rows:
        pin = " (pinned)" if row["pinned"] else ""
        body = " ".join((row["content"] or "").split())
        lines.append(f"- [{row['uid'][:8]}] {body}{pin}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle(conn: sqlite3.Connection, args: dict, ctx) -> dict:
    """One `memories` call. Raises _ToolError; tools.dispatch turns that into
    the errors-that-teach payload and never lets it reach the agent raw."""
    from ..tools import _ToolError

    command = args.get("command")
    if command not in _COMMANDS:
        raise _ToolError(
            f"unknown command {command!r}",
            "valid commands: " + "|".join(_COMMANDS) +
            ' — e.g. memories(command="view", path="/memories")',
        )
    handler = {
        "view": _view,
        "create": _create,
        "str_replace": _str_replace,
        "insert": _insert,
        "delete": _delete,
        "rename": _rename,
    }[command]
    return handler(conn, args, ctx)


def _view(conn: sqlite3.Connection, args: dict, ctx) -> dict:
    view, tag = parse_path(args.get("path"))
    if view == "root":
        return {"path": ROOT, "kind": "directory", "entries": [
            "/memories/profile.md", "/memories/index.md", "/memories/topics/"]}
    if view == "topics":
        return {"path": f"{ROOT}/topics", "kind": "directory",
                "entries": [f"{ROOT}/topics/{t}.md" for t in _all_tags(conn, ctx)]}
    if view == "index":
        from ..recall import lane1

        try:
            body = lane1.render(conn, INDEX_TOKENS)
        except Exception as e:
            logger.warning("memories: lane1 render failed: %s", e)
            body = ""
        return {"path": f"{ROOT}/index.md", "kind": "file", "readonly": True,
                "content": body or "# Index\n\n(not yet materialized — run "
                                   "'hermes brain refresh-index')\n"}
    rows = _rows_for(conn, view, tag, ctx)
    title = "Profile" if view == "profile" else f"Topic: {tag}"
    return {"path": args.get("path"), "kind": "file",
            "content": _render(rows, title), "count": len(rows)}


def _all_tags(conn: sqlite3.Connection, ctx) -> list[str]:
    scope, params = _scope_sql(ctx)
    rows = conn.execute(
        "SELECT tags FROM memories WHERE 1=1" + scope + " LIMIT 2000", params
    ).fetchall()
    tags: set[str] = set()
    for row in rows:
        tags.update(_tags_of(row))
    return sorted(tags)


def _editable(view: str, path: Any) -> None:
    from ..tools import _ToolError

    if view in ("profile", "topic"):
        return
    if view == "index":
        raise _ToolError(
            "/memories/index.md is read-only",
            "it is a rendering of what the brain already knows — edit the "
            'sources instead, e.g. memories(command="str_replace", '
            'path="/memories/profile.md", old_str="...", new_str="...")',
        )
    raise _ToolError(
        f"{path} is a directory, not an editable file",
        'edit /memories/profile.md or /memories/topics/<tag>.md',
    )


def _write_memory(conn: sqlite3.Connection, text: str, view: str, tag: str,
                  ctx) -> dict:
    """Every write funnels through brain_remember's handler — one write path,
    so the trust cap, scoping, instruction-shape quarantine, dedup, event seam
    and embedding behave identically here and there."""
    from ..tools import _remember

    kind = "preference" if view == "profile" else "fact"
    tags = [] if view == "profile" else [tag]
    result = _remember(conn, {"content": text, "kind": kind, "tags": tags}, ctx)
    if view == "topic" and result.get("deduped_against"):
        # brain_remember dedups on content_hash and leaves tags alone, which
        # would make "put this existing fact in another topic file" a silent
        # no-op — the file the user just wrote would come back empty. Adding
        # the tag to the surviving row is what the file operation MEANT.
        _add_tag(conn, result["deduped_against"], tag)
        result["note"] = (f"already known — added to topic {tag!r} "
                          f"(id {result['deduped_against']})")
    return result


def _add_tag(conn: sqlite3.Connection, uid_prefix: str, tag: str) -> None:
    row = conn.execute(
        "SELECT id, tags FROM memories WHERE uid LIKE ? AND valid_to IS NULL",
        (uid_prefix + "%",)).fetchone()
    if row is None:
        return
    tags = _tags_of(row)
    if tag in tags:
        return
    tags.append(tag)
    conn.execute("UPDATE memories SET tags=? WHERE id=?",
                 (json.dumps(tags), row["id"]))
    conn.commit()


def _create(conn: sqlite3.Connection, args: dict, ctx) -> dict:
    from ..tools import _ToolError

    view, tag = parse_path(args.get("path"))
    _editable(view, args.get("path"))
    text = args.get("file_text")
    if not isinstance(text, str) or not text.strip():
        raise _ToolError(
            "file_text is required for create",
            'e.g. memories(command="create", path="/memories/topics/deploy.md", '
            'file_text="- staging deploys need the VPN")',
        )
    # One memory per non-empty, non-heading line: a "file" here is a rendered
    # list of memories, so writing one back must round-trip to memories, not
    # to a single blob nobody can recall a piece of.
    written, notes = [], []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-* ").strip()
        if not line or line.startswith("#"):
            continue
        line = _strip_id_prefix(line)
        if not line:
            continue
        result = _write_memory(conn, line, view, tag, ctx)
        written.append(result["id"])
        if result.get("note"):
            notes.append(result["note"])
    if not written:
        raise _ToolError(
            "file_text contained no memory lines",
            'write one memory per line — e.g. file_text="- the staging DB is on host X"',
        )
    return {"path": args.get("path"), "created": written,
            "note": f"wrote {len(written)} memor{'y' if len(written) == 1 else 'ies'}"
                    + ("; " + "; ".join(notes[:3]) if notes else "")}


def _strip_id_prefix(line: str) -> str:
    """Drop a leading ``[01ABC234]`` so a viewed file can be edited and
    written back without the id becoming part of the memory text."""
    if line.startswith("[") and "]" in line:
        head, _, tail = line.partition("]")
        if head[1:].isalnum():
            return tail.strip()
    return line


def _find_one(conn: sqlite3.Connection, view: str, tag: str, needle: str,
              ctx) -> sqlite3.Row:
    from ..tools import _ToolError

    rows = [r for r in _rows_for(conn, view, tag, ctx) if needle in (r["content"] or "")]
    if not rows:
        raise _ToolError(
            f"old_str not found in that view: {needle[:60]!r}",
            'view the file first — memories(command="view", path=...) — and '
            "copy an exact substring of one line",
        )
    if len(rows) > 1:
        ids = ", ".join(r["uid"][:8] for r in rows[:5])
        raise _ToolError(
            f"old_str matches {len(rows)} memories ({ids})",
            "make old_str longer so it identifies exactly one line",
        )
    return rows[0]


def _str_replace(conn: sqlite3.Connection, args: dict, ctx) -> dict:
    from ..tools import _ToolError

    view, tag = parse_path(args.get("path"))
    _editable(view, args.get("path"))
    old = args.get("old_str")
    new = args.get("new_str")
    if not isinstance(old, str) or not old:
        raise _ToolError("old_str is required for str_replace",
                         'e.g. memories(command="str_replace", path="...", '
                         'old_str="Hetzner CX22", new_str="Hetzner CX32")')
    if not isinstance(new, str):
        raise _ToolError("new_str must be a string (use \"\" to delete the text)",
                         'e.g. new_str="Hetzner CX32"')
    row = _find_one(conn, view, tag, old, ctx)
    updated = (row["content"] or "").replace(old, new).strip()
    if not updated:
        raise _ToolError(
            "that replacement would empty the memory",
            'to remove it entirely use brain_manage(action="forget", '
            f'id="{row["uid"][:8]}", reason="...")',
        )
    # Supersede-don't-mutate: the old version is tombstoned and a new row
    # carries the edit, so `hermes brain why` can still show what changed.
    result = _write_memory(conn, updated, view, tag, ctx)
    _tombstone(conn, row, ctx, reason=f"superseded via memories str_replace "
                                      f"-> {result['id']}")
    return {"path": args.get("path"), "replaced": row["uid"][:8],
            "id": result["id"],
            "note": "superseded — the previous version is tombstoned, not erased"}


def _insert(conn: sqlite3.Connection, args: dict, ctx) -> dict:
    from ..tools import _ToolError

    view, tag = parse_path(args.get("path"))
    _editable(view, args.get("path"))
    text = args.get("insert_text")
    if not isinstance(text, str) or not text.strip():
        raise _ToolError(
            "insert_text is required for insert",
            'e.g. memories(command="insert", path="/memories/profile.md", '
            'insert_line=0, insert_text="prefers terse answers")',
        )
    result = _write_memory(conn, _strip_id_prefix(text.strip().lstrip("-* ")),
                           view, tag, ctx)
    return {"path": args.get("path"), "id": result["id"],
            "note": (result.get("note") or "") +
                    " — insert_line is advisory: these files are ordered by "
                    "pinning and recency, not by line number"}


def _tombstone(conn: sqlite3.Connection, row: sqlite3.Row, ctx, *, reason: str) -> None:
    """Soft delete only — identical to brain_manage(action='forget')."""
    from ..store import db
    from ..tools import _audit

    now = db.iso_now()
    conn.execute("UPDATE memories SET status='tombstone', valid_to=? WHERE id=?",
                 (now, row["id"]))
    conn.execute("DELETE FROM lane1_snapshot WHERE memory_id=?", (row["id"],))
    try:
        from ..store import vec as vec_store

        if vec_store.vec_available(conn):
            vec_store.delete(conn, "mem_vec", row["id"])
    except Exception as e:
        logger.warning("memories: vec delete failed for %s: %s", row["id"], e)
    _audit(conn, "memories_forget", row["uid"],
           {"reason": reason, "session": ctx.session_id}, now)
    db.bump_generation(conn, "mem")
    conn.commit()


def _delete(conn: sqlite3.Connection, args: dict, ctx) -> dict:
    from ..tools import _ToolError

    view, tag = parse_path(args.get("path"))
    if view == "profile":
        raise _ToolError(
            "refusing to delete the whole profile",
            'remove entries one at a time — memories(command="str_replace", ...) '
            'or brain_manage(action="forget", id="...", reason="...")',
        )
    _editable(view, args.get("path"))
    rows = _rows_for(conn, view, tag, ctx)
    if not rows:
        raise _ToolError(f"no memories tagged {tag!r}",
                         'list topics with memories(command="view", '
                         'path="/memories/topics")')
    for row in rows:
        _tombstone(conn, row, ctx, reason=f"memories delete {args.get('path')}")
    return {"path": args.get("path"), "deleted": [r["uid"][:8] for r in rows],
            "note": f"tombstoned {len(rows)} memor"
                    f"{'y' if len(rows) == 1 else 'ies'} (soft — reversible via "
                    f"CLI until the dream purges them)"}


def _rename(conn: sqlite3.Connection, args: dict, ctx) -> dict:
    from ..store import db
    from ..tools import _audit, _ToolError

    view, tag = parse_path(args.get("path"))
    new_view, new_tag = parse_path(args.get("new_path"))
    if view != "topic" or new_view != "topic":
        raise _ToolError(
            "rename only moves topic files",
            'e.g. memories(command="rename", path="/memories/topics/ci.md", '
            'new_path="/memories/topics/deploy.md")',
        )
    rows = _rows_for(conn, view, tag, ctx)
    if not rows:
        raise _ToolError(f"no memories tagged {tag!r}", "nothing to rename")
    now = db.iso_now()
    for row in rows:
        tags = [t for t in _tags_of(row) if t != tag]
        if new_tag not in tags:
            tags.append(new_tag)
        conn.execute("UPDATE memories SET tags=? WHERE id=?",
                     (json.dumps(tags), row["id"]))
    _audit(conn, "memories_rename", None,
           {"from": tag, "to": new_tag, "count": len(rows),
            "session": ctx.session_id}, now)
    db.bump_generation(conn, "mem")
    conn.commit()
    return {"path": args.get("new_path"), "renamed": len(rows),
            "note": f"retagged {len(rows)} memories {tag!r} -> {new_tag!r}"}
