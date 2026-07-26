"""Import memory from the other Hermes memory providers.

The `memory.provider` slot is exclusive, so adopting the brain means leaving
whatever you were on — and until this existed there was no way to bring your
memory across (docs/design/alignment-audit.md §F5). `bootstrap/` could import
MEMORY.md/USER.md, Hermes's own state.db, and Daem0n-MCP, but nothing from the
eight bundled providers.

**Scope is deliberately narrow and honest.** Adapters exist only where the data
is readable offline, with no API key and no third-party client:

  holographic  — plain SQLite at $HERMES_HOME/memory_store.db. Fully supported.
  jsonl        — one JSON object per line. The universal path: every provider
                 that can export gets in this way, and it is the documented
                 route for the cloud ones.

Not adapted, and why — a half-working importer that silently drops rows is
worse than a documented export step:

  byterover    — storage is an opaque tree owned by the `brv` CLI; importing
                 means shelling out to a binary this cannot test against.
  mem0 (OSS)   — memory lives in whichever vector store you configured
                 (qdrant/pgvector), not in a local file.
  openviking   — needs a running local server on OPENVIKING_ENDPOINT.
  honcho, retaindb, supermemory — cloud APIs requiring credentials.

Everything lands at `agent` trust (never `owner`: it did not come from the
owner speaking to THIS brain) with provenance `source=import:<provider>`, and
dedups on content hash so re-running is idempotent — the same discipline
`daemon_import.py` uses.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..capture.symbols import symbols_field
from ..store import db

logger = logging.getLogger(__name__)

PROVIDERS = ("holographic", "jsonl")

# Providers we deliberately do not adapt, with the honest reason + the route in.
UNSUPPORTED = {
    "byterover": "storage is an opaque tree owned by the `brv` CLI — export with "
                 "`brv export` and import the result with --provider jsonl",
    "mem0": "memory lives in your configured vector store (qdrant/pgvector), not a "
            "local file — export it and import with --provider jsonl",
    "openviking": "needs a running server on OPENVIKING_ENDPOINT — export and "
                  "import with --provider jsonl",
    "honcho": "cloud API (needs credentials) — export from app.honcho.dev and "
              "import with --provider jsonl",
    "retaindb": "cloud API (needs credentials) — export and import with "
                "--provider jsonl",
    "supermemory": "cloud API (needs credentials) — export and import with "
                   "--provider jsonl",
}

_EPISODIC_HALF_LIFE_DAYS = 30.0

# holographic categories -> brain kinds. Anything unmapped becomes a fact.
_HOLO_KIND_MAP = {
    "preference": "preference",
    "warning": "warning",
    "decision": "decision",
    "insight": "insight",
    "general": "fact",
}


def default_path(provider: str, hermes_home: str | Path) -> Path | None:
    if provider == "holographic":
        return Path(hermes_home) / "memory_store.db"
    return None


def _open_ro(path: Path) -> sqlite3.Connection:
    uri = "file:" + quote(str(path).replace("\\", "/")) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _tags(raw: object, provider: str) -> list[str]:
    """Provider tag fields vary (CSV, JSON array, absent). Normalize, and always
    stamp the origin so an import is identifiable and reversible later."""
    out: list[str] = []
    if isinstance(raw, str) and raw.strip():
        text = raw.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    out = [str(t).strip() for t in parsed if str(t).strip()]
            except ValueError:
                pass
        if not out:
            out = [t.strip() for t in text.split(",") if t.strip()]
    elif isinstance(raw, list):
        out = [str(t).strip() for t in raw if str(t).strip()]
    out.append(f"{provider}-import")
    return out


def _insert(conn: sqlite3.Connection, *, content: str, kind: str, tags: list[str],
            provider: str, created_at: str | None, pinned: bool,
            semantic: bool) -> bool:
    """One memory. Returns False when it deduped against existing content."""
    chash = db.content_hash(content)
    if conn.execute(
        "SELECT 1 FROM memories WHERE content_hash=? AND valid_to IS NULL LIMIT 1",
        (chash,),
    ).fetchone():
        return False
    now = db.iso_now()
    conn.execute(
        "INSERT INTO memories (uid, epistemic, memory_type, kind, status, live,"
        " content, content_hash, symbols, tags, token_len, trust_tier, created_by,"
        " source_platform, source_refs, valid_from, recorded_at, pinned,"
        " half_life_days)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            db.new_ulid(), "observation",
            "semantic" if semantic else "episodic",
            kind, "active", 1,
            content, chash, symbols_field(content),
            json.dumps(tags), db.approx_tokens(content),
            # NOT owner: this did not come from the owner speaking to THIS
            # brain, and importing a foreign store at owner trust would let it
            # straight into lane 1.
            "agent", "migration",
            f"import:{provider}",
            json.dumps([f"import:{provider}"]),
            created_at or now, now,
            1 if pinned else 0,
            None if semantic else _EPISODIC_HALF_LIFE_DAYS,
        ),
    )
    return True


def _iso(raw: object) -> str | None:
    """Best-effort timestamp normalization; None keeps the import time."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if "T" in text and text.endswith("Z"):
        return text
    # 'YYYY-MM-DD HH:MM:SS' (SQLite CURRENT_TIMESTAMP)
    if len(text) >= 19 and text[4] == "-" and text[10] in " T":
        return text[:10] + "T" + text[11:19] + ".000Z"
    if len(text) == 10 and text[4] == "-":
        return text + "T00:00:00.000Z"
    return None


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

def _import_holographic(conn: sqlite3.Connection, path: Path,
                        counts: dict[str, Any]) -> None:
    src = _open_ro(path)
    try:
        rows = [dict(r) for r in src.execute(
            "SELECT content, category, tags, trust_score, helpful_count, created_at"
            " FROM facts").fetchall()]
    except sqlite3.Error as e:
        counts["error"] = (f"{path} has no readable holographic `facts` table ({e}) — "
                           f"is this a memory_store.db?")
        return
    finally:
        src.close()

    for row in rows:
        content = str(row.get("content") or "").strip()
        if not content:
            counts["skipped"] += 1
            continue
        category = str(row.get("category") or "general").lower()
        # holographic's trust_score is 0..1 with 0.5 neutral; treat a
        # well-above-neutral fact as worth pinning rather than inventing a
        # trust tier for it (trust here means PROVENANCE, not confidence).
        try:
            pinned = float(row.get("trust_score") or 0.5) >= 0.9
        except (TypeError, ValueError):
            pinned = False
        if _insert(conn, content=content,
                   kind=_HOLO_KIND_MAP.get(category, "fact"),
                   tags=_tags(row.get("tags"), "holographic"),
                   provider="holographic",
                   created_at=_iso(row.get("created_at")),
                   pinned=pinned, semantic=True):
            counts["imported"] += 1
        else:
            counts["skipped"] += 1


def _import_jsonl(conn: sqlite3.Connection, path: Path,
                  counts: dict[str, Any]) -> None:
    """One JSON object per line. Recognized keys, all optional but `content`:

        {"content": "...", "kind": "fact", "tags": ["a"], "created_at": "...",
         "pinned": false}

    Also accepts `text`/`memory` as aliases for `content`, so a raw export from
    most providers works without reshaping.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as e:
        counts["error"] = f"cannot read {path} ({e})"
        return

    valid_kinds = ("fact", "decision", "preference", "warning", "insight")
    for lineno, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except ValueError:
            counts["malformed"] = counts.get("malformed", 0) + 1
            logger.debug("jsonl import: line %d is not JSON", lineno)
            continue
        if not isinstance(row, dict):
            counts["malformed"] = counts.get("malformed", 0) + 1
            continue
        content = str(row.get("content") or row.get("text")
                      or row.get("memory") or "").strip()
        if not content:
            counts["skipped"] += 1
            continue
        kind = str(row.get("kind") or "fact").lower()
        if kind not in valid_kinds:
            kind = "fact"
        if _insert(conn, content=content, kind=kind,
                   tags=_tags(row.get("tags"), "jsonl"),
                   provider="jsonl",
                   created_at=_iso(row.get("created_at") or row.get("timestamp")),
                   pinned=bool(row.get("pinned")), semantic=True):
            counts["imported"] += 1
        else:
            counts["skipped"] += 1


_ADAPTERS = {
    "holographic": _import_holographic,
    "jsonl": _import_jsonl,
}


def import_provider(
    conn: sqlite3.Connection,
    provider: str,
    *,
    hermes_home: str | Path,
    path: str | Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Import one provider's memory. Dry-run unless ``apply``.

    Returns ``{'imported', 'skipped', ...}`` (+ ``'error'``). Never raises: this
    is an operator command that should explain itself, not traceback.
    """
    counts: dict[str, Any] = {"imported": 0, "skipped": 0, "provider": provider,
                              "applied": bool(apply)}
    name = (provider or "").strip().lower()
    if name in UNSUPPORTED:
        counts["error"] = f"{name}: {UNSUPPORTED[name]}"
        return counts
    if name not in _ADAPTERS:
        counts["error"] = (f"unknown provider {provider!r} — supported: "
                           f"{', '.join(PROVIDERS)}")
        return counts

    source = Path(path) if path else default_path(name, hermes_home)
    if source is None:
        counts["error"] = f"--path is required for provider '{name}'"
        return counts
    if not source.is_file():
        counts["error"] = f"no such file: {source}"
        return counts
    counts["source"] = str(source)

    try:
        _ADAPTERS[name](conn, source, counts)
    except Exception as e:
        logger.warning("import-provider %s failed: %s", name, e, exc_info=True)
        counts["error"] = f"import failed: {e}"
        _rollback(conn)
        return counts

    if apply and counts["imported"]:
        db.bump_generation(conn, "mem")
        conn.commit()
    else:
        # Dry run: everything above ran for real against the transaction so the
        # counts are truthful, then is discarded.
        _rollback(conn)
    return counts


def _rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.rollback()
    except sqlite3.Error:
        pass
