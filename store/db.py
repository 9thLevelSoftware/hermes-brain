"""brain.db connection management, schema, migrations, capability probe.

Single SQLite file at ``<hermes_home>/brain/brain.db``. WAL, short
transactions, multi-process (provider worker thread + short-lived dream/
sweep processes + MCP server). Every process opens its own connections;
every thread gets its own connection (sqlite3 default check_same_thread).

Capability probing (critique item 19): FTS5 and extension loading are not
universal. We probe once per open and record results in ``meta``; callers
degrade (no FTS -> LIKE search; no vec -> FTS-only retrieval) instead of
failing. The vec0 virtual table is created lazily by store/vec.py (P2),
never in schema.sql.

Migration policy (critique item 34): forward-only numbered migrations,
``VACUUM INTO`` backup before migrating, refuse to open databases with a
schema_version newer than this code.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
import unicodedata
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _migration_step(version: int):
    """Return a callable applying migrations/<NNN>_*.sql.

    Deltas live in their own files rather than being regex-sliced out of
    schema.sql. Slicing looked cheaper and was wrong for the same reason
    review findings #7/#16 were wrong: a ';' inside a statement body (here,
    an inline comment) truncates a [^;]+ match mid-DDL and silently emits a
    fragment. schema.sql (fresh create) and these deltas (upgrade path) are
    instead kept honest by tests/test_migrations.py, which asserts a fresh
    database and a migrated one have identical structure.
    """

    def _apply(conn: sqlite3.Connection) -> None:
        matches = sorted(_MIGRATIONS_DIR.glob(f"{version:03d}_*.sql"))
        if not matches:
            raise RuntimeError(
                f"migration file {version:03d}_*.sql is missing from "
                f"{_MIGRATIONS_DIR} — the plugin install is incomplete; "
                f"reinstall or run: git -C <plugin dir> pull"
            )
        for path in matches:
            conn.executescript(path.read_text(encoding="utf-8"))

    return _apply


# Forward-only migrations: {target_version: SQL script or callable(conn)}.
# Version N upgrades a version N-1 database.
MIGRATIONS: dict[int, Any] = {
    2: _migration_step(2),   # migrations/002_proposals.sql
    3: _migration_step(3),   # migrations/003_facts_events.sql
}


class FutureSchemaError(RuntimeError):
    """brain.db was created by a newer hermes-brain than this one."""


# ---------------------------------------------------------------------------
# Time / ids / hashing helpers (shared by every module)
# ---------------------------------------------------------------------------

def iso_now() -> str:
    """ISO-8601 UTC with millisecond precision, e.g. 2026-07-16T21:04:05.123Z"""
    t = time.time()
    ms = int((t % 1) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{ms:03d}Z"


_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32
_ulid_lock = threading.Lock()
_ulid_last = [0, b""]


def new_ulid() -> str:
    """Monotonic-enough ULID (26 chars) with stdlib only."""
    with _ulid_lock:
        ts = int(time.time() * 1000)
        rand = os.urandom(10)
        if ts == _ulid_last[0]:
            # bump randomness to keep same-ms ids unique and sortable-ish
            as_int = int.from_bytes(_ulid_last[1], "big") + 1
            rand = as_int.to_bytes(10, "big", signed=False)
        _ulid_last[0], _ulid_last[1] = ts, rand

    val = (ts << 80) | int.from_bytes(rand, "big")
    chars = []
    for _ in range(26):
        chars.append(_ULID_ALPHABET[val & 0x1F])
        val >>= 5
    return "".join(reversed(chars))


_WS_RE = re.compile(r"\s+")


def normalize_for_hash(text: str) -> str:
    """Stable normalization for content_hash: NFC, casefold, collapse ws."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def content_hash(text: str) -> str:
    return sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def approx_tokens(text: str) -> int:
    """Cheap token estimate (chars/4) — used only for budget packing."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def brain_dir(hermes_home: str | Path) -> Path:
    return Path(hermes_home) / "brain"


def db_path(hermes_home: str | Path) -> Path:
    return brain_dir(hermes_home) / "brain.db"


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------

def probe_capabilities(conn: sqlite3.Connection) -> dict[str, Any]:
    caps: dict[str, Any] = {
        "sqlite_version": sqlite3.sqlite_version,
        "fts5": False,
        "load_extension": False,
        "vec": False,  # set true by store/vec.py once the extension loads
    }
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS temp._caps_fts USING fts5(x)"
        )
        conn.execute("DROP TABLE temp._caps_fts")
        caps["fts5"] = True
    except sqlite3.Error:
        pass
    caps["load_extension"] = hasattr(conn, "enable_load_extension")
    return caps


# ---------------------------------------------------------------------------
# Open / create / migrate
# ---------------------------------------------------------------------------

_PRAGMAS = (
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("busy_timeout", "5000"),      # critique item 11: 5s, not 30s
    ("foreign_keys", "ON"),
    ("temp_store", "MEMORY"),
)


def connect(
    hermes_home: str | Path,
    *,
    create: bool = True,
    cache_kb: int = 8000,
    read_only: bool = False,
) -> sqlite3.Connection:
    """Open a brain.db connection with house pragmas, creating/migrating
    the database if needed. One connection per thread — do not share.
    """
    path = db_path(hermes_home)
    if not path.exists() and not create:
        raise FileNotFoundError(f"brain.db not found at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if read_only:
        # Percent-encode: SQLite URIs decode %XX and treat '#'/'?' as
        # delimiters, so raw paths containing '%' or '#' silently open a
        # DIFFERENT (empty) database (review finding #22).
        from urllib.parse import quote

        uri = "file:" + quote(str(path).replace("\\", "/")) + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    else:
        conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row

    for key, val in _PRAGMAS:
        if read_only and key in ("journal_mode",):
            continue
        try:
            conn.execute(f"PRAGMA {key}={val}")
        except sqlite3.Error:
            pass
    conn.execute(f"PRAGMA cache_size=-{int(cache_kb)}")

    if not read_only:
        _ensure_schema(conn, path)
    return conn


def _ensure_schema(conn: sqlite3.Connection, path: Path) -> None:
    caps = probe_capabilities(conn)

    have_meta = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
    ).fetchone()

    if not have_meta:
        _create_fresh(conn, caps)
        return

    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None:
        # Interrupted fresh create (crash between executescript and the meta
        # seed): all DDL is IF NOT EXISTS, so re-running is the self-heal
        # (review finding #13).
        _create_fresh(conn, caps)
        return
    current = int(row["value"])

    if current > SCHEMA_VERSION:
        raise FutureSchemaError(
            f"brain.db is schema v{current}, this hermes-brain understands "
            f"up to v{SCHEMA_VERSION}. Update the plugin: "
            f"git -C <plugin dir> pull  (or pip install -U hermes-brain)"
        )
    if current < SCHEMA_VERSION:
        _migrate(conn, path, current)

    # Reconcile FTS objects with the CURRENT interpreter's capabilities and
    # refresh the cache — best-effort: a contended write lock must never make
    # a healthy database unopenable (review findings #2, #10). Skip the write
    # entirely when nothing changed (review finding #20): every tool call
    # opens a fresh connection, and an unconditional capabilities UPSERT put
    # a WAL writer-lock acquisition on the turn-blocking read path.
    try:
        _reconcile_fts(conn, caps)
        caps_json = json.dumps(caps, sort_keys=True)
        if get_meta(conn, "capabilities") != caps_json:
            conn.execute(
                "INSERT INTO meta(key,value) VALUES('capabilities',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (caps_json,),
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.warning("brain.db capability reconcile deferred (%s)", e)
        try:
            conn.rollback()
        except sqlite3.Error:
            pass


# FTS objects are stripped/extracted by statement, never by fragile
# body-crossing regexes (review findings #7/#16 — the old trigger pattern
# used [^;] classes that could not cross the ';' inside trigger bodies and
# therefore matched nothing).
_FTS_TABLE_RE = re.compile(r"CREATE VIRTUAL TABLE[^;]+USING fts5[^;]+;", re.S)
_FTS_TRIGGER_RE = re.compile(
    r"CREATE TRIGGER IF NOT EXISTS (?:episodes|memories)_a[idu]\b.*?\bEND;", re.S
)
_FTS_TRIGGER_NAMES = (
    "episodes_ai", "episodes_ad", "episodes_au",
    "memories_ai", "memories_ad", "memories_au",
)


def _fts_ddl() -> str:
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    return "\n".join(_FTS_TABLE_RE.findall(sql) + _FTS_TRIGGER_RE.findall(sql))


def _schema_sql(caps: dict[str, Any]) -> str:
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    if not caps["fts5"]:
        # Strip FTS virtual tables + their triggers; search degrades to LIKE.
        sql = _FTS_TABLE_RE.sub("", sql)
        sql = _FTS_TRIGGER_RE.sub("", sql)
    return sql


def _reconcile_fts(conn: sqlite3.Connection, caps: dict[str, Any]) -> None:
    """Bring FTS tables/triggers in line with what THIS interpreter supports.

    A brain.db moves between interpreters (system Python without FTS5,
    python.org build with it, Termux). Two failure directions (review
    finding #10):
      * fts5 available but objects missing (DB born on a lesser Python):
        create them and rebuild the index from content.
      * fts5 unavailable but triggers exist (DB born on a better Python):
        the INSERT triggers would make every capture fail — drop the
        triggers (plain DROP works; the orphaned virtual tables stay, since
        dropping them needs the missing module, and are recreated/rebuilt on
        the next capable open).
    """
    have_fts_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_fts'"
    ).fetchone()
    have_triggers = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='episodes_ai'"
    ).fetchone()

    if caps["fts5"]:
        if not have_fts_table or not have_triggers:
            logger.info("brain.db: (re)creating FTS objects and rebuilding index")
            conn.executescript(_fts_ddl())
            conn.execute("INSERT INTO episode_fts(episode_fts) VALUES('rebuild')")
            conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('rebuild')")
    elif have_triggers:
        logger.warning(
            "brain.db has FTS triggers but this Python lacks FTS5 — dropping "
            "triggers so capture keeps working; search degrades to LIKE"
        )
        for name in _FTS_TRIGGER_NAMES:
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")


def _create_fresh(conn: sqlite3.Connection, caps: dict[str, Any]) -> None:
    logger.info("Creating brain.db schema v%d (fts5=%s)", SCHEMA_VERSION, caps["fts5"])
    conn.executescript(_schema_sql(caps))
    now = iso_now()
    conn.executemany(
        "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
        [
            ("schema_version", str(SCHEMA_VERSION)),
            ("created_at", now),
            ("mem_generation", "0"),
            ("graph_generation", "0"),
            ("capabilities", json.dumps(caps, sort_keys=True)),
        ],
    )
    conn.commit()


def _migrate(conn: sqlite3.Connection, path: Path, current: int) -> None:
    backup = path.with_name(f"brain.pre-v{SCHEMA_VERSION}.{int(time.time())}.db")
    logger.info("Migrating brain.db v%d -> v%d (backup: %s)", current, SCHEMA_VERSION, backup)
    try:
        conn.execute("VACUUM INTO ?", (str(backup),))
    except sqlite3.Error as e:  # pragma: no cover - best effort
        logger.warning("Pre-migration backup failed: %s", e)

    for version in range(current + 1, SCHEMA_VERSION + 1):
        step = MIGRATIONS.get(version)
        if step is None:
            raise RuntimeError(f"No migration registered for schema v{version}")
        if callable(step):
            step(conn)
        else:
            conn.executescript(step)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
            (str(version),),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# meta helpers
# ---------------------------------------------------------------------------

def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def bump_generation(conn: sqlite3.Connection, which: str = "mem") -> None:
    conn.execute(
        "UPDATE meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "
        "WHERE key = ?",
        (f"{which}_generation",),
    )


def capabilities(conn: sqlite3.Connection) -> dict[str, Any]:
    raw = get_meta(conn, "capabilities", "{}")
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def touch_activity(conn: sqlite3.Connection, source: str) -> None:
    conn.execute(
        "INSERT INTO activity(source, last_seen) VALUES(?,?) "
        "ON CONFLICT(source) DO UPDATE SET last_seen=excluded.last_seen",
        (source, iso_now()),
    )
