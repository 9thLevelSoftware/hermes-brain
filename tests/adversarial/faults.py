"""Fault-injection toolkit for the adversarial pytest gauntlet.

Every helper here induces ONE hostile condition the brain is supposed to
survive — a corrupt row, a poisoned vector, a wedged write lock, a dead
archive, a missing capability — so a test can assert the degraded sentinel
(``[]`` / ``None`` / ``{"error"}`` / no-op) and, crucially, that the turn or
pipeline still completes. Stdlib + ``unittest.mock`` only, so any helper works
on any tier and needs no fixture wiring.

Lives under tests/ (not docker/) because it is pytest-only: pytest prepends
``tests/adversarial/`` to sys.path, so a test imports it as ``from faults
import ...``. The Docker phases use their own process-level fault injection
(SIGKILL, ``--memory`` limits, read-only mounts) — this module is the
in-process counterpart.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from unittest import mock

# ---------------------------------------------------------------------------
# monkeypatch-style raising (works without the monkeypatch fixture)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def raising(target, attr, exc=None):
    """Patch ``target.attr`` with a callable that raises ``exc``.

    ``target`` is a module or object; ``attr`` the callable name. Use to force
    a specific seam to throw — e.g. ``with raising(archive, "append",
    OSError("disk full")): ...`` — and assert the caller degrades rather than
    propagating.
    """
    exc = exc or RuntimeError("injected fault")

    def _boom(*_a, **_k):
        raise exc

    with mock.patch.object(target, attr, _boom):
        yield


@contextlib.contextmanager
def returning(target, attr, value):
    """Patch ``target.attr`` to a callable that returns a fixed ``value``
    (e.g. force ``archive.append`` to return ``None`` — the "archiving failed"
    sentinel — without a real disk fault)."""
    with mock.patch.object(target, attr, lambda *_a, **_k: value):
        yield


# ---------------------------------------------------------------------------
# row / index corruption
# ---------------------------------------------------------------------------

def corrupt_half_life(conn: sqlite3.Connection, rowid: int) -> None:
    """Store a non-numeric string in a memory's REAL ``half_life_days`` column
    (SQLite keeps text in a REAL-affinity column verbatim), so any lifecycle
    arithmetic on it raises — a memory-row bug that must not break the turn."""
    conn.execute("UPDATE memories SET half_life_days='not-a-number' WHERE id=?", (rowid,))
    conn.commit()


def poison_mem_vec(conn: sqlite3.Connection, rowid: int) -> bool:
    """Overwrite a memory's stored vector with a wrong-length int8 blob so the
    vec leg throws mid-scan. Only meaningful when sqlite-vec is loaded (full
    tier); returns False (no-op) when the vec table is absent."""
    from brain.store import vec

    if not vec.vec_available(conn):
        return False
    try:
        conn.execute("DELETE FROM mem_vec WHERE id=?", (rowid,))
        # 8 bytes instead of DIM(256) — a dimension mismatch on the next MATCH.
        conn.execute("INSERT INTO mem_vec(id, emb) VALUES (?, vec_int8(?))",
                     (rowid, b"\x01\x02\x03\x04\x05\x06\x07\x08"))
        conn.commit()
        return True
    except sqlite3.Error:
        return False


# ---------------------------------------------------------------------------
# capability removal (simulate a lesser interpreter without rebuilding Python)
# ---------------------------------------------------------------------------

def simulate_no_fts5(conn: sqlite3.Connection) -> None:
    """Make this DB look like it was opened on a Python without FTS5: rewrite
    the cached capabilities so search routes to the LIKE fallback, and drop the
    FTS triggers so a stray capture cannot fail on the missing module. Mirrors
    store/db._reconcile_fts's no-fts5 branch."""
    import json

    from brain.store import db

    caps = db.capabilities(conn)
    caps["fts5"] = False
    db.set_meta(conn, "capabilities", json.dumps(caps, sort_keys=True))
    for name in ("episodes_ai", "episodes_ad", "episodes_au",
                 "memories_ai", "memories_ad", "memories_au"):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.commit()


# ---------------------------------------------------------------------------
# a genuinely wedged write lock (integration-truthful; ~busy_timeout slow)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def held_write_lock(hermes_home):
    """Hold an exclusive write transaction on brain.db from a second connection
    for the duration of the block, so any writer on another connection contends
    (and, past its 5s busy_timeout, raises ``database is locked``). Use to prove
    a capture-path writer degrades to a no-op instead of raising into the turn.
    """
    from brain.store import db

    holder = db.connect(hermes_home)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("UPDATE meta SET value=value WHERE key='mem_generation'")
    try:
        yield holder
    finally:
        with contextlib.suppress(sqlite3.Error):
            holder.rollback()
        holder.close()


@contextlib.contextmanager
def sqlite_write_error(conn: sqlite3.Connection):
    """Fast alternative to held_write_lock: make every WRITE on ``conn`` raise
    OperationalError for the duration, for unit tests that only need to prove
    the swallow-and-no-op behavior without waiting on busy_timeout.

    Uses ``PRAGMA query_only`` rather than patching ``conn.execute`` —
    ``sqlite3.Connection.execute`` is a read-only C slot that cannot be
    monkeypatched (``AttributeError: attribute 'execute' is read-only``). Under
    query_only, reads still work and any INSERT/UPDATE/DELETE raises "attempt to
    write a readonly database"."""
    conn.execute("PRAGMA query_only=ON")
    try:
        yield
    finally:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("PRAGMA query_only=OFF")


# ---------------------------------------------------------------------------
# dead archive (make store.archive.append/append_batch fail -> None sentinel)
# ---------------------------------------------------------------------------

def break_archive_dir(hermes_home) -> None:
    """Create ``$HERMES_HOME/brain/archive`` as a regular FILE, so the archive
    module's ``path.parent.mkdir`` / ``gzip.open`` fail and append() returns
    None. The load-bearing consequence: forget must then preserve the memory's
    raw content (never null it). Cross-platform (no chmod needed)."""
    from pathlib import Path

    d = Path(hermes_home) / "brain" / "archive"
    d.parent.mkdir(parents=True, exist_ok=True)
    if d.is_dir():
        # can't easily turn a dir into a file if it has children; assert clean
        for child in d.iterdir():  # pragma: no cover - defensive
            raise RuntimeError(f"archive dir already populated: {child}")
        d.rmdir()
    d.write_bytes(b"not a directory")


# ---------------------------------------------------------------------------
# concurrency helper: run a callable on N threads against one home
# ---------------------------------------------------------------------------

def race(fn, n: int = 2):
    """Run ``fn(i)`` on ``n`` threads and return the list of results in thread
    order. Each thread gets its own connection inside ``fn`` (never share a
    sqlite3 connection across threads). Used for lease / buffer-claim races."""
    results: list = [None] * n
    barrier = threading.Barrier(n)

    def _worker(i):
        barrier.wait()  # maximize the overlap
        results[i] = fn(i)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return results
