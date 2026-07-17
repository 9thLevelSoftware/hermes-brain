"""sqlite-vec vector store: int8[256] KNN over memories and episodes.

Created lazily at runtime — NEVER in schema.sql — because extension loading
is not universal (critique item 19). Callers check ``vec_available(conn)``;
every entry point degrades to no-op/[] when the extension is absent, so the
retrieval pipeline composes the vector leg only when it exists.

Quantization: embedders produce unit-normalized float vectors; we store
symmetric int8 (round(x*127)). L2 distance over int8 is rank-equivalent to
cosine for unit vectors at this precision, and 256 dims × 1 byte keeps 100k
memories ≈ 26MB — brute-force scans in milliseconds (research:infra).
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from collections.abc import Sequence

logger = logging.getLogger(__name__)

DIM = 256

_TABLES = {
    "mem_vec": "memories",
    "epi_vec": "episodes",
}


def load_extension(conn: sqlite3.Connection) -> bool:
    """Load sqlite-vec into this connection. Returns availability.

    Stateless: sqlite3.Connection is a C type that rejects attribute
    stamping, so "already loaded?" is answered by probing vec_version().
    """
    try:
        conn.execute("SELECT vec_version()")
        return True
    except sqlite3.Error:
        pass
    if not hasattr(conn, "enable_load_extension"):
        return False
    try:
        import sqlite_vec  # lazy optional dep
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception as e:
        logger.warning("sqlite-vec failed to load: %s", e)
        return False


def ensure_tables(conn: sqlite3.Connection, dim: int = DIM,
                  embedder_name: str = "", *, allow_rebuild: bool = False) -> bool:
    """Create the vec0 tables if the extension is available.

    Embedder IDENTITY (not just dim) is the compatibility key (review
    findings #4/#26): all realistic embedders here share 256d, so a model
    swap with matching dims would silently compare vectors across
    incompatible spaces. Identity or dim mismatch =>
      * allow_rebuild=False (live provider path): return False — the caller
        runs FTS-only and logs the remedy; NOTHING is destroyed mid-session.
      * allow_rebuild=True (CLI reindex): drop + recreate; the caller
        immediately re-embeds (vectors are derived data).
    """
    if not load_extension(conn):
        return False
    from . import db as store_db  # local import: avoid cycle at module load

    stored_dim = store_db.get_meta(conn, "vec_dim")
    stored_name = store_db.get_meta(conn, "vec_embedder") or ""
    mismatch = (
        (stored_dim is not None and int(stored_dim) != dim)
        or (stored_name and embedder_name and stored_name != embedder_name)
    )
    if mismatch:
        if not allow_rebuild:
            logger.warning(
                "vec index was built by %s (%sd) but the active embedder is %s (%dd); "
                "vector legs disabled — run 'hermes brain reindex' to rebuild",
                stored_name or "?", stored_dim, embedder_name or "?", dim,
            )
            return False
        logger.warning("vec rebuild: %s (%sd) -> %s (%dd); dropping vector tables",
                       stored_name or "?", stored_dim, embedder_name or "?", dim)
        for table in _TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    for table in _TABLES:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} "
            f"USING vec0(id INTEGER PRIMARY KEY, emb int8[{dim}])"
        )
    store_db.set_meta(conn, "vec_dim", str(dim))
    if embedder_name:
        store_db.set_meta(conn, "vec_embedder", embedder_name)
    conn.commit()
    return True


def vec_available(conn: sqlite3.Connection) -> bool:
    if not load_extension(conn):
        return False
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='mem_vec'"
    ).fetchone()
    return row is not None


def quantize(vector: Sequence[float]) -> bytes:
    """Unit-normalized float sequence -> symmetric int8 bytes."""
    return struct.pack(
        f"{len(vector)}b",
        *(max(-127, min(127, round(v * 127.0))) for v in vector),
    )


def upsert(conn: sqlite3.Connection, table: str, row_id: int,
           vector: Sequence[float]) -> None:
    """Insert/replace one embedding. vec0 has no ON CONFLICT: delete+insert."""
    assert table in _TABLES, table
    blob = quantize(vector)
    conn.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
    conn.execute(f"INSERT INTO {table}(id, emb) VALUES (?, vec_int8(?))", (row_id, blob))


def delete(conn: sqlite3.Connection, table: str, row_id: int) -> None:
    assert table in _TABLES, table
    conn.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))


def knn(conn: sqlite3.Connection, table: str, query_vector: Sequence[float],
        limit: int) -> list[tuple[int, float]]:
    """Top-`limit` (row_id, distance) by L2 over int8. [] on any failure."""
    assert table in _TABLES, table
    if limit <= 0:
        return []
    try:
        rows = conn.execute(
            f"SELECT id, distance FROM {table} WHERE emb MATCH vec_int8(?) "
            "ORDER BY distance LIMIT ?",
            (quantize(query_vector), limit),
        ).fetchall()
        return [(r["id"], r["distance"]) for r in rows]
    except sqlite3.Error as e:
        logger.warning("vec knn over %s failed: %s", table, e)
        return []


def missing_ids(conn: sqlite3.Connection, table: str, limit: int = 500) -> list[int]:
    """Base-table rows (current truth) that have no vector yet — backfill feed."""
    assert table in _TABLES, table
    base = _TABLES[table]
    where = "m.valid_to IS NULL AND m.status='active' AND m.live=1" if base == "memories" else "1=1"
    rows = conn.execute(
        f"SELECT m.id FROM {base} m LEFT JOIN {table} v ON v.id = m.id "
        f"WHERE v.id IS NULL AND {where} ORDER BY m.id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [r["id"] for r in rows]


def stats(conn: sqlite3.Connection) -> dict | None:
    if not vec_available(conn):
        return None
    out = {}
    for table in _TABLES:
        try:
            out[table] = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]
        except sqlite3.Error:
            out[table] = -1
    return out
