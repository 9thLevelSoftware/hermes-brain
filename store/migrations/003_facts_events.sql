-- Migration v2 -> v3 (best-of-three, Phase B): the temporal `facts` layer
-- and the `memory_events` append-only log.
--
-- This file is the DELTA. store/schema.sql carries the full current schema
-- for fresh creates. The two are kept honest by tests/test_migrations.py,
-- which asserts a fresh v3 database and a migrated one have byte-identical
-- structure — drift fails CI rather than lurking.
--
-- Every statement is IF NOT EXISTS: re-running a migration is a no-op.
--
-- `facts` is an INDEX OVER memories, not a parallel store (critique item 9):
-- every triple also lands as a `memories` row (kind='fact'); facts.memory_id
-- references it, so the facts leg feeds memory ids into RRF and re-fetch
-- gives trust/scope enforcement for free. Supersession is single-current-
-- truth per (subject,predicate): add closes the current row (valid_until +
-- superseded_by) then inserts the new one — the same versions-are-rows
-- discipline the memories table uses.

CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY,
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    object        TEXT NOT NULL,
    memory_id     INTEGER REFERENCES memories(id),   -- the NL memory row this indexes
    entity_id     INTEGER REFERENCES entities(id),   -- optional canonical subject entity
    confidence    REAL NOT NULL DEFAULT 1.0,
    source        TEXT,                              -- 'extract'|'dream:facts'|'sync'|...
    valid_from    TEXT NOT NULL,                     -- ISO-8601; when this became true
    valid_until   TEXT,                              -- NULL = current truth
    recorded_at   TEXT NOT NULL,                     -- ISO-8601; when the row was written
    superseded_by INTEGER REFERENCES facts(id)       -- the row that closed this one
);

-- Point-in-time truth pivots on (subject, predicate): exactly one current row
-- per pair. Partial indexes keep the current-truth probes (the hot path) cheap.
CREATE INDEX IF NOT EXISTS idx_facts_sp_current ON facts(subject, predicate)
    WHERE valid_until IS NULL;
CREATE INDEX IF NOT EXISTS idx_facts_obj_current ON facts(object)
    WHERE valid_until IS NULL;
CREATE INDEX IF NOT EXISTS idx_facts_memory ON facts(memory_id);

-- ------------------------------------------------------------
-- memory_events: append-only log of memory lifecycle ops, the seam every
-- later phase writes from day one (Phase G sync drains it). Off by default
-- (sync_events config); when off the writers are a no-op, so floor-tier
-- write cost stays visible rather than hidden behind triggers.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_events (
    id         INTEGER PRIMARY KEY,
    event_id   TEXT NOT NULL UNIQUE,               -- ULID (monotonic, sortable)
    ts         TEXT NOT NULL,                       -- ISO-8601
    op         TEXT NOT NULL
                 CHECK (op IN ('create','supersede','tombstone','purge')),
    memory_uid TEXT NOT NULL,                       -- the memories.uid affected
    payload    TEXT,                                -- JSON (surface-safe delta)
    origin     TEXT,                                -- device id that authored it
    synced_at  TEXT                                 -- NULL = not yet pushed to a relay
);

-- Cursor paging orders by (ts, event_id); the partial index serves the
-- "what hasn't been pushed yet" outbox scan.
CREATE INDEX IF NOT EXISTS idx_events_cursor ON memory_events(ts, event_id);
CREATE INDEX IF NOT EXISTS idx_events_unsynced ON memory_events(ts)
    WHERE synced_at IS NULL;
