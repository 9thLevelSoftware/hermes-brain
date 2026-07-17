-- Migration v1 -> v2 (P5): the `proposals` table.
--
-- This file is the DELTA. store/schema.sql carries the full current schema
-- for fresh creates. The two are kept honest by tests/test_migrations.py,
-- which asserts a fresh v2 database and a migrated v1 database have
-- byte-identical structure — drift fails CI rather than lurking.
--
-- Every statement is IF NOT EXISTS: re-running a migration is a no-op.

CREATE TABLE IF NOT EXISTS proposals (
    id          INTEGER PRIMARY KEY,
    uid         TEXT NOT NULL UNIQUE,          -- ULID
    kind        TEXT NOT NULL
                  CHECK (kind IN ('skill_draft','skill_revision','skill_retire',
                                  'strategy_deprecate','tuning')),
    target      TEXT,                          -- skill name / memory uid / artifact id
    title       TEXT NOT NULL,
    rationale   TEXT,
    payload     TEXT NOT NULL DEFAULT '{}',    -- JSON (draft dir, diff, weights, ...)
    evidence    TEXT NOT NULL DEFAULT '[]',    -- JSON: memory uids / session ids
    -- pending    -> proposed, not yet validated
    -- shadow     -> accumulating evidence, gates not met yet
    -- validated  -> gates passed, awaiting approval (auto or CLI)
    -- approved   -> approved but not yet on disk
    -- applied    -> live (skill promoted into the skills tree)
    -- rejected   -> gates failed or the user said no (KEPT: this is the archive)
    -- superseded -> a newer variant replaced this one
    status      TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','shadow','validated','approved',
                                    'applied','rejected','superseded')),
    validation  TEXT,                          -- JSON: per-gate results + verdict
    supersedes  TEXT REFERENCES proposals(uid),
    shift_id    TEXT,
    created_at  TEXT NOT NULL,
    decided_at  TEXT,
    decided_by  TEXT                           -- auto|cli|user
);

CREATE INDEX IF NOT EXISTS idx_proposals_open ON proposals(kind, status)
    WHERE status IN ('pending','shadow','validated','approved');
CREATE INDEX IF NOT EXISTS idx_proposals_target ON proposals(target);
