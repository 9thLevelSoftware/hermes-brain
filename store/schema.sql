-- Hermes-Brain unified schema v2
-- ============================================================
-- v2 (P5) adds ONE table: `proposals` (propose-validate-archive for
-- self-modification). Strategy items and cases are deliberately NOT new
-- tables — they are `memories` rows (memory_type='procedural'/'episodic',
-- kind='strategy'|'guardrail'|'case') indexed in the one mem_vec index,
-- per critique item 9 ("no parallel vector paths").
--
-- Merged from docs/design/memory-engine.md (normative base) with every
-- schema-affecting resolution from docs/design/critique.md:
--   item 1  — one schema: `epistemic` 3-value column (not lane), TEXT trust
--             tiers, single ingest_buffer DDL
--   item 2  — DB lease row (brain_lease) is the sole mutual-exclusion
--             mechanism for dream processes
--   item 4  — retrieval_log carries (session_id, user_turn_count, ts,
--             content hash) so the nightly miner can resolve turn_id
--             against state.db messages
--   item 9  — retrieval_log absorbs injection_ledger; strategies/cases are
--             memories rows (no parallel vector or embedding columns)
--   item 19 — FTS5 is EXTERNAL-CONTENT (+ triggers), not contentless:
--             works on every SQLite ≥ 3.20; no contentless_delete needed.
--             vec0 virtual tables are created at runtime by store/vec.py
--             only when the sqlite-vec extension is loadable (capability
--             probe) — they are deliberately absent from this file.
--   item 25 — buffer rows carry promoted_at; only set when their memories
--             go live (observations survive shift rollback)
--   item 33 — identities table roots the trust model (owner enumeration)
--   item 34 — meta.schema_version + forward-only migrations; the code
--             refuses to open databases from the future
--
-- Conventions:
--   * All timestamps are TEXT ISO-8601 UTC ("2026-07-16T21:04:05.123Z").
--   * Versions are rows: an update INSERTs a replacement row and closes the
--     old one (valid_to + superseded_by). Current truth for reads is
--     `valid_to IS NULL AND status='active'`. Point-in-time (`as_of`) is a
--     flat predicate — never a per-row lookup.
--   * Nothing is ever DELETEd on the normal path: supersede, tombstone, or
--     expire. Hard purge exists only for compliance (`forget --hard`) and
--     tombstone grace expiry in the dream's forgetting pass.

PRAGMA user_version = 3;

-- ------------------------------------------------------------
-- meta: schema version, generation counters, capability cache
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

-- Seeded by store/db.py on create:
--   schema_version    '1'
--   created_at        ISO ts
--   mem_generation    '0'   (bumped on any memories write — cheap cross-
--   graph_generation  '0'    process cache invalidation, Daem0n pattern)
--   capabilities      JSON  (probe results: vec, fts5, load_extension)

-- ------------------------------------------------------------
-- identities: the root of the trust model (critique item 33)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS identities (
    principal_id     TEXT NOT NULL,             -- stable person id, e.g. 'owner' or ULID
    platform         TEXT NOT NULL,             -- 'telegram'|'discord'|'slack'|'cli'|...
    platform_user_id TEXT NOT NULL,             -- platform-native id
    display_name     TEXT,
    is_owner         INTEGER NOT NULL DEFAULT 0,
    added_at         TEXT NOT NULL,
    added_by         TEXT NOT NULL DEFAULT 'setup',  -- setup|cli|migration
    PRIMARY KEY (platform, platform_user_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_identities_principal ON identities(principal_id);

-- ------------------------------------------------------------
-- episodes: the raw episodic lane (critique item 22)
-- Every non-incognito primary-context turn lands here verbatim and is
-- FTS-indexed at capture time — "remember everything" holds even on the
-- no-LLM floor tier. Distilled memories cite these via source_refs.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS episodes (
    id                INTEGER PRIMARY KEY,
    uid               TEXT NOT NULL UNIQUE,      -- ULID
    session_id        TEXT NOT NULL,
    turn_no           INTEGER,                   -- provider-side user-turn counter
    platform          TEXT,
    source_channel    TEXT,
    source_author     TEXT,                      -- platform_user_id of the human speaker
    principal_id      TEXT,                      -- resolved via identities (NULL if unknown)
    trust_tier        TEXT NOT NULL DEFAULT 'known_user'
                        CHECK (trust_tier IN ('owner','agent','known_user','tool','untrusted')),
    user_content      TEXT NOT NULL,
    assistant_content TEXT NOT NULL,
    symbols           TEXT NOT NULL DEFAULT '',  -- code-symbol tokenizer expansion (FTS aux)
    token_len         INTEGER NOT NULL DEFAULT 0,
    salience          REAL,                      -- heuristic score at capture (0..1)
    ts                TEXT NOT NULL,
    archive_ref       TEXT,                      -- 'YYYY-MM.jsonl.gz:<offset>' once archived
    extracted_at      TEXT                       -- set when the sweep/dream has processed it
);

CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id, turn_no);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);
CREATE INDEX IF NOT EXISTS idx_episodes_unextracted ON episodes(id) WHERE extracted_at IS NULL;

-- External-content FTS over episodes (critique item 19: no contentless).
CREATE VIRTUAL TABLE IF NOT EXISTS episode_fts USING fts5(
    user_content, assistant_content, symbols,
    content='episodes', content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episode_fts(rowid, user_content, assistant_content, symbols)
    VALUES (new.id, new.user_content, new.assistant_content, new.symbols);
END;
CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episode_fts(episode_fts, rowid, user_content, assistant_content, symbols)
    VALUES ('delete', old.id, old.user_content, old.assistant_content, old.symbols);
END;
CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE OF user_content, assistant_content, symbols ON episodes BEGIN
    INSERT INTO episode_fts(episode_fts, rowid, user_content, assistant_content, symbols)
    VALUES ('delete', old.id, old.user_content, old.assistant_content, old.symbols);
    INSERT INTO episode_fts(rowid, user_content, assistant_content, symbols)
    VALUES (new.id, new.user_content, new.assistant_content, new.symbols);
END;

-- ------------------------------------------------------------
-- memories: the envelope (versions are rows)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY,          -- rowid; shared key for FTS/vec
    uid             TEXT NOT NULL UNIQUE,         -- ULID; stable across versions? NO —
                                                  -- each version row has its own uid;
                                                  -- chains link via supersedes_id
    epistemic       TEXT NOT NULL DEFAULT 'observation'
                      CHECK (epistemic IN ('observation','inference','belief')),
    memory_type     TEXT NOT NULL
                      CHECK (memory_type IN ('core','episodic','semantic','procedural','resource')),
    kind            TEXT,                         -- finer grain: fact|decision|preference|
                                                  -- warning|insight|profile, plus the
                                                  -- dream-owned P5 kinds: strategy |
                                                  -- guardrail (ReasoningBank items) and
                                                  -- case (Memento case bank)
    status          TEXT NOT NULL DEFAULT 'active'
                      CHECK (status IN ('active','summarized','tombstone','quarantined','expired')),
    -- dream staging (critique items 1+25): rows written by a shift are
    -- live=0 until the shift's probes pass; observations promote
    -- independently of inference/belief rows.
    live            INTEGER NOT NULL DEFAULT 1,
    shift_id        TEXT,

    content         TEXT,                         -- NULL only for tombstones
    summary         TEXT,                         -- dense one-liner (consolidation writes)
    content_hash    TEXT NOT NULL,                -- sha256 of normalized content
    symbols         TEXT NOT NULL DEFAULT '',     -- code-symbol expansion (FTS aux)
    tags            TEXT NOT NULL DEFAULT '[]',   -- JSON array of strings
    token_len       INTEGER NOT NULL DEFAULT 0,

    -- provenance
    source_platform TEXT,
    source_channel  TEXT,
    source_author   TEXT,
    source_session  TEXT,
    source_refs     TEXT NOT NULL DEFAULT '[]',   -- JSON: episode uids, state.db msg ids,
                                                  -- archive refs, evidence memory uids
                                                  -- (for inference/belief rows)
    trust_tier      TEXT NOT NULL DEFAULT 'untrusted'
                      CHECK (trust_tier IN ('owner','agent','known_user','tool','untrusted')),
    created_by      TEXT NOT NULL,                -- extraction|user_explicit|memory_tool|
                                                  -- delegation|consolidation|distillation|
                                                  -- migration|bootstrap
    instruction_shaped INTEGER NOT NULL DEFAULT 0,

    -- scope (NULL = unrestricted on that axis)
    scope_user      TEXT,                         -- principal_id
    scope_project   TEXT,
    scope_platform  TEXT,
    scope_session   TEXT,                         -- set => session-ephemeral lane

    -- version chain + bi-temporal
    version         INTEGER NOT NULL DEFAULT 1,
    supersedes_id   INTEGER REFERENCES memories(id),
    superseded_by   INTEGER REFERENCES memories(id),
    valid_from      TEXT NOT NULL,                -- happened_at backfill lands here
    valid_to        TEXT,                         -- NULL = currently valid
    recorded_at     TEXT NOT NULL,                -- transaction time
    invalidated_by  INTEGER REFERENCES memories(id),

    -- lifecycle policy
    pinned          INTEGER NOT NULL DEFAULT 0,
    core_block      TEXT CHECK (core_block IN (NULL,'identity','user_profile','guidelines','projects','warnings','open_loops')),
    core_rank       INTEGER,
    half_life_days  REAL,                         -- NULL = no decay (this IS the
                                                  -- time-sensitivity flag)
    ttl_at          TEXT,                         -- hard expiry (incognito/session lane)
    needs_review    INTEGER NOT NULL DEFAULT 0,   -- belief whose evidence was invalidated

    -- learning signals
    outcome            TEXT CHECK (outcome IN (NULL,'worked','partial','failed')),
    outcome_confidence REAL,
    outcome_note       TEXT,
    helpful_count      INTEGER NOT NULL DEFAULT 0,
    harmful_count      INTEGER NOT NULL DEFAULT 0,
    recall_count       INTEGER NOT NULL DEFAULT 0,
    last_recalled_at   TEXT,
    verification_count INTEGER NOT NULL DEFAULT 1,
    surprise           REAL,                      -- kNN cosine distance at write
    importance         REAL,                      -- consolidation-time value score
    emotional_salience REAL,                      -- stored only; no v1 consumer (critique)

    embedded_with   TEXT,                         -- 'embeddinggemma-300m-q8:256' etc.
    prompt_version  TEXT,                         -- extraction/consolidation prompt tag
    meta            TEXT                          -- JSON escape hatch; NOT a feature home
);

-- Hot-path partial indexes: the current-truth working set.
CREATE INDEX IF NOT EXISTS idx_mem_current
    ON memories(memory_type, scope_user, scope_project)
    WHERE valid_to IS NULL AND status = 'active' AND live = 1;
CREATE INDEX IF NOT EXISTS idx_mem_hash ON memories(content_hash) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS idx_mem_core
    ON memories(core_block, core_rank)
    WHERE memory_type = 'core' AND valid_to IS NULL AND status = 'active';
CREATE INDEX IF NOT EXISTS idx_mem_ttl ON memories(ttl_at) WHERE ttl_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_mem_review ON memories(needs_review) WHERE needs_review = 1;
CREATE INDEX IF NOT EXISTS idx_mem_session ON memories(scope_session) WHERE scope_session IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_mem_shift ON memories(shift_id) WHERE shift_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_mem_quarantine ON memories(status) WHERE status = 'quarantined';

-- External-content FTS over memories.
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content, summary, symbols, tags,
    content='memories', content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, content, summary, symbols, tags)
    VALUES (new.id, coalesce(new.content,''), coalesce(new.summary,''), new.symbols, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, summary, symbols, tags)
    VALUES ('delete', old.id, coalesce(old.content,''), coalesce(old.summary,''), old.symbols, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content, summary, symbols, tags ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, summary, symbols, tags)
    VALUES ('delete', old.id, coalesce(old.content,''), coalesce(old.summary,''), old.symbols, old.tags);
    INSERT INTO memory_fts(rowid, content, summary, symbols, tags)
    VALUES (new.id, coalesce(new.content,''), coalesce(new.summary,''), new.symbols, new.tags);
END;

-- ------------------------------------------------------------
-- edges: 5-type closed vocabulary, bi-temporal (Daem0n's vocabulary,
-- with the temporal validity Daem0n's edges lacked)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS edges (
    id          INTEGER PRIMARY KEY,
    src_id      INTEGER NOT NULL REFERENCES memories(id),
    dst_id      INTEGER NOT NULL REFERENCES memories(id),
    edge_type   TEXT NOT NULL
                  CHECK (edge_type IN ('led_to','supersedes','depends_on','conflicts_with','related_to')),
    confidence  REAL NOT NULL DEFAULT 1.0,
    created_by  TEXT NOT NULL,                    -- extraction|consolidation|dream|user
    valid_from  TEXT NOT NULL,
    valid_to    TEXT,
    recorded_at TEXT NOT NULL,
    UNIQUE (src_id, dst_id, edge_type, valid_from)
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id) WHERE valid_to IS NULL;

-- ------------------------------------------------------------
-- entities: global IDs + per-scope mentions (PPR substrate, P4+)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entities (
    id            INTEGER PRIMARY KEY,
    canonical     TEXT NOT NULL UNIQUE,           -- normalized name, globally scoped
    display_name  TEXT NOT NULL,
    entity_type   TEXT NOT NULL DEFAULT 'concept',-- person|project|tool|file|concept|org
    principal_id  TEXT,                           -- set when entity IS a known person
    mention_count INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_id  INTEGER NOT NULL REFERENCES entities(id),
    memory_id  INTEGER NOT NULL REFERENCES memories(id),
    scope_project TEXT,
    ts         TEXT NOT NULL,
    PRIMARY KEY (entity_id, memory_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_mentions_memory ON entity_mentions(memory_id);

-- ------------------------------------------------------------
-- ingest_buffer: the single durable capture buffer (critique item 1).
-- Rows are extraction work units; promoted_at is set only when the
-- extracted memories go live (critique item 25).
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingest_buffer (
    id          INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL
                  CHECK (kind IN ('turn','session_end_marker','pre_compress','memory_write','delegation')),
    session_id  TEXT NOT NULL,
    episode_id  INTEGER REFERENCES episodes(id),
    payload     TEXT NOT NULL,                    -- JSON (kind-specific)
    ts          TEXT NOT NULL,
    claimed_by  TEXT,                             -- shift_id/sweep run currently processing
    promoted_at TEXT                              -- set when resulting memories are live
);

CREATE INDEX IF NOT EXISTS idx_buffer_pending ON ingest_buffer(id) WHERE promoted_at IS NULL;

-- ------------------------------------------------------------
-- retrieval_log: one table for candidacy + injection + outcome joins
-- (absorbs Design B's injection_ledger — critique items 4 and 9)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS retrieval_log (
    id              INTEGER PRIMARY KEY,
    session_id      TEXT NOT NULL,
    user_turn_count INTEGER,                      -- provider-side counter; the miner
    query_hash      TEXT,                         -- resolves state.db turn_id via
    user_msg_hash   TEXT,                         -- (session, hash, ts window)
    ts              TEXT NOT NULL,
    memory_id       INTEGER NOT NULL REFERENCES memories(id),
    leg             TEXT NOT NULL,                -- fts|vec|fts+vec|like|ppr|pinned
    rank_score      REAL,
    injected        INTEGER NOT NULL DEFAULT 0,
    resolved_turn_id TEXT                         -- filled by the nightly miner
);

CREATE INDEX IF NOT EXISTS idx_rlog_session ON retrieval_log(session_id, user_turn_count);
CREATE INDEX IF NOT EXISTS idx_rlog_injected ON retrieval_log(memory_id) WHERE injected = 1;
CREATE INDEX IF NOT EXISTS idx_rlog_unresolved ON retrieval_log(id)
    WHERE injected = 1 AND resolved_turn_id IS NULL;

-- ------------------------------------------------------------
-- lane1_snapshot: the dream-materialized system-prompt index.
-- provider.initialize() renders lane 1 from THIS table only (one
-- renderer, deterministic — critique items 5 and 17).
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lane1_snapshot (
    section    TEXT NOT NULL,                     -- warnings|open_loops|facts|stats
    rank       INTEGER NOT NULL,
    memory_id  INTEGER REFERENCES memories(id),
    line       TEXT NOT NULL,                     -- pre-rendered index line
    rendered_at TEXT NOT NULL,
    PRIMARY KEY (section, rank)
) WITHOUT ROWID;

-- ------------------------------------------------------------
-- brain_lease: sole mutual-exclusion mechanism for dream/sweep
-- processes (critique item 2). Acquire = atomic UPDATE ... WHERE
-- expired; renew every 30s; TTL 120s.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS brain_lease (
    name       TEXT PRIMARY KEY,                  -- 'dream' | 'sweep'
    holder     TEXT,                              -- run_id
    acquired_at TEXT,
    expires_at TEXT
) WITHOUT ROWID;

INSERT OR IGNORE INTO brain_lease(name, holder, acquired_at, expires_at)
    VALUES ('dream', NULL, NULL, NULL), ('sweep', NULL, NULL, NULL);

-- ------------------------------------------------------------
-- activity: cross-process idle detection (provider heartbeats;
-- dream checks before/while running — cooperative preemption)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS activity (
    source     TEXT PRIMARY KEY,                  -- 'provider:<session_id>' etc.
    last_seen  TEXT NOT NULL
) WITHOUT ROWID;

-- ------------------------------------------------------------
-- sweep_state: watermarks for state.db mining + session sweeps
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sweep_state (
    key        TEXT PRIMARY KEY,                  -- e.g. 'state_db:<session_id>'
    watermark  TEXT NOT NULL,                     -- kind-specific cursor (JSON)
    updated_at TEXT NOT NULL
) WITHOUT ROWID;

-- ------------------------------------------------------------
-- shift bookkeeping (P4+): runs, per-strategy state, LLM spend,
-- deferred work
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS shift_runs (
    shift_id    TEXT PRIMARY KEY,
    kind        TEXT NOT NULL DEFAULT 'dream',    -- dream|sweep
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    outcome     TEXT,                             -- completed|preempted|failed|rolled_back
    phases_done TEXT NOT NULL DEFAULT '[]',       -- JSON list (idempotent rerun cursor)
    notes       TEXT
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS strategy_state (
    strategy    TEXT PRIMARY KEY,                 -- flush|consolidate|distill|cases|profile|forget|skills
    -- NULL mode = "use dream.shift.DEFAULT_MODES" (the ship-inert defaults);
    -- a bookkeeping row created by _mark_run must NOT pin a strategy to 'off'
    -- and silently disable the pipeline. Only an explicit --enable/--disable
    -- (or setup) writes a concrete mode.
    mode        TEXT DEFAULT NULL
                  CHECK (mode IS NULL OR mode IN ('off','shadow','dry_run','active')),
    last_run_at TEXT,
    cooldown_until TEXT,
    stats       TEXT NOT NULL DEFAULT '{}'        -- JSON counters
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS llm_ledger (
    id         INTEGER PRIMARY KEY,
    shift_id   TEXT,
    strategy   TEXT NOT NULL,
    model      TEXT NOT NULL,
    tokens_in  INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    est_usd    REAL NOT NULL DEFAULT 0,
    ts         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ledger_ts ON llm_ledger(ts);

CREATE TABLE IF NOT EXISTS work_queue (
    id         INTEGER PRIMARY KEY,
    task       TEXT NOT NULL,                     -- embed|reembed|archive|extract
    payload    TEXT NOT NULL,                     -- JSON
    created_at TEXT NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    done_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_work_pending ON work_queue(id) WHERE done_at IS NULL;

-- ------------------------------------------------------------
-- proposals (schema v2, P5): the propose-validate-archive record for
-- every self-modification — skill drafts/revisions/retirements, strategy
-- deprecations, retrieval-weight tuning (learning-system.md §3).
--
-- Darwin-Gödel archive discipline: rejected and superseded variants are
-- KEPT (supersedes/status), never deleted, so a variant can be revived
-- when context shifts. Nothing here is applied without passing the gates
-- recorded in `validation`; `decided_by='auto'` requires skill_auto_approve
-- (the user's 2026-07-16 decision) AND a passing validation record.
--
-- This is also the unified review queue's proposal half (critique item 12):
-- `hermes brain review` shows open proposals + quarantined memories as one
-- typed list.
-- ------------------------------------------------------------
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

-- ------------------------------------------------------------
-- audit_log: every autonomous mutation, quarantine decision, and
-- self-modification proposal (PendingOutcomeResolver discipline)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY,
    actor     TEXT NOT NULL,                      -- provider|sweep|dream:<shift_id>|cli|mcp
    action    TEXT NOT NULL,                      -- e.g. 'merge','supersede','tombstone',
                                                  -- 'quarantine','skill_draft','rollback'
    target    TEXT,                               -- memory uid(s) / path
    detail    TEXT,                               -- JSON
    ts        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);

-- ------------------------------------------------------------
-- facts: temporal subject-predicate-object layer, an INDEX OVER memories
-- (critique item 9 — every triple also lands as a kind='fact' memories row;
-- facts.memory_id references it). Single-current-truth per (subject,predicate):
-- `add` closes the current row then inserts the new one (versions-are-rows,
-- like memories). Point-in-time queries pivot on valid_from/valid_until.
-- See store/facts.py (adapted from mnemosyne triples.py, MIT).
-- ------------------------------------------------------------
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

CREATE INDEX IF NOT EXISTS idx_facts_sp_current ON facts(subject, predicate)
    WHERE valid_until IS NULL;
CREATE INDEX IF NOT EXISTS idx_facts_obj_current ON facts(object)
    WHERE valid_until IS NULL;
CREATE INDEX IF NOT EXISTS idx_facts_memory ON facts(memory_id);

-- ------------------------------------------------------------
-- memory_events: append-only log of memory lifecycle ops — the event seam
-- Phase G sync drains. Off by default (sync_events); writers no-op when off.
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

CREATE INDEX IF NOT EXISTS idx_events_cursor ON memory_events(ts, event_id);
CREATE INDEX IF NOT EXISTS idx_events_unsynced ON memory_events(ts)
    WHERE synced_at IS NULL;
