I have everything I need — research sections extracted (daem0n:memory-core, daem0n:graph-temporal, research:infra, academic/frameworks recommendations) and Daem0n source verified (surprise.py kNN-cosine math, fusion.py RRF k=60, similarity.py code-symbol tokenizer + decay half-life 30d/floor 0.3 + polarity conflict check, models.py bi-temporal columns). Here is the design document.

---

# Hermes-Brain Core Memory Engine — Design Document

**Scope:** storage, retrieval, write path, forgetting. Out of scope here (but designed against): sleep-time consolidation/dreaming internals, skill-forge, MCP server plumbing, provider wiring details.

**Verified against source:** Daem0n's `surprise.py` (surprise = mean cosine *distance* to k=5 nearest neighbors, clamped [0,1], first memory = 1.0), `fusion.py` (RRF k=60, rank starts at 1 — correct, was never wired in), `similarity.py` (`extract_code_symbols` regexes for backticks/CamelCase/lowerCamel/snake_case/SCREAMING_SNAKE/`.method`, both-case emission; `calculate_memory_decay` exp half-life 30d floor 0.3; `detect_conflict` polarity/negation heuristic), `models.py` (`valid_from`/`valid_to`/`invalidated_by_version_id`/`change_type` bi-temporal columns).

**Design principles (anti-Daem0n-death rules):**
1. **Every module has a named call site in a core flow.** The call-site table in §5 is normative — anything not in it doesn't get built.
2. **One of each:** one keyword index (FTS5), one vector store (sqlite-vec), one fusion (RRF→ convex later), one conflict mechanism, one version mechanism. Daem0n had three keyword engines and two contradiction detectors; zero of that.
3. **Append-only versions in the main table**, not a snapshot sidecar. `as_of` becomes a WHERE predicate, not N+1 lookups.
4. **Nothing gates on an LLM at read time.** LLM calls happen only at flush (batched) and sleep time.
5. **Cache-stability is invariant #1:** core blocks are frozen per session (`system_prompt_block`); everything dynamic rides the per-turn `<memory-context>` fence via `prefetch`.

---

## 1. Data model / SQL schema

One SQLite file: `<hermes_home>/brain/brain.db` (WAL, `synchronous=NORMAL`, `busy_timeout=30000`, `foreign_keys=ON`, `temp_store=MEMORY`, `cache_size=-32000`). Append-only raw archive: `<hermes_home>/brain/archive/YYYY-MM.jsonl.gz` (stdlib gzip, never pruned — the distill-don't-delete substrate). Requires SQLite ≥ 3.45 (FTS5 `contentless_delete`, ships with CPython ≥3.12 everywhere; on Python 3.11/old-distro combos, `pysqlite3-binary` is the documented fallback).

### 1.1 The envelope: `memories`

Versions are rows. An UPDATE writes a new row and closes the old one (`valid_to`, `superseded_by`). "Current truth" = `valid_to IS NULL AND status='active'`. This is the single most important structural decision: it makes bi-temporal queries flat predicates and makes supersede-don't-delete (Graphiti) the *only* way to change anything.

```sql
CREATE TABLE memories(
  id              INTEGER PRIMARY KEY,          -- rowid; shared key for FTS/vec
  uid             TEXT NOT NULL UNIQUE,         -- ULID (26 chars, stdlib impl)
  lane            TEXT NOT NULL DEFAULT 'evidence'
                    CHECK(lane IN ('evidence','belief')),
  memory_type     TEXT NOT NULL
                    CHECK(memory_type IN ('core','episodic','semantic','procedural','resource')),
  status          TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active','summarized','tombstone','quarantined','expired')),
  content         TEXT,                         -- NULL only for tombstones
  summary         TEXT,                         -- dense one-liner (written at consolidation)
  content_hash    TEXT NOT NULL,                -- sha256, exact-dup NOOP fast path
  token_len       INTEGER NOT NULL DEFAULT 0,   -- for budget packing

  -- provenance
  source_platform TEXT,                         -- telegram|discord|slack|cli|mcp|...
  source_channel  TEXT,
  source_author   TEXT,
  source_session  TEXT,
  source_refs     TEXT,                         -- JSON: state.db msg ids + archive offsets
  trust_tier      TEXT NOT NULL DEFAULT 'untrusted'
                    CHECK(trust_tier IN ('owner','agent','known_user','tool','untrusted')),
  created_by      TEXT NOT NULL,                -- extraction|user_explicit|memory_tool|delegation|consolidation|migration
  instruction_shaped INTEGER NOT NULL DEFAULT 0,-- SpAIware quarantine flag

  -- scope (NULL = unrestricted on that axis)
  scope_user      TEXT,
  scope_project   TEXT,
  scope_platform  TEXT,
  scope_session   TEXT,                         -- set => ephemeral/TTL lane

  -- version chain + bi-temporal (valid time vs transaction time)
  version         INTEGER NOT NULL DEFAULT 1,
  supersedes_id   INTEGER REFERENCES memories(id),
  superseded_by   INTEGER REFERENCES memories(id),
  valid_from      TEXT NOT NULL,                -- happened_at backfill lands here
  valid_to        TEXT,                         -- NULL = still valid
  recorded_at     TEXT NOT NULL,                -- transaction time
  invalidated_by  INTEGER REFERENCES memories(id), -- winner of a conflict, if any

  -- lifecycle policy knobs
  pinned          INTEGER NOT NULL DEFAULT 0,
  core_block      TEXT,                         -- identity|user_profile|guidelines|projects (core type only)
  core_rank       INTEGER,
  half_life_days  REAL,                         -- NULL = no decay (the "time-sensitive" flag IS this column)
  ttl_at          TEXT,                         -- hard expiry (incognito/session lane)
  needs_review    INTEGER NOT NULL DEFAULT 0,   -- belief with invalidated evidence

  -- learning signals (ACE + Daem0n outcome loop)
  outcome            TEXT CHECK(outcome IN ('worked','partial','failed')),
  outcome_confidence REAL,
  helpful_count      INTEGER NOT NULL DEFAULT 0,
  harmful_count      INTEGER NOT NULL DEFAULT 0,
  recall_count       INTEGER NOT NULL DEFAULT 0,
  last_recalled_at   TEXT,
  verification_count INTEGER NOT NULL DEFAULT 1, -- re-observations of same fact
  surprise           REAL,                       -- kNN distance at write (Daem0n math, finally wired)
  importance         REAL,                       -- consolidation-time value score (§4)
  emotional_salience REAL,                       -- extractor-tagged, 0..1

  embedded_with   TEXT,                         -- 'embeddinggemma-300m-q8:256' | 'potion-retrieval-32m:256'
  meta            TEXT                          -- JSON escape hatch; NOT a place for features
);

-- hot-path indexes (partial: the current-truth working set)
CREATE INDEX idx_mem_current ON memories(memory_type, scope_user, scope_project)
  WHERE valid_to IS NULL AND status = 'active';
CREATE UNIQUE INDEX idx_mem_hash ON memories(content_hash) WHERE valid_to IS NULL;
CREATE INDEX idx_mem_core ON memories(core_block, core_rank)
  WHERE memory_type = 'core' AND valid_to IS NULL;
CREATE INDEX idx_mem_ttl ON memories(ttl_at) WHERE ttl_at IS NOT NULL;
CREATE INDEX idx_mem_review ON memories(needs_review) WHERE needs_review = 1;
CREATE INDEX idx_mem_session ON memories(scope_session) WHERE scope_session IS NOT NULL;
```

**The `as_of` query is a flat predicate — no join, no per-row lookups** (fixes Daem0n's N+1):

```sql
-- point-in-time recall (valid-time only by default — Daem0n's pragmatic choice, kept,
-- so backfilled facts are findable; pass :tx to also pin transaction time)
SELECT * FROM memories
WHERE valid_from <= :t AND (valid_to IS NULL OR valid_to > :t)
  AND recorded_at <= COALESCE(:tx, recorded_at)
  AND status IN ('active','summarized');
```

**Type policies (in code, `envelope.py`, not DB):**

| type | decay (`half_life_days` default) | injection | update policy |
|---|---|---|---|
| `core` | none | always (system_prompt_block, frozen per session) | ACE delta-ops only: itemized rows, add/edit/deprecate, never monolithic rewrite; owner/agent trust required |
| `episodic` | 30d, floor 0.3 (read-time only) | retrieval only | append-only; outcomes attach |
| `semantic` | none by default; extractor sets 30–90d **only** on time-sensitive claims ("is traveling this week") | retrieval | supersede on contradiction; `verification_count++` on re-observation |
| `procedural` | none | retrieval + feeds skill-forge (pointer in `meta.skill_ref`) | graded outcomes modulate rank; never decays |
| `resource` | none | retrieval (chunked; parent doc in archive) | immutable; superseded by re-upload |
| vault | n/a | **never** — separate table, no index membership | explicit tool access only |

### 1.2 Beliefs vs evidence (Hindsight split)

`lane='belief'` rows are agent inferences (dream/reflexion/consolidation outputs, typically `semantic` or `procedural` type, `created_by='consolidation'`, `trust_tier='agent'`). They **must** cite evidence:

```sql
CREATE TABLE belief_support(
  belief_id   INTEGER NOT NULL REFERENCES memories(id),
  evidence_id INTEGER NOT NULL REFERENCES memories(id),
  stance      TEXT NOT NULL DEFAULT 'supports' CHECK(stance IN ('supports','contradicts')),
  weight      REAL NOT NULL DEFAULT 1.0,
  PRIMARY KEY(belief_id, evidence_id)
);
```

Write-path rule (enforced in `dal.py`): inserting a `lane='belief'` row without ≥1 `belief_support` row in the same transaction is an error. When an evidence row is invalidated, the same transaction runs `UPDATE memories SET needs_review=1 WHERE id IN (SELECT belief_id FROM belief_support WHERE evidence_id=:old)` — consolidation re-derives or retires the belief. This is what stops the agent's own inferences laundering into facts.

### 1.3 Edges (bi-temporal, 5-type closed vocabulary)

```sql
CREATE TABLE edges(
  id           INTEGER PRIMARY KEY,
  src_id       INTEGER NOT NULL REFERENCES memories(id),
  dst_id       INTEGER NOT NULL REFERENCES memories(id),
  rel          TEXT NOT NULL
                 CHECK(rel IN ('led_to','supersedes','depends_on','conflicts_with','related_to')),
  confidence   REAL NOT NULL DEFAULT 1.0,
  created_by   TEXT NOT NULL,               -- adjudicator|consolidation|tool
  valid_from   TEXT NOT NULL,
  valid_to     TEXT,                        -- edges get temporal validity (Daem0n lacked this)
  recorded_at  TEXT NOT NULL,
  UNIQUE(src_id, dst_id, rel, valid_from)
);
CREATE INDEX idx_edges_src ON edges(src_id) WHERE valid_to IS NULL;
CREATE INDEX idx_edges_dst ON edges(dst_id) WHERE valid_to IS NULL;
```

No `mentions` in this table — entity mentions are the bipartite table below (Daem0n conflated them; the PPR builder wants them separate anyway).

### 1.4 Entities: global IDs, per-scope mentions

```sql
CREATE TABLE entities(
  id            INTEGER PRIMARY KEY,
  name          TEXT NOT NULL,              -- original casing preserved
  norm_name     TEXT NOT NULL,              -- type-specific normalization (lifted from Daem0n resolver)
  entity_type   TEXT NOT NULL,              -- person|handle|org|project|tool|code_symbol|file|concept|place
  canonical_id  INTEGER REFERENCES entities(id),  -- alias merge: NULL = is canonical
  mention_count INTEGER NOT NULL DEFAULT 0, -- cheap salience (Daem0n idea, kept)
  first_seen    TEXT NOT NULL,
  last_seen     TEXT NOT NULL,
  UNIQUE(entity_type, norm_name)            -- NOTE: no project_path in the key — global identity
);
CREATE TABLE entity_mentions(
  entity_id  INTEGER NOT NULL REFERENCES entities(id),
  mem_id     INTEGER NOT NULL REFERENCES memories(id),
  scope_user TEXT, scope_project TEXT,      -- denormalized for scoped PPR seeding
  cnt        INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY(entity_id, mem_id)
);
CREATE INDEX idx_mentions_mem ON entity_mentions(mem_id);
```

Alias resolution: exact `norm_name` fast path at write; embedding-similarity merge candidates (≥0.90, same type) are *queued* for the consolidation pass to confirm — never auto-merged inline. Merges set `canonical_id`; readers resolve through it (one self-join).

### 1.5 FTS5 — contentless + trigram side-index

All writes flow through one DAL writer, so app-managed contentless FTS (no triggers, no stored duplication):

```sql
CREATE VIRTUAL TABLE mem_fts USING fts5(
  content, summary, tags, symbols,
  content='', contentless_delete=1,
  tokenize = "porter unicode61 remove_diacritics 2 tokenchars '_'"
);
-- query-time weighting: symbols > tags > summary > content
-- rank = bm25(mem_fts, 1.0, 2.0, 3.0, 4.0)

CREATE VIRTUAL TABLE mem_fts_tri USING fts5(
  names,                                    -- entity names, @handles, file paths, symbols ONLY (not content — 3x bloat)
  content='', contentless_delete=1,
  tokenize = 'trigram'
);
```

`tokenchars '_'` keeps `get_user_by_id` and `snake_case` whole; the `symbols` column is populated at write time by the **lifted Daem0n tokenizer** (`embed/tokenizer_code.py` = `extract_code_symbols` + `tokenize` from `similarity.py`, verbatim semantics: backticks, CamelCase, lowerCamel, snake_case, SCREAMING_SNAKE, `.method`, both original and lowercase, plus split-piece variants so partial matches hit). Query side runs the same tokenizer to expand identifiers. The trigram index serves typo/handle recall (Telegram usernames, file paths) and is only consulted when the query contains identifier-ish or @-shaped tokens, or when the porter leg returns <5 hits.

### 1.6 Vectors — sqlite-vec int8[256] with metadata pre-filter

```sql
CREATE VIRTUAL TABLE mem_vec USING vec0(
  mem_id        INTEGER PRIMARY KEY,        -- = memories.id
  embedding     int8[256] distance_metric=cosine,
  memory_type   TEXT partition key,         -- physical partition: type filters are free
  scope_user    TEXT,                       -- metadata columns: pre-filter in KNN WHERE
  scope_project TEXT,
  is_current    INTEGER                     -- 1 = valid_to IS NULL AND status='active'
);
-- KNN: WHERE embedding MATCH :q AND k = 50 AND is_current = 1 AND (scope_user IS NULL OR scope_user = :u)
```

Embeddings: EmbeddingGemma-300M ONNX int8, matryoshka-truncated 256d, L2-normalized then symmetrically quantized to int8 (scale 127). Prompts (EmbeddingGemma's trained task prefixes): documents `title: none | text: {content}`, queries `task: search result | query: {q}`. Fallback tier (Termux/512MB): potion-retrieval-32M via model2vec — same 256d schema slot, `embedded_with` records which; janitor re-embeds potion rows with Gemma when a stronger tier comes online. Brute-force scan; 100k int8×256d ≈ 26MB ≈ single-digit ms. No ANN until sqlite-vec DiskANN is stable AND corpus >500k (documented escape hatch: LanceDB >1M).

Vault rows, tombstones, and quarantined rows are **never** inserted into `mem_fts`/`mem_vec` — exclusion is structural, not a query filter.

### 1.7 Vault

```sql
CREATE TABLE vault_items(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
  ciphertext BLOB NOT NULL,                 -- Fernet; key from ~/.hermes/.env BRAIN_VAULT_KEY
  created_at TEXT NOT NULL, rotated_at TEXT, meta TEXT
);
```

Separate table, encrypted at rest, zero index membership, no retrieval path can reach it. Access only via explicit `brain_vault_get(name)` tool call (and the tool response is marked do-not-store so the extractor's quarantine rule skips it). Requires the optional `cryptography` extra; without it, vault tools simply don't register.

### 1.8 Operational tables

```sql
-- Memobase-style flush buffer (also the incognito choke point, §2.6)
CREATE TABLE ingest_buffer(
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, session_id TEXT, platform TEXT,
  kind TEXT NOT NULL CHECK(kind IN ('turn','pre_compress','memory_write','delegation','session_end')),
  payload TEXT NOT NULL,                    -- JSON
  tokens INTEGER NOT NULL DEFAULT 0,
  processed INTEGER NOT NULL DEFAULT 0
);

-- retrieval logging: design now, tune later (convex fusion weights, §3.6)
CREATE TABLE retrieval_log(
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, session_id TEXT,
  query_hash TEXT NOT NULL, mem_id INTEGER NOT NULL,
  r_fts INTEGER, r_vec INTEGER, r_ppr INTEGER,          -- per-leg ranks (NULL = absent from leg)
  s_rrf REAL, s_rerank REAL, s_final REAL,
  injected INTEGER NOT NULL DEFAULT 0,
  feedback INTEGER NOT NULL DEFAULT 0                    -- +1/-1, backfilled by attribution
);

CREATE TABLE quarantine_queue(              -- SpAIware review queue
  mem_id INTEGER PRIMARY KEY REFERENCES memories(id),
  reason TEXT NOT NULL, requested_promotion TEXT,        -- e.g. 'core','procedural'
  created_at TEXT NOT NULL, resolved TEXT                -- approved|rejected|NULL
);

CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
-- keys: schema_version, graph_generation (PPR cache invalidation),
--       mem_generation (cross-process staleness, Daem0n trigger pattern), model tags
```

---

## 2. Write path

```
sync_turn / on_pre_compress / on_memory_write / on_delegation / on_session_end
        │  (< 5ms: archive append + 1 INSERT into ingest_buffer; NO LLM, NO embedding)
        ▼
[incognito gate]  →  archive/*.jsonl.gz  +  ingest_buffer
        ▼  flush triggers: ≥4k buffered tokens │ session_end │ 10-min idle │ pre_compress(immediate)
FLUSH (background worker, serialized — Hermes already guarantees single worker):
  Phase 1  batched LLM extraction (1 call)  → candidates{type, lane=evidence, content, entities,
           relations, happened_at?, time_sensitive?, salience, emotional_salience, scope hints}
  Phase 2  local: embed all candidates → kNN top-10 each → surprise, hash check, neighbor set
  Phase 3  batched LLM adjudication (1 call, all candidates + their top-3 neighbors)
           → op ∈ ADD | UPDATE(target) | INVALIDATE(target) | NOOP  (+ conflict verdicts)
  Phase 4  transactional apply: memories + mem_fts + mem_fts_tri + mem_vec + edges +
           entity_mentions + belief_support in ONE transaction
```

**2.1 Capture (all five sources, no per-message LLM).** `sync_turn(user, assistant, messages)` buffers the turn pair (+ tool-call summaries from the transcript tail); `on_pre_compress` buffers doomed messages with an immediate-flush flag (they're about to leave the model's context — last chance to extract); `on_memory_write` bypasses extraction: the model already decided — written directly as `created_by='memory_tool'`, `trust_tier='agent'`, still passing Phase 2–4 (dedup/adjudication apply to everyone); `on_delegation` buffers subagent results (`kind='delegation'`, provenance `source_author='subagent:<name>'`, trust `tool`); `on_session_end` force-flushes the session's lane. Cron/subagent sessions never reach us (`skip_memory=True`) — by design.

**2.2 Batched extraction (Memobase).** One LLM call per flush over the buffered spans (via Hermes MCP sampling or auxiliary provider; stronger model welcome, it's background). Prompt contract: emit **only** memory-worthy items (explicitly instructed to skip phatic chatter — this is forgetting lever #1's LLM half), each with type, entities (typed), typed relations between candidates (`led_to`/`depends_on` finally get auto-populated — Daem0n never did), `happened_at` when content references past events (→ `valid_from` backfill), `time_sensitive` (→ sets `half_life_days`), salience 0–1, emotional_salience 0–1. Floor tier (no LLM available): deterministic capture only — `on_memory_write` mirrors, explicit `brain_remember` calls, and regex entity extraction; no synthetic candidates.

**2.3 Adjudication (Mem0 ops, deterministic first).** Before the Phase-3 LLM call: exact `content_hash` match → NOOP + `verification_count++`; top-1 cosine ≥0.95 same type/scope → UPDATE (merge counters, union entities). Only the ambiguous band (0.80–0.95) and candidates with outcome/polarity tension reach the LLM. UPDATE is always append-a-version: new row, old row gets `valid_to=now`, `superseded_by=new`, plus a `supersedes` edge. INVALIDATE never deletes — `valid_to` + `invalidated_by`. On floor tier, the ambiguous band defaults to ADD with a `related_to` edge to the neighbor (safe: dedup happens later at consolidation).

**2.4 Surprise gating (Daem0n's math, finally wired).** `surprise = mean cosine distance to k=5 NN` computed from the Phase-2 kNN results already in hand (zero extra cost — exactly the fix the post-mortem prescribed). Effects: `surprise < 0.15` and salience < 0.4 → NOOP/merge (routine); `surprise ≥ 0.6` → stored with `half_life_days` doubled (resists decay), `meta.consolidation_priority=1` (sleep-time pass looks at it first), and initial `importance = 0.5 + surprise/2`.

**2.5 Conflict detection (non-blocking, supersede-don't-delete).** Among kNN neighbors: (a) polarity/negation mismatch (Daem0n `detect_conflict` heuristic, kept — cheap and verified in source), (b) similar content with `outcome='failed'` neighbor ("similar approach failed before" surfacing), (c) Phase-3 LLM verdict on the ambiguous band. Result: `conflicts_with` edge + warning attached to the flush report (and to the `brain_remember` tool response when synchronous). Auto-invalidation of the loser happens **only** when the LLM verdict is confident AND `trust(new) ≥ trust(old)`; otherwise both stay valid with the conflict edge, and readers render a conflict marker. This wires Daem0n's dead `check_and_invalidate_contradictions` intent into the live path with a trust guard.

**2.6 Instruction-shaped quarantine (SpAIware defense).** Deterministic classifier at Phase 4 (regex tier: second-person imperatives, "always/never/you must/ignore previous/from now on", tool-invocation shapes; plus extractor's own `is_instruction` flag). If `instruction_shaped=1` AND `trust_tier ∉ {owner, agent}`: the row may exist as retrievable *evidence* (rendered with a provenance banner "unverified instruction from @X on discord"), but any attempt to write it as `memory_type ∈ {core, procedural}` or `pinned=1` is diverted to `status='quarantined'` + `quarantine_queue` row. Promotion requires explicit confirmation: `brain_confirm(uid)` tool (the model asks the *owner* in-chat) or `hermes brain review` CLI. Quarantined rows are not in FTS/vec (structural exclusion, §1.6), so they cannot even be retrieved until resolved. This is the wall between "someone in a group chat said it" and "it silently enters every future system prompt."

**2.7 Incognito / TTL (provable bypass).** The incognito check is at the *single* capture entry point (`ingest/buffer.py::capture()` — every hook funnels through it): if the session is incognito (provider config, `hermes brain incognito on`, or platform tag), `capture()` returns before archive append and before `ingest_buffer` insert. Nothing downstream can extract what was never buffered — that's the proof, and a unit test asserts zero rows and zero archive bytes after an incognito session. TTL lane: `scope_session`-tagged rows get `ttl_at`; the janitor hard-DELETEs expired rows from all tables (with vec/FTS delete) — the only routine hard delete in the system.

---

## 3. Read path

Entry points: `prefetch(query)` (8s cap; budget default 1200 tokens, hard 2000, injected verbatim into the per-turn `<memory-context>` fence — the cache-safe channel), and `brain_recall` tool (model-driven, supports `as_of`, scope override, type filter).

**3.1 Query prep (<5ms + embed).** Code-symbol tokenizer expands identifiers for the FTS leg; identifier/@-handle detection arms the trigram leg; query embedded with the query prefix (LRU cache, 64 entries — Daem0n encoded every query twice; we don't). Scope filter resolved: default = `(scope_user = current OR scope_user IS NULL)` + platform/project axes analogous; explicit `scope=global|user|platform|project|session` overrides.

**3.2 Candidate legs (all local, no LLM):**
- **FTS leg:** `mem_fts` MATCH with `bm25(mem_fts, 1.0, 2.0, 3.0, 4.0)`, join `memories` for scope/status predicates, top 50. Trigram leg (when armed) top 20, treated as a fourth RRF list.
- **Vector leg:** `mem_vec` KNN k=50 with `is_current=1` + scope metadata pre-filter (partition key on `memory_type` when the caller filters type).
- **Graph leg (HippoRAG-2 PPR):** query entities matched against `entities` (exact norm + trigram); seeds = matched entities weighted by 1/log(1+mention_count) (specific entities pull harder) **+** the top-3 vector hits as passage seeds (HippoRAG 2's dense-passage integration). Personalized PageRank over the bipartite adjacency (entity↔memory mentions, weight=cnt; memory↔memory edges weighted by type: `led_to`/`depends_on` 1.0, `related_to` 0.5, `supersedes` 0.3, `conflicts_with` 0.2; only `valid_to IS NULL` edges), damping 0.5, ≤20 power iterations, ε=1e-6. Hand-rolled CSR + numpy (~60 lines — deliberately **no scipy**: one heavy dep saved on every tier, trivially fast at 20–50k nodes). Adjacency cached in RAM, invalidated by `meta.graph_generation` (incremental append for new rows; full rebuild only in janitor). Top 50 memory nodes. Leg skipped when no entity matches — zero cost for entity-free queries.

**3.3 Fusion:** RRF k=60 (Daem0n's `fusion.py` semantics, lifted, actually called this time) over 2–4 lists → top 50.

**3.4 Rerank (degradable):** mxbai-edge-colbert-v0-32M ONNX int8, MaxSim over query×doc tokens, docs truncated to 256 tokens, computed on the fly for ≤50 candidates (~50–150ms CPU). Fallbacks in order: answerai-colbert-small-v1 → skip stage entirely (floor tier / model missing / >2s elapsed budget pressure). Rerank never *adds* candidates — pure precision stage — so skipping degrades gracefully.

**3.5 Score modulation (multiplicative on min-max-normalized rerank/RRF score):**
- **Decay (read-time only, time-sensitive rows only):** `w = max(2^(-age_days/half_life_days), 0.3)` with age from `valid_from`; `half_life_days IS NULL` → 1.0. Verified Daem0n formula, floor kept.
- **Outcome:** `failed` → ×1.5 (failure knowledge is warning knowledge — the crown jewel, kept); `partial` ×1.0; `worked` ×1.1; scaled by `outcome_confidence`.
- **Feedback (ACE counters):** `× (1 + 0.3·(helpful−harmful)/(helpful+harmful+5))` — Laplace-smoothed so young memories aren't whipsawed.
- **Pinned/active-context:** pinned rows ×1.5 **and** guaranteed inclusion (up to 10 pinned slots before ranked fill — Daem0n's active-context cap, kept; failures auto-pin at consolidation).
- Belief-lane rows with `needs_review=1` → ×0.5 + rendered with a "under revision" marker.

**3.6 Packing + injection:** top-down fill to token budget, verbatim, one header line each: `[semantic · 2026-05-02 · telegram/@dan · worked · 0.91]`. **No runtime compression, ever** — density is the write path's job (LLMLingua-class compression, if used at all, runs at consolidation). Conflict pairs surfacing together get a joint marker instead of silent dedup.

**3.7 Logging + the self-tuning hook (log now, fit later):** every candidate that survives RRF is logged to `retrieval_log` (async, same writer). Feedback attribution: injected uids are remembered per turn; next `sync_turn` + nightly state.db mining (READ-ONLY attach of `turn_outcomes`: outcome, retries, reaction feedback) backfill `feedback` and bump `helpful_count`/`harmful_count`. Once ≥500 labeled rows exist, `jobs/fit_weights.py` fits a logistic model over `(r_fts, r_vec, r_ppr, s_rerank)` → convex leg weights (ACM TOIS result: tuned convex beats RRF), written to `meta`; `scoring.py` reads them, RRF remains the cold-start default and the fallback whenever fit quality is poor. The tuner is a v2 job; the *log schema* is v1 and normative.

**3.8 `recall_count`/`last_recalled_at`** bump on injection (not on candidacy) — usage signal for the forgetting engine.

---

## 4. Forgetting engine — the four levers

Doctrine: **the HOT index is prunable because the archive is not.** Raw episodic log (archive JSONL + superseded version rows) is append-only and kept forever; `hermes brain reindex --from-archive` can rebuild the entire hot DB. Regret-free forgetting = index management, not data loss.

**Lever 1 — importance gating at write (§2.2, §2.4):** extractor emits only memory-worthy items; deterministic guards (min length, hash dup, surprise<0.15 ∧ salience<0.4 → merge/NOOP). Nothing else is ever blocked at write — cheap storage, expensive attention.

**Lever 2 — entity/duplicate merge at write (§2.3, §1.4):** hash NOOP, ≥0.95 cosine merge with counter accumulation (`verification_count` makes repeated observation *strengthen* one row instead of spawning near-dups); entity alias merges via `canonical_id`, confirmed at consolidation.

**Lever 3 — decay only on time-sensitive claims (§3.5):** `half_life_days` is set *only* by the extractor's time-sensitivity judgment (episodic default 30d; semantic only when flagged). Decay is a read-time multiplier with a 0.3 floor — it reorders attention, it never destroys. Permanent-ish types (`semantic` untimed, `procedural`, `core`) have `half_life_days=NULL` and are untouched.

**Lever 4 — eviction only for compliance/user request:** `brain_forget(uid|filter)` tool + `hermes brain forget` CLI: hard delete from all hot tables, tombstone marker retained (uid + hash, so re-ingest of the same content is refused), `--and-archive` flag additionally redacts matching archive lines (GDPR-grade). TTL/incognito expiry (§2.7) is the only other hard-delete path.

**Consolidation-time value scoring (runs in the sleep-time shift; scoring function lives here in `forget/value.py`):**

```
V = w1·reliability(trust_tier, verification_count, outcome_confidence)
  + w2·user_relevance(scope match, entity overlap with user-profile entities)
  + w3·task_utility(outcome grade, helpful−harmful)
  + w4·usage(log(1+recall_count), last_recalled_at recency)
  + w5·recency(valid_from)
  + w6·surprise
  + w7·emotional_salience
```

Hand-set weights at ship (w3, w1 heaviest); fittable from `retrieval_log`+`turn_outcomes` by the same `fit_weights.py` harness (the 2026 evidence: learned weighting retains 0.770 of gold evidence vs 0.368 for recency-only). **Tiered demotion**, protection ladder enforced (pinned > has-outcome > recall_count≥5 > worked — Daem0n's ladder, kept), always via the version chain:

1. **full → summary:** low-V clusters get an LLM-written dense summary row (`lane='belief'` if synthetic across rows, citing originals via `belief_support`; `supersedes` edges); originals → `status='summarized'`, dropped from `mem_fts`/`mem_vec` (still SQL-queryable, still in archive).
2. **summary → tombstone:** content nulled, envelope + entity links + hash retained; exits all indexes.
3. **never → gone** (except Lever 4).

Janitor (companion daemon via `gateway:startup` hook, or Hermes cron): TTL expiry, quarantine-queue nagging, `retrieval_log` rollup (>90d → aggregates), belief `needs_review` sweep, adjacency rebuild, `PRAGMA optimize`/vacuum, potion→Gemma re-embed backfill.

---

## 5. Module layout, dependencies, performance envelope

### 5.1 Repo layout (every module with its call site — nothing without one)

```
Hermes-Brain/
├── pyproject.toml                    # extras: [embed], [rerank], [vault]
├── hermes_brain/
│   ├── provider.py        # MemoryProvider impl; called by Hermes memory_manager
│   ├── mcp_server.py      # MCP surface (locked decision #1); calls retrieve/ + ingest/
│   ├── cli.py             # 'hermes brain' status|review|forget|incognito|reindex|fit
│   ├── config.py          # tier detection, budgets; get_config_schema/save_config
│   ├── envelope.py        # Memory dataclass, type policies, trust tiers, ULID
│   ├── db/
│   │   ├── connection.py  # PRAGMAs, sqlite-vec load, migrations, meta generations
│   │   ├── schema.sql
│   │   └── dal.py         # ALL SQL; single writer queue; belief-support invariant
│   ├── ingest/
│   │   ├── buffer.py      # capture() incognito gate; archive append   ← all 5 hooks
│   │   ├── extractor.py   # Phase-1 batched LLM extraction             ← buffer flush
│   │   ├── adjudicator.py # Phase 2-4: kNN, surprise, Mem0 ops,
│   │   │                  #   conflicts, quarantine                    ← extractor
│   │   └── entities.py    # regex tier + resolution + alias queue      ← adjudicator
│   ├── retrieve/
│   │   ├── query.py       # orchestrator                               ← prefetch, brain_recall, mcp
│   │   ├── fts.py         # porter + trigram legs                      ← query.py
│   │   ├── vec.py         # KNN leg + kNN for adjudicator              ← query.py, adjudicator
│   │   ├── ppr.py         # numpy CSR PPR + adjacency cache            ← query.py
│   │   ├── rerank.py      # colbert ONNX, degradable                   ← query.py
│   │   └── scoring.py     # RRF, modulation, packing, retrieval_log    ← query.py
│   ├── embed/
│   │   ├── encoder.py     # Gemma-ONNX / potion tiers, int8 quantize   ← adjudicator, query.py
│   │   └── tokenizer_code.py  # lifted Daem0n similarity.py            ← fts.py, ingest
│   ├── forget/
│   │   ├── value.py       # V-score                                    ← consolidation shift, janitor
│   │   ├── demote.py      # summarize/tombstone transitions            ← consolidation shift
│   │   └── evict.py       # brain_forget, TTL, archive redaction       ← tools, cli, janitor
│   ├── jobs/
│   │   ├── janitor.py     # daemon/cron entry
│   │   └── fit_weights.py # convex fusion + V-weights fitting (v2)     ← cli, janitor
│   ├── llm.py             # MCP-sampling / auxiliary-provider client, spend caps  ← extractor, adjudicator, demote
│   └── tools.py           # get_tool_schemas/handle_tool_call: brain_remember,
│                          #   brain_recall, brain_outcome, brain_pin, brain_forget,
│                          #   brain_confirm, brain_vault_get/set (7+2 tools, that's all)
└── tests/                 # incl. test_incognito_zero_capture, test_cache_stability
```

Explicitly **not built** (Daem0n dead-code lessons): no retrieval router/query classifier (one path, always), no communities/Leiden (PPR replaces it), no compression module in the read path, no BM25-in-RAM (FTS5 only), no per-project federation (scopes replace it).

### 5.2 Dependencies (exact, tiered)

| Tier | Deps | Justification |
|---|---|---|
| **Floor** (Termux, 512MB, always installed) | stdlib (`sqlite3`, `gzip`, `hashlib`, `json`, `struct`), `sqlite-vec` (~0.5MB wheel, aarch64 incl.), `numpy` (PPR CSR, int8 quantize; already a transitive dep of everything in this space) | Whole engine minus neural models works: FTS+static-vec+PPR+RRF, deterministic write path |
| + `[embed]` floor encoder | `model2vec`, `safetensors`, `tokenizers` (potion-retrieval-32M, ~30MB, no ONNX runtime needed) | Semantic leg on 512MB |
| **Standard** ($5 VPS 1–2GB, default) | + `onnxruntime` (~40MB), `tokenizers` | EmbeddingGemma-300M q8 (<200MB resident) + mxbai-edge-colbert-32M q8 (~40MB) |
| `[vault]` | `cryptography` (Fernet) | Vault only; tools unregistered without it |
| optional | `pysqlite3-binary` (Linux, old-SQLite escape hatch) | FTS5 contentless_delete needs ≥3.43 |

**Not** dependencies: torch, sentence-transformers, scipy, networkx, igraph/leidenalg, rank_bm25, qdrant-client, SQLAlchemy/aiosqlite (plain `sqlite3` + one writer thread — the provider contract already serializes writes), any server. All model loads lazy (`lazy_deps` precedent); models downloaded to `<hermes_home>/brain/models/` on first use with checksums.

### 5.3 Performance envelope

| Metric | Floor (Termux/512MB) | Standard (1–2GB VPS) |
|---|---|---|
| Resident RAM (brain total) | < 120MB (potion 30 + numpy + sqlite + adjacency) | < 500MB (Gemma 200 + colbert 60 + rest) |
| `sync_turn` capture | < 5ms (1 INSERT + gz append) | same |
| `prefetch` p50 / p99 | < 150ms / 500ms (no rerank) | < 450ms / 2s (embed ~40–80ms, legs ~10ms, PPR ~5ms, rerank ~100ms) — vs 8s cap |
| Flush cost | 0 LLM calls (deterministic) | 2 LLM calls per ~4k buffered tokens |
| DB size @ 100k memories | ~130MB (text+FTS ~100, vec int8 ~26, trigram names ~5) + archive | same |
| Vector scan @ 100k int8·256d | < 10ms | < 5ms |
| PPR @ 50k nodes / 200k edges | < 10ms (numpy CSR, cached adjacency) | same |

Concurrency: WAL + `busy_timeout=30000`; all provider writes through the in-process writer queue; the MCP server process writes through its own short transactions (WAL handles cross-process); `meta.mem_generation`/`graph_generation` counters give cheap cross-process cache staleness (Daem0n's trigger pattern, kept). Windows-safe: no fcntl anywhere, SQLite locking only. Cache-stability: `system_prompt_block()` renders core blocks once at `initialize()` and returns the identical bytes all session; all core edits take effect next session; `prefetch` output rides the ephemeral fence only — the two-lane contract that fixes hermes-agent #13631.

---

**Key deviations from the brief's ancestors, made deliberately:** (1) versions live in `memories` itself rather than a sidecar — `as_of` beats even "single-JOIN" (it's zero-join); (2) no scipy — 60 lines of numpy CSR PPR is faster to ship, lighter on Termux, and trivially testable; (3) `mentions` excluded from the 5-type edge table (bipartite `entity_mentions` is the PPR substrate and keeps the edge vocabulary purely memory↔memory); (4) quarantine is structural (no index membership) rather than a rank penalty — instruction-shaped untrusted content cannot be retrieved at all until reviewed, which is the only defense that actually closes the SpAIware loop.
