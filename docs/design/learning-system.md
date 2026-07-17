I have verified all the mechanics I need against source. Now I'll compose the design document as my final deliverable.

# Hermes-Brain Learning System Design
**Design area: sleep-time compute, consolidation, strategy/skill learning, self-tuning**
Status: implementation-ready specification. Verified against `hermes-agent` and `Daem0n-MCP` source on 2026-07-16.

---

## 0. Stance and grounding

Everything in this document obeys one test: **the loop must provably close**. A learning loop closes when (signal → artifact → injection → measured effect on the same signal) is traceable in data. Daem0n's post-mortem (research `daem0n:learning-loop`) is the cautionary baseline: its Reflexion LangGraph was 100% dead code, its "debates" compared two floats, and its dream insights were formulaic template sentences that polluted recall. What actually worked in Daem0n — and what we keep — is small: the `worked/outcome` pair with retrieval boost, the IdleDreamScheduler's cooperative preemption, dream provenance tags, per-strategy cooldowns, and the PendingOutcomeResolver's conservative decision tree with `dry_run=True` default.

**Loops we refuse to build:** no LangGraph or any graph-framework dependency; no multi-agent "debate"; no regex claim extraction; no sandbox code-verification of claims; no heuristics dressed as cognition (if no LLM is available, the brain queues work — it never fakes insights); no autonomous edits to Hermes core prompts or MEMORY.md/USER.md; no cadence-based refresh of the stable prompt block (issue #13631 — the cache invariant is sacred).

**Verified contract facts this design depends on** (re-checked in source):
- `turn_outcomes` (state.db, `hermes_state.py:815`): PK `(session_id, turn_id)`, columns `outcome` (8-value enum from `turn_outcome.py`), `outcome_reason`, `turn_exit_reason`, `api_calls`, `tool_iterations`, `retry_count`, `guardrail_halt`, `cost_usd_delta`, token deltas, `skills_loaded` (JSON array), `model`, `feedback_kind/value/source/at/event_id`. Helper queries already exist: `get_outcome_trends()`, `get_skill_outcome_counts(days)` (`hermes_state.py:2618`, uses `json_each(skills_loaded)`).
- Reaction feedback lands via `annotate_turn_feedback()`; `reflection_triggers.py` defines the negative-reaction set and the four trigger kinds (`failure`, `correction`, `tool_failure_streak`, `reaction`) with per-`(session, kind)` cooldowns.
- Skill telemetry: `tools/skill_usage.py` already maintains `.usage.json` per skill with `use_count`, `helped/hurt/neutral`, `outcome_counts`, `outcome_cost_usd`, `state` (active/stale/archived), `pinned`, and `bump_outcome()` fed from turn outcomes. The curator (`agent/curator.py`) is inactivity-triggered, only touches agent-created skills, never deletes (archive-only), pinned bypasses everything, LLM consolidation pass is opt-in (`DEFAULT_CONSOLIDATE = False`).
- `auxiliary_client.call_llm(task=..., messages=..., ...)` resolves provider/model per `auxiliary.{task}` config — the sanctioned precedent for background LLM calls (background_review, curator both use it).
- MCP sampling is real and configured per server: `sampling: {enabled, model, max_tokens_cap}` with rate limiting (`tools/mcp_tool.py:52`, `SamplingHandler` at line 1114).
- `background_review.py` enforces a thread-scoped tool whitelist and, when routed to a cheaper aux model, replays a *digest* instead of the full transcript — a pattern we reuse for shift prompts.
- Cron/subagent sessions run `skip_memory=True` (`cron/scheduler.py:3251`) — the brain never sees them as provider sessions; subagent results arrive via `on_delegation`.

---

## 1. The Night Shift: sleep-time consolidation

### 1.1 Scheduler — `BrainScheduler`

Evolves Daem0n's `IdleDreamScheduler` (monotonic-clock idle detection, cooperative `user_active` event, 1s poll — all kept) with three changes: it is **cross-process**, **dual-trigger** (idle micro-shifts + nightly full shift), and **budgeted in dollars, not vibes**.

**Process model.** The brain may be live in several processes at once (gateway provider, CLI provider, MCP server for Claude Code). All share `brain.db` (single WAL SQLite). Exactly one process runs consolidation at a time, enforced by a DB lease (mirrors Hermes's `session_leases` pattern — no fcntl, works on native Windows):

```sql
CREATE TABLE brain_lease (
  name TEXT PRIMARY KEY,          -- 'consolidator'
  holder TEXT NOT NULL,           -- pid@host:mode
  acquired_at REAL, expires_at REAL   -- 120s TTL, renewed every 30s
);
CREATE TABLE activity (k TEXT PRIMARY KEY, v REAL);  -- 'last_turn_at' updated by every process
```

**Idle signal.** MemoryProvider never sees tool calls mid-turn, so idle = "no `sync_turn`/`on_turn_start`/`prefetch` across ALL processes for `idle_minutes` (default 20)". Every hook touches `activity.last_turn_at` (cheap UPDATE, already on the write worker). In MCP mode, every tool call updates it. Preemption check = `last_turn_at > shift_start_activity`, tested between every work unit and before every LLM call; on preemption the strategy marks `interrupted=true` and returns (Daem0n's exact yield discipline, translated from asyncio to the brain's worker thread).

**Two shift types.**

| | Idle micro-shift | Nightly full shift |
|---|---|---|
| Trigger | 20 min global idle | wall-clock window (default 03:00–05:00 local, ±20 min jitter), plus idle gate |
| Strategies | (a) buffer flush, (h) precompute refresh only | full pipeline (a)–(h) + probes |
| Model tier | extract tier only (cheap) | extract + consolidate tiers |
| Budget | ≤ $0.03 / micro-shift | `brain.night_budget_usd` (default $0.50) |
| LLM offline | skips LLM steps, queues work | same — queue drains next night with LLM |

There is deliberately **no dependency on Hermes cron** for scheduling (cron runs full agent sessions with `skip_memory=True` — wrong tool). The brain ships `hermes brain shift [--strategy X] [--dry-run]` as a plugin CLI subcommand so users/ops can force a shift or wire OS cron on headless boxes; the gateway-mode daemon variant can be launched by a `gateway:startup` hook.

**The shift transaction.** Every shift gets `shift_id`. All writes during a shift are **staged**, not live:

```sql
-- every memory/artifact row carries:
status TEXT CHECK(status IN ('staged','active','reverted','superseded')),
origin TEXT,        -- 'agent' | 'user' | 'shift:<strategy>'
shift_id TEXT, evidence_ids TEXT /*json*/, epistemic TEXT
    CHECK(epistemic IN ('observation','inference','belief')),
model TEXT, prompt_version TEXT, trust_tier INTEGER;
CREATE TABLE shift_writes (shift_id TEXT, table_name TEXT, row_id INTEGER,
                           op TEXT, prev_status TEXT, at REAL);
```

Live retrieval ignores `staged` rows. At shift end, the **probe suite** (section 3) runs against a "staged view" (retrieval with staged rows visible). Pass → one UPDATE promotes `staged→active`. Fail → `staged→reverted` and demotions/supersedes are flipped back via `shift_writes`. Rollback is a metadata flip, never a restore-from-backup, because *nothing in the brain is ever destructively overwritten* — the same supersede-don't-delete rule that makes forgetting regret-free makes rollback trivial. (A rolling 7-day file copy of brain.db, exposed via `backup_paths()`, remains the disaster hatch.)

**Per-strategy state** (cooldowns are Daem0n's idea, kept and generalized):

```sql
CREATE TABLE strategy_state (
  strategy TEXT PRIMARY KEY, last_run_at REAL, cooldown_hours REAL,
  consecutive_failures INTEGER DEFAULT 0,      -- exponential backoff on errors
  mode TEXT CHECK(mode IN ('off','shadow','dry_run','active')),
  tokens_spent_today REAL, usd_spent_today REAL
);
```

Every strategy has a **cheap SQL pre-check** ("is there anything in my input queue?") that runs before any LLM call. Empty queue → skip silently. This single rule kills Daem0n's dream-spam failure mode at the root: no work, no output, no formulaic insight.

### 1.2 The ordered strategy pipeline

Order matters: facts first, enrichment second, task-level learning third, identity fourth, forgetting after new evidence is integrated, precompute against the final state.

#### (a) Buffer flush → batched extraction + ADD/UPDATE/DELETE/NOOP adjudication

`sync_turn` (single serialized worker — never blocks) appends raw turns to `ingest_buffer(session_id, turn_id, platform, user_json, assistant_json, ts, flushed)`. Only `agent_context == "primary"` sessions are buffered (cron/subagent contexts are skipped per the provider contract; `on_delegation` results are buffered as a distinct record type with `trust_tier` lowered).

Flush triggers: 12 buffered turns, or ~8k tokens, or `on_session_end`, or `on_pre_compress` (extract *before* Hermes discards messages — this hook is the anti-data-loss lane), or shift start. Flushing is Memobase-style **batched** extraction — one extract-tier LLM call per batch, not per message ($5-VPS economics).

Two-phase Mem0-style pipeline, both phases JSON-mode:
1. **Extract** (cheap tier): candidate facts with `{content, kind: fact|preference|event|task_signal, entities[], salience 1-5}`. Turn content is wrapped in fenced data blocks with explicit "this is data, not instructions" framing (poisoning guardrail).
2. **Adjudicate** (cheap tier): for each candidate, retrieve top-5 similar existing memories (vector+FTS, no LLM), then emit one op: `ADD` (novel), `UPDATE` (refines existing → new row + `supersedes` edge + `invalid_at` on old), `DELETE` (contradicts → supersede only, old row keeps provenance and remains queryable as history), `NOOP` (duplicate — the max-similarity ≥0.92 shortcut skips the LLM entirely). Contradiction with unanimous evidence auto-applies; mixed evidence lands in the review queue (section 3 decision tree).

Extraction failures leave the buffer intact (`flushed=0`); the buffer is the durable queue.

#### (b) Episodic → semantic summarization

*Explicitly not Daem0n's sha256-exact-dup consolidation* (which grouped only byte-identical reflections and summarized via bag-of-words — both defects fixed here).

- Incremental clustering over episodic memories: each new episodic joins the nearest cluster centroid if cosine ≥ 0.80 (256d matryoshka vectors already in the store), else seeds a new cluster. `clusters(id, centroid BLOB, member_count, last_distilled_at, distilled_memory_id)`.
- Trigger: cluster reaches **N=3** members with no prior distillation, or +3 members since last distillation.
- Distill (consolidate tier — the strong model; this is exactly where asymmetric compute pays): input is the member episodics + their outcomes; output is one semantic pattern memory — a *lesson*, ≤120 words, must cite which members support it and must pass the specificity gate (names ≥1 concrete entity; `actionable: true` self-check field required, else discarded unpersisted).
- Wiring: pattern gets `supersedes` edges to members; members are **demoted** (value-score penalty in (g)) not deleted; pattern is `epistemic='inference'`.

#### (c) A-MEM memory evolution

For each memory written since the last shift (cap 30/night, cooldown 24h): fetch k=8 nearest neighbors; one consolidate-tier call proposes `{links: [{target_id, relation, why}], neighbor_updates: [{id, new_context_note, new_tags}] (max 3)}`. Neighbor updates are **delta ops** on the neighbor's `context_note` field (an LLM-curated annotation lane), never rewrites of original content — original text is immutable. This is the constructive counterpart to contradiction detection: the graph's understanding of old memories refines as new ones arrive.

#### (d) ReasoningBank strategy distillation

The heart of "true learning." Signal source is **Hermes's own ledger**, mined read-only from state.db:

1. **Episode assembly** (no LLM): group `turn_outcomes` rows by session into task episodes — an episode closes on outcome in `{verified, failed, blocked}` or session end. Join `messages` for the transcript slice. Attach `feedback_kind/value` (a `thumbs_down` reaction overrides a `completed_unverified` label to failure; a positive reaction upgrades it).
2. **Judge** (cheap tier, only for ambiguous episodes — `partial`, `completed_unverified` with no feedback): "did this accomplish the user's goal? success/failure/unclear + one-line reason". `unclear` → no distillation (never guess).
3. **Distill** (consolidate tier): from **successes AND failures**, titled items:

```sql
CREATE TABLE strategy_items (
  id TEXT PRIMARY KEY,            -- 'S-0042'
  kind TEXT CHECK(kind IN ('strategy','guardrail')),
  title TEXT,                     -- <=80 chars
  description TEXT, actionable_insight TEXT,
  scope_tags TEXT,                -- 'coding','telegram','user:<id>',...
  helpful INTEGER DEFAULT 0, harmful INTEGER DEFAULT 0,
  status TEXT, embedding BLOB, evidence_ids TEXT, superseded_by TEXT
);
```
Failures produce `guardrail` items ("when X, do NOT Y because Z — seen in episode E"); successes produce `strategy` items. Contrastive bonus: when a cluster contains a failure followed by a success on a similar task, the distiller gets both trajectories and is instructed to state *what differed* — the highest-value distillation in the ReasoningBank ablations.
4. **Inject**: `prefetch(query)` retrieves top-3 strategy items by (similarity × Wilson-score of helpful/harmful) into the per-turn ephemeral `<memory-context>` fence (cache-safe lane). Every injection is logged: `injection_ledger(session_id, turn_id, item_id, item_kind, at)`.
5. **Close the loop** (this is the edge Daem0n never had): next shift joins `injection_ledger` to `turn_outcomes` — item injected into a turn that ended `verified` → `helpful++`; `failed/blocked` or negative reaction → `harmful++`. Items with `harmful > helpful` and n≥5 are auto-deprecated (delta op, supersede lineage). Attribution is noisy per-turn but unbiased in aggregate; counters gate ranking, not existence.

#### (e) Memento case bank

Full task episodes stored whole, successes and failures alike:

```sql
CREATE TABLE cases (
  id TEXT PRIMARY KEY, task_summary TEXT, plan_sketch TEXT,
  outcome TEXT, cost_usd REAL, tool_iterations INTEGER,
  platform TEXT, session_id TEXT, turn_span TEXT,
  embedding BLOB, created_at REAL, distilled INTEGER DEFAULT 0
);
```
Written at episode close from (d)'s assembly (summary+plan by the cheap tier, one call, batched). Retrieval: when the prefetch classifier scores the incoming query as "task-like" (imperative verb + concrete object — a 20-line heuristic, no LLM), top-2 cases (cos ≥ 0.75) are rendered into the prefetch fence as `Similar past task (FAILED|SUCCEEDED): <summary> → <what happened>`. Failed cases are *more* valuable than successes here. The case bank doubles as the replay/eval set for every self-modification gate (section 3) — one artifact, two jobs.

#### (f) Profile refresh — user model + agent self-model

Two **bounded core blocks** (default 1,500 bytes user / 1,000 bytes self), each an ACE-itemized artifact: entries with IDs, `helpful/harmful` counters, delta ops only. Nightly (cooldown 24h, and only if new evidence exists): consolidate-tier call sees current items + the week's relevant extracted facts, emits `add/edit/deprecate` ops. A renderer packs active items into the byte budget by score; overflow items stay in the store, not in the block.

**Cache invariant compliance:** the rendered block feeds `system_prompt_block()` frozen **at session start only**. Mid-session refreshes never touch it; a changed profile is picked up by the next session. The prefetch fence may carry "profile updated: <delta>" ephemerally if material.

The self-model is populated from measured data only — outcome trends per task type, per-model quirks observed in `turn_outcomes.model`, known failure modes with episode citations. No aspirational prose.

#### (g) Forgetting pass — value scoring and demotion

Runs late in the shift, after new evidence has landed. Score per memory (arithmetic, no LLM):

`value = w_r·recency(Ebbinghaus, half-life per type) + w_u·usage(retrievals that reached context) + w_o·outcome_link(contributed to verified episode via injection/citation) + w_s·surprise_at_write(kNN distance, computed at ingest — Daem0n's orphaned idea, finally wired) + w_p·provenance_trust + w_pin·pinned`

Weights start hand-set; every demotion decision is logged with features (`retention_log`), and once ≥500 decisions have 30-day "was it ever needed again" labels, a logistic fit replaces the hand weights (learned weighting retained 0.770 vs 0.368 for recency-only in the cited study; query-similarity is provably the wrong signal and is not a feature).

Demotion tiers, never deletion: **active → summary-only** (content compressed to 1-2 lines by cheap tier, original moved to cold table, still restorable) **→ tombstone** (id + one-line + provenance, excluded from default retrieval, findable by explicit search). A tombstone that gets restored counts as a **regret event**; regret rate >2%/month auto-raises the demotion threshold. Hard deletion exists only as an explicit user command (compliance path).

#### (h) Anticipatory pre-computation

Per-user query patterns are highly predictable (same platforms, same projects, same morning questions). Mine session-opening messages per (user, platform, weekday/time bucket) from state.db; cheap-tier call generates 5-10 likely next queries; run the **full retrieval pipeline** (RRF + rerank) now; cache rendered context packs:

```sql
CREATE TABLE prefetch_cache (query_hash TEXT PRIMARY KEY, user_id TEXT,
  rendered TEXT, built_at REAL, ttl_hours REAL, hits INTEGER);
```
`prefetch()` checks the cache (cos ≥ 0.90 against cached query embeddings) before running live retrieval — sub-100ms hits against the 8s budget, and the packs were built by the strong pipeline at zero interactive latency. Cache invalidated by any shift that changes underlying memories. Hit-rate is tracked; buckets with <20% hit rate stop being precomputed (the loop that tunes the loop).

### 1.3 ACE delta discipline (shared machinery, enforced in code)

Every LLM-curated artifact — strategy items, profile blocks, skill drafts, prompt sections, context notes — lives as **itemized entries with IDs and helpful/harmful counters**. The single write API is:

```python
apply_ops(artifact_id, ops: list[AddOp|EditOp|DeprecateOp], provenance) -> OpResult
```

There is deliberately **no** `replace_artifact(text)` function anywhere in the codebase. Monolithic rewrite is impossible by construction, not by convention — this is the context-collapse prevention the ACE and "Do Self-Evolving Agents Forget?" papers independently demand. Compaction happens only as `deprecate + add` with supersede lineage, so any collapse is reversible.

---

## 2. Skill-forge pipeline

Feeds Hermes's existing agentskills.io system; never replaces it. The brain is the *detector and drafter*; Hermes's loader, curator, and `.usage.json` telemetry remain the runtime authority.

**Detection (nightly, no LLM until triggered).** Over the case bank: find clusters of ≥3 similar task cases (cos ≥ 0.78) with ≥2 successes — or the gold pattern, 1 failure followed by ≥2 successes (a *learned fix*). Guard: embed existing SKILL.md descriptions; if a live skill covers the cluster (cos ≥ 0.80) — or `turn_outcomes.skills_loaded` shows a skill already loaded in those episodes — route to **revision** instead of creation.

**Drafting (consolidate tier).** One call per candidate (max 1/night):
- SKILL.md at **two abstraction levels**: `## When and why` (class-level workflow, transfers across tools) and `## Procedure` (concrete steps with pitfalls from the failure case).
- `references/exemplar-N.md`: few-shot exemplars distilled from the *real* episodes (secret-scrubbed via Hermes's `redact` patterns).
- Frontmatter: `name`, `description` (≤60 chars — HARDLINE standard, validated by a linter before write, reject-and-retry once on violation), `created_by: hermes-brain`, `evidence_count`, `evidence_sessions`, `success_rate_at_creation`.

**Lifecycle:**
```
candidate → draft (written to ~/.hermes/skills/.brain-drafts/<name>/ — NOT loaded by Hermes)
        → shadow (draft's description injected via prefetch as a hint when matching tasks appear;
                   brain logs whether matching episodes succeed — "would this have helped")
        → approval: `hermes brain skills approve <name>` (default: human required;
                   config brain.skills.auto_approve=true promotes after 3 shadow-period successes, 0 failures)
        → active (moved into ~/.hermes/skills/<category>/<name>/, mark_agent_created() called
                   so the curator governs it; takes effect next session, per Hermes contract)
```

**Outcome tracking and degradation.** Per-skill health is read from what already exists: `get_skill_outcome_counts(days=30)` in state.db plus `.usage.json` `helped/hurt` (fed by `bump_outcome`). Degradation rule: `hurt/(helped+hurt) > 0.4` with n≥10 → the brain drafts a **revision** as delta ops against SKILL.md sections (logged diff in `prompt_revisions`, same discipline as everything else), re-entering shadow for the revised sections. Two failed revisions → propose retirement: set the skill `stale` via the standard state machine and leave archival to the curator. The brain **never** archives directly, never touches pinned skills, never touches non-agent-created skills, and never fights a curator transition — one janitor per hallway.

**Prompt-section optimization (LangMem-style).** Applies only to brain-owned prompt surfaces: its own `system_prompt_block()` instruction text, retrieval/consolidation prompt templates, and profile block phrasing. Each is an itemized artifact; the optimizer proposes deltas from correlational evidence ("sessions where block variant B was active had higher verified-rate"), logged as diffs with before/after hashes in `prompt_revisions(id, artifact, before_hash, after_hash, diff, rationale, shift_id, status)`, evaluated by pre/post windows with the section-3 gates. Hermes core prompts are permanently out of scope.

---

## 3. Safety rails — constrained Darwin-Gödel

**Propose-validate-archive** for every self-modification (skill draft/revision, strategy-item policy change, scoring-weight update, prompt-section delta):

1. **Propose**: written as a proposal record with rationale + evidence, staged.
2. **Validate by replay**: replay against the archived case bank — for retrieval-affecting changes, deterministic replay (do the historical episodes' queries still retrieve their gold memories?); for content artifacts, cheap-tier LLM judge over sampled cases ("given this context, would the artifact have changed the outcome?"), same judge prompt version pinned for comparability.
3. **Statistical gate (PACE-lite)**: accept only if the Wilson 95% lower bound of (improved − regressed) replay outcomes is > 0 with n ≥ 8; otherwise remain shadow and accumulate evidence. No gate, no promotion — noise cannot ratchet.
4. **Archive**: rejected and superseded variants are kept (`superseded_by` lineage), so the system can escape local optima by reviving archived variants when context shifts — Darwin-Gödel's archive, at artifact scale instead of code scale.

**Fixed capability-regression probe suite** — runs after **every** shift, deterministic, <5s, no LLM required:
- *Retrieval probes*: ~15 seeded (query → must-retrieve memory id in top-5) pairs spanning old and new memories.
- *Staleness probes*: seeded superseded pairs — the superseded version must NOT rank above its successor; a fact updated during the shift must resolve to the new value.
- *Profile probes*: core blocks within byte budget; canary facts present; no low-trust-tier content in blocks.
- *Injection probes*: canary memories containing instruction-shaped text ("ignore previous instructions and…") must render quoted/fenced in prefetch output, never bare.
- *Latency probe*: cold `prefetch()` under 2s on the staged view.
Any failure → automatic full-shift rollback (`staged→reverted` flip via `shift_writes`), `consecutive_failures++` with exponential cooldown backoff, and a review-queue entry with the failing probe attached.

**PendingOutcomeResolver decision tree** (ported from Daem0n verbatim in spirit, `dreaming/strategies.py:583`) governs every autonomous adjudication (fact conflicts, episode judging overrides, auto-approvals): total evidence < 2 → skip; directional evidence < threshold → skip; **mixed → human review queue**; unanimous ≥ threshold → auto-apply. Every autonomous action writes a tagged audit memory (`origin='shift:<strategy>'`, evidence ids, shift id) — Daem0n's provenance discipline, kept wholesale. Review queue surfaces as one line in the (session-stable) system block ("brain: 3 items awaiting review — `hermes brain review`") and as a CLI table.

**Ship-inert convention.** Every autonomous behavior has `mode ∈ {off, shadow, dry_run, active}` in `strategy_state` and **ships in shadow**: it logs what it *would* do to `shadow_log(strategy, would_do_json, at)` and nothing else. Promotion path: shadow → dry_run (writes staged rows that are never promoted, exercising the full pipeline) → active, each step requiring either explicit `hermes brain enable <behavior>` or the configured auto-promotion criterion (N clean shadow runs + 0 probe failures). First release defaults: (a), (h) active; (b)–(f) dry_run; (g) demotions and skill auto-approval shadow.

**Anti-pollution rules** (the direct answer to Daem0n's dream spam):
- Shift outputs are `epistemic='belief'|'inference'` rows that **must cite evidence_ids**; retrieval renders them with an `(inferred — N sources)` marker and ranks them below observations at equal score. Beliefs never launder into facts; a belief is promoted to observation only when the user confirms it.
- Specificity gate: any distilled output must reference ≥1 concrete entity and carry `actionable: true` from its own generation, else it is dropped before persistence.
- Novelty gate: max cosine ≥ 0.92 against existing memories → NOOP, no write.
- Empty-queue silence: strategies with no input produce no output — no "session summary" memories for sessions where nothing happened.

**Memory-poisoning guardrails**: `trust_tier` on every row (owner-direct=0, delegated/subagent=1, group-chat-other-user=2, web/tool content=3); tier ≥2 content can never enter profile blocks, strategy items, or skill drafts without passing the review queue; all consolidation prompts wrap memory content in fenced blocks labeled as data; the injection probe suite is the tripwire.

---

## 4. LLM access plumbing

| Mode | Primary path | Fallback | Notes |
|---|---|---|---|
| In-process MemoryProvider | `agent.auxiliary_client.call_llm(task="brain_extract"/"brain_consolidate", ...)` — brain registers `auxiliary.brain_extract` and `auxiliary.brain_consolidate` slots in config.yaml via its `get_config_schema()`/`save_config()` | Own key: `BRAIN_LLM_API_KEY` + `brain.llm.base_url` (OpenAI-compatible), minimal internal client | Sanctioned precedent: background_review and curator both resolve aux slots this way; inherits Hermes provider auth, rate guards, api_mode handling |
| MCP server (Claude Code etc.) | MCP sampling `sampling/createMessage` — serviced by the connected client's provider; per-server config already supports `model` override and `max_tokens_cap` + rpm caps | Own key (same as above) | Sampling exists only while a client is connected; disconnected sleep-time work drains the queue via own key, or defers to the next connection / to the in-process sibling holding the consolidator lease |

**Model tiering (asymmetric compute, locked decision #2).** Two logical tiers, resolved per mode:
- `extract` tier — cheap/fast (flash-class), temperature 0, JSON mode: buffer extraction, adjudication, episode judging, case summaries, query generation, replay judging.
- `consolidate` tier — strong model, ideally stronger than the chat model: pattern distillation (b), memory evolution (c), strategy distillation (d), profile refresh (f), skill drafting (2). In MCP-sampling mode the tier maps to the per-server `sampling.model` override; in-process it maps to the two auxiliary slots. Config: `brain.models.{extract,consolidate}` with `"auto"` defaulting to the aux-slot resolution.

**Cost budgeting.** Every call is metered into `llm_ledger(shift_id, strategy, model, tokens_in, tokens_out, est_usd, at)` (prices from Hermes `usage_pricing` when resolvable, else token caps only). Enforcement: per-shift cap (`night_budget_usd`, default $0.50), per-day cap (default $1.50), per-strategy sub-caps; pipeline order is the priority order, so when the budget hits mid-shift, later strategies (profile, precompute) yield to earlier ones (extraction, distillation). Concurrency is 1 (the shift is sequential by design — simpler preemption, kinder to rpm caps). Digest-not-transcript rule from `background_review.py`: shift prompts always operate on compact digests, never full transcript replays — the brain never has a warm cache to exploit, so cold-written tokens are minimized structurally.

**Offline/degraded (no LLM at all — Termux tier, keys revoked, provider down).** The brain stays **honest**: it keeps doing everything mechanical — buffering, FTS/vector indexing, clustering (queued for later distillation), forgetting arithmetic, injection-ledger counter updates, precompute from *literal* past queries, probes — and enqueues every LLM-dependent unit into `work_queue(kind, payload, enqueued_at, attempts)`, drained FIFO when a tier returns. It never substitutes heuristic text for cognitive output. Retrieval, the profile blocks last rendered, strategy injection, and case retrieval all continue to work untouched — degraded mode loses *learning velocity*, never *memory*.

---

## 5. The learning flywheel, end to end

### One concrete week

**Monday 15:40 (CLI).** Task: "add async retry logic to the uploader." Agent flails on pytest-asyncio fixtures; turn ends `failed` (`turn_outcomes` row: outcome=failed, retry_count=2, tool_iterations=19, skills_loaded=["python-testing"]). `reflection_triggers` fires `failure` (Hermes's own background review runs as usual — orthogonal). Brain: `sync_turn` buffers turns; episode E-101 closes as failure.

**Monday 03:10 Tue (night shift #1, $0.31 spent).** (a) flushes the buffer: 3 facts extracted (project uses pytest-asyncio 0.23, uploader lives in `net/upload.py`, user prefers tenacity for retries) — 2 ADD, 1 UPDATE. (d) assembles E-101, judge not needed (outcome=failed is unambiguous), distiller writes guardrail **G-17**: *"pytest-asyncio ≥0.23: fixtures need explicit loop_scope — event_loop fixture override is removed. Seen failing in E-101."* — staged, `epistemic=belief`, evidence=[E-101]. (e) writes case C-88 (FAILED, cost $0.42, 19 iterations). Probes pass; staged→active.

**Tuesday 10:15 (CLI).** "Write tests for the downloader's retry path." `prefetch` fires: G-17 matches (cos 0.83) → injected into the `<memory-context>` fence along with case C-88 ("Similar past task FAILED: async retry tests — fixture loop_scope issue"). Injection ledger logs (session, turn, G-17, C-88). Agent starts with the loop_scope fix in view; turn ends `verified`, 6 tool iterations, no retries.

**Tuesday night shift #2.** Injection ledger ⋈ turn_outcomes: G-17 `helpful=1`, C-88 credited. Case C-89 (SUCCESS) written; cluster K-12 now holds C-88(fail)+C-89(success) — the **contrastive pair**: distiller revises G-17 via an `edit` op, adding the exact working fixture snippet. (h) notices "async test" queries trending for this user and precomputes the pack.

**Thursday 14:00.** Third similar task (mocking async httpx in tests) — prefetch cache hit (87ms). Turn `verified`. Night shift #3: cluster K-12 hits the skill-forge trigger (3 similar cases, 2 successes after a failure, no covering skill above 0.80 — existing "python-testing" skill scores 0.61). Draft skill `testing/async-python-tests` written to `.brain-drafts/`: two abstraction levels, exemplar from C-89's real transcript (redacted), `description: "Async pytest: fixtures, loop scope, httpx mocking"` (52 chars — passes hardline lint), `created_by: hermes-brain`, `evidence_count: 3`. Enters shadow.

**Friday.** `hermes brain review` shows the draft; user approves. Skill moves into `~/.hermes/skills/testing/async-python-tests/`, `mark_agent_created()` called — live next session, curator now governs it.

**Following week.** `skills_loaded` shows the new skill in 4 turns, all verified; `get_skill_outcome_counts` and `.usage.json` `helped=4`. Night shift deprecates G-17 with `superseded_by_skill=testing/async-python-tests` (the knowledge graduated from strategy bank to procedural memory — the strategy fence slot frees up for newer lessons). The Monday-class failure is now structurally impossible to repeat unnoticed: guardrail → contrastive strategy → skill, each step measured.

**What measurably improved:** verified-rate on the "async testing" cluster 0% → 100%; mean tool_iterations 19 → 6; cost per verified task $0.42 → $0.11; time-to-first-useful-context 8s live retrieval → 87ms cached.

### Metrics — the definition of "it's learning"

**Primary longitudinal metric (internal, from Hermes's own logs — the metric nobody else measures):**

> **TSI (Task-Success Improvement):** over 28-day rolling windows, the verified-rate (`verified` + `completed_unverified`-with-positive-feedback, over all non-cancelled episodes) computed **per task cluster**, compared window-over-window — reported overall and, critically, on the **redemption subset**: clusters containing ≥1 prior failure. Redemption-subset TSI is the direct measurement of "learns from mistakes." Secondary series from the same rows: retry_count, tool_iterations, cost_usd per verified episode, negative-reaction rate. All from read-only state.db queries (`get_outcome_trends` + episode assembly); rendered by `hermes brain report`.

Guard against self-deception: TSI is computed on *matched clusters* (same task type before/after), never raw aggregate (which task-mix drift can inflate); the report always prints n per cluster and withholds judgment under n=5.

**External sanity checks (regression detection, not leaderboard chasing):** a `bench/` adapter runs **LongMemEval-V2** and **BEAM** subsets against the brain's retrieval API quarterly and before any release — pass criterion is "no regression vs. our own last run," per the research consensus that LoCoMo-class absolute numbers are configuration-gamed.

**Continuous health probes (every shift, section 3):** retrieval@5 on seeded pairs, **staleness probes** (superseded facts must never win — the metric that catches slow poisoning), injection canaries, prefetch latency, profile-budget compliance. Plus the two flywheel-specific gauges: strategy-item Wilson scores (is the strategy bank net-positive?) and forgetting regret rate (<2%/month or the demotion threshold auto-tightens).

---

## Appendix: module map and config surface

```
hermes_brain/learning/
  scheduler.py        # BrainScheduler: lease, idle+nightly triggers, budgets, preemption
  shift.py            # shift transaction, staged writes, promote/rollback
  strategies/         # a_flush.py, b_distill.py, c_evolve.py, d_reasoningbank.py,
                      # e_casebank.py, f_profile.py, g_forgetting.py, h_precompute.py
  ace.py              # apply_ops() — the only write path for curated artifacts
  skill_forge.py      # detection, drafting, shadow, approval, revision
  probes.py           # fixed regression suite + rollback hook
  llm.py              # tiered client: aux-slot | MCP-sampling | own-key; llm_ledger
  outcomes.py         # read-only state.db mining: episodes, injection-ledger join, TSI
  safety.py           # decision tree, review queue, trust tiers, shadow_log
```

Config keys (all non-secret in config.yaml via `get_config_schema`; secrets in `.env`): `brain.night_window`, `brain.night_budget_usd` (0.50), `brain.day_budget_usd` (1.50), `brain.idle_minutes` (20), `brain.models.{extract,consolidate}` ("auto"), `brain.skills.auto_approve` (false), `brain.strategy_modes.{a..h}`, `brain.profile.{user_bytes,self_bytes}`, `brain.forgetting.{demote_threshold,regret_ceiling}`.

Key source files verified for this design: `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\dreaming\scheduler.py`, `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\dreaming\strategies.py`, `C:\Users\dasbl\hermes-agent\hermes_state.py` (turn_outcomes DDL :815, ledger API :2396–2660), `C:\Users\dasbl\hermes-agent\agent\turn_outcome.py`, `C:\Users\dasbl\hermes-agent\agent\reflection_triggers.py`, `C:\Users\dasbl\hermes-agent\agent\background_review.py`, `C:\Users\dasbl\hermes-agent\agent\curator.py`, `C:\Users\dasbl\hermes-agent\tools\skill_usage.py` (bump_outcome :666), `C:\Users\dasbl\hermes-agent\agent\auxiliary_client.py` (call_llm :6767), `C:\Users\dasbl\hermes-agent\tools\mcp_tool.py` (SamplingHandler :1114), `C:\Users\dasbl\hermes-agent\agent\memory_provider.py`.
