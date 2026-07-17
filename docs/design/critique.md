All load-bearing claims are now verified against source (`memory_provider.py` ABC, `memory_manager.py` timeouts, the plugin loader, `turn_outcome.py`, `turn_outcomes` DDL, `memory_setup.py`, cron `skip_memory`, and the research JSON for the RAM claims). Here is the adversarial review.

---

# Hermes-Brain — Adversarial Design Review (pre-build punch list)

Verified against: `hermes-agent` source (2026-07-16 checkout), `Daem0n-MCP` v6.6.6, research JSON (`research:infra` extracted). Legend: **[BLOCKER]** = will not work / three docs cannot coexist as written; **[MAJOR]** = will work but wrongly, expensively, or insecurely; **[MINOR]** = fix cheaply before it fossilizes.

## A. Cross-document contradictions (attack vector 7 — these gate everything else)

1. **[BLOCKER] Three incompatible schemas for the same tables.** A's `memories.status ∈ {active,summarized,tombstone,quarantined,expired}` vs B's `status ∈ {staged,active,reverted,superseded}` — same column, disjoint enums, and B's `staged/reverted` is a *write-visibility* axis while A's is a *lifecycle* axis. Also: A `trust_tier TEXT ('owner','agent','known_user','tool','untrusted')` vs B `trust_tier INTEGER 0–3` vs C `operator/agent/external-mcp/untrusted-platform-peer`; A `lane ∈ {evidence,belief}` vs B `epistemic ∈ {observation,inference,belief}`; A `ingest_buffer(kind,payload JSON,processed)` vs B `ingest_buffer(session_id,turn_id,user_json,assistant_json,flushed)`. **Fix:** declare Design A's schema normative; fold B in as *additions*: a separate `shift_id TEXT` + `live INTEGER` (or `stage` column) orthogonal to `status`; one trust enum (A's TEXT, with C's `external-mcp` mapped to `tool` or added); merge lane/epistemic into one 3-value `epistemic` column ('observation','inference','belief') since A's evidence/belief is a strict subset. One `ingest_buffer` DDL. Write the unified `schema.sql` before any code.

2. **[BLOCKER] Three different consolidation process models.** A: janitor as "companion daemon via `gateway:startup` hook, or Hermes cron" + LLM flush *inside the provider* at 4k tokens. B: resident `BrainScheduler` thread, idle micro-shifts, a 03:00–05:00 wall-clock nightly window, DB `brain_lease` (120s TTL). C: **no resident anything**, short-lived `hermes brain dream` processes, cron `no_agent` script + opportunistic `--if-due` spawn, `O_EXCL` lockfile. B's nightly window structurally never fires for CLI-only users (no process alive at 3am — C's F12 point) and dies on laptops that sleep; A's daemon contradicts C's headline decision. **Fix:** C's process/trigger model is normative (it's the only one that covers every deployment shape). B's shift machinery — staging, probes, budgets, strategy pipeline (a)–(g), cheap SQL pre-checks — runs *inside* the dream process. Delete "micro-shift" as a concept; the in-process worker does only what C's §1.1 allows (embed, salience, bounded idle sweep). Pick **one** mutual-exclusion mechanism — recommend the DB lease (it lives in the shared brain.db, is transactional, and needs no PID-liveness heuristics); delete the lockfile.

3. **[BLOCKER] Extraction timing: three answers, one is contract-violating.** A flushes with 2 LLM calls in the provider at ≥4k tokens / 10-min idle / **on_session_end force-flush**; B runs it as shift strategy (a); C forbids provider-side LLM except a bounded idle sweep. A's session-end flush collides head-on with the verified 5s shutdown drain (`_SYNC_DRAIN_TIMEOUT_S = 5.0`, `memory_manager.py:46,1169`) — the LLM call gets killed mid-flight, exactly the Hindsight-298s class of bug C designed around. **Fix:** capture-to-durable-buffer always (<5ms, A §2.1 is right); extraction *only* out-of-band (dream/sweep) or opportunistically in-process when idle ≥60s and bounded; `on_session_end` writes a marker row only. A's Phase 1–4 pipeline is kept as the *algorithm*; C decides *when it runs*.

4. **[BLOCKER] The loop-closing join key does not exist.** B's flagship edge — `injection_ledger ⋈ turn_outcomes` — assumes the provider knows `turn_id`. Verified: `prefetch(query, *, session_id)` and `sync_turn(..., session_id, messages)` carry **no turn id**; `turn_outcomes` PK is `(session_id, turn_id TEXT)` where `turn_id = agent._current_turn_id`, a runtime value never passed to providers. `on_turn_start` gives `turn_number` (the user-turn counter), which is not `turn_id`. **Fix (concrete):** log injections as `(session_id, user_turn_count, ts, sha256(user_content), item_ids)`; the nightly miner resolves `turn_id` via state.db's `messages` table (which carries `turn_id`, verified `hermes_state.py:875`) by matching session + content hash/timestamp window. Specify this resolution step in `outcomes.py` or the "provably closes" claim is hollow. Accept that some turns won't resolve; counters are aggregate signals anyway.

5. **[MAJOR] Lane-1 ownership is triplicated.** A: `core` rows with `core_block/core_rank` rendered at `initialize()`. B: two ACE profile artifacts with byte budgets (1500/1000 bytes). C: `lane1_snapshot` materialized table with four sections (warnings/open loops/standing facts/stats) and a 1200-token budget. Three renderers = three different system prompts. **Fix:** one renderer (C's section layout, C's snapshot-table mechanism — materialized by the dream, rendered once at `initialize()`), whose *inputs* are A's core rows, and B's profile items simply *are* core rows (ACE-itemized, `core_block='user_profile'|'self'`). One token budget knob.

6. **[MAJOR] Two tool surfaces.** A ships 8–9 tools (`brain_pin`, `brain_forget`, `brain_confirm`, `brain_vault_get/set`, …); C ships exactly 5 (`brain_manage` folds pin/forget/incognito; adds `memories`; has no confirm/vault). Params for the shared tools also differ (A `brain_recall(as_of, type)` vs C `brain_recall(depth, kind, project, limit, id)`). **Fix:** C's 5-tool surface is normative (minus item 8's cut of `memories` and item 7's vault verdict); merge A's `as_of` into `brain_recall` as one optional param; A's write-path semantics (dedup report, quarantine diversion) live behind C's schemas.

7. **[MAJOR] Vault leaks secrets into every transcript.** `brain_vault_get` returns plaintext into the model's context → it is persisted verbatim in Hermes `state.db.messages`, in platform history (Telegram/Discord servers!), and in any transcript-derived artifact. A's "do-not-store" flag only guards the brain's *own* extractor — the one store it controls is the only place the secret *won't* land. **Fix:** cut vault from v1 entirely (secrets already live in `.env`). If ever revived, it needs a non-context delivery mechanism (e.g. env injection into tool execution), which doesn't exist in the provider contract.

8. **[MAJOR] `memories` virtual-file tool is a large fiddly surface with unproven payoff.** `str_replace`/`insert` against materialized views: which memory row does a line-level edit in a rendered digest target? The mapping is ambiguous by construction, and the trained-behavior transfer claim is weakened by the tool being named `memories` (the trained tool is `memory`, which is reserved — F6, verified `memory_manager.py:429`). **Fix:** defer past P3; if kept, ship read-only (`view` only) first.

9. **[MAJOR] Duplicated subsystems across B and A.** (i) state.db mining appears three times (A §3.7 nightly mining + `fit_weights`, B `outcomes.py`, C `dream/mine_state.py`) — build one module. (ii) B's `injection_ledger` duplicates A's `retrieval_log(injected=1)` — same rows, two tables; merge. (iii) B's `strategy_items` and `cases` tables carry their own raw `embedding BLOB`s outside `mem_vec` — a second, Python-brute-forced vector path. **Fix:** strategy items and cases are `memories` rows (`memory_type='procedural'` / a `case` episodic subtype) indexed in `mem_vec`, with thin side-tables for counters only — or at minimum their vectors go into `mem_vec`. (iv) B's clusters table vs A's consolidation clustering — same thing, specify once.

10. **[MAJOR] B's A-MEM strategy (c) writes to a column that doesn't exist.** `context_note` delta-ops on neighbors have no home in A's schema, and A explicitly bans `meta` for features. **Fix:** cut strategy (c) from v1 (see cut list) or add the column to the unified schema now.

11. **[MINOR] Numeric knob disagreements.** `busy_timeout` 30000 (A) vs 5000 (C); lane-2 budget 1200/2000 (A) vs 600 (C); flush trigger 4k tokens (A) vs 12 turns/8k (B); ID grammar ULID (A) vs `S-0042` (B) vs `w-0412` (C). **Fix:** one constants file; recommend busy_timeout 5000 (30s makes C's latency probes meaningless), lane2 default 600, one kind-prefixed short-id display scheme mapped to ULIDs.

12. **[MINOR] CLI verb collision:** `hermes brain review` means the quarantine queue in A and the proposal/review queue in B. Merge into one review queue with typed entries; one argparse tree unifying A/B/C verb lists.

## B. Hermes contract violations (attack vector 1)

13. **[MAJOR] skip_memory blind spot is subtler than A claims.** A: "cron/subagent sessions never reach us — by design." Verified ABC docstring: `initialize()` kwargs *may include* `agent_context ∈ {"primary","subagent","cron","flush"}` and "providers should skip writes for non-primary contexts" — i.e. the provider **can** be initialized in non-primary contexts; skipping is the provider's job. B has the guard; A doesn't. **Fix:** the `agent_context != "primary"` write-guard goes in `capture()` next to the incognito gate, with a test.

14. **[MAJOR] Quarantine semantics contradict themselves inside Design A.** §2.6: instruction-shaped untrusted rows "may exist as retrievable *evidence* (rendered with a provenance banner)"; §1.6 + "key deviation (4)": quarantined rows "cannot even be retrieved at all until resolved." Both can't be true, and C has a third rule (never in lanes, tool-recall flagged). Note also: a "banner" on injected lane-2 content is weak — the injection payload still enters model context. **Fix:** adopt C's rule as normative: instruction-shaped + untrusted → *never* rendered into lane 1 or lane 2; retrievable only via explicit `brain_recall` with a `⚠ quarantined` flag; promotion to core/procedural/pinned blocked pending review. A's structural index-exclusion then applies only to the *promotion-blocked quarantine queue* rows, not to all instruction-shaped evidence — say so explicitly.

15. **[MAJOR] `brain_confirm` is a prompt-injectable promotion path.** The model can be induced (by the very content being quarantined, or by a group-chat peer) to call `brain_confirm(uid)` without genuine owner consent — the tool trusts the model to have asked. **Fix:** remove the tool; quarantine promotion is CLI-only (`hermes brain review`). If in-chat confirmation is wanted later, it must verify the *author* of the confirming message is the owner (gateway `user_id`), not trust the model.

16. **[MINOR] A ignores `on_pre_compress`'s return contract.** Verified: the hook returns a string included in the compression summary prompt (`memory_provider.py:220–230`). A treats it as capture-only; C returns ≤300 tokens of insights. Adopt C's behavior — it's free signal preservation.

17. **[MINOR] Cache economics are honored everywhere — one residual trap.** All three docs respect the two-lane contract (lane 1 frozen at `initialize()`, kept byte-identical across `on_session_switch(reset=False)`/compression; lane 2 ephemeral). The trap: any lane-1 render that reads *live* tables at initialize (A renders "core blocks at initialize") is fine for caching but means two concurrent processes (gateway + CLI, same profile) show different blocks — acceptable, but C's `lane1_snapshot` (render from a materialized table the dream maintains) is the deterministic version; make it normative. Keep C's golden test (50 turns + compression → one distinct string) in CI from P1, as specified.

## C. SQLite multi-process reality (attack vector 2)

18. **[MAJOR] Dream-end `wal_checkpoint(TRUNCATE)` can stall live sessions.** TRUNCATE blocks until no readers; after a long dream the WAL is large and an interactive prefetch/sync can stall behind it. **Fix:** `PASSIVE` (or `RESTART`) by default; `TRUNCATE` only when the activity table shows global idle.

19. **[MAJOR] Extension loading is not universal.** python.org macOS builds of 3.11/3.12 compile `sqlite3` without `enable_load_extension` (fixed in 3.13) → sqlite-vec cannot load at all on a supported platform tier. A's "ships with CPython ≥3.12 everywhere" claim for `contentless_delete` (needs SQLite ≥3.43/3.45) is also false on Linux where CPython links the system SQLite (Debian 11 = 3.34). **Fix:** runtime capability probe at first open (extension loading? FTS5 contentless_delete? trigram?), automatic degrade to FTS-content-table/FTS-only modes, and a `doctor` finding naming the exact remedy (`pysqlite3-binary`, or Python ≥3.13). Don't gate install on it.

20. **[MINOR] Live-file backup corruption.** B's "rolling 7-day file copy of brain.db" — a plain copy of a WAL database under write is not consistent. Use `VACUUM INTO` or the SQLite backup API from the dream process.

21. **[MINOR] Writer discipline is fine, but say what it actually is.** The real design is *WAL multi-writer with short transactions* (provider worker, dream ≤50-row transactions, MCP short transactions), not "single-writer" (A's principle statement). Rename the principle so nobody "enforces" a false invariant later; keep `meta.mem_generation/graph_generation` polling — and specify the cadence (check on each prefetch: one indexed SELECT, cheap).

## D. Dead code & feature halo — Daem0n's disease (attack vector 3)

22. **[MAJOR] A's write path makes "remember everything" false between flushes and on the floor tier.** A indexes only *extracted* memories; raw turns live in the archive/buffer, unsearchable. On the floor tier (no LLM), nothing conversational ever becomes searchable except explicit writes — the product promise fails exactly on Termux. C's P1 indexes verbatim episodic turns in FTS immediately. **Fix:** raw episodic lane is always FTS-indexed at capture (embeddings optional/deferred); the extraction lane builds distilled memories on top. This also reconciles A's `sync_turn <5ms` with C's `p95 <200ms` (FTS insert of a turn is well under 200ms; embedding rides the worker).

23. **[MAJOR] Phase-1 minimality: C's P1 is genuinely minimal — protect it.** The audit of guaranteed call sites turns up these components with **no call site in a core flow at v1**: convex-weight fitting (`fit_weights.py` — correctly marked v2; keep only the log schema), B's prompt-section optimizer (correlational evidence, weakest gate in the doc), B's agent self-model block, B's precompute strategy (h) (a cache with its own invalidation lifecycle to beat a sub-500ms live path), the trigram side-index (useful, but nothing breaks without it), `emotional_salience` (a column the extractor fills and one scoring term reads — fine to keep as a column, don't build UX on it). See cut list.

24. **[MINOR] B ships (h) precompute "active" at first release while (b)–(f) are dry_run — inverted priorities.** The cache serves nothing until retrieval is trusted. Ship (h) shadow or cut.

## E. Learning-loop integrity (attack vector 5)

25. **[MAJOR] Shift-wide rollback nukes legitimate extraction.** B: any probe failure flips the *entire* shift `staged→reverted` — including strategy (a)'s plain fact extraction, whose buffer rows were already marked flushed. A probe failure caused by, say, a bad profile render would silently discard the day's facts. **Fix:** promote strategy (a) rows independently (they're observations, not inferences — probe only staleness/injection canaries against them); or don't mark buffer rows flushed until their memories promote.

26. **[MAJOR] Skill-forge lifecycle vs curator race.** New brain-created skills are `mark_agent_created()` → the curator's inactivity-based archiving governs them immediately; a skill drafted for a monthly task can be archived as "stale" before its second use. Also `.brain-drafts/` under `~/.hermes/skills/` assumes the skills loader skips dot-dirs — verify at build or place drafts under `~/.hermes/brain/drafts/` (safer, zero assumptions). **Fix:** drafts outside the skills tree; check curator grace period and set `pinned` or a `created_at`-aware grace for brain-created skills if needed.

27. **[MINOR] Idle detection is blind mid-turn.** No provider hook fires during a long multi-hour tool-grinding turn, so `activity.last_turn_at` goes stale and a sweep/dream can start while the agent is mid-task. Impact is bounded (WAL + budgets), but note it; optionally treat an open turn in state.db (`messages` newer than last `sync_turn`) as activity.

28. **[MINOR] Attribution noise honesty.** B's helpful/harmful counters credit *every* injected item with the turn outcome. Fine in aggregate as B says — but add the guard that items co-injected fewer than N times with divergent outcomes don't auto-deprecate (B has n≥5; keep it, and log the co-injection confound).

29. **[MINOR] Dream-spam defenses are good — one gap.** The specificity gate ("names ≥1 concrete entity, `actionable: true`") is self-reported by the same LLM call it gates. Cheap hardening: the entity must exact-match a row in `entities`, not merely appear in the text.

## F. Resource envelope (attack vector 6)

30. **[MAJOR] The 1GB tier cannot run "standard".** The research does back EmbeddingGemma <200MB at int8 QAT (so A's per-model numbers are fine), but the brain lives **inside the Hermes agent process**: Hermes + Python + deps (~150–300MB) + onnxruntime & arenas + Gemma (<200MB) + ColBERT (~60MB) + numpy + SQLite cache (32MB) + adjacency ≈ total process well over 700MB before the OS. C's lite auto-detect at <1.5GB already concedes this. **Fix:** A's perf table changes: Standard = ≥1.5–2GB; 1GB VPS = lite (potion, no reranker). Add C's planned RAM-capped container test to P2 acceptance. Also drop `cache_size=-32000` to -8000 on lite.

31. **[MINOR] Background token budget sanity: passes.** ~$0.50/night with digest-not-transcript prompts and flash-class extract tier is realistic (a day's episodes ≈ tens of kilotokens). One caution: "consolidate tier stronger than chat model" on opus-class pricing eats the budget in 2–3 calls — the ledger + per-strategy sub-caps (B) handle it; keep pipeline-order-as-priority. Also cap `work_queue` growth for devices offline for weeks (drain oldest-first with a max-age discard-to-archive rule).

32. **[MINOR] `retrieval_log` write amplification.** 50 rows/turn per prefetch; on Termux this is measurable. Batch in one transaction; floor tier logs injected rows only.

## G. Missing entirely (attack vector 8)

33. **[MAJOR] Multi-user identity resolution is unspecified in all three docs.** `initialize()` provides `user_id`/`user_id_alt` (verified in ABC) and A has `scope_user`, but nothing maps `telegram:123 == discord:456 == owner`, and nothing specifies how `trust_tier='owner'` is assigned. Without this, cross-platform memory (the "money shot" in C's P5) fragments per platform and the whole trust model has no root. **Fix:** an `identities(principal_id, platform, platform_user_id, is_owner)` table; owner ids declared in setup (default: the sole CLI user + configured gateway owner ids); unknown platform users default to `known_user` at best, never `owner`. This is day-one schema.

34. **[MAJOR] brain.db migration/versioning policy is a mention, not a story.** `meta.schema_version` exists in A and "migrations" in C's `db.py`, but no policy: forward-only numbered migrations, `VACUUM INTO` backup before migrate, refuse-to-open on future versions (older plugin vs newer DB), and a pinned `state.db SCHEMA_VERSION` compatibility check (C mentions the pin — make it a hard gate with a doctor message, since v22 is verified today and *will* drift). Also specify: embeddings re-embed policy on model change is covered (`embedded_with` + janitor) — good; extend the same tag discipline to reranker and prompt versions (B has `prompt_version` — put it in the unified schema).

35. **[MINOR] MCP surface security is mostly right; pin it.** stdio-only (no network listener) — state this as an invariant, not an accident; any future SSE transport requires auth design first. External writes → `trust_tier` external and never lane-eligible without review (C has this). Add: the MCP server must open brain.db with the same capability probe (item 19) and must never run consolidation (only the dream process does — make normative; B's table currently implies MCP-side sleep-time work via sampling, which Claude Code doesn't even service).

36. **[MINOR] Model file location conflict is also a backup bug.** A puts models in `<hermes_home>/brain/models/` → `hermes backup` (walks HERMES_HOME, F15) archives 300MB of ONNX per backup. C's `~/.cache/hermes-brain/models` (LOCALAPPDATA on Windows) is correct; A must change. Similarly note the append-forever archive grows backups unboundedly — document, and consider excluding old archive months via a documented restore path.

37. **[MINOR] Install-path nuance.** User memory plugins load from `$HERMES_HOME/plugins/<name>/` (verified), not literally `~/.hermes/plugins/` — multi-profile users need a per-profile clone or symlink; docs must say so. Directory name `brain` = provider name = CLI verb (verified loader behavior) — lock the name.

38. **[MINOR] Aux-slot registration mechanism.** B's `auxiliary.brain_extract/brain_consolidate` in Hermes config.yaml can't be created via `get_config_schema()` alone; it must happen in `post_setup(hermes_home, config)` mutating the config dict (verified `memory_setup.py:225–232` saves config after post_setup). Specify that, plus the fallback when the slot is absent (aux client default model — verified supported).

---

## Cut list (remove from v1; most can return with evidence)

- **Vault** (`vault_items`, `brain_vault_get/set`, `cryptography` extra) — leaks secrets into transcripts (item 7).
- **`brain_confirm` tool** — injectable promotion path; CLI review instead (item 15).
- **`memories` Anthropic-shaped file tool** — defer; read-only view at most (item 8).
- **A-MEM memory evolution (B strategy c)** — needs a schema column nobody defined; lowest-evidence strategy in the pipeline (item 10).
- **Anticipatory precompute (B strategy h)** + `prefetch_cache` — a cache lifecycle to beat an already-fast path; ship shadow at most (item 24).
- **Prompt-section optimizer (LangMem-style)** — correlational gates; v2.
- **Agent self-model core block** — keep the user profile block; self-model returns when TSI exists to feed it.
- **Convex/logistic weight fitting (`fit_weights.py`)** — already v2; keep only `retrieval_log` schema (both docs agree).
- **Trigram FTS side-index** — v1.1; porter FTS + code tokenizer covers the core; add when typo/handle recall demonstrably misses.
- **`emotional_salience`** as anything beyond a stored column — no v1 consumer.
- **Idle "micro-shifts" as a distinct mechanism** — collapses into C's bounded idle sweep (item 2).
- **LLM query decomposition via MCP sampling in `brain_recall deep`** — sampling is absent in the primary MCP client anyway.

Phase 1 as specified in Design C survives review intact **provided item 22's raw-turn FTS lane is in it** (it is, in C; it must be adopted over A's extraction-only indexing).

## Open questions for the user (decisions only you can make)

1. **Sleep-time LLM keys:** when no Hermes aux provider is reachable (headless box, gateway off), may the brain use its own `BRAIN_LLM_API_KEY` from `.env`, or should learning simply wait? (Defaults proposed: wait unless key present; night budget $0.50, day $1.50 — confirm the dollars.)
2. **Group-chat capture policy:** extract facts from *other people's* messages in group chats by default (stored as facts-about-them, trust-gated), or owner-messages-only with peers reduced to entity mentions? Privacy posture, not engineering.
3. **Skill auto-approve:** confirm human-approval default for brain-drafted skills (config `auto_approve` exists but ships false)?
4. **1GB VPS verdict:** accept that 1GB runs lite tier (potion embeddings, no reranker) and "standard" means ≥1.5–2GB? Affects how the README sells the $5-VPS story.
5. **Archive retention:** raw archive is append-forever by default (and grows `hermes backup`); acceptable, or cap with an age-based cold-storage/exclusion policy?
6. **Owner identity bootstrap:** at setup, are you willing to enumerate your platform user IDs (Telegram/Discord/Slack) so `trust_tier='owner'` has a root? Without it, everything from gateway platforms lands at `known_user` trust.
