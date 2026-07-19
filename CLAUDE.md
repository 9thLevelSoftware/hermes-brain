# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Hermes-Brain is an out-of-tree `MemoryProvider` plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) — global memory + continual learning across every session and platform. **The repo root *is* the plugin.** It installs to `$HERMES_HOME/plugins/brain/`, and the directory name `brain` is load-bearing: it is simultaneously the provider name, the config key, and the `hermes brain` CLI verb. Do not rename it.

A **second, companion** plugin lives in `observer/` (`brain_observer`) — a general-purpose `register(ctx)` plugin that taps host hooks the `MemoryProvider` contract can't reach. It installs as its OWN top-level `$HERMES_HOME/plugins/brain_observer/`, NOT nested under `brain/`.

The floor tier is **stdlib-only by design** (no required pip deps). Vector/ONNX/reranker tiers are optional extras. Requires **Python >=3.11, <3.14**.

`_compat.py` vendors a fallback `MemoryProvider` ABC so the plugin loads standalone (tests, CI, MCP) without `hermes-agent` installed.

## Commands

```bash
pip install -e .[dev]            # pytest + ruff
pip install -e .[full]           # onnxruntime, tokenizers, sqlite-vec, numpy (full retrieval tier)
pip install -e .[rerank]         # onnxruntime, tokenizers, numpy (ColBERT rerank stage)
pip install -e .[sync]           # cryptography (multi-device encrypted delta sync; never floor tier)

python -m pytest                 # full suite (testpaths=tests)
python -m pytest tests/test_migrations.py::test_fresh_and_migrated_schemas_are_identical   # single test
python -m pytest -p no:cacheprovider -q                                                    # quiet, no cache

python -m ruff check .           # lint (must be clean before shipping)
python -m ruff check --fix .     # autofix — line-length=100, rules: E,F,I,W,UP,B,SIM

# Replay harness — drives the REAL provider hook sequence; the CI invariants live here.
# Run as a direct script (it self-registers the `brain` package via importlib).
python replay/run.py --fixture tests/fixtures/session_basic.json --assert-lane1-stable --budget-check

# Docker smoke tests (each `docker build` IS the test). See docker/README.md.
docker build -f docker/Dockerfile      -t hermes-brain:floor .   # stdlib floor tier, true standalone
docker build -f docker/Dockerfile.full -t hermes-brain:full  .   # onnx+vec+numpy tier
# Live integration (real hermes-agent + brain; needs a staged context — see docker/README.md):
#   Dockerfile.hermes (loads under real Hermes) + Dockerfile.hermes-mock (mock LLM drives a real turn+dream, offline).

# Adversarial gauntlet (multi-phase invariant testing — see docker/adversarial/README.md):
docker/adversarial/run-suite.sh
```

There is **no build step** (pure-Python plugin). On Windows, prefix with `PYTHONIOENCODING=utf-8` if console output contains non-ASCII. `.gitattributes` forces LF line endings for `.py`, `.sh`, `Dockerfile*`, `.sql`, `.yaml`, `.md`, `.json` — CRLF breaks Docker shebangs and BuildKit heredocs.

Tests run standalone: `tests/conftest.py` registers the repo root as package `brain` the way the Hermes loader does (`spec_from_file_location` + `submodule_search_locations`), so imports are `from brain.x import ...`. Cross-test helpers (`seed_memory`, `seed_episode`, `poll_until`) come `from conftest import ...`.

**Testing gotcha — fake the LLM around EVERY step.** If `hermes-agent` happens to be importable in your dev environment, `llm.py`'s lazy `from agent import auxiliary_client` succeeds, so a brain LLM call that is *not* faked hits the real aux client (and fails on auth). Tests must install `llm.set_llm_for_tests(...)` around every step that can call the LLM, not only the step under test. The `docker/Dockerfile` floor image is the only guaranteed clean-standalone env (no hermes-agent, no optional deps) — use it to catch tests that secretly depend on `agent` being importable.

**Adversarial test layer** (`tests/adversarial/`): 8 test modules exercising trust scoping, budget caps, concurrency, capture-path never-raise, capability degradation, anti-spam, and crash recovery. The Docker adversarial gauntlet (`docker/adversarial/`) runs multi-phase invariant tests with a mock LLM (valid, spam, malformed, huge, prompt-echo scenarios).

## Load-bearing architectural rules

These are invariants, not preferences. Violating them breaks Hermes silently.

1. **The Hermes loader eagerly imports every *root* `*.py` on every CLI invocation**, and registers the brain under a SYNTHETIC package name `_hermes_user_memory.brain` — an empty `ModuleSpec` shell whose `__init__.py` is **never executed** (so it has no `__version__` attribute, `__file__ is None`). Consequences: root modules (`cli.py`, `tools.py`, `provider.py`, `mcp_server.py`, `brain_setup.py`, `config.py`, `llm.py`, `__init__.py`) MUST keep module level **stdlib-only** (heavy sibling imports deferred into function bodies). No eager-loaded code may `from .. import __version__` or otherwise assume the package name/attributes exist — guard such imports with a fallback (see `recall/lane1.py`). Subpackages (`dream/`, `recall/`, `skillforge/`, `store/`, `capture/`, `bootstrap/`, `replay/`) are not eagerly imported and may import freely. Ruff ignores related to this pattern: `E402` (deferred imports), `B904` (intentionally dropped tracebacks from LLMUnavailable/ToolError), `SIM105` (`try/except sqlite3.Error: pass` rollback idiom kept over `contextlib.suppress`).

2. **Two-lane cache-safe injection is invariant #1** (prompt caching is Hermes's top invariant). Lane 1 = `system_prompt_block()`, rendered ONCE at `initialize()` and **byte-identical for the whole session** (from the materialized `lane1_snapshot` table). Lane 2 = `prefetch()`, a per-turn ephemeral fence — the only dynamic channel. A golden test (`tests/test_provider.py`) asserts byte-stability across 50 turns + compression; never make lane 1 read live/dynamic data at render time.

3. **No LLM in the turn path.** Hooks (`sync_turn`, `queue_prefetch`, etc.) enqueue and return in microseconds. One owned worker thread ("brain-bg") holds the only long-lived connection and does all real work; tool calls, the CLI, and the observer plugin use short-lived connections. Capture-path functions (`search`, `log_retrieval`, extraction, guidance) must **never raise into a turn** — they log and degrade.

4. **`store/db.py` runs a capability probe** because FTS5 and sqlite-vec extension loading are not universal. Code must degrade: no FTS5 → LIKE fallback; no sqlite-vec → FTS-only; no models/deps → each retrieval stage skips. Tiers: `full` (ONNX) / `lite` (static embeddings) / `fts-only` / `stub` (tests).

5. **`store/schema.sql` is law** (currently **v3**). Storage is ONE SQLite file (WAL) holding memories + FTS5 external-content + sqlite-vec int8[256] + graph tables (`entities`/`entity_mentions`/`edges`) + the temporal `facts` layer + the `memory_events` append-only sync log (both added in migration 003). **Versions are rows** (supersede-don't-delete): current truth = `valid_to IS NULL AND status='active' AND live=1`. Schema changes are forward-only numbered files in `store/migrations/`; `tests/test_migrations.py` fails if a fresh DB and a migrated DB diverge — add both the `schema.sql` change and the migration (the test drops *every* migration's objects and re-applies, so v3+ deltas are genuinely exercised). Prefer a free-text `kind` or the documented `source_refs` JSON over a migration when possible (e.g. `kind='peer_card'`/`kind='fact'`; archive refs in `source_refs`).

6. **`llm.py` is the SOLE gateway for brain-initiated LLM calls**, resolved through the active Hermes profile's `auxiliary_client` via the `brain_extract`/`brain_consolidate` aux task slots (registered by `brain_setup.post_setup`). It records REAL token counts + priced `est_usd` from the response (host `usage_pricing`), falling back to a char/4 token proxy; the daily budget gate counts priced USD **plus** a proxy for unpriced rows. Standalone/tests raise `LLMUnavailable` unless a fake is installed via `llm.set_llm_for_tests(...)`; every autonomous caller handles `LLMUnavailable` by skipping and retrying next run.

## Retrieval stack (`recall/`)

Hybrid retrieval, degrading stage-by-stage: FTS/BM25 and vector-KNN (int8, 256-d) each yield a ranked leg → **RRF fusion** (`fusion.py`, k=60) over one keyspace → optional **ColBERT reranker** (`rerank.py`, late-interaction MaxSim) reorders the fused top-K *before* lifecycle modulation → optional **graph/PPR leg** (`graph.py`, pure-Python Personalized PageRank over the entity co-mention + `edges` graph, seeded by the fused candidates) surfaces connected memories keyword/vector missed → optional **facts leg** (`facts_leg.py`, current-truth S-P-O rows whose subject/object match, emitting their backing memory ids) → optional MMR **diversify** hook (`mmr.py`) → lifecycle modulation (decay, outcome, feedback, pinned) + the 0.6× episode factor. Entities are populated by `store/entities.py` (from `consolidate`); learned fusion weights (`fit_weights.py`) are *proposed* shadow-only via the `tune` strategy and NEVER auto-applied. `search()` never raises (capture path) and excludes internal kinds (`strategy`/`guardrail`/`case`/`peer_card`) from generic facts recall.

## The learning system (dream cycle)

`dream/` runs sleep-time work in a `hermes brain dream` process — **cron + manual only; the brain never auto-spawns background processes** (a deliberate security decision — do not add process-spawning to `provider.py`). Mutual exclusion is the `brain_lease` DB row (no lockfiles; Windows-safe).

- `shift.py` — the per-run `Shift` context: `PIPELINE` (ordered strategy names), `DEFAULT_MODES`, plus preemption (`tick`/`keepalive`), budget, and mode gating.
- `run.py` — the phase machine: acquire lease → run pipeline → idempotent cursor → release. `_strategy_fn` dispatches (and mode-gates) each strategy.
- **Active by default** (modes `off | shadow | dry_run | active`): the mutating strategies (`cases`/`distill`/`consolidate`/`contradict`/`forget`/`forge`/`revise`/`peers`) default to **`active`** — they learn live on every dream run. `tune` stays `shadow` (it only ever *proposes* retrieval weights, never applies — a hard invariant). Roll a strategy back with `hermes brain dream --disable <strategy>`, or neutralize a run with `--dry-run`. When adding a strategy, register it in `PIPELINE`, `DEFAULT_MODES`, and `run.py:_strategy_fn`.
- `PIPELINE` order: `flush` (extraction) · `mine` (outcome credit) · `cases` (Memento case bank) · `distill` (ReasoningBank strategy/guardrail items) · `forge` (draft skills from the case bank) · `revise` (propose skill revisions/retirements) · `consolidate` (episodic→semantic patterns) · `facts` (s-p-o triple extraction — **shadow** default) · `contradict` (incl. deterministic zero-LLM knowledge-update for triple-backed same-(s,p) pairs) · `peers` (group-chat theory-of-mind `peer_card`s — typed-prefix spec, 40-line cap) · `forget` (tiered demotion — **non-destructive**: archives raw text via `store/archive.py` before nulling content; per-kind Weibull decay shapes; compacts synced events when sync is on) · `tune` (shadow) · `probes` (capability-regression health check) · `lane1` (re-render the snapshot). `consolidate` also seeds anomalies via the `surprise` column when `dream_surprisal` is on.

**`dream/mine_state.py` is the single read-only reader of Hermes's `state.db`** (critique item 9 — do not open a second one). `assemble_episodes()` groups `turn_outcomes` into task episodes; its watermark (`advance_watermark()`) must never move past an *unresolved* episode's `started_at` (an episode spans multiple turns; skipping loses it).

**The injection→outcome loop** is load-bearing: `recall/search.log_retrieval` writes `retrieval_log` rows **pending** (a block computed after turn N is injected into turn N+1); the next `sync_turn` calls `stamp_pending_injections()` to attach the raw user text of the turn that consumed the block; the nightly miner joins `user_msg_hash` against `state.db.messages` to resolve a `turn_id` and move helpful/harmful counters (also reused by `recall/fit_weights.py` for its labels).

## Trust & safety model

- `identities` table roots trust; `trust_tier ∈ {owner, agent, known_user, tool, untrusted}`. Recall is scoped on **every** id path (`recall/search.py:_scope_memories`) — a non-owner caller sees only unscoped or their-own-principal rows, and NEVER a `peer_card` (the owner's private theory-of-mind of a person, which would otherwise leak to the very peer it describes).
- Instruction-shaped / untrusted content is **quarantined** — never rendered into lane 1 or lane 2; retrievable only via explicit `brain_recall` with a flag.
- Strategy items are planning-eligible, so `distill`/`peers` only run on **owner-trusted** episodes.
- The **MCP server** (`mcp_server.py`, stdio JSON-RPC, no SDK) exposes the brain at `tool` trust: reads see the owner's global/scoped memories (the cross-platform "money shot"); writes are capped, quarantined when instruction-shaped, never lane-1 eligible. stdio-only is an invariant; it never runs consolidation.
- The **observer plugin** (`observer/`) registers `post_tool_call`/`subagent_stop` host hooks that enqueue lightweight signals into the `work_queue` table (drained by the brain-bg worker); it never blocks a turn, never opens the long-lived connection, and honors a live `BRAIN_OBSERVER_DISABLE` kill switch. Its `pre_llm_call` context-injection lane ships OFF.

## Skill-forge (`skillforge/`)

`forge` drafts `SKILL.md` from the case bank into `$HERMES_HOME/brain/drafts/` (outside the skills tree — curator-race safety), validates (replay + Wilson + probes), and on pass promotes into Hermes's skills tree written `created_by: hermes-brain` (NOT `agent`) so Hermes's curator never garbage-collects it (refuses any name in `.bundled_manifest`). `revise` (`skillforge/revise.py`) reads each brain-forged skill's `.usage.json` health and PROPOSES a revision (net-harmful over a Wilson-gated decisive-sample floor) or, after repeated rejections, a retirement — proposals only, applied via the CLI `hermes brain review --approve` (which patches SKILL.md sections / marks stale and resets the health window). Retired (`state='stale'`) skills are skipped by the brain-owned scan.

## CLI surface

```
hermes brain status | doctor | search <q> | why <id>
hermes brain remember/forget/pin/unpin/incognito ...
hermes brain dream-now [--phase X] [--dry-run]     # run a consolidation shift
hermes brain dream --if-due                          # cron entry point
hermes brain dream --enable/--disable <strategy>     # promote/demote a strategy
hermes brain insights                                # longitudinal learning metrics
hermes brain review [--approve/--reject <uid>]       # proposals + quarantine queue
hermes brain skills list|forge|approve|reject        # forged-skill lifecycle
hermes brain mcp                                      # stdio MCP server for external agents
hermes brain adopt-memory [--apply]                  # hand memory ownership to the brain
```

## "Best-of-three" subsystems (Mnemosyne + Honcho ports)

Layered on the floor via free-text `kind`/`meta` and migration 003. **Every new behavior is config-gated** (`config.py` DEFAULTS) and **degrades by tier** (plain-SQL/stdlib legs work at fts-only/stub; LLM/embedder legs skip when absent). Vendored Mnemosyne files (MIT) carry in-file attribution headers; Honcho contributions are re-implemented from concept (AGPL — never copy code). See `docs/design/` and the plan for rationale.

- **Retrieval upgrades (`recall/`)** — `mmr.py` (Jaccard MMR diversify, λ=0.7), `query_cache.py` (in-process cache keyed on `meta.mem_generation`), `intent.py` (query-intent classifier — **shadow-only**, logs proposals to `audit_log`, never applied), `blend.py` (working-representation blend: semantic + most-reinforced + most-recent, RRF-fused, memory-first then episodes). Wired into `provider.py:_do_retrieve` (gates `lane2_blend`/`query_cache`/`mmr_lambda`) + `capture/extract.py` info-content dedup contest (`dedup_contest`).
- **Temporal facts (`store/facts.py`, `recall/facts_leg.py`, `dream/facts.py`)** — S-P-O triples with close-then-insert supersession + as-of queries. **Facts are an index OVER memories** (critique item 9): every triple links a `kind='fact'` memory row; the facts leg emits memory ids into `fusion.rrf()` and re-fetches via `_memories_by_ids` (scope enforced centrally). Triple supersession retires the linked memory in lockstep. `hermes brain fact <subject> [--as-of]`; `why` shows facts + reasoning chain.
- **Dialectic ask (`recall/ask.py`)** — bounded tool-loop LLM agent (dual explicit/derived prefetch via the `epistemic` filter; actions search_memory/grep_episodes/get_reasoning_chain/date_range_search/answer; enumeration/contradiction/update/abstention protocols; `fast`→`extract` tier cap 2, `deep`→`consolidate` cap 6). **Every action re-applies caller scope**; citations only reference registry-gathered evidence (no leak). `LLMUnavailable` → recall-only result, never raises. Surfaces: `hermes brain ask`, MCP `brain_ask`, and the agent-facing tool schema **off by default** (`ask_tool_agent`).
- **Context assembly (`recall/context.py`)** — token-budgeted block: fixed identity + relevant peer-card subtracted first, remainder split 40/60 summary/recent-extracts, hard budget wall. Wired into `provider.on_pre_compress` (short-lived read conn, salience fallback), `hermes brain context`, MCP `brain_context`.
- **Eval harness (`tests/eval/`)** — BEAM-pattern stages (ingest→dream→retrieve P@k/MRR→answer→judge) over a fixture; CI-hermetic with a scripted fake LLM, `BRAIN_EVAL_REAL=1` for a real model. `hermes brain eval --fixture`.
- **Sync (`sync/`, `[sync]` extra — `cryptography`; never floor tier)** — `crypto.py` (Fernet + Argon2id/PBKDF2 KDF + HMAC, all lazy-imported), `relay.py` (stdlib zero-knowledge ciphertext relay), `engine.py` (cursor paging + outbox-ack over `memory_events`, LWW+causal-chain conflict resolution). **Surface-only deny-list is the load-bearing invariant** (`engine.is_syncable`, re-checked at push time): a scoped/`peer_card`/quarantined/instruction-shaped row NEVER serializes — a synced memory is global by construction. `hermes brain sync init|push|pull|status`; off by default (`sync_enabled`). Adversarial coverage: `tests/adversarial/test_sync_namespace.py`, `test_ask_scope.py`.

## Design record

`docs/design/{memory-engine,learning-system,integration,critique,substrate-spike}.md` are the normative design; `critique.md` is the resolved adversarial punch list (findings referenced by number throughout the code comments); `substrate-spike.md` records deferred/gated escape hatches (KV/LoRA, ANN scale). `docs/research/` holds the source research. When a code comment cites "critique item N" or "review finding #N", that document is the authority.
