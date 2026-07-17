# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Hermes-Brain is an out-of-tree `MemoryProvider` plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) — global memory + continual learning across every session and platform. **The repo root *is* the plugin.** It installs to `$HERMES_HOME/plugins/brain/`, and the directory name `brain` is load-bearing: it is simultaneously the provider name, the config key, and the `hermes brain` CLI verb. Do not rename it.

The floor tier is **stdlib-only by design** (no required pip deps). Vector/ONNX tiers are optional extras.

## Commands

```bash
pip install -e .[dev]            # pytest + ruff
pip install -e .[full]           # optional: onnxruntime, tokenizers, sqlite-vec, numpy (full retrieval tier)

python -m pytest                 # full suite (testpaths=tests)
python -m pytest tests/test_p5_learning.py::test_cases_dry_run_writes_nothing   # single test
python -m pytest -p no:cacheprovider -q                                         # quiet, no cache

python -m ruff check .           # lint (must be clean before shipping)
python -m ruff check --fix .     # autofix

# Replay harness — drives the REAL provider hook sequence; the CI invariants live here.
# Run as a direct script (it self-registers the `brain` package via importlib).
python replay/run.py --fixture tests/fixtures/session_basic.json --assert-lane1-stable --budget-check
```

There is **no build step** (pure-Python plugin). On Windows, prefix with `PYTHONIOENCODING=utf-8` if console output contains non-ASCII.

Tests run standalone (no hermes-agent needed): `tests/conftest.py` registers the repo root as package `brain` exactly the way the Hermes loader does (`spec_from_file_location` + `submodule_search_locations`), so imports are `from brain.x import ...`. It also refuses to run outside pytest. Cross-test helpers (`seed_memory`, `seed_episode`, `poll_until`) come `from conftest import ...`.

## Load-bearing architectural rules

These are invariants, not preferences. Violating them breaks Hermes silently.

1. **The Hermes loader eagerly imports every *root* `*.py` on every CLI invocation.** So root modules (`cli.py`, `tools.py`, `provider.py`, `mcp_server.py`, `brain_setup.py`, `config.py`, `llm.py`, `__init__.py`) MUST keep module level **stdlib-only** — every heavy sibling import is deferred into function bodies. This is why ruff ignores `E402`. Subpackages (`dream/`, `recall/`, `skillforge/`, `store/`, `capture/`, `bootstrap/`, `replay/`) are *not* eagerly imported and may import freely at module level.

2. **Two-lane cache-safe injection is invariant #1** (prompt caching is Hermes's top invariant). Lane 1 = `system_prompt_block()`, rendered ONCE at `initialize()` and **byte-identical for the whole session** (from the materialized `lane1_snapshot` table). Lane 2 = `prefetch()`, a per-turn ephemeral fence — the only dynamic channel. A golden test (`tests/test_provider.py`) asserts byte-stability across 50 turns + compression; never make lane 1 read live/dynamic data at render time.

3. **No LLM in the turn path.** Hooks (`sync_turn`, `queue_prefetch`, etc.) enqueue and return in microseconds. One owned worker thread ("brain-bg") holds the only long-lived connection and does all real work; tool calls and the CLI use short-lived connections. Capture-path functions (`search`, `log_retrieval`, extraction, guidance) must **never raise into a turn** — they log and degrade.

4. **`store/db.py` runs a capability probe** because FTS5 and sqlite-vec extension loading are not universal (system Python, Termux, python.org macOS builds). Code must degrade: no FTS5 → LIKE fallback; no sqlite-vec → FTS-only. Tiers: `full` (ONNX) / `lite` (static embeddings) / `fts-only` / `stub` (tests).

5. **`store/schema.sql` is law.** Storage is ONE SQLite file (WAL) holding memories + FTS5 external-content + sqlite-vec int8[256] + graph tables. **Versions are rows** (supersede-don't-delete): current truth = `valid_to IS NULL AND status='active' AND live=1`. Schema changes are forward-only numbered files in `store/migrations/`; `tests/test_migrations.py` fails if a fresh DB and a migrated DB diverge — add both the `schema.sql` change and the migration.

6. **`llm.py` is the SOLE gateway for brain-initiated LLM calls**, resolved through the active Hermes profile's `auxiliary_client` (per-profile provider config; cloud or local). Standalone/tests: raises `LLMUnavailable` unless a fake is installed via `llm.set_llm_for_tests(...)`. Every autonomous LLM caller must handle `LLMUnavailable` by skipping and retrying next run.

## The learning system (dream cycle)

`dream/` runs sleep-time work in a `hermes brain dream` process — **cron + manual only; the brain never auto-spawns background processes** (a deliberate security decision — do not add process-spawning to `provider.py`). Mutual exclusion is the `brain_lease` DB row (no lockfiles; Windows-safe).

- `shift.py` — the per-run `Shift` context: `PIPELINE` (ordered strategy names), `DEFAULT_MODES`, plus preemption (`tick`/`keepalive`), budget, and mode gating.
- `run.py` — the phase machine: acquire lease → run pipeline → idempotent cursor → release.
- **Ship-inert**: every mutating strategy has a mode `off | shadow | dry_run | active`. Mutating ones default to `dry_run`/`shadow`; they compute honest counts and audit what they *would* do but write no live memory until promoted (`hermes brain dream --enable <strategy>`). When adding a strategy, register it in `PIPELINE`, `DEFAULT_MODES`, and `run.py:_strategy_fn`, and prove dry_run/shadow write nothing.
- Strategies: `flush` (extraction) · `mine` (credit injections) · `cases` (Memento case bank) · `distill` (ReasoningBank strategy/guardrail items) · `consolidate` · `contradict` · `forget` (tiered demotion, never destructive) · `tune` (retrieval-weight tuning — shadow only, never applies) · `probes` (capability-regression health check) · `lane1` (re-render the snapshot).

**`dream/mine_state.py` is the single read-only reader of Hermes's `state.db`** (critique item 9 — do not open a second one). `assemble_episodes()` groups `turn_outcomes` into task episodes and is shared by `distill` and `cases`. Its watermark advances via `advance_watermark()` — it must never move past an *unresolved* episode's `started_at` (an episode spans multiple turns; skipping loses it).

**The injection→outcome loop** is subtle and load-bearing: `recall/search.log_retrieval` writes `retrieval_log` rows **pending** (a block computed after turn N is injected into turn N+1); the next `sync_turn` calls `stamp_pending_injections()` to attach the raw user text of the turn that actually consumed the block; the nightly miner joins `user_msg_hash` against `state.db.messages` to resolve a `turn_id` and move helpful/harmful counters.

## Trust & safety model

- `identities` table roots trust; `trust_tier ∈ {owner, agent, known_user, tool, untrusted}`. Recall is scoped on **every** id path — non-owner callers see only unscoped or their-own-principal rows.
- Instruction-shaped / untrusted content is **quarantined** — never rendered into lane 1 or lane 2; retrievable only via explicit `brain_recall` with a flag.
- Strategy items are planning-eligible, so `distill` only distills from **owner-trusted** episodes.
- The **MCP server** (`mcp_server.py`, stdio JSON-RPC, no SDK) exposes the brain to external agents at `tool` trust: reads see the owner's global/scoped memories (the cross-platform "money shot"); writes are capped, quarantined when instruction-shaped, never lane-1 eligible. stdio-only is an invariant; it never runs consolidation.
- **Skill-forge** (`skillforge/`) drafts `SKILL.md` into `$HERMES_HOME/brain/drafts/` (outside the skills tree — curator-race safety), validates (replay + Wilson + probes), and on pass promotes into Hermes's skills tree written `created_by: hermes-brain` (NOT `agent`) so Hermes's curator never garbage-collects it. It refuses any name in `.bundled_manifest` (the one curator trap).

## Design record

`docs/design/{memory-engine,learning-system,integration,critique}.md` are the normative design; `critique.md` is the resolved 38-item adversarial punch list (findings are referenced by number throughout the code comments). `docs/research/` holds the source research. When a code comment cites "critique item N" or "review finding #N", that document is the authority.
