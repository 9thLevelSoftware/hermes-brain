# Research: hermes:integration-map

## Summary

Hermes Agent (C:\Users\dasbl\hermes-agent) is a synchronous single-loop agent (AIAgent in run_agent.py, ~3,900-line run_conversation in agent/conversation_loop.py) shared by CLI/TUI/gateway/desktop, with two sacred invariants: byte-stable system prompt per session (prompt caching) and strict role alternation. Its learning loop = (a) tiny file-backed curated memory (MEMORY.md/USER.md, ~2200/1375 chars, frozen into the system prompt at session start), (b) interval 'nudges' that no longer inject text but instead fork a background review agent that writes memory/skills, (c) FTS5 session search over a SQLite state.db, (d) an optional single external MemoryProvider plugin (Honcho et al.) with a rich lifecycle-hook ABC, and (e) skills-as-files with a background curator. The cleanest deep integration for Hermes-Brain is a standalone MemoryProvider plugin repo installed into ~/.hermes/plugins (implements agent/memory_provider.py ABC, activated via memory.provider config) — it gets system-prompt injection, per-turn prefetch (fenced into the user message at API-call time, cache-safe), full-transcript sync_turn, session/compression/delegation hooks, its own model tools, setup-wizard and CLI integration. Secondary hooks: general plugin lifecycle hooks (pre_llm_call context injection, post_tool_call), gateway event hooks (gateway:startup to launch a companion service), MCP server registration (with sampling = server-initiated LLM calls through Hermes' configured provider), cron jobs, direct read access to the WAL SQLite session store, and direct authorship of skills on disk.

## Current memory system

STORAGE + FORMAT — Built-in memory is tools/memory_tool.py (MemoryStore): two plain-text files under get_hermes_home()/memories/ — MEMORY.md (agent notes, default 2,200 chars ≈ 800 tokens) and USER.md (user profile, 1,375 chars ≈ 500 tokens), entries delimited by "\n§\n". Atomic temp-file+rename writes, fcntl/msvcrt lock files (Windows-aware), external-drift detection with .bak snapshots, threat-pattern scanning (tools/threat_patterns.py, 'strict' scope) on both write and load, overflow archiving to memories/archive/. One `memory` tool (add/replace/remove + atomic `operations` batch checked against the FINAL char budget), optional write-approval gate (tools/write_approval.py). A FROZEN snapshot of both files is rendered into the system prompt's volatile tier at session start (agent/system_prompt.py builds stable/context/volatile tiers; cached on agent._cached_system_prompt for the whole session); mid-session writes hit disk immediately but never mutate the prompt — snapshot refreshes next session or after compression (invalidate_system_prompt reloads from disk). Config: memory.* in config.yaml (memory_enabled, user_profile_enabled, char limits, nudge_interval, flush_min_turns).

NUDGE MECHANISM — "Periodic nudges" are interval counters that trigger a BACKGROUND REVIEW FORK, not injected text ("no nudge injection" — agent/turn_context.py:371). memory.nudge_interval (default 10 user turns, counter hydrated from history on resume) sets should_review_memory in turn_context; skills.creation_nudge_interval (config default 15; agent_init fallback 10) counts tool-calling iterations (agent/conversation_loop.py:714, reset when skill_manage is used) and sets _should_review_skills in agent/turn_finalizer.py:530. Additionally agent/reflection_triggers.py fires event-based reviews: turn outcome failed/blocked/unresolved, detected user correction, 3-consecutive-tool-failure streak, negative emoji reaction — each with per-(session,kind) 300s cooldown and single-flight gating. When triggered, turn_finalizer calls agent._spawn_background_review → agent/background_review.py: a daemon thread forks a fresh AIAgent inheriting the parent's provider/model/credentials/cached system prompt (so it replays the conversation against the WARM prefix cache; optionally routed to a cheaper model via auxiliary.background_review, in which case a compact digest is replayed instead), with a tool whitelist limited to memory + skill-management tools, and one of four prompts (_MEMORY_REVIEW_PROMPT, _SKILL_REVIEW_PROMPT, _FAILURE_REVIEW_PROMPT, _COMBINED_REVIEW_PROMPT) instructing it to save user facts to memory and to actively patch/create CLASS-LEVEL skills ("a pass that does nothing is a missed learning opportunity"). Writes go straight to the stores; the main conversation and cache are untouched. There is also a memory FLUSH: memory.flush_min_turns (default 6) gives the agent one turn to save memories/skills before compression, /new, /reset, exit, and gateway session_reset wipes context.

FTS5 SESSION SEARCH — hermes_state.py SessionDB: SQLite at get_hermes_home()/state.db, WAL mode (with fallback for network FS), SCHEMA_VERSION 22. Tables: sessions (id, source platform tag, user/chat/thread ids, model_config, system_prompt snapshot, parent_session_id lineage chains for compression-splits/branches/subagents, end_reason), messages (role, content, tool_calls/tool_name, reasoning fields, timestamps, token counts), messages_fts (FTS5 virtual table), turn_outcomes (per-turn outcome/reason, api_calls, tool_iterations, cost/token deltas, skills_loaded, feedback_kind/value — reaction feedback recorded via annotate_turn_feedback), session_model_usage, gateway_routing. tools/session_search_tool.py exposes the `session_search` tool: DISCOVERY (FTS5 query → lineage-deduped top sessions with snippet, ±5-message window, bookends; cron-source hits demoted below interactive; subagent/tool sessions hidden), SCROLL (anchored ±window), BROWSE (recent sessions). The current tool is zero-LLM by design (the older LLM-summary path was merged away; auxiliary.session_search provider/model config still exists for summarization side-tasks). /insights (agent/insights.py) computes usage analytics over the same DB.

HONCHO / EXTERNAL PROVIDERS — agent/memory_provider.py defines the MemoryProvider ABC; agent/memory_manager.py (MemoryManager) orchestrates builtin + AT MOST ONE external provider (second registration rejected). Providers ship in plugins/memory/<name>/ (honcho, mem0, supermemory, byterover, hindsight, holographic, openviking, retaindb — set CLOSED by policy May 2026; new backends must be standalone repos installed to ~/.hermes/plugins/ or pip entry points) and are activated via memory.provider in config.yaml. Honcho (plugins/memory/honcho/) is optional and network-dependent: requires honcho-ai SDK + HONCHO_API_KEY (app.honcho.dev SaaS) or a self-hosted base_url, config chain $HERMES_HOME/honcho.json → ~/.honcho/config.json → env. It exposes 5 model tools (honcho_profile card read/write, honcho_search hybrid RRF search over all sessions, honcho_reasoning dialectic LLM Q&A, honcho_context snapshot, honcho_conclude persistent conclusions) and passively syncs turns. External prefetch runs on a thread with 8s timeout (skipped if still running); sync_turn is serialized on a single daemon background worker so a wedged backend can never block a turn; cron and subagent sessions run skip_memory=True (providers intentionally never see them directly — subagent results reach the parent's provider via on_delegation).

SKILL CREATION/IMPROVEMENT LOOP — Skills are directories with SKILL.md (+ scripts/, references/, templates/) in ~/.hermes/skills/ (agentskills.io-compatible; bundled in skills/, heavier ones in optional-skills/ installed via skills hub). A skills index (name + ≤60-char description) is built into the STABLE system prompt tier; /skill-name slash commands inject the skill body as a USER message (agent/skill_commands.py) to preserve caching. The agent creates/patches skills via the skill_manage tool; the background skill review (above) is the autonomous creation path, with prompts pushing update-loaded-skill > update-umbrella > add support file > create-new-umbrella and explicit do-not-capture rules; /learn (agent/learn_prompt.py) builds a standards-guided authoring prompt as a normal turn. tools/skill_usage.py tracks per-skill use/view/patch counts + state in ~/.hermes/skills/.usage.json; agent/curator.py is the background maintenance loop (polled hourly inside the long-lived gateway via maybe_run_curator, gated by curator.interval_hours/min_idle_hours) that LLM-reviews and auto-archives stale agent-created skills (never bundled/hub/pinned ones, never deletes) with tar.gz backups; hermes curator CLI for pin/archive/restore/rollback.

## Integration points

### MemoryProvider plugin (primary, deepest)

Implement the MemoryProvider ABC in a standalone plugin repo installed to ~/.hermes/plugins/<name>/ (plugin.yaml + __init__.py) or a pip entry point; activate via memory.provider in config.yaml. Hooks received: initialize(session_id, hermes_home, platform, agent_context, user_id...), system_prompt_block() (static text in the volatile prompt tier), prefetch(query)/queue_prefetch (recall injected as a <memory-context> fenced block appended to the CURRENT user message at API-call time only — never persisted, cache-safe; see conversation_loop.py:815-824), sync_turn(user, assistant, messages=<full OpenAI-format transcript>) after every verified turn on a background worker, on_turn_start, on_session_end (full history at real session boundaries), on_session_switch (/new,/resume,/branch,compression), on_pre_compress (contribute text to the compression summary before messages are discarded), on_memory_write (mirror of every built-in memory tool write with provenance metadata), on_delegation (parent-side view of subagent task+result), get_tool_schemas()/handle_tool_call() (expose brain tools to the model), backup_paths(), get_config_schema()/save_config()/post_setup() (hermes memory setup wizard), plus plugins/memory/<name>/cli.py register_cli → `hermes <name>` subcommands (only shown for the ACTIVE provider).

Files: `C:\Users\dasbl\hermes-agent\agent\memory_provider.py`, `C:\Users\dasbl\hermes-agent\agent\memory_manager.py`, `C:\Users\dasbl\hermes-agent\plugins\memory\honcho\__init__.py`, `C:\Users\dasbl\hermes-agent\plugins\memory\hindsight`, `C:\Users\dasbl\hermes-agent\agent\turn_context.py`, `C:\Users\dasbl\hermes-agent\agent\conversation_loop.py`

**Notes:** Depth: maximal — this is the sanctioned successor path and exactly what Hermes-Brain should be. Constraints: only ONE external provider at a time (brain would replace Honcho, not coexist); prefetch has an 8s timeout and is skipped while a previous prefetch is in flight (do heavy recall async, serve cached); sync/prefetch run on a single serialized daemon worker with 5s drain at shutdown — never block; tool names must not shadow core tools; cron/subagent sessions pass skip_memory=True; must be an out-of-tree repo (in-tree plugins/memory/ is closed by policy). Fragility: low — ABC is versioned, signature-introspected for backward compat (metadata kwargs, messages kwarg).

### General plugin lifecycle hooks

~/.hermes/plugins/<name>/ with plugin.yaml + register(ctx). Hooks in VALID_HOOKS: pre_llm_call (return string → appended to the outgoing user message copy at API-call time, alongside memory prefetch — a second cache-safe context-injection channel with session_id/turn_id/user_message/conversation_history kwargs), post_llm_call, pre_tool_call/post_tool_call (observe or block every tool call — invoked from model_tools.py), transform_tool_result/transform_terminal_output/transform_llm_output, pre_api_request/post_api_request/api_request_error, on_session_start/on_session_end (fired at end of EVERY run_conversation — effectively per-turn)/on_session_finalize/on_session_reset, subagent_start/subagent_stop, pre_verify, pre_gateway_dispatch (inspect/skip/rewrite every inbound gateway message), approval + kanban observer hooks. ctx.register_tool() adds tools to the registry; ctx.register_cli_command() wires `hermes <plugin> ...` argparse trees. Plugins are opt-in via plugins.enabled in config.yaml.

Files: `C:\Users\dasbl\hermes-agent\hermes_cli\plugins.py`, `C:\Users\dasbl\hermes-agent\agent\turn_context.py`, `C:\Users\dasbl\hermes-agent\model_tools.py`

**Notes:** Depth: high; complements the MemoryProvider (which does NOT see individual tool calls mid-turn — pre/post_tool_call does). Pitfall: discover_plugins() only runs when model_tools.py is imported; call it explicitly elsewhere (idempotent). Note on_session_end plugin hook fires per run_conversation call, unlike MemoryProvider.on_session_end which fires only at real session boundaries. Plugins must never modify core files.

### Shell-script hooks

config.yaml hooks: section registers shell scripts as hook callbacks (pre_tool_call, post_tool_call, pre_llm_call, on_session_end, subagent_stop, ...). JSON payload on stdin; JSON on stdout can block the tool call or inject context into the next LLM call. First-use consent allowlist (~/.hermes/shell-hooks-allowlist.json), hooks_auto_accept / HERMES_ACCEPT_HOOKS for gateway/cron.

Files: `C:\Users\dasbl\hermes-agent\cli-config.yaml.example`, `C:\Users\dasbl\hermes-agent\agent\shell_hooks.py`

**Notes:** Language-agnostic escape hatch — a brain daemon could receive turn events via a tiny relay script with zero Python coupling. Subprocess-per-event overhead; use for eventing, not hot-path recall.

### Gateway event hooks (companion-service launcher)

~/.hermes/hooks/<name>/ with HOOK.yaml + handler.py (async def handle(event_type, context)). Events: gateway:startup, session:start/end/reset, agent:start/step/end, command:*. Errors never block the pipeline. gateway:startup is the natural place to health-check/launch a Hermes-Brain daemon alongside the long-lived gateway process; agent:end delivers platform/user_id/chat_id/session_id/message/response per turn.

Files: `C:\Users\dasbl\hermes-agent\gateway\hooks.py`, `C:\Users\dasbl\hermes-agent\gateway\run.py`, `C:\Users\dasbl\hermes-agent\gateway\builtin_hooks`

**Notes:** Gateway-only (not CLI sessions). gateway/builtin_hooks/ is an extension point for always-registered hooks (none shipped). Message text truncated to 500 chars in context — use SessionDB for full transcripts.

### MCP server registration

Add the brain as an MCP server under mcp_servers: in config.yaml (stdio command/args/env, or HTTP/SSE url). tools/mcp_tool.py auto-discovers its tools into the registry as a toolset (enable per platform); dedicated background asyncio loop, reconnection, keepalive, per-server timeouts. Crucially, MCP SAMPLING is supported and on by default: the brain server can issue sampling/createMessage requests and Hermes services them with its configured LLM provider (model override, rpm caps, tool rounds configurable per server) — so the brain gets LLM access without its own API keys. There is also hermes_cli/mcp_catalog.py (curated catalog) and optional-mcps/ precedent, plus the reverse direction: mcp_serve.py exposes Hermes conversations/messages/events as an MCP server the brain could consume.

Files: `C:\Users\dasbl\hermes-agent\tools\mcp_tool.py`, `C:\Users\dasbl\hermes-agent\mcp_serve.py`, `C:\Users\dasbl\hermes-agent\hermes_cli\mcp_catalog.py`, `C:\Users\dasbl\hermes-agent\optional-mcps`

**Notes:** This is the Daem0n-MCP-shaped path and rung 5 of the Footprint Ladder. Gives tool surface + LLM sampling but NO passive lifecycle events (no sync_turn/prefetch/system-prompt block) — pair with a MemoryProvider or hooks for the write path, or run MCP purely as the cross-agent interface (Claude Code etc.) while the MemoryProvider is the Hermes-native face.

### Service-gated core tool / toolset

tools/<name>.py calling registry.register(name, toolset, schema, handler, check_fn) — check_fn gates availability on brain daemon reachability/config so schema footprint is zero when unconfigured; wire the name into toolsets.py. Per-platform enablement via hermes tools / platform_toolsets.

Files: `C:\Users\dasbl\hermes-agent\tools\registry.py`, `C:\Users\dasbl\hermes-agent\toolsets.py`, `C:\Users\dasbl\hermes-agent\tools\memory_tool.py`

**Notes:** Requires a core PR — maintainers will reject (Footprint Ladder says plugin/MCP first; in-tree memory backends closed). Prefer ctx.register_tool or MemoryProvider.get_tool_schemas — same effect, no fork.

### System-prompt assembly injection

agent/system_prompt.py builds three tiers (stable / context / volatile); built-in memory snapshots enter volatile via MemoryStore.format_for_system_prompt; external providers via MemoryManager.build_system_prompt() → MemoryProvider.system_prompt_block(). SOUL.md (persona) and context files (AGENTS.md etc.) are other stable inputs. Prompt is byte-stable for the whole session (cached on agent._cached_system_prompt); rebuilt only at session start and after compression (which also reloads memory files from disk).

Files: `C:\Users\dasbl\hermes-agent\agent\system_prompt.py`, `C:\Users\dasbl\hermes-agent\agent\prompt_builder.py`

**Notes:** The brain's standing knowledge digest goes here via system_prompt_block(); it MUST be deterministic per session — anything per-turn belongs in prefetch(). Violating byte-stability breaks prompt caching, the project's #1 invariant.

### Cron scheduler (background cognition)

cron/jobs.py store + cron/scheduler.py tick (gateway thread every 60s; file lock ~/.hermes/cron/.tick.lock; catchup/grace windows; 3-minute hard interrupt per session). Jobs run FULL agent sessions with per-job model/provider overrides, skills preload, pre-run data script (stdout injected into prompt; no_agent=True for script-only jobs), context_from chaining, workdir, multi-platform delivery. Agents/users create jobs via the cronjob tool / hermes cron / /cron. Yes — cron jobs call LLMs through the configured provider (and background_review/curator/auxiliary tasks also make LLM calls via agent/auxiliary_client.py routing).

Files: `C:\Users\dasbl\hermes-agent\cron\scheduler.py`, `C:\Users\dasbl\hermes-agent\cron\jobs.py`, `C:\Users\dasbl\hermes-agent\agent\auxiliary_client.py`

**Notes:** Ideal for nightly consolidation/dream cycles: a cron job (or no_agent script invoking the brain daemon) that mines state.db and rewrites the brain's stores. Caveat: cron sessions run skip_memory=True by default — memory providers deliberately don't observe cron runs, and cron agents get protected toolsets stripped.

### Session-end / flush hooks

Real session boundaries (CLI exit/atexit, /reset, /new, gateway session expiry/reset) fire MemoryProvider.on_session_end(full messages) via MemoryManager.commit_session_boundary_async (serialized end→switch ordering), plus the memory-flush turn (memory.flush_min_turns) where the live agent gets one turn to save before context wipe; compression fires on_pre_compress with the messages about to be discarded.

Files: `C:\Users\dasbl\hermes-agent\agent\memory_manager.py`, `C:\Users\dasbl\hermes-agent\agent\turn_finalizer.py`, `C:\Users\dasbl\hermes-agent\gateway\session.py`

**Notes:** This is where end-of-session extraction belongs (Honcho pattern). Extraction may be LLM-bound (runs on the background worker, non-blocking).

### Direct SessionDB access (offline mining)

state.db is WAL-mode SQLite readable concurrently by an external process: sessions/messages/messages_fts/turn_outcomes (per-turn outcome + cost + skills_loaded + reaction feedback) — a brain daemon can continuously index transcripts, outcomes, and feedback without touching Hermes code at all. skills/.usage.json gives skill telemetry; ~/.hermes/logs/ gives structured logs.

Files: `C:\Users\dasbl\hermes-agent\hermes_state.py`, `C:\Users\dasbl\hermes-agent\tools\skill_usage.py`

**Notes:** Read-only recommended (schema is versioned, v22, migrations happen; writing risks corruption and skew). Highest-bandwidth signal source available; pairs with cron for scheduled digestion.

### Skills standard (procedural memory output)

The brain can author/patch skills as plain directories under ~/.hermes/skills/<category>/<name>/SKILL.md — they are picked up next session into the system-prompt index and as /slash commands. Respect frontmatter standards (description ≤60 chars, created_by provenance so the curator manages them) and the curator's .usage.json sidecar.

Files: `C:\Users\dasbl\hermes-agent\tools\skill_manager_tool.py`, `C:\Users\dasbl\hermes-agent\agent\curator.py`, `C:\Users\dasbl\hermes-agent\agent\learn_prompt.py`

**Notes:** Skills are the durable 'how-to' memory tier; a brain that distills transcripts into class-level skills supersedes the prompt-driven background review. Changes take effect next session (cache-aware deferred invalidation is the house pattern).

### Context-engine plugin (compression-time capture)

plugins/context_engine/ + agent/context_engine.py ABC — replace the built-in ContextCompressor via context.engine config; owns when/how compaction happens and may expose its own tools. A brain-aware engine could archive full fidelity to the brain before summarizing.

Files: `C:\Users\dasbl\hermes-agent\agent\context_engine.py`, `C:\Users\dasbl\hermes-agent\plugins\context_engine`

**Notes:** Heavyweight; only one engine active. Usually unnecessary since MemoryProvider.on_pre_compress already delivers the to-be-discarded messages.

### Companion service configuration/startup

Recommended combo: config under a brain-owned section of config.yaml (via MemoryProvider.get_config_schema + save_config, secrets in ~/.hermes/.env via env_var fields); `hermes brain <verb>` CLI via the memory-plugin cli.py or ctx.register_cli_command; daemon auto-started lazily by the provider's initialize() (spawn/health-check local process) and/or a gateway:startup hook; systemd unit precedent in plugins/kanban/systemd/. Never a new HERMES_* env var for non-secret config.

Files: `C:\Users\dasbl\hermes-agent\hermes_cli\config.py`, `C:\Users\dasbl\hermes-agent\plugins\kanban\systemd`, `C:\Users\dasbl\hermes-agent\hermes_constants.py`

**Notes:** Use get_hermes_home() for all state paths (profiles change HERMES_HOME); each profile gets an isolated brain store for free if you do.

## Constraints

- Python >=3.11,<3.14 (uv-managed venv; ceiling is load-bearing). Core dependencies are EXACT-pinned (==X.Y.Z) post supply-chain incidents; heavy/optional deps go in extras and are lazy-installed via tools/lazy_deps.py. An out-of-tree plugin should minimize deps and vendor or lazy-install anything heavy.
- Platforms: native Windows (no fcntl — code uses msvcrt fallbacks everywhere), Linux/macOS/WSL2, Termux/Android (constraints-termux.txt curates out incompatible voice deps), Docker, $5 VPS target — resource footprint must be small; SQLite (WAL, with non-WAL fallback for network filesystems) is the house database; embedding/vector deps must be optional.
- Prompt caching is sacred: system prompt byte-stable per session; per-turn dynamic content only via the API-call-time user-message injection channel (memory prefetch fence / pre_llm_call); never inject synthetic mid-loop user messages (role alternation); slash commands that mutate prompt state default to next-session effect.
- Only ONE external memory provider may be active (MemoryManager enforces); provider tool names must not shadow core tools; prefetch budget 8s, sync on a single serialized background worker with 5s shutdown drain — provider must never block the turn.
- Config: all non-secret settings in config.yaml (DEFAULT_CONFIG in hermes_cli/config.py; three loaders — cli.py load_cli_config, hermes_cli/config.py load_config, gateway raw YAML); secrets only in ~/.hermes/.env (OPTIONAL_ENV_VARS metadata); no new HERMES_* env vars for behavior; all state paths via get_hermes_home() for profile isolation.
- Policy: plugins/memory/ in-tree set is CLOSED — new memory backends ship as standalone repos into ~/.hermes/plugins/ or pip entry points; plugins must not modify core files; new core tools are last-resort (Footprint Ladder: extend > CLI+skill > check_fn-gated tool > plugin > MCP catalog > core tool); no speculative hooks without a consumer (but a stated real use case justifies widening the plugin surface upstream).
- Cron/subagent contexts run skip_memory=True — the brain will not passively observe cron or subagent sessions (subagent results arrive via on_delegation on the parent); cron sessions have a 3-minute hard interrupt.
- Skills authoring standards are HARDLINE (description ≤60 chars, section order, platforms gating, scripts/references/templates layout) — brain-authored skills must comply or the curator/reviewers will fight them.

## Opportunities

- Capacity: the entire durable semantic memory is ~3,575 characters (MEMORY.md 2,200 + USER.md 1,375) of §-delimited plain text with LRU-ish manual consolidation — a real brain with unbounded graded storage + salience-ranked retrieval trivially supersedes it while still feeding a small high-signal digest into the same system-prompt slot.
- Retrieval: cross-session recall is keyword FTS5 (BM25) only — no embeddings, no semantic clustering, no temporal decay, no entity linking; session_search returns raw message windows the model must re-read. A brain with hybrid semantic retrieval served through prefetch() (proactive, zero-tool-call recall) changes the economics of every turn.
- The learning loop is prompt-driven and unverified: background review re-runs the whole conversation through an LLM with 'be ACTIVE' prompts, no dedup/consolidation memory of what it already saved, no confidence tracking, no forgetting policy beyond char-limit pressure and curator staleness archiving. A brain can do write-time dedup/merge, provenance, contradiction detection, and confidence decay.
- Rich but unmined signal: turn_outcomes (outcome, retries, guardrail halts, cost deltas, skills_loaded, reaction feedback), skill .usage.json telemetry, and full transcripts sit in WAL SQLite that nothing continuously mines — an offline consolidation daemon (cron 'dream cycle') can build episodic→semantic→procedural distillation Hermes simply doesn't have.
- User modeling currently outsourced to Honcho (network SaaS, API key, 8s prefetch ceiling, one-provider slot) — a local-first brain occupying the same MemoryProvider slot removes the network dependency and merges user modeling with agent self-knowledge in one store.
- Reflection triggers are shallow heuristics (regex correction detection, 3-failure streak, thumbs-down); a brain observing pre/post_tool_call streams can detect repeated inefficiencies, tool-choice mistakes, and plan-quality patterns across sessions, not just within-turn failure streaks.
- Skill library grows without semantic organization: curator only archives by staleness and flags overlap for a later LLM pass; a brain with an actual knowledge graph can own skill consolidation, cross-linking (related_skills), and promotion of episodic lessons into class-level skills with evidence counts.
- Cross-surface continuity: memory writes during a session don't reach the prompt until next session, and subagents/cron are memory-blind — a brain daemon shared across profiles/surfaces (CLI, gateway platforms, desktop, kanban workers via tenant memory-keys) can be the continuity layer the per-process MemoryStore cannot be.
