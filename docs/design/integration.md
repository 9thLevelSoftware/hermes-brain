All facts verified. Here is the design document.

---

# Hermes-Brain — Integration, Process Architecture & Delivery Design

**Design area:** how the brain ships, runs, and integrates with Hermes Agent. All contract claims below were re-verified against `C:\Users\dasbl\hermes-agent` source on 2026-07-16; file:line references are to that checkout.

**One paragraph of orientation.** Hermes-Brain is a single Python package that lives at `~/.hermes/plugins/brain/` and plays four roles from one codebase: (1) an in-process `MemoryProvider` (the fast path: lane-1 prompt block, lane-2 prefetch, passive capture); (2) a short-lived sleep-time worker invoked as `hermes brain dream` (the slow path: consolidation, distillation, forgetting, skill promotion); (3) a `hermes brain` CLI for humans; (4) a stdio MCP server (`mcp_server.py`) for non-Hermes agents. All four open the same WAL SQLite file under `get_hermes_home()/brain/`. **There is no resident daemon.** Every verified constraint that forced a decision is called out inline.

---

## 0. Verified integration facts that shaped this design

These are load-bearing; the implementer should re-confirm each at build time (§6.7 lists the files).

| # | Fact | Where verified |
|---|------|----------------|
| F1 | User memory-provider plugins load through `plugins/memory/__init__.py::_load_provider_from_dir`, which calls the plugin's `register(ctx)` with a **`_ProviderCollector` fake context whose `register_tool` / `register_hook` / `register_cli_command` are no-ops** (lines 306–347). The general `PluginManager` **skips** plugins auto-coerced to `kind="exclusive"` (any user plugin whose `__init__.py` mentions `register_memory_provider`/`MemoryProvider` — `hermes_cli/plugins.py:1594–1612`, skip at 1397–1407). **Consequence: the brain gets ONLY the MemoryProvider surface. No `pre_llm_call`, no `pre/post_tool_call`, no `ctx.llm`, no `ctx.register_cli_command` from the same plugin.** |
| F2 | This is fine for capture: `sync_turn(..., messages=...)` receives the **full OpenAI-style transcript including assistant tool calls and tool results** (`memory_provider.py:116–132`, dispatched with messages when the signature accepts it, `memory_manager.py:606–674`). Mid-turn tool observation is unavailable; post-turn observation is complete. |
| F3 | Prefetch: external providers get a **hard 8s timeout** (`_EXTERNAL_PREFETCH_TIMEOUT_S = 8.0`, `memory_manager.py:47`), run on a named thread, **skipped entirely while a previous prefetch is still in flight** (551–556). The result is fenced by `build_memory_context_block()` (`memory_manager.py:337–351`) — provider output must be **plain text, no fence tags** (pre-wrapped fences are stripped with a warning). Injection happens at API-call time only, into the current turn's user message copy — never persisted (`conversation_loop.py:806–830`). |
| F4 | `system_prompt_block()` is assembled into the system prompt via `agent._memory_manager.build_system_prompt()` (`system_prompt.py:499–501`) and the system prompt must be **byte-stable for the whole session** (issue #13631; the whole reason Honcho's cadence refresh broke caching). |
| F5 | Writes are serialized: `sync_all` / `queue_prefetch_all` / session-boundary end→switch all run on **one daemon worker** (`memory_manager.py:678–726, 846–893`) with a **5s drain at shutdown** (`_SYNC_DRAIN_TIMEOUT_S`, line 46). Anything slower than ~5s at exit can be killed mid-flight → the brain must never rely on exit-time LLM work (§4.2). |
| F6 | `"memory"` is a **reserved core tool name** (`toolsets.py:54`); `add_provider` rejects shadowing tools unconditionally (`memory_manager.py:417–434`). The Anthropic-shaped file tool must be named something else (`memories`, §3.2). |
| F7 | Provider tool names route through `MemoryManager._tool_to_provider`; schemas are OpenAI-function-shaped, normalized by `normalize_tool_schema` (bare `{"name","description","parameters"}` is the safe shape). |
| F8 | Config activation: `memory.provider: "brain"` in `config.yaml` (`hermes_cli/config.py:2261–2265`). Built-in knobs that matter: `memory.memory_enabled`, `memory.user_profile_enabled`, `memory.nudge_interval` (default 10, read in `agent_init.py:1436`; `0` disables the memory-nudge→background-review cadence, `turn_context.py:376–380`), `memory.write_approval`. |
| F9 | Setup wizard: `hermes memory setup` consumes `get_config_schema()`, calls `save_config(values, hermes_home)` for non-secrets, secrets to `.env`, and **delegates entirely to `post_setup(hermes_home, config)` if the provider defines it** (`hermes_cli/memory_setup.py:179–365`). |
| F10 | CLI: `plugins/memory/__init__.py::discover_plugin_cli_commands` loads `cli.py` of the **active** provider only and registers `hermes <dirname>` from `register_cli(subparser)` + `<dirname>_command(args)` (lines 365–461). **The plugin directory name is the provider name is the CLI verb** → directory must be `brain` to get `hermes brain …`. Bundled names win collisions; `brain` collides with nothing bundled. |
| F11 | Auxiliary LLM lane: memory plugins can't get `ctx.llm` (F1), but in-process code may import `agent.auxiliary_client.call_llm` directly — the exact engine behind `ctx.llm` and `background_review`/`curator` (per `agent/plugin_llm.py:55–57`, `config.py:1765–1773` for the per-task `{provider, model, base_url, api_key, timeout, reasoning_effort}` shape). This is the brain's LLM lane for both session-sweep extraction and dreaming. MCP sampling (on by default, `tools/mcp_tool.py:2782–2785`) is the LLM lane **only** for the external-agent MCP surface, not for the provider. |
| F12 | Cron scheduler is **gateway-resident** (`cron/scheduler.py:4` — "Provides tick()… The gateway" ticks it). CLI-only users have no cron. Cron supports `no_agent=True` script jobs where "the script IS the job" (`cron/jobs.py:1074–1139`); agent cron jobs run `skip_memory=True` and have a 3-minute interrupt — **never** schedule the dream as an agent session. |
| F13 | Gateway hooks live at `~/.hermes/hooks/<name>/{HOOK.yaml, handler.py}`, events include `gateway:startup`, `agent:end`, `session:*` (`gateway/hooks.py:1–49`). Available as an optional accelerant, not a dependency. |
| F14 | `state.db` at `get_hermes_home()/state.db`, `SCHEMA_VERSION = 22` (`hermes_state.py:132–134`), WAL; safe for concurrent read-only access. |
| F15 | `hermes backup` walks HERMES_HOME; `backup_paths()` is only for state stored **outside** it (`memory_provider.py:299–315`). |

---

## 1. Process architecture

### 1.1 Decision

**Hybrid (d), minimal form: in-process provider + short-lived spawned dream processes. Zero resident processes beyond what Hermes already runs.**

```
┌─────────────────────────────── Hermes agent process ───────────────────────────────┐
│  BrainProvider (MemoryProvider)                                                    │
│    fast path (ms):  prefetch() ← cached result   sync_turn() → episodic append     │
│    one owned worker thread ("brain-bg"):  embeds new turns, runs queued retrieval, │
│      micro-maintenance (watermarks, salience tags). No LLM calls on this thread    │
│      except the bounded session-sweep (§4.2), via auxiliary_client.call_llm.       │
│    on initialize()/shutdown(): if dream overdue → spawn detached                   │
│      `hermes brain dream --if-due` and forget about it                             │
└───────────────┬─────────────────────────────────────────────────────────────────--─┘
                │  same SQLite file, WAL, busy_timeout=5000
┌───────────────┴───────────────┐      ┌────────────────────────────────────────────┐
│  ~/.hermes/brain/brain.db     │◄─────┤ `hermes brain dream` (short-lived process) │
│  (+ brain.yaml, exports/)     │      │  triggered by: cron no_agent script (gate- │
└───────────────┬───────────────┘      │  way users) | --if-due spawn (CLI users) | │
                │                      │  `hermes brain dream-now` (manual)         │
┌───────────────┴───────────────┐      │  LLM: auxiliary_client.call_llm, stronger  │
│ mcp_server.py (stdio, spawned │      │  model via brain.yaml dream.model override │
│ per-session by Claude Code /  │      └────────────────────────────────────────────┘
│ any MCP client; dies with it) │
└───────────────────────────────┘
```

### 1.2 Why the alternatives lose

- **(a) Threads inside the gateway process (curator precedent):** the gateway is absent for CLI-only users, and heavy sleep-time work (LLM adjudication over hundreds of memories, re-embedding, graph consolidation) inside the chat-serving process risks latency and RAM on the $5-VPS/Termux floor. The curator precedent is real but curator work is small. We keep only *light* background work in-process (the provider's single worker), which is exactly the pattern `MemoryManager` already blesses (F5).
- **(b) Resident brain daemon (gateway:startup / health-check-spawn):** pays rent it can't afford. It buys mid-turn observation (unavailable anyway per F1), sub-second cross-process IPC (unneeded — same-file SQLite is our IPC), and always-warm models (solvable with lazy load + OS page cache). It costs: Windows service management (no `fcntl`, no `systemd`), Termux background-process murder by Android, crash/restart supervision, port-or-socket auth, per-profile daemon multiplexing, and a second thing for `hermes brain doctor` to debug. Rejected. **Escape hatch documented:** if a future feature genuinely needs residency (e.g. sub-100ms cross-agent shared cache), the `gateway:startup` hook (F13) can launch it for gateway users without touching this design.
- **(c) Cron alone:** correct for gateway users, nonexistent for CLI-only users (F12). Kept as *one of two triggers*, not the mechanism.

### 1.3 The dream trigger contract (covers every deployment shape)

1. **Gateway users:** `hermes brain setup` (post_setup hook, F9) offers to create a cron job: `no_agent=True`, `script=~/.hermes/scripts/brain-dream.sh` (contents: `hermes brain dream --if-due --quiet`), schedule default `03:30` local daily. `no_agent` because the dream must not be an agent session (F12: skip_memory, 3-min interrupt, token cost).
2. **CLI-only users:** `BrainProvider.initialize()` and `shutdown()` check `meta.last_dream_completed`; if older than `dream.interval_hours` (default 20h) **and** no dream lock is held, spawn `hermes brain dream --if-due` detached: Windows `creationflags=DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP|CREATE_NO_WINDOW`, POSIX `start_new_session=True`, stdout/err to `~/.hermes/brain/logs/dream.log`. Fire-and-forget; the spawn itself is <10ms and never blocks a hook.
3. **Manual:** `hermes brain dream-now` (runs foreground with progress output).
4. `--if-due` re-checks the watermark *inside* the lock so triple-triggering is harmless.

### 1.4 SQLite multi-process discipline

- **One DB:** `get_hermes_home()/brain/brain.db`, `PRAGMA journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`, `wal_autocheckpoint` default; dream process runs `PRAGMA wal_checkpoint(TRUNCATE)` at end.
- **Writer roles, not writer locks:** WAL already serializes writers; we just keep transactions short. The agent process writes only appends (episodic rows, embeddings, outcome updates) in single-statement transactions. The dream process does batched read-transform-write in transactions capped at ~50 rows so it never starves an interactive `sync_turn` for more than milliseconds.
- **At most one dream process:** an `O_CREAT|O_EXCL` lockfile `brain/dream.lock` containing `{pid, started_at, host}`. Stale detection: if the PID is dead (`psutil` if present, else `OpenProcess`/`kill(pid,0)` per platform) or `started_at` > 6h old, break the lock. **No `fcntl`** — `O_EXCL` + PID-liveness is fully portable (Windows/Termux/Linux/macOS) and is the same class of mechanism Hermes itself uses (`msvcrt` fallbacks exist repo-wide, but we don't even need byte-range locking).
- **Crash recovery:** WAL self-heals on next open. Dream idempotence via watermark tables (`sweep_state`: last processed `state.db` message rowid per session; `dream_state`: last completed phase + run id). A crashed dream re-runs its phase from the watermark; every dream mutation is tagged with `run_id` so `hermes brain doctor` can report and `why` can attribute.
- **Profile isolation:** everything keys off `hermes_home` passed to `initialize()` (F: `memory_manager.py:1178–1195` injects it) — never a hardcoded `~/.hermes`. The dream spawn passes the active profile through explicitly: `hermes brain dream` resolves `get_hermes_home()` itself, and the spawner sets `HERMES_HOME` in the child env to pin the same profile.
- **Termux/low-RAM:** `mode: lite` (auto-detected at setup: total RAM < 1.5GB or Termux markers) — potion static embeddings, no reranker, dream batch sizes halved, dream deferred to explicit `dream-now` unless on AC power is detectable (don't bother detecting; just default `--if-due` interval to 48h in lite mode).

---

## 2. The two-lane cache-safe injection contract (invariant #1)

This is the direct fix-pattern for hermes-agent issue #13631, and it maps 1:1 onto the two channels Hermes already provides (F3, F4).

### 2.1 Lane 1 — `system_prompt_block()`: frozen brain index

**Byte-stability mechanics (non-negotiable):**
- Rendered **once** in `initialize()` from the `lane1_snapshot` table (a materialized rendering the dream cycle and session-boundary hooks maintain — *not* rendered live from memory tables), cached as `self._lane1_block: str`, and returned unchanged from every `system_prompt_block()` call for the process/session lifetime.
- Refresh points, exactly: (a) `initialize()` (new session/process); (b) `on_session_switch(reset=True)` (a genuinely new conversation — `/new`, `/reset`); the refreshed block applies because Hermes rebuilds the system prompt for the new session. On `on_session_switch` with `reset=False` (resume/branch/**compression**) the cached block is **kept byte-identical** — compression must not change the prefix.
- Golden test in the repo: run 50 simulated turns incl. a compression event; assert `system_prompt_block()` returned one distinct string.

**Content & budget** (default 1,200 tokens ≈ 4,800 chars; config `lane1_tokens` 800–1500, hard-truncated by the renderer with deterministic ellipsis so the budget is a guarantee, not a hope):

```
## Brain (persistent memory) — session index
You have a persistent brain. Everything below is an INDEX — drill down with
brain_recall before acting on stale or truncated items.

### ⚠ Failures & warnings (avoid repeating)          [~350 tok, always first]
- [w-0412] pip install inside Termux session kills gateway — use pkg. (3×, last 2026-07-02)
- [w-0398] User's Slack workspace strips code blocks >80 lines. (1×)

### ◔ Open loops — outcomes unknown                    [~200 tok]
- [d-0521] Chose LanceDB fallback threshold 1M vecs (2026-07-10) — outcome unrecorded.
  If resolved, call brain_outcome(id, worked|failed).

### ● Standing facts & preferences                     [~450 tok]
- [u-0007] User: Devil; timezone America/Chicago; prefers terse answers, no emojis.
- [p-0102] Project Hermes-Brain: single SQLite, no Qdrant. …

### Index stats & drill-down                           [~100 tok]
1,204 memories (312 facts, 87 procedures, 41 warnings) across 5 projects.
brain_recall(query) searches all of it; depth:"deep" for graph traversal.
Recent context arrives automatically each turn in <memory-context>.
```

Ordering inside each section is deterministic (salience desc, then id) so re-renders are stable when content hasn't changed. IDs are short and stable — they are the drill-down handles and the vocabulary the nudges use (Daem0n's briefing-as-index lesson, minus its recency-only selection: selection is salience-scored by the dream cycle, failures pinned first).

### 2.2 Lane 2 — `prefetch()`: per-turn ephemeral recall

**Mechanics:** `queue_prefetch(query)` (called post-turn on the serialized worker) runs the real retrieval — RRF over FTS5 + sqlite-vec top-50, optional rerank — and caches the rendered string. `prefetch(query)` computes a fresh retrieval only if the cache is missing/stale for the query (first turn of a session), else returns the cache; either way it must comfortably beat 8s (target p95 < 300ms full mode, < 80ms lite). Return **plain text** (F3: the manager fences it and strips any fence we add). If nothing clears the relevance floor, return `""` — an empty lane 2 is cache-free by construction (no fence is injected at all).

**Format** (stable, parse-friendly, budget default `lane2_tokens: 600`, configurable 0–2000; 0 disables lane 2 entirely):

```
Recalled for this turn (top 4 of 23 matches; not new user input):
[w-0412 | warning | 14d | ★★★ | cli] Termux pip install kills gateway — use pkg install. 
  ↳ recorded after gateway crash 2026-07-02; brain_recall(id="w-0412") for full incident.
[f-0883 | fact | 3d | ★★☆ | telegram] User said the VPS is Hetzner CX22, 2GB RAM.
[e-1204 | episode | 21d | ★☆☆ | cli] Similar task: added FTS5 triggers to kanban.db (worked).
Nudge: open loop d-0521 matches this topic — record outcome if known.
```

Line grammar: `[id | kind | age | confidence ★ | source-platform] one-line verbatim-or-summary`. Verbatim bodies only for `warning`/`decision` kinds (they must not be paraphrased); episodes get one-line summaries + drill-down id. At most **one** nudge line per turn, and the same nudge id at most twice per session (anti-nag cap, Daem0n lesson). Provenance tier gates content: memories whose source is quarantined (untrusted platform author, §4.5) are **never** rendered into lane 1 or lane 2 — tool-recall only, flagged.

**Update timing summary:** Lane 1 changes only at session boundaries; everything volatile — retrieval, freshness, nudges, outcome prompts — rides lane 2, which is injected into a copy of the user message at API-call time and never persisted (F3). This is exactly the #13631 contract; there is no third channel and we must never want one (`pre_llm_call` is unavailable to us anyway, F1).

---

## 3. Tool surface

### 3.1 Hermes tools via `get_tool_schemas()` — five tools, no unions of unions

Daem0n's lesson: 8 verb tools beat 67, but flat 28-param unions inside a verb are the same disease one level down. Budget: **≤5 tools, ≤6 params each, every param documented for exactly one purpose.** Bare-function schemas (F7), names prefixed `brain_`/`memories` (no core-name collisions, F6).

1. **`brain_recall`** — `query` (req), `depth` (`"quick"`|`"deep"`, default quick: quick = RRF top-k index lines; deep = +graph neighbors, +episode bodies, slower), `kind` (optional filter enum), `project` (optional scope), `limit` (default 8, max 25). Returns index-format lines (same grammar as lane 2) + `total_matches` + `hint` ("refine with kind=… / drill with id=…"). Also accepts `id` for direct drill-down (mutually exclusive with `query`; the schema documents this and the error teaches it).
2. **`brain_remember`** — `content` (req), `kind` (`fact|decision|preference|warning|insight`, req), `tags` (array, opt), `project` (opt), `ttl_days` (opt — explicit expiry for known-transient facts). Returns `{id, deduped_against?}` — write-time dedup happens silently and reports what it merged with.
3. **`brain_outcome`** — `id` (req), `outcome` (`worked|failed|mixed`, req), `note` (opt). Closes open loops; feeds self-tuning. This is the only "learning" verb the model sees.
4. **`brain_manage`** — `action` (`forget|pin|unpin|incognito_on|incognito_off`, req), `id` (req for forget/pin/unpin), `reason` (opt, stored as provenance). `forget` is soft (tombstone; excluded from all retrieval; purged by dream after `forget_grace_days`, default 30 — distill-don't-delete, reversible via CLI). `incognito_on` suspends all capture for the session and is announced in the tool result so the model can confirm to the user. This tool exists because gateway users manage memory *through chat* ("forget that"); the richer audit UX is CLI-only (§5.3).
5. **`memories`** — the Anthropic-memory-tool-shaped file interface: `command` (`view|create|str_replace|insert|delete|rename`), `path` (under `/memories`), plus the standard per-command params (`file_text`, `old_str`/`new_str`, `insert_line`/`insert_text`). Named `memories` because `memory` is reserved (F6). It maps onto **virtual views** of brain storage, not real files: `/memories/profile.md` (lane-1 standing facts, editable — edits become `brain_remember`/supersede operations), `/memories/index.md` (read-only rendered index), `/memories/topics/<tag>.md` (materialized per-tag digests; `create`/`str_replace` translate to remember/edit with `tags=[tag]`). On Claude models this inherits trained memory-tool behavior nearly for free (the trained shape is the command grammar + path convention more than the exact tool name); on other models it's a harmless secondary interface. Ship in phase 3, gated by config `memories_tool: true`.

**Deliberately NOT exposed to the model:** graph surgery (edges are dream-owned), embedding/index admin, forgetting policy knobs, config, export/import, dream triggers, Daem0n-style `consult(action=…, 28 params)` anything, and no "search the raw episodic log" tool (that's `brain_recall depth:"deep"`'s job with sane defaults). The model surface is: *recall, remember, record outcome, manage, files*. Everything else is CLI or automatic.

**Errors that teach (convention, enforced by a helper):** every tool error returns `{"error": "...", "recovery_hint": "..."}` where the hint contains a *complete corrective call*, e.g. `{"error": "unknown kind 'note'", "recovery_hint": "valid kinds: fact|decision|preference|warning|insight — e.g. brain_remember(content=..., kind=\"fact\")"}`. Same convention on the MCP surface.

### 3.2 MCP server surface (external agents: Claude Code, etc.)

`mcp_server.py`, stdio, FastMCP, console-script `hermes-brain-mcp` (and `python -m hermes_brain.mcp_server`). Registered by the *client* (e.g. Claude Code `mcpServers` entry, or Hermes's own `mcp_servers:` if someone wants the brain in a second Hermes profile read-only). Spawned per client session, dies with it — no lifecycle management (§1.1).

- Exposes the **same five tools** (same names, same schemas, same store) plus `brain_status` (read-only: counts, last dream, mode) so external agents can orient.
- Profile selection via `HERMES_HOME` env in the server entry, defaulting to the default profile.
- **Writes from external agents are provenance-tagged** `source=mcp:<client-name>` and land at trust tier "external" — retrievable, but never promoted into lane 1 by the dream without corroboration (§4.5).
- MCP **sampling** (`sampling/createMessage`) is used *opportunistically and only here*: if the connected host advertises sampling (Hermes does, on by default, F11/`mcp_tool.py:2782`), `brain_recall depth:"deep"` may use one bounded sampling call for query decomposition, and a `brain/consolidate` MCP *prompt* is exposed for hosts that want to donate compute. The brain must remain fully functional with sampling absent (Claude Code today) — sampling is an accelerant, never a dependency.

### 3.3 What the brain does NOT do to Hermes

No core-file modification, no new core tools, no slash commands, no second general-purpose plugin (F1 makes a companion `standalone` plugin *possible* — a dir whose `__init__.py` avoids the magic strings — but it would only buy `pre/post_tool_call` mid-turn observation that F2 shows we don't need; KISS says don't ship two plugins to do one plugin's job).

---

## 4. Capture-side automation (the enforcement triad, Hermes-shaped)

Daem0n enforced memory with client hooks + hard gates. Hermes gives us something better: **the provider hooks make capture unconditional and invisible** — zero agent cooperation for lanes and capture; the only thing the model is ever *asked* to do is record outcomes and explicit remembers. Adaptation of the triad: (1) session-start = lane 1 index (automatic), (2) pre-action recall = lane 2 prefetch (automatic, replaces Daem0n's preflight hard-gate), (3) post-action write-back = sync_turn capture (automatic) + soft outcome nudges (ship-inert). **Nothing gates hard initially.**

### 4.1 Passive capture (zero cooperation)

- **`sync_turn(user, assistant, messages=…)`** — the workhorse. Appends an episodic turn record (user text, assistant text, tool-call digest extracted from `messages` per F2: tool name, args hash, ok/error, duration if present), computes cheap salience heuristics inline (<5ms: error markers, correction phrases, decision verbs, user-emphasis markers), and queues embedding on the brain worker. Runs on the manager's serialized worker already (F5) — must stay well under a second per turn.
- **`on_pre_compress(messages)`** — archives the about-to-be-discarded messages verbatim into the episodic store (append-only; cheap disk is the point of distill-don't-delete), returns a ≤300-token string of brain-extracted insights for the compression summary prompt (the hook's return contract, `memory_provider.py:220–230`).
- **`on_delegation(task, result, child_session_id)`** — capture the pair as an episode of kind `delegation`; this is the only visibility into subagent work (cron/subagents are `skip_memory=True`).
- **`on_memory_write(action, target, content, metadata)`** — during the transition period (built-in memory still on), mirror every built-in write into the brain with provenance `builtin-mirror` so nothing is lost when the built-in is later disabled.
- **Incognito:** when active (via `brain_manage` or `hermes brain incognito`), `sync_turn`/`on_pre_compress`/`on_session_end` write nothing; a session-scoped marker with hard TTL guarantees it can't leak into the dream sweep (table-stakes UX per products research; provably bypasses capture because capture is one code path).

### 4.2 Session-end extraction — designed around the 5s drain (F5)

`on_session_end` does **not** run LLM extraction. It synchronously writes a `session_closed` marker row (<10ms) and returns. Extraction runs out-of-band as the **sweep**: the dream process (and, opportunistically, the brain worker when the agent process is alive and idle ≥60s with `sweep_inline: true`, default on for non-lite mode) reads unswept sessions from the episodic store + `state.db` transcripts (read-only, F14), runs one bounded `auxiliary_client.call_llm` extraction per session (facts/decisions/warnings/preferences, JSON-schema output), and advances the `sweep_state` watermark. Rationale: CLI exits, crashes, and gateway session expiry all produce identical results — extraction is never lost to a dying process, merely deferred. This one decision removes the entire class of "shutdown raced my LLM call" bugs the Hindsight 298s incident exemplifies (`memory_manager.py:630–641`).

### 4.3 Reflection-trigger equivalents (mined, not hooked)

The dream/sweep mines `state.db` **read-only**: `turn_outcomes` (outcome, retries, guardrail halts, cost, `skills_loaded`, reaction feedback) joined to transcripts. Failure streaks, user-correction turns, and thumbs-down reactions auto-generate `warning`/`insight` candidates tagged `auto`, at reduced confidence, deduped against existing warnings. This supersedes nothing in Hermes — `reflection_triggers.py` keeps firing background review for *skills* (locked decision 4); the brain just mines the same signals for *memory* on its own cadence, with no regex-transcript-scraping in the hot path (Daem0n's low-quality Stop-hook capture lesson: extraction quality comes from the offline LLM pass, not inline regex).

### 4.4 When we still nudge (ship-inert)

- Open decisions with unrecorded outcomes surface in lane 1 (§2.1 "Open loops") and as the capped lane-2 nudge line. That's it. No preflight tokens, no blocking. If observation shows outcomes go unrecorded, escalation options exist (e.g. a stronger lane-1 instruction) — but they ship off.
- Every autonomous dream behavior (auto-warning promotion, forgetting purges, skill drafting) ships with `dry_run: true` for its first release phase, logging intended actions to `~/.hermes/brain/logs/` for review via `hermes brain why` (Daem0n's ship-inert convention).

### 4.5 Security posture (capture is the attack surface)

Every memory row carries `source` (platform, author id, session), `trust` (`operator` > `agent` > `external-mcp` > `untrusted-platform-peer`). Instruction-shaped content ("ignore previous…", imperative-to-the-assistant heuristics) from non-operator sources is quarantined at write time: stored, never rendered into lane 1/lane 2, tool-recall returns it flagged `⚠ quarantined`. Group-chat peers get per-peer write isolation (their statements become facts *about them*, never global preferences). This is day-one schema, not a later phase (SpAIware/ZombieAgent lesson).

### 4.6 Hermes built-ins: disable/keep matrix (locked decision 4)

| Built-in | Setting | Phase 1–2 (transition) | Phase 3+ (brain owns memory) |
|---|---|---|---|
| Built-in MEMORY.md/USER.md + `memory` tool | `memory.memory_enabled`, `memory.user_profile_enabled` | **Keep on**; brain mirrors via `on_memory_write` | **Off** (`false`/`false`) after `hermes brain bootstrap` imports both files; brain's lane 1 replaces the frozen blocks |
| Memory nudge → background-review fork | `memory.nudge_interval` | Keep default (10) | **`0`** — the brain's sweep replaces cadence-driven memory review |
| Honcho / any external provider | `memory.provider` | n/a | **`"brain"`** (the slot is exclusive — setting it *is* the Honcho off-switch) |
| Skill nudges, `reflection_triggers` → background review for skills, curator, skills telemetry | skills/curator config | **Keep, untouched** | **Keep, untouched** — the brain *feeds* this loop (drafts SKILL.md candidates with provenance frontmatter, §6 P5); curator remains the janitor |
| `session_search` core tool | — | Keep | Keep (harmless; brain_recall is better but session_search costs nothing) |

`hermes brain setup` prints this matrix and offers to apply the phase-appropriate column; `hermes brain doctor` warns on drift (e.g. brain active but nudge_interval still 10 in phase 3+).

---

## 5. Config, CLI, setup, packaging

### 5.1 Repo layout (this repo, `C:\Users\dasbl\Documents\Hermes-Brain`)

The repo root **is** the plugin directory — install is `git clone <repo> ~/.hermes/plugins/brain` (dir name `brain` is load-bearing, F10).

```
Hermes-Brain/                     # → ~/.hermes/plugins/brain/
  plugin.yaml                     # name: brain, description, pip_dependencies (lazy-safe list)
  __init__.py                     # register(ctx): ctx.register_memory_provider(BrainProvider())
                                  #   MUST be import-light: no onnx/vec imports at module load
  cli.py                          # register_cli(subparser) + brain_command(args)   (F10)
  provider.py                     # BrainProvider (MemoryProvider impl; thin — delegates)
  store/                          # db.py (open/pragmas/migrations), schema.sql, fts.py, vec.py, graph.py
  recall/                         # retrieve.py (RRF), rerank.py, embed.py (tiered), render.py (lane1/lane2/index grammar)
  capture/                        # turns.py (sync_turn path), sweep.py (session extraction), salience.py, quarantine.py
  dream/                          # run.py (phases+locks+watermarks), consolidate.py, distill.py, forget.py, mine_state.py, skills.py
  mcp_server.py                   # stdio FastMCP entry
  bootstrap/                      # memory_md.py, state_db.py, daemon_import.py
  pyproject.toml                  # name: hermes-brain; console_scripts: hermes-brain-mcp; extras: [full],[lite]
  replay/                         # harness (§6) — dev tool, ships in repo
  tests/
```

Pip-install path (secondary): `pip install hermes-brain` + `hermes-brain init-plugin` which symlinks/copies the package into `~/.hermes/plugins/brain/` (the memory-plugin discovery scans directories, not entry points — F: `plugins/memory/__init__.py:109–119`; the pip entry-point lane is for `hermes-brain-mcp` only).

**Import discipline:** `__init__.py` + `provider.py` import only stdlib + `agent.memory_provider` at module load (the discovery availability check imports the module, `plugins/memory/__init__.py:178–187`). `is_available()` checks config + files only, no network, no heavy imports (contract at `memory_provider.py:54–59`). onnxruntime/sqlite-vec load lazily on first use; missing deps degrade to FTS-only mode with a one-line lane-1 notice and a `doctor` finding — the brain must never make Hermes fail to start.

### 5.2 `get_config_schema()` wizard fields (consumed by `hermes memory setup`, F9)

```
mode              choices [auto, full, lite, fts-only]   default auto     (auto = RAM/platform detect)
lane1_tokens      default 1200                                            (800–1500)
lane2_tokens      default 600                                             (0 disables lane 2)
dream_schedule    choices [cron, on-idle, manual]        default auto     (cron if gateway detected)
dream_time        default "03:30"
dream_model       default ""                                              (auxiliary override — stronger model; empty = active model)
bootstrap_import  choices [yes, no]                      default yes      (MEMORY.md/USER.md + state.db backfill on first run)
memories_tool     default true                                            (the Anthropic-shaped file tool)
```
No secrets → nothing to `.env`; `save_config()` writes `~/.hermes/brain/brain.yaml`. `post_setup(hermes_home, config)` (F9) then: creates dirs, downloads/validates the embedding model with a progress bar and an explicit skip option (skip → fts-only until `hermes brain doctor --fix`), runs bootstrap if accepted, offers the cron job (§1.3), sets `memory.provider: "brain"`, and prints the built-ins matrix (§4.6).

Model files live in a **shared, non-backed-up cache**: `~/.cache/hermes-brain/models/` (Windows `%LOCALAPPDATA%\hermes-brain\models`), shared across profiles, re-downloadable — deliberately outside HERMES_HOME so `hermes backup` archives memories, not 300MB of ONNX (F15). `backup_paths()` returns `[]`; all state is under HERMES_HOME already.

### 5.3 `hermes brain` CLI verbs (via F10; one `cli.py`, argparse subcommands)

- `status` — counts by kind/trust, DB size, mode, last dream/sweep, lane budgets, lock state.
- `search <query> [--kind --project --deep]` — same retrieval as `brain_recall`, human-rendered.
- `why <id>` — full provenance chain: source turn/session/platform, dedup merges, dream runs that touched it, outcome links. (The auditability answer to "why does it believe this?")
- `remember / forget <id> [--hard] / pin <id>` — human-side writes; `forget --hard` purges immediately; plain `forget` tombstones (§3.1).
- `export [--format jsonl|md] [--out dir]` / `import <file>` — plain-file exports: JSONL (lossless, re-importable) + a human-readable markdown snapshot tree (`profile.md`, `warnings.md`, `topics/*.md`) into `~/.hermes/brain/exports/<date>/`.
- `doctor [--fix]` — integrity check (PRAGMA quick_check), FTS/vec index consistency, model presence/hash, stale locks, config drift vs §4.6 matrix, orphaned watermarks.
- `dream-now [--phase consolidate|distill|forget|skills] [--dry-run]` and internal `dream --if-due --quiet`.
- `bootstrap [--daemon <path>]` — re-runnable first-run import; `--daemon` triggers Daem0n import (below).
- `incognito [on|off|status]`.

### 5.4 Bootstrap story

First `initialize()` with an empty DB (or `hermes brain bootstrap`):

1. **MEMORY.md / USER.md import** — parse the §-delimited entries (`~/.hermes/memories/` per built-in layout; re-verify exact paths in `agent/` at build time), each becomes a memory (`kind` inferred, `source=builtin-import`, trust `operator`). USER.md entries seed the lane-1 profile section.
2. **state.db backfill** — read-only walk of `sessions`/`messages`/`turn_outcomes` (F14; open with `mode=ro` URI + its own busy_timeout), oldest→newest, writing episodic rows + salience tags and setting `sweep_state` watermarks; extraction of old sessions is then just normal sweep work the next dreams chew through (rate-limited: `dream.backfill_sessions_per_run`, default 20, so a 2-year history doesn't produce a 4-hour first dream).
3. **Optional Daem0n-MCP import** (`--daemon C:\path\to\project\.daem0nmcp\memory.db`, repeatable per project). Schema mapping sketch (verify against Daem0n v6.6.6 at build time): `memories(id, category, content, tags, importance, created_at)` → brain memories with `kind` mapped (`warning→warning`, `decision→decision`, `pattern/insight→insight`, else `fact`), `project=<dirname>` scope, `source=daemon-import:<project>`, importance → initial salience; `rules`/covenant `must_not` → `warning` kind, pinned; `outcomes` → outcome links on the imported decisions; Daem0n's federation links → inter-project `related` edges. Auto-captured Daem0n spam (`Auto-captured from conversation`) imported at floor confidence, first in line for the forgetting pipeline.

Bootstrap is idempotent (content-hash dedup) and always re-runnable.

---

## 6. Phased implementation roadmap

Five phases; each independently shippable, each verified before the next starts. Size estimates are net-new Python (excl. tests). **Golden rule carried through all phases: the byte-stability test and the sync_turn-latency test run from phase 1 and never leave CI.**

**Replay harness first (part of P1):** `replay/run.py` — loads a recorded session from `state.db` (read-only) or a JSON fixture, instantiates `BrainProvider` against a throwaway HERMES_HOME, and drives the real hook sequence (`initialize → [on_turn_start → prefetch → sync_turn → queue_prefetch]* → on_pre_compress? → on_session_end → shutdown`), printing per-turn: lane-2 text + token count, sync latency, rows written. Flags: `--assert-lane1-stable`, `--budget-check`, `--no-llm` (stubs auxiliary calls). This is how every phase is verified without burning tokens, and it doubles as the regression suite against future hermes-agent contract drift.

**P1 — Passive capture + FTS recall (skeleton that already earns its keep).** ~1,800 LOC.
Provider skeleton (all hooks stubbed safe), `store/` with schema v1 (memories, episodes, FTS5, provenance/trust columns, watermarks, meta), `sync_turn` capture + salience heuristics, `prefetch` lane 2 from FTS5+BM25 only, static minimal lane 1 (fixed instructions block — trivially byte-stable), `cli.py` with `status|search|doctor`, `plugin.yaml`, install docs.
*Verify:* clone to `~/.hermes/plugins/brain`, set `memory.provider: "brain"`, run a CLI session → `hermes brain status` shows captured turns; `hermes brain search` finds them; replay harness over 3 recorded sessions passes stability+budget asserts; `sync_turn` p95 < 200ms; Hermes starts cleanly with onnxruntime absent.

**P2 — Real retrieval + real lane 1 + bootstrap.** ~2,200 LOC.
Tiered embeddings (EmbeddingGemma ONNX int8 256d / potion lite / none), sqlite-vec, RRF fusion, optional mxbai ColBERT rerank, `lane1_snapshot` materialization + renderer with hard budgets, bootstrap (MEMORY.md/USER.md + state.db backfill), model download UX, `mode` auto-detect.
*Verify:* recall@10 on a hand-labeled 50-query set from own history beats P1 FTS-only; lane 1 golden test across compression; fresh-profile bootstrap produces a sensible index (eyeball + `hermes brain why` on 10 items); Termux/lite mode smoke run (or RAM-capped container).

**P3 — Tool surface + sweep extraction + memory UX.** ~2,000 LOC.
Five tools (§3.1) incl. `memories` file view, errors-that-teach helper, session-close marker + out-of-band sweep via `auxiliary_client.call_llm` (bounded, JSON-schema output), `on_pre_compress` archival, `on_delegation`, `on_memory_write` mirroring, incognito, forget/pin, quarantine gate. Transition column of §4.6 becomes the recommended default.
*Verify:* live session — ask the agent to remember/recall/forget and confirm via CLI; kill -9 a session mid-conversation → sweep still extracts it on next run; quarantine test (instruction-shaped memory from a fake group peer never appears in lanes); replay harness `--no-llm` covers the tool dispatch paths.

**P4 — The dream cycle.** ~2,500 LOC.
`dream/run.py` phase machine (lock, watermarks, run_id), consolidation (dedup/merge/contradiction via stronger `dream_model`), distill-don't-delete forgetting (tombstone→purge with grace; episodic log stays append-only), `turn_outcomes` mining (§4.3), lane-1 snapshot refresh as a dream output, cron job creation in setup + `--if-due` opportunistic spawn, all mutating behaviors `dry_run: true` this release.
*Verify:* `hermes brain dream-now --dry-run` over the P1–P3 accumulated corpus → review the action log for a week, then flip dry-run off; concurrent-writer test (dream running while replay harness hammers sync_turn — assert no `database is locked` errors at busy_timeout=5s); crash the dream mid-phase (kill) → rerun completes idempotently; Windows detached-spawn test.

**P5 — Learning flywheel + MCP surface.** ~1,800 LOC.
Skill promotion: dream drafts `SKILL.md` folders (agentskills.io layout, ≤60-char description, `created_by: hermes-brain` + evidence-count frontmatter) into `~/.hermes/skills/` for the curator/review loop to accept or archive — the brain proposes, the existing Hermes skill loop disposes (locked decision 4). Outcome-driven self-tuning (retrieval weight/threshold adjustment from `brain_outcome` + reaction signals, shadow-logged first). `mcp_server.py` + `hermes-brain-mcp`, external trust tier, optional sampling accelerant. Phase-3+ column of §4.6 (built-in memory off) becomes default guidance.
*Verify:* a repeated successful trajectory across ≥3 sessions produces a draft skill that passes `hermes skills` validation and survives curator; Claude Code connects via MCP, recalls a memory written from Telegram (the cross-platform money shot); tuning shadow-log reviewed before activation.

**~10,300 LOC total**, front-loaded so that after P1 (a weekend-scale milestone) the system is already remembering everything.

### 6.7 Files an implementer must re-read at build time

- `agent/memory_provider.py` (whole ABC — the contract), `agent/memory_manager.py` (timeouts, serialization, fencing, reserved names, session-boundary ordering)
- `agent/conversation_loop.py:800–840` (injection), `agent/turn_context.py` (prefetch call site + nudge counters), `agent/system_prompt.py:490–560` (block placement)
- `plugins/memory/__init__.py` (loader reality: `_ProviderCollector`, dir-name = provider name, cli.py discovery) and `plugins/memory/honcho/{__init__,cli}.py` + `plugins/memory/openviking/` (largest local-first reference)
- `hermes_cli/memory_setup.py` (wizard/post_setup), `hermes_cli/config.py:978–2400` (DEFAULT_CONFIG: `memory`, `auxiliary` blocks), `agent/agent_init.py:1420–1445` (nudge wiring)
- `agent/auxiliary_client.py` (`call_llm` signature), `agent/background_review.py` + `agent/reflection_triggers.py` (what stays on), `agent/curator.py` (skill janitor the brain feeds)
- `hermes_state.py` (state.db open discipline, SCHEMA_VERSION guard — pin a version check in `mine_state.py`), `cron/jobs.py:1040–1200` (no_agent job creation), `gateway/hooks.py` (escape hatch), `tools/mcp_tool.py` sampling config keys, `tools/lazy_deps.py` (lazy-dep precedent)

---

**Summary of the five opinionated calls:** (1) no daemon — WAL + short-lived `hermes brain dream` processes with a portable lockfile cover every platform Hermes supports; (2) the two-lane contract is implemented as *materialized snapshot rendered once per session* (lane 1) + *cached background retrieval* (lane 2), making cache-safety structural rather than disciplinary; (3) five model-facing tools, nothing else — management depth lives in the CLI; (4) all LLM-dependent extraction is out-of-band (sweep/dream via `auxiliary_client.call_llm`) because the loader gives us no `ctx.llm` and shutdown gives us 5 seconds; (5) the brain proposes skills and the existing Hermes loop disposes — supersede memory, feed skills, touch nothing else.
