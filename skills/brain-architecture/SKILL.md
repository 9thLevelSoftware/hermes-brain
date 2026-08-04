---
name: brain-architecture
description: "Memory system architecture: brain↔agent boundary, ownership model, and data flow."
version: 1.0.1
metadata:
  hermes:
    tags: [brain, memory, architecture, debugging]
    category: mlops
---

# Brain ↔ Agent Memory Architecture

## Ownership Model

The Hermes memory system has TWO layers that are easy to confuse:

### Layer 1: hermes-agent (the host)
Owns:
- `tools/memory_tool.py` — the `memory` tool (add/replace/remove)
- `MEMORY.md` / `USER.md` — file-backed curated memory stores
- Char limits (`memory_char_limit`, `user_char_limit`) and overflow handling
- Write approval gate (`tools/write_approval.py`)
- Memory tool JSON schema (`MEMORY_SCHEMA`)
- System prompt injection of memory blocks
- MemoryManager (`agent/memory_manager.py`) — the provider orchestrator

The agent core is the **only** code path for MEMORY.md/USER.md mutations.
A `memory(action="add")` call always goes through `memory_tool.py` first.
Brain's `on_memory_write()` hook is then notified asynchronously.

Brain also writes DIRECTLY to brain.db via its own capture pipeline
(episode logging, extraction, skill forge) — these writes bypass
`memory_tool.py` entirely and are NOT subject to the MEMORY.md char limit.

### Layer 2: hermes-brain (the plugin)
Owns:
- `brain.db` — SQLite store for episodes, memories, vectors, graph
- Embedding, retrieval, reranking (recall/)
- Dream consolidation (dream/)
- Skill forge (skillforge/)
- Two-lane context injection:
  - Lane 1: `system_prompt_block()` — static, rendered once at init
  - Lane 2: `prefetch()` — per-turn, budget-capped, serves cached results
- Episode capture and sync (capture/, sync/)
- `memories` tool (file view, pin/forget, correct, etc.)

Brain does NOT own the `memory` tool, the char limits, or the
MEMORY.md/USER.md files. It plugs into the agent via the
`MemoryProvider` ABC.

## Data Flow: A Memory Write

```
Model calls memory(action="add", content="User prefers dark mode")
  → tools/memory_tool.py: validate, check capacity, write MEMORY.md
  → agent/memory_manager.py: notify_memory_tool_write()
  → BrainProvider.on_memory_write() enqueues to brain-bg worker thread
    → (async) brain-bg stores in brain.db with provenance "builtin-mirror"
```

Note: `on_memory_write()` returns immediately (microseconds). The actual
brain.db write happens on the dedicated "brain-bg" daemon thread. This
keeps the agent turn path non-blocking.

## Data Flow: Brain Context Injection

```
Session start:
  → BrainProvider.initialize()
    → render lane1 from materialized lane1_snapshot table → system prompt
    → (byte-stable for entire session — cache-safety invariant #1)

Per turn:
  → BrainProvider.prefetch(query=user_message)
    → returns cached lane2 results (not an immediate DB search)
    → budget-capped, injected as <memory-context> fence in user message
    → NOT injected into system prompt (that would break prefix cache)
```

## Config Boundary

In `config.yaml`:
```yaml
memory:
  memory_char_limit: 4000    # ← agent core reads this
  user_char_limit: 2500      # ← agent core reads this
  provider: brain            # ← agent loads BrainProvider
```

Brain's own config lives in `brain.yaml` (separate file, flat keys):
```yaml
lane1_tokens: 400
lane2_tokens: 400
# ... brain-specific settings (flat key-value, not nested)
```

## Key Files

| File | Repo | Purpose |
|------|------|---------|
| `tools/memory_tool.py` | hermes-agent | Memory tool handler + schema |
| `hermes_cli/config_defaults.py` | hermes-agent | Default char limits |
| `tools/write_approval.py` | hermes-agent | Write approval gate |
| `agent/memory_manager.py` | hermes-agent | Provider orchestrator |
| `provider.py` | hermes-brain | BrainProvider implementation |
| `store/db.py` | hermes-brain | SQLite storage |
| `recall/` | hermes-brain | Retrieval pipeline |
| `dream/` | hermes-brain | Consolidation |

## Skill Installation

**Installation required:** These skills ship inside the brain plugin repo
(`plugins/brain/skills/`) but the Hermes skill loader only walks
`$HERMES_HOME/skills/`. They must be copied or symlinked into
`~/.hermes/skills/` to be discoverable. The brain plugin does NOT
auto-install them — this is a manual step after cloning the repo.
