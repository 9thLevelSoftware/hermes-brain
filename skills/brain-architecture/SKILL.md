---
name: brain-architecture
description: "Memory system architecture: brain↔agent boundary, ownership model, and data flow."
version: 1.0.0
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

The agent core is the **only** code path for memory mutations. Even when
brain is the provider, a `memory(action="add")` call goes through
`memory_tool.py` first, then brain's `on_memory_write()` hook is notified.

### Layer 2: hermes-brain (the plugin)
Owns:
- `brain.db` — SQLite store for episodes, memories, vectors, graph
- Embedding, retrieval, reranking (recall/)
- Dream consolidation (dream/)
- Skill forge (skillforge/)
- Two-lane context injection:
  - Lane 1: `system_prompt_block()` — static, rendered once at init
  - Lane 2: `prefetch()` — per-turn, dynamic, budget-capped
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
  → BrainProvider.on_memory_write("add", "memory", "User prefers dark mode")
    → brain stores in brain.db with provenance "builtin-mirror"
```

## Data Flow: Brain Context Injection

```
Session start:
  → BrainProvider.initialize()
    → render lane1 (static facts) → system prompt
  → Per turn:
    → BrainProvider.prefetch(query=user_message)
      → search brain.db → return <memory-context> fence
      → injected into current user message (NOT system prompt)
```

## Config Boundary

In `config.yaml`:
```yaml
memory:
  memory_char_limit: 4000    # ← agent core reads this
  user_char_limit: 2500      # ← agent core reads this
  provider: brain            # ← agent loads BrainProvider
```

Brain's own config lives in `brain.yaml` (separate file):
```yaml
brain:
  lane1_budget: 400
  lane2_budget: 400
  # ... brain-specific settings
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
