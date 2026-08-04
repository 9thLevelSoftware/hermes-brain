---
name: brain-troubleshooting
description: "Debug memory system failures: trace errors to the right repo and fix."
version: 1.0.1
metadata:
  hermes:
    tags: [brain, memory, debugging, troubleshooting]
    category: mlops
---

# Brain / Memory Troubleshooting

## Decision Tree: Which Repo Has the Bug?

```
Memory failure detected
  │
  ├─ Error mentions "Unknown action" or "action is missing"
  │   → hermes-agent: tools/memory_tool.py
  │
  ├─ Error mentions "chars" / "limit" / "overflow" in MEMORY.md context
  │   → hermes-agent: tools/memory_tool.py (char limits)
  │   → Check config.yaml memory.memory_char_limit
  │   NOTE: brain.db has NO char limit — if the error involves brain.db
  │   memories, it's in hermes-brain (store/db.py)
  │
  ├─ Error mentions "staged" / "pending" / "approval"
  │   → hermes-agent: tools/write_approval.py
  │
  ├─ Error mentions "threat" / "blocked" / "injection" from memory writes
  │   → hermes-agent: tools/threat_patterns.py
  │   NOTE: brain-side content filtering (quarantine gate) is in
  │   hermes-brain: store/db.py status='quarantined'
  │
  ├─ Error mentions "drift" / "round-trip" / "corrupt"
  │   → hermes-agent: tools/memory_tool.py (_reload_target)
  │
  ├─ Brain context not appearing in prompts
  │   → hermes-brain: provider.py (lane1/lane2)
  │   → Check brain.yaml lane1_budget/lane2_budget
  │
  ├─ Dream/consolidation failures
  │   → hermes-brain: dream/
  │
  ├─ Retrieval quality (wrong memories recalled)
  │   → hermes-brain: recall/
  │
  ├─ Vector/embedding errors
  │   → hermes-brain: store/db.py, recall/
  │   → Check if sqlite-vec extension loaded
  │
  └─ Brain plugin not loading
      → Check hermes brain status
      → Check Python >=3.11
      → Check plugin.yaml exists
```

## Common Failure Patterns

### 1. "Unknown action 'None'" (hermes-agent)

**Symptom:** `memory()` returns `"Unknown action 'None'. Use: add, replace, remove"`

**Cause:** Model omitted the `action` parameter. The schema can't make it
required (Codex backend rejects allOf/oneOf combinators).

**Fix:** In `tools/memory_tool.py`, the handler should detect missing action
and return a recoverable error with current entries. If you see this in
Dojo analysis, the fix is in hermes-agent, not brain.

**PR reference:** https://github.com/9thLevelSoftware/hermes-agent/pull/47

### 2. Capacity Overflow (hermes-agent)

**Symptom:** `memory(action="add")` returns `"Memory at 4,047/4,000 chars"`

**Cause:** MEMORY.md has hit the configured char limit. Default is 4000
(raised from 2200 in PR #47). The `add` operation hard-fails when the
new total would exceed the limit. This limit applies ONLY to MEMORY.md
— brain.db has no char limit.

**Fix options:**
- Increase `memory.memory_char_limit` in config.yaml
- Use `operations` array to batch remove+add atomically
- Consolidate entries manually via `replace`

**Note:** Brain's dream process operates on brain.db, NOT on MEMORY.md.
It cannot compact or consolidate MEMORY.md entries. MEMORY.md compaction
must be done manually via the `memory` tool or by the agent itself.

### 3. brain.db Locked / WAL Errors (hermes-brain)

**Symptom:** `database is locked` or WAL checkpoint failures

**Cause:** Multiple processes accessing brain.db concurrently without
proper WAL mode or busy_timeout.

**Fix:** Check `store/db.py` — it applies WAL + busy_timeout=5000. If
you're running brain in Docker, ensure the volume mount supports SQLite
locking (NFS doesn't work).

### 4. Lane 1 Instability (hermes-brain)

**Symptom:** Cache invalidation warnings, or brain context changing
mid-session

**Cause:** Lane 1 (`system_prompt_block()`) must be byte-identical for
the entire session. If it reads live/dynamic data at render time, it
violates the cache-safety invariant.

**Fix:** Lane 1 must read from the materialized `lane1_snapshot` table,
never from live data. See `tests/test_provider.py` golden test.

### 5. Dream Not Running (hermes-brain)

**Symptom:** Memories not being consolidated, brain.db growing unbounded

**Cause:** The `hermes brain dream --if-due` command not executing on
schedule. The cron job uses a shell wrapper script; verify it exists
and the cron job references it correctly.

**Fix:** Check `hermes brain status` for `lease.dream` and `last dream`
fields. Run `hermes brain dream --if-due` manually to test. Verify the
cron job is enabled (`hermes cron list`).

## Debugging Commands

```bash
# Brain status (db size, episodes, memories, last dream)
hermes brain status

# Memory store status
hermes memory status

# Check config
hermes config get memory

# Run dream manually
hermes brain dream --if-due

# Check brain plugin loading
hermes plugins list
```

## Dojo Integration

When the nightly Dojo reports memory tool failures, classify by origin:

1. Parse the error message from the Dojo's `weakest_tools` output
2. Match against the decision tree above (be specific — generic tokens
   like "chars" can appear in brain.db errors too)
3. File the fix in the correct repo (hermes-agent vs hermes-brain)
4. Update `references/memory-tool-persistent-failure.md` in the Dojo skill
   with the new error class and fix reference
