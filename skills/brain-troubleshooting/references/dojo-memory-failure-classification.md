# Dojo Memory Failure Classification Guide

When the nightly Dojo reports memory tool failures, use this to trace the
fix to the correct repository.

## Classification Rules

### hermes-agent fixes (the host)
These error patterns are in hermes-agent when they involve MEMORY.md/USER.md:
- "Unknown action" — tools/memory_tool.py handler
- "Memory at N/N chars" (with MEMORY.md entries) — tools/memory_tool.py
- "Replacement would put memory at N/N chars" — tools/memory_tool.py
- "Staged for approval" — tools/write_approval.py
- "content contained threat pattern" — tools/threat_patterns.py
- "external drift" / "round-trip" — tools/memory_tool.py drift guard
- "action is missing" — tools/memory_tool.py (PR #47)
- MEMORY_SCHEMA validation — tools/memory_tool.py
- Config defaults — hermes_cli/config_defaults.py

### hermes-brain fixes (the plugin)
These error patterns are in hermes-brain:
- Brain context not appearing in system prompt — provider.py lane1
- Brain context not appearing in user message — provider.py lane2
- "database is locked" on brain.db — store/db.py
- Dream/consolidation failures — dream/
- Retrieval quality (wrong memories recalled) — recall/
- Vector/embedding errors — store/db.py, recall/
- Plugin load failure (missing plugin.yaml, Python version) — __init__.py
- Quarantine gate blocking valid memories — store/db.py

### Config issues (user's config.yaml)
- "Memory at N/N chars" → increase memory.memory_char_limit
- Provider not loading → ensure memory.provider: brain
Brain budget too low → adjust brain.yaml lane1_tokens/lane2_tokens

## How to Distinguish MEMORY.md vs brain.db Errors

The key question: does the error involve a char limit?
- MEMORY.md has a configurable char limit (default 4000)
- brain.db has NO char limit — it stores unlimited memories

If the error mentions "chars" or "limit" with a specific N/N ratio,
it's almost always MEMORY.md (hermes-agent). If it mentions "database",
"sqlite", "WAL", or "quarantine", it's brain.db (hermes-brain).

## Updating This Reference

After fixing a memory failure, add the error class to the appropriate
section above. Include:
1. The exact error message pattern
2. The file and function where the fix lives
3. The PR number if applicable
4. The date the fix was applied

This reference is read by the Dojo's overnight analysis to classify
future failures without human intervention.
