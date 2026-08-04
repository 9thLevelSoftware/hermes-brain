# Dojo Memory Failure Classification Guide

When the nightly Dojo reports memory tool failures, use this to trace the
fix to the correct repository.

## Classification Rules

### hermes-agent fixes (the host)
These error patterns are ALWAYS in hermes-agent:
- "Unknown action" — tools/memory_tool.py handler
- "chars" / "limit" / "overflow" — tools/memory_tool.py capacity check
- "staged" / "pending" / "approval" — tools/write_approval.py
- "threat" / "blocked" / "injection" — tools/threat_patterns.py
- "drift" / "round-trip" — tools/memory_tool.py drift guard
- Schema issues — MEMORY_SCHEMA in tools/memory_tool.py
- Config defaults — hermes_cli/config_defaults.py

### hermes-brain fixes (the plugin)
These error patterns are in hermes-brain:
- Brain context not appearing — provider.py lane1/lane2
- Dream/consolidation failures — dream/
- Retrieval quality — recall/
- Vector/embedding errors — store/db.py, recall/
- Plugin loading — brain.yaml, __init__.py
- brain.db locks — store/db.py WAL/locking

### Config issues (user's config.yaml)
- Char limits too low — increase memory.memory_char_limit
- Provider not set — ensure memory.provider: brain
- Brain budget too low — adjust brain.yaml lane1/lane2_budget

## Updating This Reference

After fixing a memory failure, add the error class to the appropriate
section above. Include:
1. The exact error message pattern
2. The file and function where the fix lives
3. The PR number if applicable
4. The date the fix was applied

This reference is read by the Dojo's overnight analysis to classify
future failures without human intervention.
