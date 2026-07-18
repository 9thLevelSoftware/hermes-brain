"""Skill-forge: the brain drafts agentskills.io skills from proven task
patterns and, on validation, promotes them into Hermes's own skills tree
(learning-system.md §2). The brain is the detector and drafter; Hermes's
loader and curator remain the runtime authority (locked decision #4).

This is a SUBPACKAGE, not a root module, so the plugin loader does not
eagerly import it — the filesystem + LLM machinery loads only when a
`hermes brain skills` verb or the dream's forge step actually runs.
"""

from __future__ import annotations

from .forge import forge_once, promote_draft
from .revise import revise_once

__all__ = ["forge_once", "promote_draft", "revise_once"]
