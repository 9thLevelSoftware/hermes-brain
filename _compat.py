"""MemoryProvider ABC import with a vendored fallback.

Inside a Hermes process, ``agent.memory_provider`` is importable and is
the real contract. Outside (pytest in this repo, the standalone MCP
server, CI without hermes-agent), we fall back to a minimal structural
mirror so the provider class can still be defined and unit-tested.

The mirror carries NO behavior — only the method surface we implement.
If hermes-agent's ABC gains methods we override, nothing here needs to
change; if it renames ones we implement, tests against the real Hermes
(replay harness on a machine with hermes-agent) catch it.
"""

from __future__ import annotations

try:  # inside a Hermes process
    from agent.memory_provider import MemoryProvider  # type: ignore
    HAVE_HERMES = True
except ImportError:  # standalone: tests, MCP server, CI
    HAVE_HERMES = False

    class MemoryProvider:  # type: ignore[no-redef]
        """Structural stand-in for agent.memory_provider.MemoryProvider."""

        @property
        def name(self) -> str:
            raise NotImplementedError

        def is_available(self) -> bool:
            raise NotImplementedError

        def initialize(self, session_id: str, **kwargs) -> None:
            raise NotImplementedError

        def system_prompt_block(self) -> str:
            return ""

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            return ""

        def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
            pass

        def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None) -> None:
            pass

        def get_tool_schemas(self):
            return []

        def handle_tool_call(self, tool_name, args, **kwargs) -> str:
            raise NotImplementedError

        def shutdown(self) -> None:
            pass

        def on_turn_start(self, turn_number, message, **kwargs) -> None:
            pass

        def on_session_end(self, messages) -> None:
            pass

        def on_session_switch(self, new_session_id, *, parent_session_id="",
                              reset=False, rewound=False, **kwargs) -> None:
            pass

        def on_pre_compress(self, messages) -> str:
            return ""

        def on_delegation(self, task, result, *, child_session_id="", **kwargs) -> None:
            pass

        def get_config_schema(self):
            return []

        def save_config(self, values, hermes_home) -> None:
            pass

        def on_memory_write(self, action, target, content, metadata=None) -> None:
            pass

        def backup_paths(self):
            return []
