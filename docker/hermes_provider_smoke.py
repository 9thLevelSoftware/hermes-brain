"""Live-integration smoke: load the brain memory provider through HERMES'S OWN
loader (plugins.memory.load_memory_provider) and drive the MemoryProvider hook
lifecycle the way the host's MemoryManager does — proving the real contract,
not the brain's replay shim. No LLM key needed: sync_turn enqueues and the
extraction LLM path degrades to LLMUnavailable (logged, never fatal)."""

import os
import sys
import traceback


def main() -> int:
    from plugins.memory import _get_active_memory_provider, load_memory_provider

    active = _get_active_memory_provider()
    print(f"active memory provider (config): {active!r}")
    assert active == "brain", f"expected active provider 'brain', got {active!r}"

    prov = load_memory_provider("brain")
    assert prov is not None, "load_memory_provider('brain') returned None"
    print(f"loaded: name={prov.name!r}  is_available={prov.is_available()}")
    assert prov.name == "brain"

    home = os.environ["HERMES_HOME"]
    prov.initialize(session_id="smoke-1", platform="cli", hermes_home=home,
                    agent_context="primary")
    block = prov.system_prompt_block()          # lane 1 (byte-stable)
    print(f"system_prompt_block: {len(block)} chars")
    schemas = prov.get_tool_schemas()
    print(f"tool schemas: {[s.get('name') for s in schemas]}")
    assert isinstance(schemas, list)

    prov.prefetch("what is my staging database?", session_id="smoke-1")  # lane 2
    prov.sync_turn("my staging db is postgres 14 on fly.io",
                   "Noted — postgres 14 on fly.io.", session_id="smoke-1")

    # Optional hooks the host fans out (each may be a no-op on the brain).
    for name, args in (("on_turn_start", (1, "hi")),
                       ("queue_prefetch", ("staging db",)),
                       ("on_session_end", ([],))):
        fn = getattr(prov, name, None)
        if fn is None:
            continue
        try:
            fn(*args)
        except TypeError:
            fn(*args[:1])  # signature tolerance (args is always non-empty here)

    prov.shutdown()   # joins the brain-bg worker (<=5s), LLM-free at shutdown
    print("PROVIDER LIFECYCLE SMOKE OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
