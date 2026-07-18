"""Hermes plugin-loader parity: the two live-integration bugs that surface
ONLY when the brain is loaded under the host's SYNTHETIC package name
``_hermes_user_memory.brain`` (not the standalone ``brain`` that conftest.py
and replay/run.py register).

Why the synthetic name matters (hermes-agent ``plugins/memory/__init__.py``):
user-installed providers import under a synthetic parent namespace so they
never collide with bundled providers. On the ``hermes brain dream-now`` CLI
path, ``discover_plugin_cli_commands()`` registers ``_hermes_user_memory.brain``
via ``_register_synthetic_package()`` as an EMPTY package shell —
``ModuleSpec(name, None, is_package=True)`` with ``submodule_search_locations``
but NO execution of the plugin's ``__init__.py``. That shell therefore exposes
no ``__version__`` attribute and has ``__file__ is None``, which is precisely
what made ``from .. import __version__`` on the lane1 path raise
``ImportError: cannot import name '__version__' from '_hermes_user_memory.brain'
(unknown location)`` (bug 1). This test recreates that shell byte-for-byte.

Bug 2 is shape-only and package-name-independent, but we exercise it through a
provider loaded under the same synthetic namespace for fidelity.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SYN = "_hermes_user_memory"
_BRAIN = f"{_SYN}.brain"


def _host_normalize_tool_schema(schema):
    """Faithful replica of hermes-agent ``agent/memory_manager.py``
    :func:`normalize_tool_schema` (lines 50-80) — the SOLE path every host
    consumer of ``get_tool_schemas()`` routes through. Inlined because
    ``agent.memory_manager`` transitively imports ``tools.registry``, which is
    not importable from within the Hermes-Brain test tree; ``normalize`` itself
    is pure and stdlib-only, so this is byte-equivalent to the host's.

    Returns a bare function schema with a resolvable top-level ``name``, or
    ``None`` for anything nameless.
    """
    if not isinstance(schema, dict):
        return None
    # Unwrap an already-wrapped OpenAI tool entry — exactly the shape the brain
    # emits ({"type":"function","function":{...}}).
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        schema = schema["function"]
        if not isinstance(schema, dict):
            return None
    name = schema.get("name", "")
    if not name or not isinstance(name, str):
        return None
    return schema


def _register_synthetic_package(name: str, search_locations: list[str]) -> None:
    """Mirror hermes-agent's ``_register_synthetic_package``: an empty package
    shell with ``__path__`` but no executed ``__init__`` — so it exposes no
    ``__version__`` and ``__file__ is None`` (the bug-1 trigger condition)."""
    if name in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = search_locations
    sys.modules[name] = importlib.util.module_from_spec(spec)


@pytest.fixture(scope="module")
def synthetic_brain():
    """Register ``_hermes_user_memory.brain`` exactly as the dream-now CLI path
    does: empty shells, no ``__init__.py`` execution."""
    _register_synthetic_package(_SYN, [])
    _register_synthetic_package(_BRAIN, [str(REPO_ROOT)])
    shell = sys.modules[_BRAIN]
    # Prove we reproduced the failing condition, not a working one: the shell
    # must NOT carry __version__, and __file__ must be None ("unknown location").
    assert not hasattr(shell, "__version__")
    assert getattr(shell, "__file__", None) is None
    yield shell
    # Drop everything under the synthetic namespace so it can't leak a
    # half-loaded module into later test modules (which use ``brain``).
    for mod_name in [m for m in sys.modules if m == _SYN or m.startswith(_SYN + ".")]:
        del sys.modules[mod_name]


@pytest.fixture(autouse=True)
def _fake_llm(synthetic_brain):
    """No real LLM in a turn/worker (P1 has none anyway) — install a fake on
    the SYNTHETIC package's llm gateway so any opportunistic sweep degrades."""
    llm = importlib.import_module(f"{_BRAIN}.llm")
    llm.set_llm_for_tests(lambda *a, **k: "")
    yield
    llm.set_llm_for_tests(None)


# ---------------------------------------------------------------------------
# Bug 1 — lane1 __version__ import under the synthetic shell
# ---------------------------------------------------------------------------

def test_lane1_materialize_survives_synthetic_shell(synthetic_brain, tmp_path):
    """Importing + running the lane1 materialize path under the empty-shell
    parent must NOT raise the ``__version__`` ImportError.

    Before the fix, ``importlib.import_module`` below raised at module import:
    ``cannot import name '__version__' from '_hermes_user_memory.brain'
    (unknown location)``.
    """
    # The import itself is the regression surface (module-level ``from ..
    # import __version__`` ran here).
    lane1 = importlib.import_module(f"{_BRAIN}.recall.lane1")
    db = importlib.import_module(f"{_BRAIN}.store.db")

    home = tmp_path / "home"
    home.mkdir()
    conn = db.connect(home)
    try:
        # One current-truth fact so the facts section (not just stats) renders.
        from conftest import seed_memory

        seed_memory(conn, "the staging DB lives on host neptune", kind="fact")

        rows = lane1.materialize(conn, {})          # <- previously ImportError
        assert rows >= 1

        rendered = lane1.render(conn, 1200)
        # The version stamp is emitted; under the shell it degrades to the
        # fallback rather than crashing.
        assert "brain v" in rendered
        assert isinstance(lane1._BRAIN_VERSION, str) and lane1._BRAIN_VERSION
    finally:
        conn.close()


def test_synthetic_brain_shell_has_no_version_attr(synthetic_brain):
    """Guard the premise: the shell genuinely lacks ``__version__`` (so a naive
    ``from .. import __version__`` WOULD fail) — the fix must tolerate it."""
    assert not hasattr(synthetic_brain, "__version__")
    with pytest.raises(ImportError) as excinfo:
        # The exact statement lane1 used to run unguarded (absolute form of
        # ``from .. import __version__``): it raises the reported error.
        exec("from _hermes_user_memory.brain import __version__", {})
    assert "__version__" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Bug 2 — get_tool_schemas() shape vs. the host contract
# ---------------------------------------------------------------------------

def test_get_tool_schemas_match_host_expected_shape(synthetic_brain, tmp_path):
    """The brain emits the nested OpenAI-tool shape ``{"type":"function",
    "function":{"name",...}}``. The host's ``normalize_tool_schema`` (the sole
    path every host consumer routes through) unwraps that and resolves a
    top-level name — so the shape is CORRECT and no brain change is needed. The
    driver that saw ``[None, None, None, None]`` simply read the wrong key
    (top-level ``name``) instead of normalizing / reading ``function.name``.
    """
    provider_mod = importlib.import_module(f"{_BRAIN}.provider")

    home = tmp_path / "home"
    home.mkdir()
    provider = provider_mod.BrainProvider()
    provider.initialize("loader-test", hermes_home=str(home), platform="replay",
                        agent_context="primary")
    try:
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 4

        # The DRIVER's mistake, reproduced: top-level "name" is absent because
        # the shape is nested — hence [None, None, None, None].
        assert [s.get("name") for s in schemas] == [None, None, None, None]
        # ...but the name IS present one level down, in the host-expected slot.
        assert all(s["type"] == "function" for s in schemas)
        assert all(isinstance(s["function"]["name"], str) for s in schemas)

        # The HOST contract: every consumer routes raw schemas through
        # normalize_tool_schema() (memory_manager.py:50-80), which unwraps the
        # nested shape and returns a bare function schema with a resolvable
        # name (replica above — the real one needs tools.registry).
        resolved = [_host_normalize_tool_schema(s) for s in schemas]
        assert all(r is not None for r in resolved), \
            "host normalize_tool_schema() must resolve every brain schema"
        names = [r["name"] for r in resolved]
        assert names == [
            "brain_recall", "brain_remember", "brain_outcome", "brain_manage",
        ]

        # Those resolved names are exactly what the provider dispatches on:
        # a valid name never returns the 'unknown tool' errors-that-teach
        # payload (an empty-arg call yields a *validation* error instead).
        for name in names:
            out = provider.handle_tool_call(name, {}, session_id="loader-test")
            assert "unknown tool" not in out
    finally:
        provider.shutdown()
