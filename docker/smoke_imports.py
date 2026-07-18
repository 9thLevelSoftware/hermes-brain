"""Eager-import smoke test.

The Hermes loader eagerly imports every ROOT ``*.py`` on every CLI invocation,
so those modules MUST stay stdlib-only at module level (invariant #1). This
script mirrors the loader: it registers the repo root as package ``brain`` via
``spec_from_file_location`` (exactly as the loader and tests/conftest.py do) and
imports every root module, plus the separate ``observer`` plugin. It exits
non-zero the moment anything explodes — which, on the floor-tier image (no
numpy/onnx/sqlite-vec installed), is precisely where a stray heavy module-level
import would blow up.
"""

import glob
import importlib
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _register_brain():
    spec = importlib.util.spec_from_file_location(
        "brain", os.path.join(ROOT, "__init__.py"),
        submodule_search_locations=[ROOT])
    module = importlib.util.module_from_spec(spec)
    sys.modules["brain"] = module
    spec.loader.exec_module(module)


def main() -> int:
    _register_brain()
    roots = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(ROOT, "*.py"))
        if not os.path.basename(p).startswith("_"))
    for name in roots:
        importlib.import_module(f"brain.{name}")
        print(f"  ok  brain.{name}")

    # The observer is a SEPARATE general-purpose plugin; the host eagerly
    # imports its __init__ too, so its module level must also be stdlib-only.
    ospec = importlib.util.spec_from_file_location(
        "brain_observer_smoke", os.path.join(ROOT, "observer", "__init__.py"))
    obs = importlib.util.module_from_spec(ospec)
    ospec.loader.exec_module(obs)
    assert hasattr(obs, "register"), "observer must expose register(ctx)"
    print("  ok  observer.register")

    print("EAGER-IMPORT SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
