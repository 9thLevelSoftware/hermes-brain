"""Host capability detection for mode auto-selection (stdlib only).

Tier policy (docs/design/integration.md §5.2 + critique item 30):
  full     >= 1.5 GB RAM and onnxruntime importable
  lite     model2vec importable (static embeddings, ~30MB)
  fts-only everything else — still captures everything, upgrades in place
"""

from __future__ import annotations

import importlib.util
import logging
import sys

logger = logging.getLogger(__name__)

_FULL_TIER_MIN_GB = 1.5


def total_ram_gb() -> float | None:
    """Best-effort total physical RAM. None when undetectable."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024 ** 3)
            return None
        # Linux / Termux / WSL / macOS-with-procfs won't exist on mac; try both.
        try:
            with open("/proc/meminfo", encoding="ascii") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024 ** 2)  # kB -> GB
        except OSError:
            pass
        if sys.platform == "darwin":
            import subprocess

            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) / (1024 ** 3)
    except Exception as e:
        logger.debug("RAM detection failed: %s", e)
    return None


def importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def resolve_mode(configured: str) -> str:
    """Resolve config 'mode' to a concrete tier: full | lite | fts-only.

    Explicit settings are honored (degrading only if imports are missing);
    'auto' applies the tier policy. Values are normalized (case/whitespace);
    an unknown value warns and behaves as 'auto' — a typo in brain.yaml must
    degrade gracefully, not silently disable retrieval.
    """
    configured = (configured or "").strip().lower()
    if configured not in ("full", "lite", "auto", "stub", "fts-only"):
        logger.warning("unknown mode %r in brain.yaml; treating as 'auto' "
                       "(valid: auto|full|lite|fts-only|stub)", configured)
        configured = "auto"
    if configured == "full" and importable("onnxruntime") and importable("tokenizers"):
        return "full"
    if configured in ("full", "lite", "auto"):
        if configured == "auto":
            ram = total_ram_gb()
            if (ram is None or ram >= _FULL_TIER_MIN_GB) and \
                    importable("onnxruntime") and importable("tokenizers"):
                return "full"
        if importable("model2vec"):
            return "lite"
        # 'stub' is never auto-selected: it is a test tier, config-only.
    return "fts-only" if configured != "stub" else "stub"
