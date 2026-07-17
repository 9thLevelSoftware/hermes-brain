"""Hermes-Brain — global memory & continual learning for Hermes Agent.

This repo root IS the plugin directory: install with

    git clone <repo> $HERMES_HOME/plugins/brain

and set ``memory.provider: "brain"`` in Hermes config.yaml (or run
``hermes memory setup``). The directory name ``brain`` is load-bearing:
it is the provider name, the config key, and the CLI verb.

MUST stay import-light: the Hermes plugin loader eagerly imports every
root-level .py in this directory at discovery time. No onnx, no
sqlite-vec, no model loading here — heavy work lives in subpackages and
loads lazily on first use.
"""

from __future__ import annotations

__version__ = "0.1.0"


def register(ctx) -> None:
    """Hermes plugin entry point (register(ctx) pattern)."""
    from .provider import BrainProvider

    ctx.register_memory_provider(BrainProvider())
