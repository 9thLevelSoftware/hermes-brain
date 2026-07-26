"""brain.yaml config: load/save with defaults. Stdlib only (JSON-in-YAML
subset written by us; parsed with a tolerant line parser so we never need
PyYAML at the floor tier — Hermes has yaml, standalone tests may not).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "mode": "auto",              # auto | full | lite | fts-only | stub (tests only)
    "embed_model": "modernbert-embed-base",  # or embeddinggemma-300m (gated; needs HF_TOKEN)
    "rerank": "auto",            # auto | off — late-interaction ColBERT rerank (full tier)
    "rerank_model": "",          # '' = mxbai-edge-colbert default (fallback answerai); 'stub' for tests
    "lane1_tokens": 1200,        # 800-1500; hard-truncated by the renderer
    "lane2_tokens": 600,         # 0 disables lane 2
    # How the AGENT reaches memory. Mirrors the cross-provider convention
    # (Honcho recallMode, Hindsight memory_mode) so a user migrating from
    # either finds the knob where they expect it:
    #   hybrid  = injection lanes + tools (default, today's behavior)
    #   context = lanes only; get_tool_schemas() returns []
    #   tools   = tools only; lane 1 static, lane 2 empty
    # Operator surfaces (CLI, MCP) are deliberately unaffected.
    "recall_mode": "hybrid",     # hybrid | context | tools
    # -- cross-profile links (store/links.py, recall/linked.py) --
    # Linked profiles are searched only for an OWNER-trust caller, always
    # read-only, and fused at a slight discount: another profile is relevant
    # context but it is not THIS conversation's.
    "link_weight": 0.85,
    # Include linked profiles in the per-turn lane-2 injection. OFF: lane 2 is
    # the cache-safe hot path, and a second database read per turn is latency
    # and blast radius for a feature whose value is mostly on-demand.
    "link_lane2": False,
    "dream_schedule": "auto",    # cron | on-idle | manual | auto
    "dream_time": "03:30",
    "dream_min_interval_hours": 6,  # dream --if-due no-ops within this window
    "dream_model": "",           # auxiliary override; empty = active model
    "extract_model": "",         # cheap tier override; empty = auxiliary default
    "extract_mode": "active",    # off | shadow | active — the P3 sweep
    "extract_search_aids": True, # D2: fold LLM paraphrase aids into tags + embed text
    "extract_max_aids": 4,       # per-item cap on search aids
    "bootstrap_import": True,
    # An `ended_at IS NULL` session is only withheld from bootstrap while it is
    # plausibly LIVE. Hermes stamps ended_at on a clean close only, so without a
    # staleness reaper "still running" also meant "abandoned months ago" — and
    # on the install that motivated this, that was 72% of all history, skipped
    # permanently (docs/design/alignment-audit.md §F1).
    "bootstrap_stale_days": 7,
    # Import messages compacted out of the live context (active=0). Right for
    # bootstrap and wrong for live capture: compacted history is exactly what an
    # external memory system exists to preserve.
    "bootstrap_include_compacted": True,
    "night_budget_usd": 0.50,
    "day_budget_usd": 1.50,
    "forget_grace_days": 30,
    "forget_demote_below": 0.15,  # value score under which a memory is demoted
    "skill_auto_approve": True,  # user decision 2026-07-16: auto-approve after validation
    "capture_peers": True,       # user decision: trust-gated peer capture in group chats
    "incognito": False,
    # -- Phase A: retrieval upgrades (best-of-three) --
    "dedup_contest": True,       # info-content contest on near-dup merge (else exact-hash merge)
    "lane2_blend": True,         # compose lane-2 via semantic+reinforced+recent blend
    "lane2_blend_recent_days": 14,  # recency window for the "most-recent" blend leg
    "query_cache": True,         # in-process recall cache, invalidated on mem_generation
    "mmr_lambda": 0.7,           # MMR diversity/relevance tradeoff (1.0 = pure relevance)
    # off | shadow (log proposed deltas, nothing reads them) | active (apply
    # per-query intent multipliers on top of the approved base weights, see
    # recall/weights.py). Never auto-promoted past shadow.
    "intent_weighting": "shadow",
    # Rewrite the raw user turn into a retrieval query via the host's shared
    # plugins/memory/query_rewrite.py helper (the same one honcho uses). OFF by
    # default: it is one auxiliary LLM call per turn against day_budget_usd.
    # Runs on the brain-bg worker only — never the turn path.
    "query_rewrite": False,
    # -- Phase B: temporal fact layer + event seam --
    "facts_extract": True,       # sweep extracts s-p-o triples alongside memories
    "facts_leg": True,           # facts retrieval leg feeds memory ids into fusion
    "sync_events": False,        # write memory_events on lifecycle ops (Phase G seam; off)
    # -- Phase C: dream upgrades --
    "dream_surprisal": True,     # seed consolidate with top-surprise/anomaly hints
    "contradict_knowledge_update": True,  # deterministic same-(s,p) fact resolution (no LLM)
    "forget_weibull": True,      # per-kind Weibull decay shapes in the forget value score
    # -- Phase D: dialectic "ask" agent --
    # The Anthropic-memory-tool-shaped `memories` file interface over virtual
    # views of brain storage (integration.md §3.1 tool #5). Master switch:
    # with it off, tools.dispatch refuses the tool and no surface advertises it.
    "memories_tool": True,
    "ask_tool": True,            # expose brain_ask via CLI + MCP (tool trust)
    "ask_tool_agent": False,     # agent-facing brain_ask schema (LLM-in-turn) — OFF by default
    "ask_max_iterations": 6,     # hard cap on the ask tool-loop iterations (deep level)
    # -- Phase E: token-budgeted context assembly --
    "precompress_tokens": 300,   # budget for the on_pre_compress contribution
    "context_summary_ratio": 0.4,  # remainder split: 40% summary / 60% recent extracts
    # -- Phase G: multi-device encrypted delta sync (needs [sync] extra) --
    "sync_enabled": False,       # master switch for push/pull (off by default)
    "sync_url": "",              # relay base URL (opaque-ciphertext store)
    "sync_device_id": "",        # this device's origin id (set at `sync init`)
    "sync_account": "",          # shared relay namespace across a user's devices
    "sync_salt": "",             # base64 KDF salt (shared across devices; set at init)
}


def config_path(hermes_home: str | Path) -> Path:
    return Path(hermes_home) / "brain" / "brain.yaml"


def load_config(hermes_home: str | Path | None) -> dict[str, Any]:
    """Defaults overlaid with brain.yaml (flat key: value lines).

    Degrades to defaults for a missing/unusable home rather than raising: the
    provider calls this from hooks that the host invokes even when
    ``initialize()`` failed (``on_session_switch`` fires on /reset, /branch and
    compression regardless), and ``Path(None)`` would raise TypeError from
    OUTSIDE the try below — the host would swallow it at debug level and the
    real cause would never surface.
    """
    cfg = dict(DEFAULTS)
    if not hermes_home:
        return cfg
    try:
        path = config_path(hermes_home)
        if not path.exists():
            return cfg
    except (TypeError, ValueError, OSError) as e:
        logger.warning("brain.yaml path unresolvable (%s); using defaults", e)
        return cfg
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key not in DEFAULTS:
                continue
            default = DEFAULTS[key]
            try:
                if isinstance(default, bool):
                    cfg[key] = val.lower() in ("1", "true", "yes", "on")
                elif isinstance(default, int):
                    cfg[key] = int(val)
                elif isinstance(default, float):
                    cfg[key] = float(val)
                else:
                    cfg[key] = val
            except ValueError:
                logger.warning("brain.yaml: bad value for %s: %r (using default)", key, val)
    except (OSError, UnicodeDecodeError) as e:
        # Bad encoding must degrade to defaults, not abort provider
        # initialization (review finding #21).
        logger.warning("brain.yaml unreadable (%s); using defaults", e)
    return cfg


def mirror_path(hermes_home: str | Path) -> Path:
    return Path(hermes_home) / "brain" / "config.json"


def save_config(hermes_home: str | Path, values: dict[str, Any]) -> None:
    """Write flat YAML (only known keys; atomic replace), then the JSON mirror."""
    path = config_path(hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# hermes-brain configuration (flat key: value)"]
    merged = {**load_config(hermes_home), **{k: v for k, v in values.items() if k in DEFAULTS}}
    for key in DEFAULTS:
        val = merged[key]
        if isinstance(val, bool):
            rendered = "true" if val else "false"
        elif isinstance(val, str):
            rendered = json.dumps(val) if (":" in val or val == "") else val
        else:
            rendered = str(val)
        lines.append(f"{key}: {rendered}")
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    _write_mirror(hermes_home, merged)


def _write_mirror(hermes_home: str | Path, merged: dict[str, Any]) -> None:
    """Mirror the saved config to ``<hermes_home>/brain/config.json``.

    Purely for the Hermes dashboard: hermes_cli/web_server.py's
    ``_read_memory_provider_existing_values`` prefills the provider form from
    ``<hermes_home>/<provider>.json`` or ``<hermes_home>/<provider>/config.json``
    — the second path is exactly ours. Without the mirror the dashboard renders
    schema DEFAULTS regardless of what brain.yaml says, and saving that form
    writes those defaults back through save_config, silently discarding the
    user's real settings.

    brain.yaml remains the single source of truth. NEVER read this file back:
    a second read path is how the two copies start disagreeing. It is
    write-only, and best-effort — a failed mirror must not fail a config save.
    """
    try:
        mirror = mirror_path(hermes_home)
        tmp = mirror.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        tmp.replace(mirror)
    except (OSError, TypeError, ValueError) as e:
        logger.warning("brain: config.json mirror not written (%s); the Hermes "
                       "dashboard may show stale defaults", e)
