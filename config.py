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
    "dream_schedule": "auto",    # cron | on-idle | manual | auto
    "dream_time": "03:30",
    "dream_min_interval_hours": 6,  # dream --if-due no-ops within this window
    "dream_model": "",           # auxiliary override; empty = active model
    "extract_model": "",         # cheap tier override; empty = auxiliary default
    "extract_mode": "active",    # off | shadow | active — the P3 sweep
    "extract_search_aids": True, # D2: fold LLM paraphrase aids into tags + embed text
    "extract_max_aids": 4,       # per-item cap on search aids
    "bootstrap_import": True,
    "memories_tool": False,      # deferred past P3 (critique item 8)
    "night_budget_usd": 0.50,
    "day_budget_usd": 1.50,
    "forget_grace_days": 30,
    "skill_auto_approve": True,  # user decision 2026-07-16: auto-approve after validation
    "capture_peers": True,       # user decision: trust-gated peer capture in group chats
    "incognito": False,
    # -- Phase A: retrieval upgrades (best-of-three) --
    "dedup_contest": True,       # info-content contest on near-dup merge (else exact-hash merge)
    "lane2_blend": True,         # compose lane-2 via semantic+reinforced+recent blend
    "lane2_blend_recent_days": 14,  # recency window for the "most-recent" blend leg
    "query_cache": True,         # in-process recall cache, invalidated on mem_generation
    "mmr_lambda": 0.7,           # MMR diversity/relevance tradeoff (1.0 = pure relevance)
    "intent_weighting": "shadow",  # off | shadow (log proposed deltas) — never applied in v1
    # -- Phase B: temporal fact layer + event seam --
    "facts_extract": True,       # sweep extracts s-p-o triples alongside memories
    "facts_leg": True,           # facts retrieval leg feeds memory ids into fusion
    "sync_events": False,        # write memory_events on lifecycle ops (Phase G seam; off)
    # -- Phase C: dream upgrades --
    "dream_surprisal": True,     # seed consolidate with top-surprise/anomaly hints
    "contradict_knowledge_update": True,  # deterministic same-(s,p) fact resolution (no LLM)
    "forget_weibull": True,      # per-kind Weibull decay shapes in the forget value score
    # -- Phase D: dialectic "ask" agent --
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


def load_config(hermes_home: str | Path) -> dict[str, Any]:
    """Defaults overlaid with brain.yaml (flat key: value lines)."""
    cfg = dict(DEFAULTS)
    path = config_path(hermes_home)
    if not path.exists():
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


def save_config(hermes_home: str | Path, values: dict[str, Any]) -> None:
    """Write flat YAML (only known keys; atomic replace)."""
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
