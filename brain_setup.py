"""Setup wizard glue for `hermes memory setup` (F9 contract).

Contract verified against hermes_cli/memory_setup.py: the wizard calls
``provider.get_config_schema()`` (which delegates to ``config_schema()``
here) to walk the field prompts, and — because BrainProvider defines
``post_setup`` — delegates activation entirely to ``post_setup(hermes_home,
config)`` where ``config`` is the *Hermes* config.yaml dict (not brain.yaml).

Module name note: this file is deliberately NOT ``setup.py`` — pip treats a
top-level setup.py as a build script, and the plugin dir may be pip-installed.

Design (docs/design/integration.md §5.2): no secrets, so nothing touches
.env; per-field values land in ``brain.yaml`` via config.save_config. Every
post_setup step is skippable and failure-tolerant — a failed model download
or bootstrap leaves a working fts-only brain, never a broken install.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def config_schema() -> list[dict[str, Any]]:
    """The §5.2 wizard field set. Descriptions/choices/defaults track
    config.DEFAULTS — the provider's get_config_schema delegates here."""
    from .config import DEFAULTS
    from .recall.embed import REGISTRY

    return [
        {"key": "mode",
         "description": "Retrieval tier (auto = detect RAM and installed deps)",
         "choices": ["auto", "full", "lite", "fts-only"],
         "default": DEFAULTS["mode"]},
        {"key": "lane1_tokens",
         "description": "System-prompt brain index budget, 800-1500 tokens",
         "default": DEFAULTS["lane1_tokens"]},
        {"key": "lane2_tokens",
         "description": "Per-turn recall injection budget (0 disables lane 2)",
         "default": DEFAULTS["lane2_tokens"]},
        {"key": "embed_model",
         "description": "Embedding model (embeddinggemma-300m is license-gated: "
                        "needs HF_TOKEN)",
         "choices": sorted(REGISTRY),
         "default": DEFAULTS["embed_model"]},
        {"key": "rerank",
         "description": "Late-interaction ColBERT rerank of results "
                        "(auto = on when the model is present; full tier only)",
         "choices": ["auto", "off"],
         "default": DEFAULTS["rerank"]},
        {"key": "dream_schedule",
         "description": "When the dream cycle runs (auto = cron if gateway detected)",
         "choices": ["auto", "cron", "on-idle", "manual"],
         "default": DEFAULTS["dream_schedule"]},
        {"key": "dream_time",
         "description": "Nightly dream time, HH:MM local",
         "default": DEFAULTS["dream_time"]},
        {"key": "dream_model",
         "description": "Model override for dream consolidation (empty = active model)",
         "default": DEFAULTS["dream_model"]},
        {"key": "bootstrap_import",
         "description": "Import MEMORY.md/USER.md + state.db history on first run",
         "choices": ["yes", "no"],
         "default": "yes" if DEFAULTS["bootstrap_import"] else "no"},
        {"key": "night_budget_usd",
         "description": "Nightly LLM spend cap in USD (dream consolidation)",
         "default": DEFAULTS["night_budget_usd"]},
    ]


# §4.6 built-ins matrix, printed verbatim — P2 never auto-edits Hermes config.
_TRANSITION_MATRIX = """
  Built-ins transition matrix (docs/design/integration.md §4.6) — what to
  flip when. Nothing below is changed automatically; edit config.yaml when
  you reach the right-hand phase:

    setting                        now (P1-P2 transition)   P3+ (brain owns memory)
    memory.memory_enabled          keep on (brain mirrors)  false
    memory.user_profile_enabled    keep on (brain mirrors)  false
    memory.nudge_interval          keep default (10)        0 (sweep replaces nudges)
    memory.provider                "brain"                  "brain"
    skills / curator settings      keep untouched           keep untouched
    session_search core tool       keep                     keep
"""


def post_setup(hermes_home: str, config: dict[str, Any]) -> None:
    """Wizard finish: field prompts, dirs, optional model download, bootstrap,
    lane 1, activation, transition matrix, identity reminder. Never raises.

    The field walk lives HERE, not in the wizard: hermes memory_setup
    delegates entirely to post_setup when the attribute exists, so the
    get_config_schema()-driven prompt loop in hermes_cli never runs for
    brain. Values are persisted to brain.yaml BEFORE the steps below, which
    read the saved config."""
    from . import config as brain_config
    from .store import db, sysinfo

    home = Path(hermes_home)
    _prompt_schema_fields(home)

    for sub in ("", "exports", "logs"):
        (db.brain_dir(home) / sub).mkdir(parents=True, exist_ok=True)

    cfg = brain_config.load_config(home)
    mode = sysinfo.resolve_mode(str(cfg.get("mode", "auto")))
    print(f"\n  hermes-brain: retrieval tier resolved to '{mode}'")

    if mode == "full":
        _offer_model_download(cfg)

    conn = None
    try:
        conn = db.connect(home)
        if cfg.get("bootstrap_import"):
            _run_bootstrap(conn, home, cfg, mode)
        _materialize_lane1(conn, cfg)
    except Exception as e:
        print(f"  Could not open brain.db: {e}\n"
              f"  Remedy: run 'hermes brain doctor' after setup.")
    finally:
        if conn is not None:
            conn.close()

    _register_aux_slots(config)
    _activate_provider(config)
    print(_TRANSITION_MATRIX)
    print("  Gateway users: enroll yourself as owner or your messages can never\n"
          "  be owner-trusted (finding #33):\n"
          "    hermes brain identity add <platform> <your-user-id> --owner\n")


def _prompt_schema_fields(home: Path) -> None:
    """Walk config_schema() prompts and persist answers to brain.yaml.

    Empty answer / EOF / Ctrl-C keeps the default; save_config writes the
    merged file either way so brain.yaml exists after setup."""
    from . import config as brain_config

    values: dict[str, Any] = {}
    print("\n  hermes-brain configuration (Enter keeps the default):")
    for field in config_schema():
        key = field["key"]
        default = field.get("default")
        choices = field.get("choices")
        if choices:
            print(f"  {field.get('description', key)}")
            for i, choice in enumerate(choices, 1):
                marker = "  (default)" if str(choice) == str(default) else ""
                print(f"    {i}. {choice}{marker}")
            prompt = f"  Choice [1-{len(choices)}] [{default}]: "
        else:
            prompt = f"  {field.get('description', key)} [{default}]: "
        try:
            answer = input(prompt).strip()
        except (EOFError, KeyboardInterrupt, OSError):  # non-tty: keep defaults
            answer = ""
        if not answer:
            continue
        if choices:
            if answer.isdigit() and 1 <= int(answer) <= len(choices):
                values[key] = choices[int(answer) - 1]
            elif answer in [str(c) for c in choices]:
                values[key] = answer
            else:
                print(f"    (not one of the choices — keeping {default})")
            continue
        values[key] = answer
    try:
        brain_config.save_config(home, values)
    except Exception as e:
        print(f"  Could not write brain.yaml ({e}) — defaults apply; edit "
              f"{brain_config.config_path(home)} later.")


def _offer_model_download(cfg: dict[str, Any]) -> None:
    from .recall.embed import REGISTRY, ModelDownloadError, ensure_files, models_cache_dir

    spec = REGISTRY.get(str(cfg.get("embed_model"))) or REGISTRY["modernbert-embed-base"]
    model_dir = models_cache_dir() / spec.key
    if all((model_dir / n).exists() and (model_dir / n).stat().st_size > 0
           for n in spec.files):
        print(f"  Embedding model {spec.key}: already downloaded ({model_dir})")
        return
    try:
        answer = input(f"  Download embedding model {spec.repo} (~90 MB, one-time)? [Y/n] ")
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    if answer.strip().lower() in ("n", "no"):
        print("  Skipped — search runs FTS-only until you run "
              "'hermes brain models --download'.")
        return
    try:
        ensure_files(spec, download=True, progress=True)
        print(f"  Model ready: {model_dir}")
    except ModelDownloadError as e:
        print(f"  {e}")
        print("  Continuing — search runs FTS-only until models are present.")


def _run_bootstrap(conn, home: Path, cfg: dict[str, Any], mode: str) -> None:
    try:
        from . import bootstrap
    except ImportError:
        print("  bootstrap module missing — run 'hermes brain bootstrap' after updating.")
        return
    try:
        from .recall.embed import get_embedder

        embedder = get_embedder(cfg, mode, allow_download=False)
        counts = bootstrap.run_bootstrap(conn, home, cfg, embedder=embedder)
        for key, value in (counts or {}).items():
            print(f"  bootstrap: {key:<20} {value}")
    except Exception as e:
        print(f"  Bootstrap failed: {e} — re-run later with 'hermes brain bootstrap' "
              f"(it is idempotent).")


def _materialize_lane1(conn, cfg: dict[str, Any]) -> None:
    try:
        from .recall import lane1
    except ImportError:
        print("  lane1 module missing — run 'hermes brain refresh-index' after updating.")
        return
    try:
        lane1.materialize(conn, cfg)
        print("  lane 1 index materialized.")
    except Exception as e:
        print(f"  lane 1 materialize failed: {e} — run 'hermes brain refresh-index' later.")


# Auxiliary task slots the brain routes sleep-time LLM work through (mirrors
# llm._TIER_TASK): 'extract' -> brain_extract, 'dream'/'consolidate' ->
# brain_consolidate. Registering these blocks lets `hermes model → Configure
# auxiliary models` and aux.call_llm("brain_extract"/"brain_consolidate") pin a
# cheap/local model per task. An absent or empty block means "use the auxiliary
# default": the host resolver (agent.auxiliary_client._get_auxiliary_task_config
# + _resolve_provider_and_model) reads auxiliary.<task>, and an empty
# provider/model resolves to the auto-detected main provider.
_AUX_TASK_SLOTS = ("brain_extract", "brain_consolidate")


def _register_aux_slots(config: dict[str, Any]) -> None:
    """Idempotently seed auxiliary.<task> routing blocks for the brain's LLM
    tiers into the Hermes config dict.

    Writes an empty routing block (provider/model = '', i.e. inherit the
    auxiliary default) for each brain task, but never clobbers a block the
    user has already configured. Persisted by _activate_provider's
    hermes_save_config(config) call (this must run before it)."""
    if not isinstance(config, dict):
        return
    aux = config.setdefault("auxiliary", {})
    if not isinstance(aux, dict):
        return
    for task in _AUX_TASK_SLOTS:
        existing = aux.get(task)
        if isinstance(existing, dict) and existing:
            continue  # user-set (or previously seeded) — leave untouched
        aux[task] = {"provider": "", "model": ""}


def _activate_provider(config: dict[str, Any]) -> None:
    """Set memory.provider: brain in Hermes config.yaml (§5.2). Inside the
    wizard hermes_cli is importable; standalone runs get the manual line."""
    try:
        if isinstance(config, dict):
            memory = config.setdefault("memory", {})
            if isinstance(memory, dict):
                memory["provider"] = "brain"
        from hermes_cli.config import save_config as hermes_save_config  # type: ignore

        hermes_save_config(config)
        print("  memory.provider set to 'brain' in config.yaml — "
              "start a new session to activate.")
    except ImportError:
        print("  To activate, set in ~/.hermes/config.yaml:  memory.provider: \"brain\"")
    except Exception as e:
        print(f"  Could not write config.yaml ({e}) — set memory.provider: \"brain\" "
              f"manually.")
