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
        {"key": "context_budget_tokens",
         "description": "Combined Brain injection cap across stable + per-turn lanes",
         "default": DEFAULTS["context_budget_tokens"]},
        {"key": "lane1_tokens",
         "description": "Stable system-prompt index request (clamped under total cap)",
         "default": DEFAULTS["lane1_tokens"]},
        {"key": "lane2_tokens",
         "description": "Per-turn recall request (uses remaining total; 0 disables)",
         "default": DEFAULTS["lane2_tokens"]},
        {"key": "recall_mode",
         "description": "How the agent reaches memory (hybrid = inject + tools, "
                        "context = inject only, tools = tools only)",
         "choices": ["hybrid", "context", "tools"],
         "default": DEFAULTS["recall_mode"]},
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
        {"key": "memories_tool",
         "description": "Expose the Anthropic-shaped 'memories' file tool "
                        "(/memories/*.md views over stored memories)",
         "choices": ["yes", "no"],
         "default": "yes" if DEFAULTS["memories_tool"] else "no"},
        {"key": "query_rewrite",
         "description": "LLM-rewrite each turn into a retrieval query "
                        "(one auxiliary call per turn; costs budget)",
         "choices": ["no", "yes"],
         "default": "yes" if DEFAULTS["query_rewrite"] else "no"},
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


# §4.6 ownership contract, printed verbatim — setup never disables built-ins.
_TRANSITION_MATRIX = """
  Automatic ownership handoff (docs/design/integration.md §4.6):

    memory.provider                "brain"
    MEMORY.md / USER.md prompt     retained until bootstrap marker is healthy;
                                    then suppressed at a safe prompt rebuild
    memory tool stays operational; built-in writes keep mirroring
    flat memory files              stay enabled as recoverable mirrors/fallback
    built-in memory review nudge   suppressed only while Brain owns context
    skills / curator / search      untouched

  If Brain cannot open or bootstrap is incomplete, Hermes retains its built-in
  prompt automatically. `hermes brain adopt-memory` is a compatibility path
  only for older Hermes versions without the ownership capability.
"""


def post_setup(hermes_home: str, config: dict[str, Any]) -> None:
    """Wizard finish: field prompts, dirs, optional model download, bootstrap,
    lane 1, activation, ownership summary, identity reminder. Never raises.

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

    _offer_cron_job(home, cfg)
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


# ---------------------------------------------------------------------------
# Dream scheduling (docs/design/integration.md §1.3)
# ---------------------------------------------------------------------------

# The dream runs as a `no_agent=True` cron SCRIPT job, never an agent session:
# agent cron jobs run skip_memory=True with a 3-minute interrupt and cost
# tokens for a job whose whole point is to run offline (F12).
_CRON_JOB_NAME = "hermes-brain dream"
# .py, not the .sh the design sketched: cron/scheduler.py picks the interpreter
# by extension (bash for .sh/.bash, Python otherwise), and a Python script needs
# no bash on Windows hosts. Relative names resolve under HERMES_HOME/scripts/.
_CRON_SCRIPT_NAME = "brain-dream.py"
_CRON_SCRIPT = '''\
"""Nightly hermes-brain consolidation. Created by `hermes memory setup`.

`--if-due` re-checks the interval watermark inside the dream lease, so extra
runs (a manual dream-now, a second scheduler) are harmless no-ops.
"""
import subprocess
import sys

sys.exit(subprocess.run(
    ["hermes", "brain", "dream", "--if-due", "--quiet"], check=False).returncode)
'''


def _cron_schedule_for(dream_time: str) -> str:
    """'03:30' -> a cron expression, falling back to a plain interval.

    Cron expressions need croniter (cron/jobs.py:parse_schedule raises without
    it), which is not a guaranteed dependency — so degrade to 'every 24h'
    rather than failing the whole setup over a scheduling nicety.
    """
    try:
        hh, _, mm = str(dream_time or "").partition(":")
        hour, minute = int(hh), int(mm)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(dream_time)
    except ValueError:
        return "every 24h"
    try:
        import croniter  # noqa: F401
    except ImportError:
        return "every 24h"
    return f"{minute} {hour} * * *"


def _offer_cron_job(home: Path, cfg: dict[str, Any]) -> None:
    """Offer to schedule the nightly dream (§1.3 step 1). Never raises.

    Without this the learning system only ever runs when a human types
    `hermes brain dream-now`: `dream_schedule`/`dream_time` were collected by
    the wizard and read by nothing.

    Cron is gateway-resident (F12), so `cron.jobs` is absent for CLI-only
    installs — there we point at `dream_schedule: on-idle`, which the provider's
    own worker honours without spawning anything.
    """
    schedule_pref = str(cfg.get("dream_schedule", "auto"))
    if schedule_pref == "manual":
        print("\n  dream_schedule=manual — no job created; run "
              "'hermes brain dream-now' when you want a consolidation shift.")
        return
    if schedule_pref == "on-idle":
        print("\n  dream_schedule=on-idle — the provider's background worker "
              "runs one bounded strategy while a session sits idle.\n"
              "  No cron job needed.")
        return

    try:
        from cron.jobs import create_job, list_jobs
    except ImportError:
        print("\n  No cron scheduler on this install (it is gateway-resident).\n"
              "  For automatic learning either set 'dream_schedule: on-idle' in\n"
              f"  {Path(home) / 'brain' / 'brain.yaml'}, or schedule this yourself:\n"
              "    hermes brain dream --if-due --quiet")
        return

    try:
        for job in list_jobs(include_disabled=True) or []:
            if str(job.get("name") or "") == _CRON_JOB_NAME:
                print(f"\n  Nightly dream cron job already exists "
                      f"(id {job.get('id')}) — leaving it alone.")
                return
    except Exception as e:
        logger.debug("brain: could not list cron jobs (%s)", e)

    dream_time = str(cfg.get("dream_time", "03:30"))
    schedule = _cron_schedule_for(dream_time)
    try:
        answer = input(f"\n  Schedule the nightly dream at {dream_time} "
                       f"({schedule})? [Y/n] ")
    except (EOFError, KeyboardInterrupt, OSError):
        answer = "n"
    if answer.strip().lower() in ("n", "no"):
        print("  Skipped — run 'hermes brain dream-now' manually, or re-run "
              "'hermes memory setup' later.")
        return

    try:
        scripts_dir = Path(home) / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / _CRON_SCRIPT_NAME).write_text(_CRON_SCRIPT, encoding="utf-8")
        job = create_job(
            None,                    # no_agent: the script IS the job
            schedule,
            name=_CRON_JOB_NAME,
            script=_CRON_SCRIPT_NAME,
            no_agent=True,
        )
        print(f"  Scheduled: {_CRON_JOB_NAME} (id {job.get('id')}, {schedule})\n"
              f"  Disable any time with 'hermes cron' or delete the job.")
    except Exception as e:
        print(f"  Could not create the cron job ({e}).\n"
              f"  Schedule this yourself instead:  hermes brain dream --if-due --quiet")


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
