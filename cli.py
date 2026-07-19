"""`hermes brain ...` CLI verbs — P1: status | search | doctor; P2 adds
bootstrap | remember | forget | pin | unpin | why | identity | refresh-index |
reindex | models | export | import | incognito.

Loader contract (verified against plugins/memory/honcho/cli.py and the
discover_plugin_cli_commands loop in plugins/memory/__init__.py): Hermes
imports this module at argparse setup on EVERY invocation and calls
``register_cli(subparser)``; the selected handler is ``brain_command(args)``
routed via ``dest='brain_command'``. Therefore module level stays feather-
light — stdlib only, all sibling imports deferred into the command bodies.

Errors-that-teach convention (docs/design/integration.md): every failure
names the exact corrective action, and ``doctor`` prints a one-line remedy
next to every WARN/FAIL. Plain aligned text, no emoji.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_KINDS = ("fact", "decision", "preference", "warning", "insight")
# Full schema vocabulary for direct writes (schema.sql memories.kind comment).
_ALL_KINDS = _KINDS + ("strategy", "case", "profile")

# lane1_snapshot staleness threshold for doctor (days).
_LANE1_STALE_DAYS = 7

# Documented budget ranges (config.py DEFAULTS comments): lane 1 is
# 800-1500 hard-truncated by the renderer; lane 2 is 0 (disabled) to 1500.
_LANE1_RANGE = (800, 1500)
_LANE2_RANGE = (0, 1500)


def _hermes_home() -> Path:
    """Resolve HERMES_HOME: hermes_constants inside Hermes, env, ~/.hermes."""
    try:
        from hermes_constants import get_hermes_home  # type: ignore[import-not-found]
        return Path(get_hermes_home())
    except ImportError:
        pass
    env = os.environ.get("HERMES_HOME")
    return Path(env) if env else Path.home() / ".hermes"


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Attach the P1 verbs to the `hermes brain` subparser."""
    cmds = subparser.add_subparsers(dest="brain_command")

    cmds.add_parser("status", help="capture/recall counters, DB health at a glance")

    p_search = cmds.add_parser("search", help="search memories and episodes")
    p_search.add_argument("query", nargs="+", help="search terms")
    p_search.add_argument("--kind", choices=_KINDS, default=None,
                          help="restrict to one memory kind")
    p_search.add_argument("--project", default=None, help="restrict to a project scope")
    p_search.add_argument("--limit", type=int, default=8, help="max results (default 8)")
    p_search.add_argument("--episodes", dest="episodes", action="store_true", default=True,
                          help="include raw episode hits (default)")
    p_search.add_argument("--no-episodes", dest="episodes", action="store_false",
                          help="memories only")

    cmds.add_parser("doctor", help="run health checks (PASS/WARN/FAIL + remedies)")

    p_boot = cmds.add_parser("bootstrap", help="first-run import (re-runnable, dedup-safe)")
    p_boot.add_argument("--daemon", default=None, metavar="PATH",
                        help="also import a Daem0n-MCP memory.db")
    p_boot.add_argument("--max-sessions", type=int, default=None,
                        help="cap state.db sessions backfilled this run")

    p_rem = cmds.add_parser("remember", help="write a memory directly (owner trust)")
    p_rem.add_argument("text", nargs="+", help="the memory text")
    p_rem.add_argument("--kind", choices=_ALL_KINDS, default=None,
                       help="memory kind (default fact)")

    p_forget = cmds.add_parser("forget", help="tombstone a memory (--hard purges)")
    p_forget.add_argument("id", help="uid prefix, at least 6 characters")
    p_forget.add_argument("--hard", action="store_true",
                          help="permanent delete (compliance purge)")
    p_forget.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    p_pin = cmds.add_parser("pin", help="pin a memory (x1.3 recall boost)")
    p_pin.add_argument("id", help="uid prefix, at least 6 characters")
    p_unpin = cmds.add_parser("unpin", help="remove a memory's pin")
    p_unpin.add_argument("id", help="uid prefix, at least 6 characters")

    p_why = cmds.add_parser("why", help="provenance + retrieval history for a memory")
    p_why.add_argument("id", help="uid prefix, at least 6 characters")

    p_fact = cmds.add_parser("fact", help="current (or as-of) s-p-o facts for a subject")
    p_fact.add_argument("subject", help="the subject to look up")
    p_fact.add_argument("--as-of", dest="as_of", default=None,
                        help="ISO timestamp for point-in-time truth (default: now)")
    p_fact.add_argument("--predicate", default=None, help="filter to one predicate")

    p_ask = cmds.add_parser("ask", help="ask a natural-language question over the brain (cited)")
    p_ask.add_argument("question", nargs="+", help="the question (quotes optional)")
    p_ask.add_argument("--level", choices=["fast", "deep"], default="deep",
                       help="fast = quick lookup (<=2 steps); deep = multi-step (default)")
    p_ask.add_argument("--json", action="store_true", help="emit the raw result as JSON")

    p_ctx = cmds.add_parser("context",
                            help="assemble a token-budgeted context block from the brain")
    p_ctx.add_argument("--tokens", type=int, default=None,
                       help="token budget (default: precompress_tokens config)")

    p_eval = cmds.add_parser("eval", help="run the retrieval/answer eval harness on a fixture")
    p_eval.add_argument("--fixture", default=None,
                        help="eval fixture JSON (default: bundled eval_basic.json)")
    p_eval.add_argument("--real", action="store_true",
                        help="use the real aux LLM instead of the scripted fake")

    p_id = cmds.add_parser("identity", help="manage platform identities (trust roots)")
    id_sub = p_id.add_subparsers(dest="identity_command")
    p_id_add = id_sub.add_parser("add", help="enroll a platform user")
    p_id_add.add_argument("platform", help="telegram|discord|slack|...")
    p_id_add.add_argument("platform_user_id", help="platform-native user id")
    p_id_add.add_argument("--owner", action="store_true",
                          help="this user IS the owner (full trust)")
    p_id_add.add_argument("--name", default=None, help="display name")
    id_sub.add_parser("list", help="list enrolled identities")
    p_id_rm = id_sub.add_parser("rm", help="remove an identity")
    p_id_rm.add_argument("platform")
    p_id_rm.add_argument("platform_user_id")

    cmds.add_parser("refresh-index", help="rematerialize the lane 1 snapshot")

    p_re = cmds.add_parser("reindex", help="backfill vector embeddings")
    p_re.add_argument("--limit", type=int, default=500, help="max rows per table (default 500)")
    p_re.add_argument("--all", dest="all_rows", action="store_true",
                      help="also re-embed rows written by an older embedder")

    p_models = cmds.add_parser("models", help="show embedding model cache")
    p_models.add_argument("--download", action="store_true",
                          help="download the configured embed_model now")

    p_exp = cmds.add_parser("export", help="JSONL + markdown snapshot of current memories")
    p_exp.add_argument("--out", default=None, help="output directory (default brain/exports/<date>)")

    p_imp = cmds.add_parser("import", help="re-import a memories.jsonl (content-hash dedup)")
    p_imp.add_argument("file", help="path to a memories.jsonl produced by export")
    p_imp.add_argument("--trust-owner", dest="trust_owner", action="store_true",
                       help="keep the file's trust/pinned/status values verbatim "
                            "(only for files YOU exported; default caps trust at "
                            "'agent' and quarantines steering-shaped rows)")

    p_inc = cmds.add_parser("incognito", help="pause capture (applies at next session)")
    p_inc.add_argument("state", nargs="?", choices=("on", "off", "status"), default="status")

    # -- dream cycle (P4): user-invoked or cron-invoked; never auto-spawned --
    p_dream = cmds.add_parser("dream-now", help="run a consolidation shift now")
    p_dream.add_argument("--phase", choices=list(_dream_phases()), default=None,
                         help="run just one strategy (default: full pipeline)")
    p_dream.add_argument("--dry-run", action="store_true",
                         help="force dry-run over every strategy's mode")

    p_ifdue = cmds.add_parser("dream", help="run a shift if one is due (for cron/ops)")
    p_ifdue.add_argument("--if-due", action="store_true",
                         help="no-op unless the min interval has elapsed")
    p_ifdue.add_argument("--quiet", action="store_true", help="log only, no stdout")
    p_ifdue.add_argument("--enable", metavar="STRATEGY", default=None,
                         help="promote a strategy to active mode")
    p_ifdue.add_argument("--disable", metavar="STRATEGY", default=None,
                         help="set a strategy's mode to off")

    # -- P5: learning surface --------------------------------------------------
    p_ins = cmds.add_parser("insights", help="longitudinal learning metrics from turn_outcomes")
    p_ins.add_argument("--days", type=int, default=30, help="window in days (default 30)")

    p_rev = cmds.add_parser("review", help="the review queue: proposals + quarantined memories")
    p_rev.add_argument("--approve", metavar="UID", default=None, help="approve a proposal")
    p_rev.add_argument("--reject", metavar="UID", default=None, help="reject a proposal")

    p_sk = cmds.add_parser("skills", help="brain-forged skills: list | forge | approve | reject")
    sk_sub = p_sk.add_subparsers(dest="skills_command")
    sk_sub.add_parser("list", help="list forged skills + open drafts")
    sk_forge = sk_sub.add_parser("forge", help="detect+draft one skill now (manual)")
    sk_forge.add_argument("--no-approve", action="store_true",
                          help="draft only, do not auto-approve even if configured")
    sk_ok = sk_sub.add_parser("approve", help="promote a validated draft into the skills tree")
    sk_ok.add_argument("uid", help="proposal uid (from 'hermes brain review')")
    sk_no = sk_sub.add_parser("reject", help="reject a draft")
    sk_no.add_argument("uid", help="proposal uid")

    p_sync = cmds.add_parser("sync", help="multi-device encrypted delta sync")
    sync_sub = p_sync.add_subparsers(dest="sync_command")
    sync_sub.add_parser("init", help="generate device id + shared account/salt (run once)")
    sync_sub.add_parser("push", help="encrypt and push local deltas to the relay")
    sync_sub.add_parser("pull", help="pull, decrypt, and apply remote deltas")
    sync_sub.add_parser("status", help="show sync config, cursors, and unsynced count")

    cmds.add_parser("mcp", help="run the stdio MCP server (for external agents)")

    p_adopt = cmds.add_parser("adopt-memory",
                              help="apply the 'brain owns memory' matrix to Hermes config")
    p_adopt.add_argument("--apply", action="store_true",
                         help="write the changes (default: dry-run preview)")


def brain_command(args: argparse.Namespace) -> int:
    """Route `hermes brain <verb>`; bare `hermes brain` shows status."""
    cmd = getattr(args, "brain_command", None) or "status"
    handler = {
        "status": cmd_status, "search": cmd_search, "doctor": cmd_doctor,
        "bootstrap": cmd_bootstrap, "remember": cmd_remember, "forget": cmd_forget,
        "pin": cmd_pin, "unpin": cmd_unpin, "why": cmd_why, "fact": cmd_fact,
        "ask": cmd_ask, "context": cmd_context, "eval": cmd_eval,
        "identity": cmd_identity,
        "refresh-index": cmd_refresh_index, "reindex": cmd_reindex,
        "models": cmd_models, "export": cmd_export, "import": cmd_import,
        "incognito": cmd_incognito, "dream-now": cmd_dream_now, "dream": cmd_dream,
        "insights": cmd_insights, "review": cmd_review, "skills": cmd_skills,
        "sync": cmd_sync,
        "mcp": cmd_mcp, "adopt-memory": cmd_adopt_memory,
    }.get(cmd)
    if handler is None:
        print(f"Unknown brain command: {cmd}. Try: hermes brain status|search|doctor|"
              f"remember|forget|why|fact|ask|identity|reindex|models|export|import|incognito|"
              f"dream-now|dream|insights|review|skills|context|sync|mcp|adopt-memory",
              file=sys.stderr)
        return 1
    return handler(args)


# Derived from the one source of truth so the CLI choices can never drift
# from the pipeline the dream actually runs.
def _dream_phases() -> tuple[str, ...]:
    from .dream.shift import PIPELINE

    return PIPELINE


_DREAM_MIN_INTERVAL_HOURS = 6


def _open_db(home: Path):
    """Open brain.db for a CLI verb; on failure teach and return None."""
    from .store import db

    try:
        return db.connect(home)
    except db.FutureSchemaError as e:
        print(f"brain.db is from a newer plugin version.\n  {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Cannot open brain.db: {e}\n"
              f"  Remedy: run 'hermes brain doctor' to diagnose.", file=sys.stderr)
        return None


def _audit_cli(conn, action: str, uid: str, detail: dict | None = None) -> None:
    from .store import db

    conn.execute(
        "INSERT INTO audit_log (actor, action, target, detail, ts) VALUES ('cli',?,?,?,?)",
        (action, uid, json.dumps(detail) if detail else None, db.iso_now()),
    )


def _resolve_uid(conn, prefix: str, *, current_only: bool = True):
    """uid-prefix -> single memories row, or None after an error-that-teaches."""
    prefix = (prefix or "").strip().upper()
    if len(prefix) < 6:
        print(f"id '{prefix}' is too short — pass at least 6 characters of the uid "
              f"(shown by 'hermes brain search').", file=sys.stderr)
        return None
    sql = "SELECT * FROM memories WHERE uid LIKE ?"
    if current_only:
        sql += " AND valid_to IS NULL"
    rows = conn.execute(sql + " LIMIT 5", (prefix + "%",)).fetchall()
    if not rows:
        print(f"No {'current ' if current_only else ''}memory matches id '{prefix}'.\n"
              f"  Remedy: find the uid with 'hermes brain search <words>'.", file=sys.stderr)
        return None
    if len(rows) > 1:
        listing = ", ".join(r["uid"][:12] for r in rows)
        print(f"id '{prefix}' is ambiguous ({listing}).\n"
              f"  Remedy: pass more characters of the uid.", file=sys.stderr)
        return None
    return rows[0]


def _first_archive_ref(source_refs: str | None) -> str | None:
    """The archive ref ('<YYYY-MM>.jsonl.gz:<uid>') a purge stored in
    source_refs, or None. Used by 'why' to recover purged content."""
    try:
        for entry in json.loads(source_refs or "[]"):
            if isinstance(entry, str) and entry.startswith("archive:"):
                return entry[len("archive:"):]
    except (ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def _tier_label(cfg: dict, resolved: str) -> str:
    """One-line tier description: resolved mode + embedder identity when the
    model files are actually on disk (no heavy imports, no downloads)."""
    from .recall import embed

    if resolved == "full":
        spec = embed.REGISTRY.get(str(cfg.get("embed_model"))) \
            or embed.REGISTRY["modernbert-embed-base"]
        model_dir = embed.models_cache_dir() / spec.key
        if all((model_dir / n).exists() and (model_dir / n).stat().st_size > 0
               for n in spec.files):
            return f"full ({spec.key}-q8:{spec.truncate_to or spec.native_dim})"
        return "full (model files missing — run 'hermes brain models --download')"
    if resolved == "lite":
        return "lite (potion-retrieval-32m static embeddings)"
    if resolved == "stub":
        return "stub (hash embedder — tests only)"
    return "fts-only (no embedder)"


def cmd_status(args: argparse.Namespace) -> int:
    from . import config
    from .store import db, sysinfo
    from .store import vec as vec_store

    home = _hermes_home()
    try:
        conn = db.connect(home)
    except db.FutureSchemaError as e:
        print(f"brain.db is from a newer plugin version.\n  {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Cannot open brain.db at {db.db_path(home)}: {e}\n"
              f"  Remedy: check the file/directory permissions, or move the file aside "
              f"to let hermes-brain recreate it.", file=sys.stderr)
        return 1

    try:
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 86400)) + ".000Z"
        ep_total = conn.execute("SELECT count(*) AS n FROM episodes").fetchone()["n"]
        ep_24h = conn.execute(
            "SELECT count(*) AS n FROM episodes WHERE ts >= ?", (cutoff,)
        ).fetchone()["n"]

        by_type = conn.execute(
            "SELECT memory_type, count(*) AS n FROM memories "
            "WHERE valid_to IS NULL AND status='active' "
            "GROUP BY memory_type ORDER BY memory_type"
        ).fetchall()
        quarantined = conn.execute(
            "SELECT count(*) AS n FROM memories WHERE status='quarantined'"
        ).fetchone()["n"]
        pending = conn.execute(
            "SELECT count(*) AS n FROM ingest_buffer WHERE promoted_at IS NULL"
        ).fetchone()["n"]

        size_mb = db.db_path(home).stat().st_size / (1024 * 1024)
        schema = db.get_meta(conn, "schema_version", "?")
        caps = db.capabilities(conn)
        cfg = config.load_config(home)
        mode = cfg["mode"]

        activity = conn.execute(
            "SELECT source, last_seen FROM activity ORDER BY last_seen DESC LIMIT 5"
        ).fetchall()
        leases = conn.execute(
            "SELECT name, holder, expires_at FROM brain_lease ORDER BY name"
        ).fetchall()

        print(f"brain.db          {db.db_path(home)}  ({size_mb:.1f} MB, schema v{schema})")
        print(f"mode              {mode}   capabilities: "
              f"fts5={'yes' if caps.get('fts5') else 'NO'} "
              f"vec={'yes' if caps.get('vec') else 'no'}")
        print(f"tier              {_tier_label(cfg, sysinfo.resolve_mode(str(mode)))}")
        vstats = vec_store.stats(conn)
        if vstats:
            print(f"vectors           mem_vec={vstats.get('mem_vec')} "
                  f"epi_vec={vstats.get('epi_vec')}")
        else:
            print("vectors           none (sqlite-vec not loaded or tables absent)")
        print(f"episodes          {ep_total}  ({ep_24h} in last 24h)")
        if by_type:
            for row in by_type:
                print(f"memories.{row['memory_type']:<10} {row['n']}")
        else:
            print("memories          0")
        print(f"quarantined       {quarantined}")
        print(f"buffer pending    {pending}")
        for row in leases:
            state = f"held by {row['holder']} until {row['expires_at']}" if row["holder"] else "free"
            print(f"lease.{row['name']:<12} {state}")
        # dreams: last shift + per-strategy effective mode.
        last_dream = conn.execute(
            "SELECT shift_id, finished_at, outcome FROM shift_runs "
            "WHERE finished_at IS NOT NULL ORDER BY finished_at DESC LIMIT 1").fetchone()
        if last_dream:
            print(f"last dream        {last_dream['finished_at']} "
                  f"({last_dream['outcome']})")
        else:
            print("last dream        never — run 'hermes brain dream-now'")
        modes = {r["strategy"]: r["mode"] for r in
                 conn.execute("SELECT strategy, mode FROM strategy_state").fetchall()}
        from .dream.shift import DEFAULT_MODES, PIPELINE
        eff = [f"{n}={modes.get(n) or DEFAULT_MODES.get(n, 'dry_run')}" for n in PIPELINE]
        print(f"strategies        {'  '.join(eff)}")
        for row in activity:
            print(f"activity          {row['source']}  last seen {row['last_seen']}")
        return 0
    except Exception as e:
        print(f"Status query failed: {e}\n"
              f"  Remedy: run 'hermes brain doctor' to diagnose brain.db.", file=sys.stderr)
        return 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> int:
    from . import config
    from .recall import render as recall_render
    from .recall import search as recall_search
    from .recall.embed import get_embedder
    from .recall.rerank import get_reranker
    from .store import db, sysinfo
    from .store import vec as vec_store

    home = _hermes_home()
    query = " ".join(args.query)
    cfg = config.load_config(home)
    mode = sysinfo.resolve_mode(str(cfg.get("mode", "auto")))
    embedder = get_embedder(cfg, mode, allow_download=False)  # never surprise-download
    reranker = get_reranker(cfg, mode, allow_download=False)
    try:
        conn = db.connect(home)
    except Exception as e:
        print(f"Cannot open brain.db: {e}\n"
              f"  Remedy: run 'hermes brain doctor' to diagnose.", file=sys.stderr)
        return 1

    try:
        vec_ok = embedder is not None and vec_store.vec_available(conn)
        legs = "fts+vec" if vec_ok else "fts"
        print(f"legs: {legs}{' +rerank' if reranker is not None else ''}")
        hits = recall_search.search(
            conn,
            query,
            kinds=[args.kind] if args.kind else None,
            scope_project=args.project,
            limit=args.limit,
            include_episodes=args.episodes,
            embedder=embedder,
            reranker=reranker,
        )
        if not hits:
            print("(no matches)")
            print("Hint: 'hermes brain status' shows capture counts — if episodes are 0, "
                  "nothing has been captured yet.")
            return 0
        print(recall_render.render_hits_text(hits))
        return 0
    except Exception as e:
        print(f"Search failed: {e}\n"
              f"  Remedy: simplify the query (plain words work best) or run "
              f"'hermes brain doctor' to check FTS health.", file=sys.stderr)
        return 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    """P1 + P2 health checks; every WARN/FAIL carries a one-line remedy."""
    from . import config
    from .store import db

    home = _hermes_home()
    results: list[tuple[str, str, str]] = []  # (PASS|WARN|FAIL, check, detail/remedy)

    def report(status: str, check: str, detail: str) -> None:
        results.append((status, check, detail))
        print(f"[{status:<4}] {check:<18} {detail}")

    conn = None
    future_schema_msg = None
    try:
        conn = db.connect(home, create=False)
    except FileNotFoundError:
        report("FAIL", "db-open", f"brain.db not found at {db.db_path(home)} — "
               f"run a hermes session (or 'hermes brain status') to create it")
    except db.FutureSchemaError as e:
        future_schema_msg = str(e)
        report("FAIL", "db-open", "brain.db is from a newer plugin — see schema-version check")
    except Exception as e:
        report("FAIL", "db-open", f"cannot open brain.db: {e} — check file permissions "
               f"or restore from a brain.pre-v*.db backup")

    if conn is not None:
        # 1. integrity
        try:
            ok = conn.execute("PRAGMA quick_check").fetchone()[0]
            if ok == "ok":
                report("PASS", "db-open", "brain.db opens, quick_check ok")
            else:
                report("FAIL", "db-open", f"quick_check: {ok} — restore from a "
                       f"brain.pre-v*.db backup or move brain.db aside to recreate")
        except Exception as e:
            report("FAIL", "db-open", f"quick_check failed: {e} — restore from backup")

        # 2. schema version
        ver = db.get_meta(conn, "schema_version")
        if ver is not None and int(ver) == db.SCHEMA_VERSION:
            report("PASS", "schema-version", f"v{ver} (current)")
        else:
            report("FAIL", "schema-version", f"found v{ver}, code expects "
                   f"v{db.SCHEMA_VERSION} — update the plugin: git -C <plugin dir> pull")

        caps = db.capabilities(conn)

        # 3. fts5 capability
        if caps.get("fts5"):
            report("PASS", "fts5", "available")
        else:
            report("WARN", "fts5", "absent — search degraded to LIKE; install a Python "
                   "with SQLite FTS5, e.g. python.org >=3.12")

        # 4. FTS consistency
        if caps.get("fts5"):
            try:
                n_ep = conn.execute("SELECT count(*) AS n FROM episodes").fetchone()["n"]
                n_fts = conn.execute("SELECT count(*) AS n FROM episode_fts").fetchone()["n"]
                if n_ep == n_fts:
                    report("PASS", "fts-consistency", f"episode_fts in sync ({n_ep} rows)")
                else:
                    report("FAIL", "fts-consistency",
                           f"episode_fts has {n_fts} rows, episodes has {n_ep} — rebuild: "
                           f"sqlite3 brain.db \"INSERT INTO episode_fts(episode_fts) "
                           f"VALUES('rebuild')\"")
            except Exception as e:
                report("FAIL", "fts-consistency", f"query failed: {e} — rebuild the FTS "
                       f"index with INSERT INTO episode_fts(episode_fts) VALUES('rebuild')")

        # 5. stale lease
        try:
            now = db.iso_now()
            stale = conn.execute(
                "SELECT name, holder, expires_at FROM brain_lease "
                "WHERE holder IS NOT NULL AND expires_at < ?", (now,)
            ).fetchall()
            if stale:
                names = ", ".join(f"{r['name']} (holder {r['holder']})" for r in stale)
                report("WARN", "lease", f"stale: {names} — a dream process may have "
                       f"crashed; lease auto-expires, no action needed")
            else:
                report("PASS", "lease", "no stale holders")
        except Exception as e:
            report("FAIL", "lease", f"query failed: {e} — brain_lease table missing; "
                   f"re-run 'hermes brain status' to re-apply schema")

        # 7. WAL
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            if str(mode).lower() == "wal":
                report("PASS", "wal", "journal_mode=wal")
            else:
                report("WARN", "wal", f"journal_mode={mode} — WAL unavailable "
                       f"(network filesystem?); move ~/.hermes/brain to a local disk")
        except Exception as e:
            report("WARN", "wal", f"cannot read journal_mode: {e}")

        # 8-12: P2 retrieval checks (model files, sqlite-vec, dims, lane 1,
        # owner identity). Config parse failures are reported by check 13.
        try:
            cfg = config.load_config(home)
        except Exception:
            cfg = dict(config.DEFAULTS)
        _doctor_p2_checks(conn, cfg, report)

        conn.close()
    elif future_schema_msg:
        report("FAIL", "schema-version", future_schema_msg)

    # 6. config (independent of the DB)
    try:
        cfg = config.load_config(home)
        l1, l2 = cfg["lane1_tokens"], cfg["lane2_tokens"]
        problems = []
        if not _LANE1_RANGE[0] <= l1 <= _LANE1_RANGE[1]:
            problems.append(f"lane1_tokens={l1} outside {_LANE1_RANGE[0]}-{_LANE1_RANGE[1]}")
        if not _LANE2_RANGE[0] <= l2 <= _LANE2_RANGE[1]:
            problems.append(f"lane2_tokens={l2} outside {_LANE2_RANGE[0]}-{_LANE2_RANGE[1]}")
        if problems:
            report("WARN", "config", "; ".join(problems) +
                   f" — edit {config.config_path(home)}")
        else:
            report("PASS", "config", "parses; lane budgets within documented ranges")
    except Exception as e:
        report("FAIL", "config", f"load failed: {e} — fix or delete "
               f"{config.config_path(home)} (defaults apply when absent)")

    n_pass = sum(1 for s, _, _ in results if s == "PASS")
    n_warn = sum(1 for s, _, _ in results if s == "WARN")
    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    print(f"\n{n_pass} pass, {n_warn} warn, {n_fail} fail")
    return 1 if n_fail else 0


def _doctor_p2_checks(conn, cfg: dict, report) -> None:
    """Doctor checks 8-12: the P2 retrieval stack. WARN-only — a missing
    embedder degrades search, it never breaks capture."""
    from .recall import embed
    from .store import db, sysinfo

    resolved = sysinfo.resolve_mode(str(cfg.get("mode", "auto")))
    expected_dim = None

    # 8. model files for the resolved tier
    if resolved == "full":
        spec = embed.REGISTRY.get(str(cfg.get("embed_model"))) \
            or embed.REGISTRY["modernbert-embed-base"]
        expected_dim = spec.truncate_to or spec.native_dim
        model_dir = embed.models_cache_dir() / spec.key
        if all((model_dir / n).exists() and (model_dir / n).stat().st_size > 0
               for n in spec.files):
            report("PASS", "model-files", f"{spec.key} present in {model_dir}")
        else:
            report("WARN", "model-files", f"{spec.key} missing from {model_dir} — "
                   f"run 'hermes brain models --download'")
    else:
        if resolved == "stub":
            expected_dim = embed.TARGET_DIM
        report("PASS", "model-files", f"tier '{resolved}' needs no ONNX model files")

    # 9. sqlite-vec importable
    if sysinfo.importable("sqlite_vec"):
        report("PASS", "sqlite-vec", "importable")
    else:
        report("WARN", "sqlite-vec", "not importable — vector recall disabled; "
               "pip install sqlite-vec")

    # 10. vec table dim vs the tier's embedder dim
    try:
        stored = db.get_meta(conn, "vec_dim")
        if stored is None:
            report("PASS", "vec-dim", "no vector tables yet (created on first embed)")
        elif expected_dim is not None and int(stored) != expected_dim:
            report("WARN", "vec-dim", f"vec tables are {stored}d but tier '{resolved}' "
                   f"produces {expected_dim}d — run 'hermes brain reindex'")
        else:
            report("PASS", "vec-dim", f"{stored}d")
    except Exception as e:
        report("WARN", "vec-dim", f"query failed: {e}")

    # 11. lane1_snapshot freshness
    try:
        row = conn.execute("SELECT max(rendered_at) AS ts FROM lane1_snapshot").fetchone()
        cutoff = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - _LANE1_STALE_DAYS * 86400)
        ) + ".000Z"
        if row["ts"] is None:
            report("WARN", "lane1-snapshot", "empty — run 'hermes brain refresh-index'")
        elif row["ts"] < cutoff:
            report("WARN", "lane1-snapshot", f"stale (rendered {row['ts']}, >"
                   f"{_LANE1_STALE_DAYS}d) — run 'hermes brain refresh-index'")
        else:
            report("PASS", "lane1-snapshot", f"rendered {row['ts']}")
    except Exception as e:
        report("WARN", "lane1-snapshot", f"query failed: {e}")

    # 12. owner identity enrolled
    try:
        n = conn.execute(
            "SELECT count(*) AS n FROM identities WHERE is_owner=1"
        ).fetchone()["n"]
        if n:
            report("PASS", "owner-identity", f"{n} owner identit{'y' if n == 1 else 'ies'}")
        else:
            report("WARN", "owner-identity", "none enrolled — gateway messages can never "
                   "be owner-trusted; run 'hermes brain identity add <platform> "
                   "<user_id> --owner'")
    except Exception as e:
        report("WARN", "owner-identity", f"query failed: {e}")


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def cmd_bootstrap(args: argparse.Namespace) -> int:
    from . import config
    from .recall.embed import get_embedder
    from .store import sysinfo

    try:
        from . import bootstrap
    except ImportError:
        print("bootstrap module missing — update the plugin: git -C <plugin dir> pull",
              file=sys.stderr)
        return 1

    home = _hermes_home()
    cfg = config.load_config(home)
    mode = sysinfo.resolve_mode(str(cfg.get("mode", "auto")))
    embedder = get_embedder(cfg, mode, allow_download=False)  # never surprise-download
    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        counts = bootstrap.run_bootstrap(
            conn, home, cfg,
            embedder=embedder,
            daemon_db=args.daemon,
            max_sessions=args.max_sessions,
        )
        for key, value in (counts or {}).items():
            print(f"{key:<24} {value}")
        return 0
    except Exception as e:
        print(f"Bootstrap failed: {e}\n"
              f"  Remedy: bootstrap is re-runnable (content-hash dedup) — fix the "
              f"cause and run 'hermes brain bootstrap' again.", file=sys.stderr)
        return 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# remember / forget / pin / unpin / why
# ---------------------------------------------------------------------------

def cmd_remember(args: argparse.Namespace) -> int:
    from .capture.turns import TurnContext, capture_memory_write

    home = _hermes_home()
    text = " ".join(args.text).strip()
    if not text:
        print("Nothing to remember: empty text.", file=sys.stderr)
        return 1
    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        ctx = TurnContext(session_id="cli", platform="cli",
                          principal_id="owner", trust_tier="owner")
        mem_id = capture_memory_write(conn, ctx, "add", "memory", text, None)
        if mem_id is None:
            print("Write failed (see log).\n"
                  "  Remedy: run 'hermes brain doctor' to check brain.db health.",
                  file=sys.stderr)
            return 1
        # capture_memory_write is the built-in-mirror path (trust 'agent',
        # created_by 'memory_tool'); an explicit CLI write IS the owner
        # speaking, so restamp provenance and apply --kind.
        conn.execute(
            "UPDATE memories SET trust_tier='owner', created_by='user_explicit',"
            " kind=COALESCE(?, kind) WHERE id=?",
            (args.kind, mem_id),
        )
        row = conn.execute("SELECT uid, kind FROM memories WHERE id=?", (mem_id,)).fetchone()
        # Sync seam: a CLI-remembered memory must enter the outbox too, or a
        # synced device never sees it. Off unless sync is on.
        from . import config as _config
        from .store import events as _events
        _events.record_event(
            conn, "create", row["uid"], payload={"kind": row["kind"]},
            enabled=_events.recording_enabled(_config.load_config(home)))
        conn.commit()
        print(f"remembered {row['uid'][:8]} ({row['kind']})")
        return 0
    finally:
        conn.close()


def cmd_forget(args: argparse.Namespace) -> int:
    from .store import db
    from .store import vec as vec_store

    home = _hermes_home()
    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        # --hard may purge already-tombstoned rows too (compliance path).
        row = _resolve_uid(conn, args.id, current_only=not args.hard)
        if row is None:
            return 1
        if not args.yes:
            snippet = (row["content"] or "")[:60].replace("\n", " ")
            verb = "PERMANENTLY DELETE" if args.hard else "tombstone"
            try:
                answer = input(f"{verb} {row['uid'][:8]} \"{snippet}\"? [y/N] ")
            except EOFError:
                answer = ""
            if answer.strip().lower() not in ("y", "yes"):
                print("cancelled")
                return 1
        now = db.iso_now()
        if args.hard:
            # Compliance purge: clear referencing rows first (FKs are ON),
            # then delete — the memories_ad trigger cleans memory_fts.
            mem_id = row["id"]
            conn.execute("DELETE FROM retrieval_log WHERE memory_id=?", (mem_id,))
            conn.execute("DELETE FROM edges WHERE src_id=? OR dst_id=?", (mem_id, mem_id))
            conn.execute("DELETE FROM entity_mentions WHERE memory_id=?", (mem_id,))
            conn.execute("DELETE FROM lane1_snapshot WHERE memory_id=?", (mem_id,))
            # facts.memory_id REFERENCES memories(id) and FKs are ON, so the
            # delete below would fail if any fact still points at this row.
            conn.execute("DELETE FROM facts WHERE memory_id=?", (mem_id,))
            for col in ("supersedes_id", "superseded_by", "invalidated_by"):
                conn.execute(f"UPDATE memories SET {col}=NULL WHERE {col}=?", (mem_id,))
            conn.execute("DELETE FROM memories WHERE id=?", (mem_id,))
            # Compliance also scrubs any archived copy of the raw text — a hard
            # purge that left the content in the episodic archive would defeat it.
            from .store import archive
            archive.purge_uid(home, row["uid"])
            try:  # vec cleanup is best-effort: absent extension must not block a purge
                if vec_store.vec_available(conn):
                    vec_store.delete(conn, "mem_vec", mem_id)
                elif db.get_meta(conn, "vec_dim") is not None:
                    # A vector index exists but sqlite-vec is not loadable on
                    # THIS connection: queue the orphan for the next reindex.
                    pending = json.loads(db.get_meta(conn, "vec_pending_delete") or "[]")
                    if mem_id not in pending:
                        pending.append(mem_id)
                    db.set_meta(conn, "vec_pending_delete", json.dumps(pending))
                    print("note: sqlite-vec not loadable here — vector cleanup deferred "
                          "to the next 'hermes brain reindex'")
            except Exception as e:
                logger.warning("vec delete for %s failed: %s", row["uid"], e)
            _audit_cli(conn, "cli_forget_hard", row["uid"])
            db.bump_generation(conn, "mem")
            conn.commit()
            print(f"forgot {row['uid'][:8]} (hard — row purged)")
        else:
            conn.execute(
                "UPDATE memories SET status='tombstone', valid_to=? WHERE id=?",
                (now, row["id"]),
            )
            # A tombstoned memory must leave lane 1 immediately, not at the
            # next dream-cycle rematerialization (mirrors the hard branch).
            conn.execute("DELETE FROM lane1_snapshot WHERE memory_id=?", (row["id"],))
            # Propagate the deletion to other devices (off unless sync_events).
            # The row stays present as a tombstone, so the sync engine gates on
            # its scope at push time — a private forget leaks nothing.
            from . import config as _config
            from .store import events as _events
            _events.record_event(
                conn, "tombstone", row["uid"],
                enabled=_events.recording_enabled(_config.load_config(home)))
            _audit_cli(conn, "cli_forget", row["uid"])
            db.bump_generation(conn, "mem")
            conn.commit()
            print(f"forgot {row['uid'][:8]} (tombstoned; purged after the grace period)")
        return 0
    except Exception as e:
        print(f"Forget failed: {e}\n"
              f"  Remedy: run 'hermes brain doctor' to check brain.db health.",
              file=sys.stderr)
        return 1
    finally:
        conn.close()


def _set_pinned(args: argparse.Namespace, pinned: int, action: str) -> int:
    from .store import db

    conn = _open_db(_hermes_home())
    if conn is None:
        return 1
    try:
        row = _resolve_uid(conn, args.id, current_only=True)
        if row is None:
            return 1
        conn.execute("UPDATE memories SET pinned=? WHERE id=?", (pinned, row["id"]))
        _audit_cli(conn, action, row["uid"])
        db.bump_generation(conn, "mem")
        conn.commit()
        print(f"{'pinned' if pinned else 'unpinned'} {row['uid'][:8]}")
        return 0
    except Exception as e:
        print(f"{action} failed: {e}\n"
              f"  Remedy: run 'hermes brain doctor' to check brain.db health.",
              file=sys.stderr)
        return 1
    finally:
        conn.close()


def cmd_pin(args: argparse.Namespace) -> int:
    return _set_pinned(args, 1, "cli_pin")


def cmd_unpin(args: argparse.Namespace) -> int:
    return _set_pinned(args, 0, "cli_unpin")


def cmd_why(args: argparse.Namespace) -> int:
    """Provenance: envelope, version chain, retrieval stats, audit trail."""
    conn = _open_db(_hermes_home())
    if conn is None:
        return 1
    try:
        row = _resolve_uid(conn, args.id, current_only=False)
        if row is None:
            return 1

        print(f"uid               {row['uid']}")
        print(f"kind              {row['kind'] or '-'} "
              f"({row['memory_type']}, {row['epistemic']})")
        print(f"status            {row['status']}   pinned={'yes' if row['pinned'] else 'no'}"
              f"   version={row['version']}")
        print(f"trust             {row['trust_tier']}   created_by={row['created_by']}")
        print(f"source            session={row['source_session'] or '-'} "
              f"platform={row['source_platform'] or '-'} "
              f"author={row['source_author'] or '-'}")
        print(f"valid             {row['valid_from']} -> {row['valid_to'] or '(current)'}"
              f"   recorded {row['recorded_at']}")
        if row["source_refs"] and row["source_refs"] != "[]":
            print(f"source_refs       {row['source_refs']}")
        if row["tags"] and row["tags"] != "[]":
            print(f"tags              {row['tags']}")
        if row["content"]:
            print(f"content           {row['content'][:200]}")
        elif (arch := _first_archive_ref(row["source_refs"])):
            from .store import archive

            recovered = archive.recover_content(_hermes_home(), arch)
            print(f"content           (purged from live index; archived at {arch})")
            if recovered:
                print(f"archived          {recovered[:200]}")
        else:
            print("content           (tombstone)")

        for label, ref_col in (("supersedes", "supersedes_id"),
                               ("superseded by", "superseded_by")):
            link_id, hops = row[ref_col], 0
            while link_id is not None and hops < 10:
                link = conn.execute("SELECT uid, version, status, "
                                    "supersedes_id, superseded_by FROM memories WHERE id=?",
                                    (link_id,)).fetchone()
                if link is None:
                    break
                print(f"{label:<17} {link['uid'][:8]} v{link['version']} ({link['status']})")
                link_id = link[ref_col]
                hops += 1

        injected = conn.execute(
            "SELECT count(*) AS n FROM retrieval_log WHERE memory_id=? AND injected=1",
            (row["id"],),
        ).fetchone()["n"]
        print(f"recalled          {row['recall_count']} times"
              f" (last {row['last_recalled_at'] or 'never'}); injected {injected} times")

        facts = conn.execute(
            "SELECT subject, predicate, object, confidence, valid_from, valid_until "
            "FROM facts WHERE memory_id=? ORDER BY valid_from",
            (row["id"],),
        ).fetchall()
        if facts:
            print("facts:")
            for f in facts:
                window = f"{f['valid_from']} -> {f['valid_until'] or '(current)'}"
                print(f"  {f['subject']} --{f['predicate']}--> {f['object']}"
                      f"   conf={f['confidence']:.2f}  [{window}]")

        try:
            from .store import facts as facts_store

            chain = facts_store.reasoning_chain(conn, row["id"])
            if chain and len(chain) > 1:
                print("reasoning chain:")
                for node in chain:
                    uid = node.get("uid") or node.get("id") or "?"
                    snippet = str(node.get("content") or node.get("summary") or "")[:80]
                    print(f"  {str(uid)[:8]}  {snippet}")
        except Exception as e:  # provenance walk is best-effort diagnostic
            logger.debug("reasoning_chain unavailable: %s", e)

        audits = conn.execute(
            "SELECT actor, action, ts FROM audit_log WHERE target=? ORDER BY ts",
            (row["uid"],),
        ).fetchall()
        if audits:
            print("audit:")
            for a in audits:
                print(f"  {a['ts']}  {a['actor']:<8} {a['action']}")
        return 0
    except Exception as e:
        print(f"why failed: {e}\n"
              f"  Remedy: run 'hermes brain doctor' to check brain.db health.",
              file=sys.stderr)
        return 1
    finally:
        conn.close()


def cmd_fact(args: argparse.Namespace) -> int:
    """Current (or point-in-time) s-p-o facts for a subject."""
    conn = _open_db(_hermes_home())
    if conn is None:
        return 1
    try:
        from .store import facts as facts_store

        as_of = getattr(args, "as_of", None)
        rows = facts_store.query_facts(
            conn, subject=args.subject,
            predicate=getattr(args, "predicate", None), as_of=as_of)
        label = f"as of {as_of}" if as_of else "current truth"
        if not rows:
            print(f"no facts for subject {args.subject!r} ({label})")
            return 0
        print(f"facts for {args.subject!r} ({label}):")
        for f in rows:
            window = f"{f.valid_from} -> {f.valid_until or '(current)'}"
            print(f"  {f.subject} --{f.predicate}--> {f.object}"
                  f"   conf={f.confidence:.2f}  [{window}]")
        return 0
    except Exception as e:
        print(f"fact failed: {e}\n"
              f"  Remedy: run 'hermes brain doctor' to check brain.db health.",
              file=sys.stderr)
        return 1
    finally:
        conn.close()


def cmd_ask(args: argparse.Namespace) -> int:
    """Dialectic ask: a cited, natural-language answer over the brain."""
    import json as _json

    from . import config
    from .recall.ask import ask as ask_fn
    from .recall.embed import get_embedder
    from .recall.rerank import get_reranker
    from .store import sysinfo

    home = _hermes_home()
    question = " ".join(args.question)
    cfg = config.load_config(home)
    mode = sysinfo.resolve_mode(str(cfg.get("mode", "auto")))
    embedder = get_embedder(cfg, mode, allow_download=False)
    reranker = get_reranker(cfg, mode, allow_download=False)
    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        result = ask_fn(
            conn, question, level=args.level, trust_tier="owner",
            embedder=embedder, reranker=reranker, config=cfg,
            max_iterations=int(cfg.get("ask_max_iterations", 6)))
        if getattr(args, "json", False):
            print(_json.dumps({
                "answered": result.answered, "answer": result.answer,
                "citations": result.citations, "iterations": result.iterations,
                "level": result.level, "degraded": result.degraded,
            }, indent=2))
            return 0
        if result.degraded:
            print("(degraded: the LLM was unavailable — showing recall only)\n")
        if result.answered and result.answer:
            print(result.answer)
        else:
            print("I don't know — the brain has no clear evidence for that.")
        if result.citations:
            print("\nsources:")
            for c in result.citations:
                uid = str(c.get("uid") or "")[:8]
                snip = " ".join(str(c.get("snippet") or "").split())[:100]
                print(f"  [{uid}] {snip}")
        print(f"\n({result.level}, {result.iterations} steps)")
        return 0
    except Exception as e:
        print(f"ask failed: {e}", file=sys.stderr)
        return 1
    finally:
        conn.close()


def cmd_context(args: argparse.Namespace) -> int:
    """Assemble a token-budgeted context block from the brain (identity +
    peer card + distilled summary), as the compression path would."""
    from . import config
    from .recall.context import assemble
    from .recall.embed import get_embedder
    from .store import sysinfo

    home = _hermes_home()
    cfg = config.load_config(home)
    budget = args.tokens if getattr(args, "tokens", None) else int(
        cfg.get("precompress_tokens", 300))
    mode = sysinfo.resolve_mode(str(cfg.get("mode", "auto")))
    embedder = get_embedder(cfg, mode, allow_download=False)
    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        block = assemble(
            conn, [], budget, principal_id=None, trust_tier="owner",
            embedder=embedder, config=cfg,
            summary_ratio=float(cfg.get("context_summary_ratio", 0.4)))
        print(block if block else "(no context assembled)")
        return 0
    except Exception as e:
        print(f"context failed: {e}", file=sys.stderr)
        return 1
    finally:
        conn.close()


def cmd_sync(args: argparse.Namespace) -> int:
    """Multi-device encrypted delta sync: init | push | pull | status.

    Client-side crypto only (the relay stores opaque ciphertext). The shared
    passphrase comes from the HERMES_BRAIN_SYNC_PASSPHRASE env var; account +
    salt are replicated across a user's devices (printed by `init`)."""
    import base64
    import os

    from . import config
    from .store import db

    home = _hermes_home()
    cfg = config.load_config(home)
    sub = getattr(args, "sync_command", None) or "status"

    if sub == "init":
        try:
            from .sync import crypto as sync_crypto
        except Exception as e:
            print(f"sync unavailable: {e}", file=sys.stderr)
            return 1
        updates: dict[str, object] = {}
        if not cfg.get("sync_device_id"):
            updates["sync_device_id"] = db.new_ulid()
        if not cfg.get("sync_account"):
            updates["sync_account"] = db.new_ulid()
        if not cfg.get("sync_salt"):
            updates["sync_salt"] = base64.b64encode(sync_crypto.new_salt()).decode()
        if updates:
            config.save_config(home, updates)
            cfg = config.load_config(home)
        print("sync initialized:")
        print(f"  device_id  {cfg['sync_device_id']}")
        print(f"  account    {cfg['sync_account']}   (copy to every device)")
        print(f"  salt       {cfg['sync_salt']}   (copy to every device)")
        print("\nNext, on THIS device: set sync_url and sync_enabled: true in "
              "brain.yaml, and export HERMES_BRAIN_SYNC_PASSPHRASE.")
        print("On OTHER devices: set the SAME account, salt, and passphrase.")
        return 0

    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        if sub == "status":
            unsynced = conn.execute(
                "SELECT count(*) FROM memory_events WHERE synced_at IS NULL").fetchone()[0]
            total = conn.execute("SELECT count(*) FROM memory_events").fetchone()[0]
            print("sync status:")
            print(f"  enabled     {bool(cfg.get('sync_enabled'))}")
            print(f"  url         {cfg.get('sync_url') or '(unset)'}")
            print(f"  account     {cfg.get('sync_account') or '(unset)'}")
            print(f"  device_id   {cfg.get('sync_device_id') or '(unset)'}")
            print(f"  salt        {'set' if cfg.get('sync_salt') else '(unset)'}")
            print(f"  passphrase  {'set' if os.environ.get('HERMES_BRAIN_SYNC_PASSPHRASE') else '(unset in env)'}")
            print(f"  events      {unsynced} unsynced / {total} total")
            print(f"  outbox={db.get_meta(conn, 'sync_outbox_cursor') or '-'}  "
                  f"pull={db.get_meta(conn, 'sync_pull_cursor') or '-'}")
            return 0

        if sub in ("push", "pull"):
            if not cfg.get("sync_enabled"):
                print("sync is disabled (set sync_enabled: true in brain.yaml)",
                      file=sys.stderr)
                return 1
            try:
                from .sync import crypto as sync_crypto
                from .sync.engine import pull as sync_pull
                from .sync.engine import push as sync_push
                from .sync.relay import RelayClient
            except Exception as e:
                print(f"sync unavailable ({e}); install: pip install -e .[sync]",
                      file=sys.stderr)
                return 1
            if not sync_crypto.crypto_available():
                print("the [sync] extra (cryptography) is not installed: "
                      "pip install -e .[sync]", file=sys.stderr)
                return 1
            passphrase = os.environ.get("HERMES_BRAIN_SYNC_PASSPHRASE")
            fields = {"sync_url": cfg.get("sync_url"),
                      "sync_account": cfg.get("sync_account"),
                      "sync_salt": cfg.get("sync_salt"),
                      "sync_device_id": cfg.get("sync_device_id"),
                      "HERMES_BRAIN_SYNC_PASSPHRASE": passphrase}
            missing = [k for k, v in fields.items() if not v]
            if missing:
                print(f"sync not configured — missing: {', '.join(missing)}. "
                      f"Run 'hermes brain sync init'.", file=sys.stderr)
                return 1
            crypto = sync_crypto.SyncCrypto.from_passphrase(
                passphrase, base64.b64decode(cfg["sync_salt"]))
            client = RelayClient(cfg["sync_url"], namespace=cfg["sync_account"])
            origin = cfg["sync_device_id"]
            if sub == "push":
                s = sync_push(conn, crypto, client, origin=origin)
                print(f"pushed {s.get('pushed', 0)} "
                      f"(skipped {s.get('skipped_private', 0)} private); "
                      f"cursor {s.get('cursor')}")
            else:
                s = sync_pull(conn, crypto, client, origin=origin)
                print(f"pulled {s.get('pulled', 0)}, applied {s.get('applied', 0)} "
                      f"({s.get('conflicts', 0)} conflicts); cursor {s.get('cursor')}")
            return 0

        print(f"unknown sync command: {sub} (use init|push|pull|status)",
              file=sys.stderr)
        return 1
    except Exception as e:
        print(f"sync {sub} failed: {e}", file=sys.stderr)
        return 1
    finally:
        conn.close()


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the eval harness (retrieval P@k/MRR + answer/abstain rubric) on a
    fixture. Loads the harness by file path so it works under any package name.
    Use --real to hit the real aux LLM instead of the scripted fake."""
    import importlib.util
    from pathlib import Path

    harness_path = Path(__file__).parent / "tests" / "eval" / "harness.py"
    if not harness_path.exists():
        print(f"eval harness not found at {harness_path}", file=sys.stderr)
        return 1
    spec = importlib.util.spec_from_file_location("brain_eval_harness", harness_path)
    if spec is None or spec.loader is None:
        print("cannot load eval harness", file=sys.stderr)
        return 1
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"cannot load eval harness: {e}", file=sys.stderr)
        return 1
    fixture = args.fixture or str(harness_path.parent / "fixtures" / "eval_basic.json")
    try:
        m = mod.run_eval(fixture, real=bool(getattr(args, "real", False)))
    except Exception as e:
        print(f"eval failed: {e}\n"
              f"  Remedy: check the fixture path and (for --real) that an aux "
              f"LLM is configured.", file=sys.stderr)
        return 1
    print(f"fixture:          {fixture}")
    print(f"tier:             {m.get('tier')}   real_llm={m.get('real')}")
    print(f"P@{m.get('k', 5)}:             {m.get('p_at_5', 0.0):.3f}")
    print(f"MRR:              {m.get('mrr', 0.0):.3f}")
    print(f"answer pass rate: {m.get('answer_pass_rate', 0.0):.3f}")
    print(f"abstain correct:  {m.get('abstain_correct', 0.0):.3f}")
    print(f"llm calls:        {m.get('llm_calls', 0)}")
    return 0


# ---------------------------------------------------------------------------
# identity (critique item 33: the trust root)
# ---------------------------------------------------------------------------

def cmd_identity(args: argparse.Namespace) -> int:
    from .store import db

    sub = getattr(args, "identity_command", None)
    if sub not in ("add", "list", "rm"):
        print("Usage: hermes brain identity add <platform> <platform_user_id> "
              "[--owner] [--name N] | list | rm <platform> <platform_user_id>",
              file=sys.stderr)
        return 1
    conn = _open_db(_hermes_home())
    if conn is None:
        return 1
    try:
        if sub == "add":
            principal = "owner" if args.owner else db.new_ulid()
            existing = conn.execute(
                "SELECT principal_id FROM identities WHERE platform=? AND platform_user_id=?",
                (args.platform, args.platform_user_id),
            ).fetchone()
            if existing and not args.owner:
                principal = existing["principal_id"]  # keep a stable person id
            conn.execute(
                "INSERT INTO identities (principal_id, platform, platform_user_id,"
                " display_name, is_owner, added_at, added_by)"
                " VALUES (?,?,?,?,?,?,'cli')"
                " ON CONFLICT(platform, platform_user_id) DO UPDATE SET"
                " principal_id=excluded.principal_id, display_name=excluded.display_name,"
                " is_owner=excluded.is_owner, added_at=excluded.added_at, added_by='cli'",
                (principal, args.platform, args.platform_user_id, args.name,
                 1 if args.owner else 0, db.iso_now()),
            )
            _audit_cli(conn, "cli_identity_add", principal,
                       {"platform": args.platform, "user": args.platform_user_id,
                        "owner": bool(args.owner)})
            conn.commit()
            print(f"identity added: {args.platform}/{args.platform_user_id} -> "
                  f"{principal}{' (OWNER)' if args.owner else ''}")
            print("Takes effect at the next session (identity is resolved at initialize).")
        elif sub == "list":
            rows = conn.execute(
                "SELECT * FROM identities ORDER BY platform, platform_user_id"
            ).fetchall()
            if not rows:
                print("(no identities enrolled)")
                print("Hint: hermes brain identity add <platform> <user_id> --owner")
            for r in rows:
                owner = "OWNER" if r["is_owner"] else "known_user"
                print(f"{r['platform']:<10} {r['platform_user_id']:<24} {owner:<10} "
                      f"{r['display_name'] or '-':<16} principal={r['principal_id']}")
        else:  # rm
            cur = conn.execute(
                "DELETE FROM identities WHERE platform=? AND platform_user_id=?",
                (args.platform, args.platform_user_id),
            )
            if cur.rowcount == 0:
                print(f"No identity {args.platform}/{args.platform_user_id}.\n"
                      f"  Remedy: 'hermes brain identity list' shows enrolled identities.",
                      file=sys.stderr)
                return 1
            _audit_cli(conn, "cli_identity_rm", args.platform_user_id,
                       {"platform": args.platform})
            conn.commit()
            print(f"identity removed: {args.platform}/{args.platform_user_id}")
        return 0
    except Exception as e:
        print(f"identity {sub} failed: {e}\n"
              f"  Remedy: run 'hermes brain doctor' to check brain.db health.",
              file=sys.stderr)
        return 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# refresh-index / reindex / models
# ---------------------------------------------------------------------------

def cmd_refresh_index(args: argparse.Namespace) -> int:
    from . import config

    try:
        from .recall import lane1
    except ImportError:
        print("lane1 module missing — update the plugin: git -C <plugin dir> pull",
              file=sys.stderr)
        return 1
    home = _hermes_home()
    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        lane1.materialize(conn, config.load_config(home))
        rows = conn.execute(
            "SELECT section, count(*) AS n FROM lane1_snapshot "
            "GROUP BY section ORDER BY section"
        ).fetchall()
        if not rows:
            print("lane 1 snapshot is empty (no eligible memories yet)")
        for r in rows:
            print(f"{r['section']:<12} {r['n']} lines")
        return 0
    except Exception as e:
        print(f"refresh-index failed: {e}\n"
              f"  Remedy: run 'hermes brain doctor' to check brain.db health.",
              file=sys.stderr)
        return 1
    finally:
        conn.close()


def _rows_to_texts(conn, table: str, ids: list) -> tuple:
    """(texts, kept_ids) for an id batch — empty-content rows are skipped."""
    marks = ",".join("?" * len(ids))
    if table == "mem_vec":
        rows = conn.execute(
            f"SELECT id, coalesce(content, summary, '') AS t FROM memories "
            f"WHERE id IN ({marks})", ids).fetchall()
    else:
        rows = conn.execute(
            f"SELECT id, user_content || char(10) || assistant_content AS t "
            f"FROM episodes WHERE id IN ({marks})", ids).fetchall()
    by_id = {r["id"]: r["t"] for r in rows}
    texts, keep = [], []
    for row_id in ids:
        text = (by_id.get(row_id) or "").strip()
        if text:
            texts.append(text[:8000])
            keep.append(row_id)
    return texts, keep


def _embed_batch(conn, embedder, table: str, ids: list) -> int:
    """Embed+upsert one table's rows in batches of 32; returns rows embedded."""
    from .store import vec as vec_store

    total = 0
    for start in range(0, len(ids), 32):
        texts, keep = _rows_to_texts(conn, table, ids[start:start + 32])
        if not keep:
            continue
        vectors = embedder.encode_documents(texts)
        for row_id, vector in zip(keep, vectors, strict=False):
            vec_store.upsert(conn, table, row_id, vector)
        if table == "mem_vec":
            conn.executemany(
                "UPDATE memories SET embedded_with=? WHERE id=?",
                [(embedder.name, row_id) for row_id in keep],
            )
        conn.commit()
        total += len(keep)
    return total


def cmd_reindex(args: argparse.Namespace) -> int:
    from . import config
    from .recall.embed import get_embedder
    from .store import db, sysinfo
    from .store import vec as vec_store

    home = _hermes_home()
    cfg = config.load_config(home)
    mode = sysinfo.resolve_mode(str(cfg.get("mode", "auto")))
    embedder = get_embedder(cfg, mode, allow_download=False)
    if embedder is None:
        print(f"No embedder available for tier '{mode}'.\n"
              f"  Remedy: run 'hermes brain models --download' to fetch model files "
              f"(tier 'full'), or pip install model2vec (tier 'lite').", file=sys.stderr)
        return 1
    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        prev_name = db.get_meta(conn, "vec_embedder")
        prev_dim = db.get_meta(conn, "vec_dim")
        if not vec_store.ensure_tables(conn, embedder.dim, embedder.name,
                                       allow_rebuild=True):
            print("sqlite-vec is not loadable — no vector index.\n"
                  "  Remedy: pip install sqlite-vec (needs a Python built with "
                  "extension loading).", file=sys.stderr)
            return 1

        # Consume vector deletions deferred by 'forget --hard' runs where
        # sqlite-vec was not loadable (orphaned vectors must not linger).
        pending = json.loads(db.get_meta(conn, "vec_pending_delete") or "[]")
        if pending:
            for row_id in pending:
                vec_store.delete(conn, "mem_vec", row_id)
                vec_store.delete(conn, "epi_vec", row_id)
            conn.execute("DELETE FROM meta WHERE key='vec_pending_delete'")
            conn.commit()
            print(f"purged            {len(pending)} deferred vector deletion(s)")

        # An identity/dim mismatch above DROPPED both vec tables: everything
        # must be re-embedded, so a rebuild implies --all for both tables.
        rebuilt = (prev_name is not None and prev_name != embedder.name) or \
                  (prev_dim is not None and int(prev_dim) != embedder.dim)
        all_rows = args.all_rows or rebuilt
        if rebuilt:
            print(f"vec index rebuilt: {prev_name or '?'} ({prev_dim or '?'}d) -> "
                  f"{embedder.name} ({embedder.dim}d); re-embedding both tables in full")

        n_mem = _embed_batch(conn, embedder,
                             "mem_vec", vec_store.missing_ids(conn, "mem_vec", args.limit))
        n_epi = _embed_batch(conn, embedder,
                             "epi_vec", vec_store.missing_ids(conn, "epi_vec", args.limit))
        n_stale = 0
        if all_rows:
            stale_ids = [r["id"] for r in conn.execute(
                "SELECT id FROM memories WHERE valid_to IS NULL AND status='active' "
                "AND live=1 AND (embedded_with IS NULL OR embedded_with != ?) "
                "ORDER BY id DESC LIMIT ?", (embedder.name, args.limit)).fetchall()]
            n_stale = _embed_batch(conn, embedder, "mem_vec", stale_ids)
        line = f"embedded          {n_mem} memories, {n_epi} episodes ({embedder.name})"
        if all_rows:
            line += f", re-embedded {n_stale} stale"
        print(line)
        vstats = vec_store.stats(conn)
        if vstats:
            print(f"vectors           mem_vec={vstats.get('mem_vec')} "
                  f"epi_vec={vstats.get('epi_vec')}")
        return 0
    except Exception as e:
        print(f"Reindex failed: {e}\n"
              f"  Remedy: run 'hermes brain doctor' to check vec health.", file=sys.stderr)
        return 1
    finally:
        conn.close()


def cmd_models(args: argparse.Namespace) -> int:
    from . import config
    from .recall.embed import REGISTRY, ModelDownloadError, ensure_files, models_cache_dir
    from .recall.rerank import RERANK_REGISTRY

    cache = models_cache_dir()
    cfg = config.load_config(_hermes_home())
    configured = str(cfg.get("embed_model") or "modernbert-embed-base")
    rr_off = str(cfg.get("rerank", "auto")).strip().lower() in ("off", "false", "no", "0", "none")
    rr_key = str(cfg.get("rerank_model") or "mxbai-edge-colbert-v0-32m").strip().lower()
    if rr_key not in RERANK_REGISTRY:
        rr_key = "mxbai-edge-colbert-v0-32m"

    def _line(key: str, spec, extra: str) -> None:
        model_dir = cache / key
        have = all((model_dir / n).exists() and (model_dir / n).stat().st_size > 0
                   for n in spec.files)
        size_mb = sum((model_dir / n).stat().st_size for n in spec.files
                      if (model_dir / n).exists()) / (1024 * 1024)
        state = f"downloaded ({size_mb:.0f} MB)" if have else "not downloaded"
        print(f"{key:<28} {spec.repo:<48} {state}{extra}")

    for key, spec in REGISTRY.items():
        _line(key, spec, (" [configured]" if key == configured else "")
              + (" [license-gated: needs HF_TOKEN]" if spec.gated else ""))
    for key, spec in RERANK_REGISTRY.items():
        _line(key, spec, " [rerank]" + (" [configured]" if key == rr_key and not rr_off else ""))
    print(f"cache             {cache}")

    if not args.download:
        return 0
    rc = 0
    spec = REGISTRY.get(configured)
    if spec is None:
        print(f"Configured embed_model '{configured}' is unknown.\n"
              f"  Remedy: set embed_model to one of {sorted(REGISTRY)} in brain.yaml.",
              file=sys.stderr)
        rc = 1
    else:
        try:
            ensure_files(spec, download=True, progress=True)
            print(f"downloaded        {spec.key} -> {cache / spec.key}")
        except ModelDownloadError as e:
            print(str(e), file=sys.stderr)  # already teaches (incl. gated-repo case)
            rc = 1
    if not rr_off:
        try:
            rr_spec = RERANK_REGISTRY[rr_key]
            ensure_files(rr_spec, download=True, progress=True)
            print(f"downloaded        {rr_spec.key} -> {cache / rr_spec.key}")
        except ModelDownloadError as e:
            print(str(e), file=sys.stderr)
            rc = 1
    return rc


# ---------------------------------------------------------------------------
# export / import
# ---------------------------------------------------------------------------

def _topic_filename(tag: str) -> str:
    import re as _re

    return (_re.sub(r"[^A-Za-z0-9_-]+", "-", tag).strip("-") or "untagged") + ".md"


def cmd_export(args: argparse.Namespace) -> int:
    from .store import db

    home = _hermes_home()
    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        out_dir = Path(args.out) if args.out else \
            db.brain_dir(home) / "exports" / time.strftime("%Y-%m-%d")
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = conn.execute(
            "SELECT * FROM memories WHERE valid_to IS NULL AND status='active' "
            "ORDER BY id"
        ).fetchall()

        jsonl_path = out_dir / "memories.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(dict(r), ensure_ascii=False) + "\n")

        profile, warnings_, topics = [], [], {}
        for r in rows:
            line = f"- {(r['summary'] or r['content'] or '').strip()}  " \
                   f"[{r['uid'][:8]} · {r['kind'] or r['memory_type']}]"
            if r["kind"] in ("profile", "preference"):
                profile.append(line)
            if r["kind"] == "warning":
                warnings_.append(line)
            try:
                tags = json.loads(r["tags"] or "[]")
            except json.JSONDecodeError:
                tags = []
            if tags:
                topics.setdefault(str(tags[0]), []).append(line)

        (out_dir / "profile.md").write_text(
            "# Profile & preferences\n\n" + "\n".join(profile) + "\n", encoding="utf-8")
        (out_dir / "warnings.md").write_text(
            "# Warnings\n\n" + "\n".join(warnings_) + "\n", encoding="utf-8")
        if topics:
            topics_dir = out_dir / "topics"
            topics_dir.mkdir(exist_ok=True)
            for tag, lines in topics.items():
                (topics_dir / _topic_filename(tag)).write_text(
                    f"# {tag}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")

        print(f"exported {len(rows)} memories to {out_dir}")
        return 0
    except Exception as e:
        print(f"Export failed: {e}\n"
              f"  Remedy: check that the output directory is writable "
              f"(--out picks another location).", file=sys.stderr)
        return 1
    finally:
        conn.close()


# Link/id columns never survive an export/import round trip: row ids differ
# across databases, so version-chain pointers would dangle.
_IMPORT_SKIP_COLS = {"id", "supersedes_id", "superseded_by", "invalidated_by"}

# Steering-shaped content: text that reads as an instruction to the agent
# rather than a fact about the world. Imported rows matching this are
# quarantined (unless --trust-owner) so a crafted JSONL cannot plant
# behavior-steering lines into lane 1.
_INSTRUCTION_SHAPED_RE = re.compile(
    r"(?i)\b(?:ignore|disregard|forget|override)\b.{0,40}"
    r"\b(?:previous|prior|all|above|earlier)\b.{0,40}"
    r"\b(?:instructions?|rules?|prompts?|memor(?:y|ies))\b"
    r"|\byou\s+(?:must|should|will)\s+(?:always|never)\b"
    r"|\bsystem\s+prompt\b"
    r"|\bfrom\s+now\s+on\b",
)


def cmd_import(args: argparse.Namespace) -> int:
    from .store import db

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}\n"
              f"  Remedy: pass the memories.jsonl written by 'hermes brain export'.",
              file=sys.stderr)
        return 1
    conn = _open_db(_hermes_home())
    if conn is None:
        return 1
    try:
        columns = [r["name"] for r in conn.execute("PRAGMA table_info(memories)").fetchall()
                   if r["name"] not in _IMPORT_SKIP_COLS]
        imported = skipped = bad = 0
        now = db.iso_now()
        for _line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            content = (rec.get("content") or "").strip()
            if not content:
                skipped += 1
                continue
            # Always recompute the hash from content: a crafted file value
            # could defeat dedup or mislabel the row (import hardening).
            chash = db.content_hash(content)
            dup = conn.execute(
                "SELECT 1 FROM memories WHERE content_hash=? AND valid_to IS NULL",
                (chash,),
            ).fetchone()
            if dup:
                skipped += 1
                continue
            rec["content_hash"] = chash
            if not rec.get("uid") or conn.execute(
                    "SELECT 1 FROM memories WHERE uid=?", (rec["uid"],)).fetchone():
                rec["uid"] = db.new_ulid()
            rec.setdefault("memory_type", "semantic")
            rec["created_by"] = "migration"  # forced: imports are never the owner speaking
            rec.setdefault("valid_from", now)
            rec.setdefault("recorded_at", now)
            if not args.trust_owner:
                # Imported rows are data, not the owner: cap trust at 'agent',
                # never pinned, never dream-staged (live=1), and quarantine any
                # row that could steer behavior (warning/insight kinds,
                # instruction-shaped text) until the owner reviews it.
                if rec.get("trust_tier") not in ("agent", "known_user", "tool",
                                                 "untrusted"):
                    rec["trust_tier"] = "agent"
                rec["pinned"] = 0
                rec["live"] = 1
                if rec.get("kind") in ("warning", "insight") or \
                        _INSTRUCTION_SHAPED_RE.search(content):
                    rec["status"] = "quarantined"
            data = {k: rec[k] for k in columns if k in rec}
            keys = ",".join(data)
            conn.execute(
                f"INSERT INTO memories ({keys}) VALUES ({','.join('?' * len(data))})",
                list(data.values()),
            )
            imported += 1
        if imported:
            db.bump_generation(conn, "mem")
        _audit_cli(conn, "cli_import", str(path),
                   {"imported": imported, "skipped": skipped, "bad": bad})
        conn.commit()
        line = f"imported {imported}, skipped {skipped} (content-hash dedup)"
        if bad:
            line += f", {bad} unparseable lines"
        print(line)
        return 0
    except Exception as e:
        print(f"Import failed: {e}\n"
              f"  Remedy: the file must be memories.jsonl from 'hermes brain export' "
              f"(one JSON object per line).", file=sys.stderr)
        return 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# incognito
# ---------------------------------------------------------------------------

def cmd_incognito(args: argparse.Namespace) -> int:
    from . import config

    home = _hermes_home()
    try:
        if args.state in ("on", "off"):
            config.save_config(home, {"incognito": args.state == "on"})
            print(f"incognito {args.state} — takes effect at the NEXT session "
                  f"(the provider reads brain.yaml at initialize).")
        else:
            current = config.load_config(home).get("incognito")
            print(f"incognito is {'on' if current else 'off'}")
        return 0
    except Exception as e:
        print(f"incognito failed: {e}\n"
              f"  Remedy: check permissions on {config.config_path(home)}.",
              file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# dream cycle (P4) — user-invoked / cron-invoked only. The brain never
# auto-spawns dream processes; scheduling is the user's explicit choice
# (this CLI, or an OS/Hermes cron job wired at setup).
# ---------------------------------------------------------------------------

def _resolve_dream_embedder(cfg):
    from .recall.embed import get_embedder
    from .store import sysinfo

    mode = sysinfo.resolve_mode(str(cfg.get("mode", "auto")))
    return get_embedder(cfg, mode, allow_download=False)


def _last_dream_finished(conn):
    row = conn.execute(
        "SELECT finished_at FROM shift_runs WHERE finished_at IS NOT NULL "
        "ORDER BY finished_at DESC LIMIT 1").fetchone()
    return row["finished_at"] if row else None


def _dream_is_due(conn, cfg) -> bool:
    from .dream.lease import held_by
    from .store import db

    if held_by(conn, "dream"):
        return False
    last = _last_dream_finished(conn)
    if not last:
        return True
    hours = float(cfg.get("dream_min_interval_hours", _DREAM_MIN_INTERVAL_HOURS))
    # due once `last + hours` is in the past.
    return _iso_add_hours(last, hours) < db.iso_now()


def _iso_add_hours(iso: str, hours: float) -> str:
    import time as _time

    try:
        t = _time.mktime(_time.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return iso
    t += hours * 3600.0
    return _time.strftime("%Y-%m-%dT%H:%M:%S", _time.gmtime(t)) + ".000Z"


def cmd_dream_now(args: argparse.Namespace) -> int:
    from . import config
    from .dream import run_dream

    home = _hermes_home()
    cfg = {**config.load_config(home), "hermes_home": str(home)}
    embedder = _resolve_dream_embedder(cfg)
    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        summary = run_dream(conn, cfg, embedder=embedder, phase=args.phase,
                            dry_run=args.dry_run or None, actor="cli")
    finally:
        conn.close()

    if summary.get("skipped") == "lease_held":
        print(f"a dream is already running (lease held by {summary.get('holder')}); "
              f"skipped.")
        return 0
    if "error" in summary:
        print(f"dream error: {summary['error']}\n  {summary.get('recovery_hint','')}",
              file=sys.stderr)
        return 1
    print(f"dream {summary['shift_id']}:")
    print(f"  {'strategy':12} {'mode':8} result")
    for name, r in summary.get("strategies", {}).items():
        mode = r.get("mode", r.get("skipped", "-"))
        detail = {k: v for k, v in r.items() if k not in ("mode",)}
        print(f"  {name:12} {mode:8} {detail}")
    return 0


def cmd_dream(args: argparse.Namespace) -> int:
    """`dream --if-due` (cron/ops) and `dream --enable/--disable <strategy>`."""
    from . import config

    home = _hermes_home()

    if args.enable or args.disable:
        strategy = args.enable or args.disable
        phases = _dream_phases()
        if strategy not in phases:
            print(f"unknown strategy '{strategy}'. valid: {'|'.join(phases)}",
                  file=sys.stderr)
            return 1
        new_mode = "active" if args.enable else "off"
        conn = _open_db(home)
        if conn is None:
            return 1
        try:
            conn.execute(
                "INSERT INTO strategy_state (strategy, mode, stats) VALUES (?,?,'{}')"
                " ON CONFLICT(strategy) DO UPDATE SET mode=excluded.mode",
                (strategy, new_mode))
            conn.commit()
            print(f"strategy '{strategy}' -> {new_mode}")
            _print_strategy_modes(conn)
        finally:
            conn.close()
        return 0

    if not args.if_due:
        print("usage: hermes brain dream --if-due | --enable X | --disable X",
              file=sys.stderr)
        return 1

    cfg = {**config.load_config(home), "hermes_home": str(home)}
    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        if not _dream_is_due(conn, cfg):
            if not args.quiet:
                print("not due yet (or a dream is already running).")
            return 0
    finally:
        conn.close()

    from .dream import run_dream

    embedder = _resolve_dream_embedder(cfg)
    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        summary = run_dream(conn, cfg, embedder=embedder, actor="cron")
    finally:
        conn.close()
    if not args.quiet:
        print(f"dream {summary.get('shift_id', '-')}: "
              f"{list(summary.get('strategies', {}))}")
    return 0


def _print_strategy_modes(conn) -> None:
    from .dream.shift import DEFAULT_MODES, PIPELINE

    rows = {r["strategy"]: r["mode"] for r in
            conn.execute("SELECT strategy, mode FROM strategy_state").fetchall()}
    print("  strategy modes:")
    for name in PIPELINE:
        print(f"    {name:12} {rows.get(name) or DEFAULT_MODES.get(name, 'dry_run')}")


# ---------------------------------------------------------------------------
# P5: insights / review / skills / mcp / adopt-memory
# ---------------------------------------------------------------------------

def cmd_insights(args: argparse.Namespace) -> int:
    """Longitudinal learning metrics — the metric that matters (plan §5):
    verified-rate, tool_iterations, and cost trends from Hermes's own
    turn_outcomes, plus what the brain has learned so far."""
    from . import config
    from .dream import mine_state
    from .store import db

    home = _hermes_home()
    cfg = {**config.load_config(home), "hermes_home": str(home)}
    path = mine_state.state_db_path(cfg)
    if path is None:
        print("No state.db found — insights need Hermes's turn history.\n"
              f"  Looked in {db.brain_dir(home).parent / 'state.db'}.", file=sys.stderr)
        return 1

    import time as _time

    since = _time.time() - args.days * 86400
    try:
        state = mine_state.open_state_ro(path)
        try:
            episodes = mine_state.assemble_episodes(state, since_epoch=since, limit=4096)
        finally:
            state.close()
    except Exception as e:
        print(f"insights: could not read state.db ({e})\n"
              f"  Remedy: run 'hermes brain doctor'.", file=sys.stderr)
        return 1

    print(f"learning insights — last {args.days} days")
    if not episodes:
        print("  (no closed task episodes in this window yet)")
    else:
        total = len(episodes)
        succ = sum(1 for e in episodes if e.verdict == "success")
        fail = sum(1 for e in episodes if e.verdict == "failure")
        amb = total - succ - fail
        rate = succ / (succ + fail) if (succ + fail) else 0.0
        mean_iters = sum(e.tool_iterations for e in episodes) / total
        cost = sum(e.cost_usd for e in episodes)
        first_half = episodes[: total // 2] or episodes
        second_half = episodes[total // 2:] or episodes

        def _rate(eps):
            s = sum(1 for e in eps if e.verdict == "success")
            f = sum(1 for e in eps if e.verdict == "failure")
            return s / (s + f) if (s + f) else 0.0

        trend = _rate(second_half) - _rate(first_half)
        arrow = "up" if trend > 0.02 else ("down" if trend < -0.02 else "flat")
        print(f"  episodes          {total} ({succ} verified, {fail} failed, {amb} unclear)")
        print(f"  verified-rate     {rate:.0%}  (trend {arrow}, {trend:+.0%} over the window)")
        print(f"  tool iterations   {mean_iters:.1f} mean per episode")
        print(f"  cost              ${cost:.2f} total")

    conn = _open_db(home)
    if conn is None:
        return 0
    try:
        learned = conn.execute(
            "SELECT kind, count(*) AS n FROM memories WHERE memory_type IN "
            "('procedural','episodic') AND kind IN ('strategy','guardrail','case') "
            "AND valid_to IS NULL AND status='active' GROUP BY kind").fetchall()
        if learned:
            print("  learned so far    " + ", ".join(f"{r['n']} {r['kind']}" for r in learned))
        forged = conn.execute(
            "SELECT count(*) AS n FROM proposals WHERE kind='skill_draft' "
            "AND status='applied'").fetchone()["n"]
        drafts = conn.execute(
            "SELECT count(*) AS n FROM proposals WHERE kind='skill_draft' "
            "AND status IN ('pending','validated','shadow')").fetchone()["n"]
        print(f"  skills            {forged} forged, {drafts} in review")
        top = conn.execute(
            "SELECT summary, content, helpful_count, harmful_count FROM memories "
            "WHERE memory_type='procedural' AND valid_to IS NULL AND status='active' "
            "AND (helpful_count + harmful_count) > 0 "
            "ORDER BY helpful_count DESC LIMIT 5").fetchall()
        if top:
            print("  top strategies by proven help:")
            for r in top:
                label = (r["summary"] or r["content"] or "")[:60]
                print(f"    +{r['helpful_count']} -{r['harmful_count']}  {label}")
        return 0
    except Exception as e:
        print(f"insights: brain.db query failed ({e})\n"
              f"  Remedy: run 'hermes brain doctor'.", file=sys.stderr)
        return 1
    finally:
        conn.close()


def cmd_review(args: argparse.Namespace) -> int:
    """The unified review queue (critique item 12): open proposals + the
    quarantine. Approve/reject a proposal by uid."""
    conn = _open_db(_hermes_home())
    if conn is None:
        return 1
    try:
        if args.approve or args.reject:
            return _review_decide(conn, args)
        props = conn.execute(
            "SELECT uid, kind, title, status, created_at FROM proposals "
            "WHERE status IN ('pending','shadow','validated','approved') "
            "ORDER BY created_at DESC LIMIT 50").fetchall()
        quar = conn.execute(
            "SELECT uid, kind, substr(content,1,60) AS snippet, source_platform "
            "FROM memories WHERE status='quarantined' ORDER BY recorded_at DESC "
            "LIMIT 50").fetchall()
        if not props and not quar:
            print("review queue is empty — nothing awaiting a decision.")
            return 0
        if props:
            print(f"proposals ({len(props)}):")
            for p in props:
                print(f"  {p['uid'][:8]}  {p['kind']:16} {p['status']:10} {p['title'][:48]}")
            print("  decide: hermes brain review --approve <uid> | --reject <uid>")
        if quar:
            print(f"\nquarantined memories ({len(quar)}):")
            for q in quar:
                print(f"  {q['uid'][:8]}  {q['kind'] or '-':10} [{q['source_platform'] or '-'}] "
                      f"{(q['snippet'] or '').strip()}")
            print("  release: hermes brain review --approve <uid> (quarantined uid)")
        return 0
    except Exception as e:
        print(f"review failed: {e}\n"
              f"  Remedy: run 'hermes brain doctor' to check brain.db health.",
              file=sys.stderr)
        return 1
    finally:
        conn.close()


def _review_decide(conn, args) -> int:
    from . import config
    from .store import db

    home = _hermes_home()
    raw = args.approve or args.reject
    if not raw or not str(raw).strip():
        print("pass a uid: hermes brain review --approve <uid> | --reject <uid>\n"
              "  (uids come from 'hermes brain review')", file=sys.stderr)
        return 1
    uid = str(raw).strip()
    approving = bool(args.approve)

    # A proposal? (prefix match, current schema uses full uid in the queue)
    prop = conn.execute("SELECT * FROM proposals WHERE uid LIKE ? LIMIT 2",
                        (uid + "%",)).fetchall()
    if len(prop) > 1:
        print(f"ambiguous uid '{uid}' — pass more characters.", file=sys.stderr)
        return 1
    if prop:
        row = prop[0]
        if not approving:
            conn.execute("UPDATE proposals SET status='rejected', decided_at=?, "
                         "decided_by='cli' WHERE uid=?", (db.iso_now(), row["uid"]))
            conn.commit()
            print(f"rejected proposal {row['uid'][:8]} ({row['kind']})")
            return 0
        if row["kind"] == "skill_draft":
            from .skillforge import promote_draft

            cfg = {**config.load_config(home), "hermes_home": str(home)}
            res = promote_draft(conn, cfg, row["uid"], decided_by="cli")
            if res.get("promoted"):
                print(f"approved: skill '{res['name']}' promoted to {res['dir']}")
                return 0
            print(f"could not promote: {res.get('reason') or res.get('error')}",
                  file=sys.stderr)
            return 1
        if row["kind"] in ("skill_revision", "skill_retire"):
            from .skillforge import skilltree

            payload = json.loads(row["payload"] or "{}")
            name = row["target"]
            if row["kind"] == "skill_retire":
                skilltree.mark_stale(home, name)
                msg = f"retired skill '{name}' (marked stale; the curator archives it)"
            else:
                sections = (payload.get("revision") or {}).get("sections") or []
                if not payload.get("path") or not skilltree.apply_revision(
                        payload["path"], sections):
                    print("could not apply revision (skill SKILL.md missing?)",
                          file=sys.stderr)
                    return 1
                rec = skilltree.read_usage(home).get(name, {})
                # Reset the health window on revision (P2): the old hurt/helped
                # counters describe the PRE-revision skill; leaving them would
                # let the next dream run re-propose a revision from stale
                # failures. The host re-accumulates from real post-revision use.
                skilltree.write_usage_record(home, name, {
                    "patch_count": int(rec.get("patch_count") or 0) + 1,
                    "last_patched_at": db.iso_now(),
                    "helped": 0, "hurt": 0, "neutral": 0, "outcome_counts": {}})
                msg = f"revised skill '{name}' ({len(sections)} section(s) replaced)"
            conn.execute("UPDATE proposals SET status='applied', decided_at=?, "
                         "decided_by='cli' WHERE uid=?", (db.iso_now(), row["uid"]))
            conn.commit()
            print(f"approved: {msg}")
            return 0
        conn.execute("UPDATE proposals SET status='approved', decided_at=?, "
                     "decided_by='cli' WHERE uid=?", (db.iso_now(), row["uid"]))
        conn.commit()
        print(f"approved proposal {row['uid'][:8]} ({row['kind']})")
        return 0

    # Else a quarantined memory: release it (approve) or tombstone (reject).
    mem = _resolve_uid(conn, uid, current_only=True)
    if mem is None:
        return 1
    if mem["status"] != "quarantined":
        print(f"{mem['uid'][:8]} is not quarantined (status={mem['status']}).",
              file=sys.stderr)
        return 1
    from .store import db as _db

    if approving:
        conn.execute("UPDATE memories SET status='active', instruction_shaped=0 "
                     "WHERE id=?", (mem["id"],))
        _audit_cli(conn, "cli_quarantine_release", mem["uid"])
        print(f"released {mem['uid'][:8]} from quarantine (now active)")
    else:
        conn.execute("UPDATE memories SET status='tombstone', valid_to=? WHERE id=?",
                     (_db.iso_now(), mem["id"]))
        _audit_cli(conn, "cli_quarantine_reject", mem["uid"])
        print(f"rejected {mem['uid'][:8]} (tombstoned)")
    _db.bump_generation(conn, "mem")
    conn.commit()
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    sub = getattr(args, "skills_command", None) or "list"
    home = _hermes_home()
    if sub == "forge":
        return _skills_forge(home, no_approve=getattr(args, "no_approve", False))
    if sub in ("approve", "reject"):
        conn = _open_db(home)
        if conn is None:
            return 1
        try:
            ns = argparse.Namespace(approve=args.uid if sub == "approve" else None,
                                    reject=args.uid if sub == "reject" else None)
            return _review_decide(conn, ns)
        finally:
            conn.close()
    return _skills_list(home)


def _skills_list(home) -> int:
    from .skillforge import skilltree

    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        applied = conn.execute(
            "SELECT target, payload, decided_at FROM proposals "
            "WHERE kind='skill_draft' AND status='applied' ORDER BY decided_at DESC"
        ).fetchall()
        drafts = conn.execute(
            "SELECT uid, target, status, validation FROM proposals "
            "WHERE kind='skill_draft' AND status IN ('pending','validated','shadow') "
            "ORDER BY created_at DESC").fetchall()
        usage = skilltree.read_usage(home)
        if applied:
            print("forged skills (live):")
            for a in applied:
                rec = usage.get(a["target"], {})
                used = rec.get("use_count", 0)
                print(f"  {a['target']:32} used {used}x  since {a['decided_at'] or '-'}")
        if drafts:
            print("drafts awaiting review:")
            approvable = False
            for d in drafts:
                gates = ""
                try:
                    v = json.loads(d["validation"] or "{}")
                    gates = "gates PASS" if v.get("passed") else "gates incomplete"
                except json.JSONDecodeError:
                    pass
                ok = d["status"] in ("validated", "approved")
                approvable = approvable or ok
                tag = "" if ok else "  (re-forge — did not pass validation)"
                print(f"  {d['uid'][:8]}  {d['target']:28} {d['status']:10} {gates}{tag}")
            if approvable:
                print("  approve: hermes brain skills approve <uid>  (validated drafts only)")
        if not applied and not drafts:
            print("no forged skills yet — 'hermes brain skills forge' drafts one from "
                  "your task history.")
        return 0
    except Exception as e:
        print(f"skills list failed: {e}\n"
              f"  Remedy: run 'hermes brain doctor' to check brain.db health.",
              file=sys.stderr)
        return 1
    finally:
        conn.close()


def _skills_forge(home, *, no_approve: bool) -> int:
    from . import config
    from .skillforge import forge_once

    cfg = {**config.load_config(home), "hermes_home": str(home)}
    if no_approve:
        cfg["skill_auto_approve"] = False
    embedder = _resolve_dream_embedder(cfg)
    conn = _open_db(home)
    if conn is None:
        return 1
    try:
        result = forge_once(conn, cfg, embedder=embedder, shift_id="cli")
    finally:
        conn.close()
    if result.get("skipped"):
        print(f"no skill forged: {result['skipped']}")
        return 0
    if result.get("error"):
        print(f"forge error: {result['error']}", file=sys.stderr)
        return 1
    if not result.get("drafted"):
        print(f"no candidate found ({result.get('candidates', 0)} clusters).")
        return 0
    outcome = result.get("outcome")
    print(f"drafted '{result['drafted']}' — validation "
          f"{'PASSED' if result['validation']['passed'] else 'incomplete'}, "
          f"outcome: {outcome}")
    if outcome == "promoted":
        print(f"  promoted to the skills tree: {result['promotion']['dir']}")
    elif outcome in ("review_queue", "awaiting_approval"):
        print(f"  draft at {result['draft_dir']}")
        print(f"  approve with: hermes brain skills approve {result['proposal']}")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    """Run the stdio MCP server. Blocks until stdin closes."""
    from .mcp_server import serve

    return serve(str(_hermes_home()))


def cmd_adopt_memory(args: argparse.Namespace) -> int:
    """Apply the 'brain owns memory' matrix to Hermes config.yaml (§4.6):
    turn OFF the built-in memory/profile/nudges so the brain is authoritative.
    Dry-run by default; --apply writes."""
    target = {
        "memory.memory_enabled": False,
        "memory.user_profile_enabled": False,
        "memory.nudge_interval": 0,
        "memory.provider": "brain",
    }
    print("Brain-owns-memory matrix (skills loop + curator + session_search untouched):")
    for key, val in target.items():
        print(f"  {key:32} -> {val}")

    if not args.apply:
        print("\nDry-run. Re-run with --apply to write these to Hermes config.yaml.")
        return 0
    try:
        from hermes_cli.config import load_config as h_load  # type: ignore
        from hermes_cli.config import save_config as h_save  # type: ignore
    except ImportError:
        print("\nHermes config API not importable (running standalone).\n"
              "  Set these manually in ~/.hermes/config.yaml:", file=sys.stderr)
        for key, val in target.items():
            print(f"    {key}: {json.dumps(val)}", file=sys.stderr)
        return 1
    try:
        cfg = h_load()
        mem = cfg.setdefault("memory", {}) if isinstance(cfg, dict) else {}
        mem["memory_enabled"] = False
        mem["user_profile_enabled"] = False
        mem["nudge_interval"] = 0
        mem["provider"] = "brain"
        h_save(cfg)
        print("\nApplied. Start a new Hermes session for it to take effect.")
        return 0
    except Exception as e:
        print(f"\nCould not write config.yaml ({e}). Set the values manually.",
              file=sys.stderr)
        return 1
