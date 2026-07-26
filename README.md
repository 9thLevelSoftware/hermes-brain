# Hermes-Brain 🧠

**Global memory & continual learning for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes-Brain is a `MemoryProvider` plugin that gives Hermes one persistent brain across
every session and platform (Telegram, Discord, Slack, CLI): it remembers everything
important, actively forgets what isn't worth keeping (without destroying data), and
genuinely improves over time — distilling strategies from successes *and* failures,
promoting repeated wins into skills, and consolidating memory during idle "dream" cycles.

Successor to [Daem0n-MCP](https://github.com/DasBluEyedDevil/Daem0n-MCP), rebuilt
agent-global on 2025–26 memory research. Design documents live in
[`docs/design/`](docs/design/), the research corpus in [`docs/research/`](docs/research/).

## Install

```bash
git clone https://github.com/DasBluEyedDevil/Hermes-Brain "$HERMES_HOME/plugins/brain"
hermes memory setup     # choose "brain"
```

The repo root **is** the plugin. The directory name `brain` is load-bearing (provider
name = config key = CLI verb). All state lives in `$HERMES_HOME/brain/` (one SQLite
file + append-only archive), so every profile gets its own brain for free via
`HERMES_HOME` — and you can [link them](#one-brain-many-profiles) when you want one
profile to draw on another.

## Tiers

| Tier | Hardware | Retrieval |
|---|---|---|
| full | ≥1.5–2 GB RAM | FTS5 + vectors (EmbeddingGemma-300M ONNX int8) + ColBERT rerank |
| lite | ~1 GB / Termux | FTS5 + static embeddings (potion-retrieval-32M) |
| fts-only | anything with SQLite | FTS5/BM25 (still captures everything; upgrades in place) |

## What it does

- **Remembers** every turn across every platform; recalls by hybrid keyword + vector
  search (fused with RRF), with cache-safe two-lane injection (a byte-stable
  system-prompt block + a per-turn ephemeral fence).
- **Learns** during idle/nightly *dream* cycles: consolidates repeated observations into
  cited lessons, distills reusable **strategy** and **guardrail** items from Hermes's own
  successes *and* failures (ReasoningBank), banks task **cases** (Memento), and mines the
  outcome ledger to reweight what actually helped.
- **Forges skills**: clusters of proven task patterns become agentskills.io `SKILL.md`
  drafts, validated (replay + statistical + capability-regression probes) and — per your
  setting — auto-approved into Hermes's skills tree, curator-safe.
- **Forgets** by tiered demotion, never destructive deletion; contradictions supersede
  (versions-are-rows), and instruction-shaped/untrusted content is quarantined out of the
  lanes.
- **Shares** across agents via a stdio **MCP server**: Claude Code can recall a memory the
  owner wrote from Telegram.

The learning strategies are **active by default** — modes `off | shadow | dry_run | active`;
the mutating strategies (`cases`/`distill`/`consolidate`/`contradict`/`forget`) learn live
on every dream run. `tune` stays `shadow` (it only ever *proposes* retrieval-weight changes,
never applies them). Roll any strategy back with `hermes brain dream --disable <strategy>`,
or neutralize a whole run with `--dry-run`.

**The brain never spawns a background process.** The dream runs when you schedule it:
`hermes memory setup` offers a `no_agent` cron job (gateway installs), `dream_schedule:
on-idle` lets the provider's existing worker run a due shift while a session sits idle
(CLI-only installs), and `hermes brain dream-now` always works. `hermes brain doctor`
warns when nothing has consolidated in a while.

## At a glance

| | |
|---|---|
| **Storage** | Local — one SQLite file under `$HERMES_HOME/brain/` |
| **Cost** | Free (optional auxiliary LLM spend for extraction/dreaming, capped by `day_budget_usd`) |
| **Tools** | 5 — `brain_recall`, `brain_remember`, `brain_outcome`, `brain_manage`, `memories`; plus opt-in `brain_ask` |
| **Key feature** | Sleep-time consolidation + skill forging — memory that improves rather than accumulates |

`recall_mode` (`hybrid` \| `context` \| `tools`) matches the convention used by Honcho
(`recallMode`) and Hindsight (`memory_mode`): `hybrid` injects *and* exposes tools,
`context` injects only, `tools` exposes tools only.

## One brain, many profiles

Profiles keep separate brains — that's the right default, and it's also why a new
`hermes profile create coder` starts out knowing nothing about you. Linking fixes that
without merging anything:

```bash
hermes brain link personal --home ~/.hermes-personal   # read-only, owner-only
hermes brain links                                     # what's linked, and reachable?
hermes brain unlink personal
```

Your coder profile can now recall what you told your personal one — the timezone you
work in, the box you rent, the deploy runbook you wrote three months ago — while each
profile keeps writing only to its own store. Results are labelled `@personal` so you
always know which brain answered, and linked memories are read-only from the other side:
`forget` on one tells you which profile owns it rather than reaching across.

Ranking is a proper merge, not a concatenation. Each profile is searched on its own and
the ranked lists are fused with RRF, so a profile with 50 memories and one with 5,000
are compared by *rank* rather than by scores that only mean something inside their own
corpus. Linked profiles are fused at a slight discount and hold a guaranteed share of the
result slots, so another profile informs your results without crowding out the one you're
working in.

Two things worth knowing before you link:

- **Links are owner-only.** A gateway peer or an MCP session at tool trust searches your
  local profile and nothing else, so a link widens what *you* see and never becomes a
  way in for anyone else.
- **Linking is genuine sharing, including the private parts.** A linked profile is read
  as its owner, so peer cards — the brain's private notes about specific people — cross
  the link too. That's the point when it's your own two profiles; it's worth a moment's
  thought if a profile is shared with anyone.

Per-turn injection stays local unless you opt in with `link_lane2: true`; on-demand
recall (`search`, `brain_recall`, `ask`) uses links immediately.

## What leaves your device

Nothing, by default. All memory lives in `$HERMES_HOME/brain/brain.db`; retrieval,
forgetting and consolidation are local. Linked profiles are local too — a link reads
another directory on the same machine, never a network. Two paths can send data
off-device, both opt-in:

- **Auxiliary LLM calls** — extraction and dreaming send memory/conversation excerpts to
  whatever model you configured under `auxiliary.brain_extract` / `auxiliary.brain_consolidate`.
  A local model keeps this on-device. `extract_mode: off` disables it entirely.
- **Multi-device sync** (`sync_enabled`, off by default, needs the `[sync]` extra) — pushes
  **ciphertext only** to a relay that never holds your key: content is encrypted with a key
  derived from your passphrase (Argon2id/PBKDF2 → Fernet + HMAC) before it leaves. The relay
  stores opaque blobs. A surface-only deny-list is the load-bearing invariant, re-checked at
  push time: a scoped, `peer_card`, quarantined or instruction-shaped row **never**
  serializes, so a synced memory is global by construction.

## CLI

```
hermes brain status | doctor | search <q> | why <id>
hermes brain remember/forget/pin/unpin/incognito ...
hermes brain dream-now [--phase X] [--dry-run]     # run a consolidation shift
hermes brain context | ask <q> | fact <subject>       # assembled context, cited answers, s-p-o facts
hermes brain why-not <query> <id>                    # why a memory did NOT surface
hermes brain eval --generate | --sample K | --compare # measure retrieval on your own corpus
hermes brain weights show | reset                    # active retrieval-leg weights
hermes brain import-provider <name> [--apply]        # migrate from another memory provider
hermes brain link <name> --home <path> | unlink | links   # read another profile's brain
hermes brain export --full | import <manifest.json>  # complete, re-importable snapshot
hermes brain dream --if-due                          # cron entry point
hermes brain dream --enable/--disable <strategy>     # promote a strategy
hermes brain insights                                # longitudinal learning metrics
hermes brain review [--approve/--reject <uid>]       # proposals + quarantine queue
hermes brain skills list|forge|approve|reject        # forged-skill lifecycle
hermes brain mcp                                      # stdio MCP server for external agents
hermes brain adopt-memory [--apply]                  # hand memory ownership to the brain
```

## Status

Phases P1–P5 complete: passive capture + FTS, hybrid retrieval + real lane 1 + bootstrap,
tool surface + sweep extraction, the dream cycle, and the learning flywheel + MCP surface.
See [`docs/design/`](docs/design/) for the normative design,
[`docs/design/critique.md`](docs/design/critique.md) for the resolved punch list, and
[`docs/design/alignment-audit.md`](docs/design/alignment-audit.md) for the audit against
the live `hermes-agent` provider contract.

## Is it actually working?

Every retrieval leg degrades independently and silently — a missing model, an
absent extension, an empty index — which is correct behavior and terrible
observability. Two commands answer it:

```bash
hermes brain doctor      # 'legs' line: fts=yes vec=no rerank=no graph=no facts=no
hermes brain eval --generate --limit 150   # paraphrase queries from YOUR memories
hermes brain eval --sample 10              # spot-check them before trusting a number
hermes brain eval --compare                # P@k, MRR, paired win/loss per leg config
```

The comparison reports a configuration whose leg is unavailable as **skipped**,
never as scoring zero improvement — an absent stage that quietly scores like the
baseline is indistinguishable from a stage that ran and did nothing.

**No claim is made here that the vector/rerank/graph stack beats plain BM25 on
your data.** It has not been measured on a real corpus with a real model. The
tooling above exists so you can find out rather than assume.

## Development

```bash
pip install -e .[dev]     # pytest + ruff
pip install -e .[full]    # onnx + tokenizers + sqlite-vec + numpy
pytest
```

`replay/run.py` drives the full provider hook sequence against recorded sessions — the
byte-stability and latency invariants are tested from Phase 1 and never leave CI.
