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
file + append-only archive); per-profile isolation comes free via `HERMES_HOME`.

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

Everything autonomous is **ship-inert** — modes `off | shadow | dry_run | active`, with
mutating strategies defaulting to `dry_run`/`shadow` until you promote them. The dream
runs on **cron + manual only**; the brain never spawns background processes on its own.

## CLI

```
hermes brain status | doctor | search <q> | why <id>
hermes brain remember/forget/pin/unpin/incognito ...
hermes brain dream-now [--phase X] [--dry-run]     # run a consolidation shift
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
See [`docs/design/`](docs/design/) for the normative design and
[`docs/design/critique.md`](docs/design/critique.md) for the resolved punch list.

## Development

```bash
pip install -e .[dev]
pytest
```

`replay/run.py` drives the full provider hook sequence against recorded sessions — the
byte-stability and latency invariants are tested from Phase 1 and never leave CI.
