# Adversarial gauntlet — try to break the brain, the proper way

This suite attacks Hermes-Brain's load-bearing invariants under hostile inputs,
missing capabilities, resource pressure, crashes, and concurrency. Every check
asserts an invariant **holds** — the brain degrades, quarantines, caps, refuses,
or recovers — and never destroys data. **A failing assertion means a real
invariant broke**, not that a test is flaky.

Two layers:

- **Adversarial pytest** (`tests/adversarial/`) — the pure-logic seams. Runs
  locally and inside the Docker tier images. Imports the shared fault toolkit
  `tests/adversarial/faults.py` (`from faults import ...`) and the conftest
  helpers (`seed_memory`, `seed_episode`, `poll_until`).
- **Docker phases** (`docker/adversarial/`) — what needs a process/environment
  boundary: the real floor/full tiers, a hard memory ceiling, multi-process
  lease exclusion, a genuine SIGKILL, and the full live hermes-agent lifecycle.

## Run it

```bash
# fast: the pytest gauntlet only (no Docker)
docker/adversarial/run-suite.sh --quick

# full gauntlet (pytest + every Docker phase); needs Docker + hermes-agent
HA=/path/to/hermes-agent docker/adversarial/run-suite.sh

# one phase, or skip the heavy live phase
docker/adversarial/run-suite.sh --phase p5
docker/adversarial/run-suite.sh --no-live
```

The orchestrator prints a per-phase PASS/FAIL table and exits non-zero if any
phase failed.

## What each phase breaks

| Phase | Layer | Attack | Invariant asserted |
|------|-------|--------|--------------------|
| pytest · never-raise | pytest | force every capture-path/dream seam to throw | swallow → sentinel (`[]`/`None`/`{"error"}`), turn/pipeline still completes; **archive failure preserves raw text** |
| pytest · trust/scope | pytest | non-owner recall on all legs; laundering; injection; MCP malformed frames | scope + quarantine hold; no existence leak; no path-traversal / FTS injection; MCP survives garbage |
| pytest · budget/caps | pytest | preload the ledger; oversized inputs; huge seed sets | daily + night budgets trip (incl. wedged provider); per-run caps stop at the cap; graph var-cap holds |
| pytest · anti-spam | pytest | adversarial/low-quality LLM output | spam/vague/coin-flip learning rejected; only well-formed, Wilson-supported items ratchet in |
| pytest · concurrency | pytest | thread races on the lease + buffer claim | exactly-once ownership; renew-authority; disjoint claims; cooperative preemption |
| pytest · recovery | pytest | future schema, interrupted create, migration, contradiction | refuse-the-future; self-heal; VACUUM backup; supersede-don't-delete |
| pytest · capability | pytest | no-fts5, cross-interpreter reconcile, embedder swap, mode typo | degrade to LIKE (still honoring exclude_kinds); reconcile both directions; vec disabled-not-destroyed |
| **p1** | Docker | build floor + full standalone images | the whole suite + eager-import + ruff + replay + golden pass at **each real tier** |
| **p4** | Docker | floor golden under `--memory=256m --cpus=1` | capture + recall + byte-stable lane-1 under a Termux-class ceiling |
| **p5** | Docker | 3 concurrent processes race the dream lease | exactly one winner (WAL-serialized mutual exclusion across processes) |
| **p6** | Docker | SIGKILL a dream after 2 phases, restart | `phases_done` cursor resumes — done phases skipped, only the rest re-run |
| **p8** | Docker | real hermes-agent + adversarial mock: flywheel, compression, session-reset, delegation, memory-write, MCP | learns + dedups; lifecycle hooks fire; MCP survives malformed frames and serves the owner's cross-platform recall |

## Harness pieces

- `mock_llm.py` — scenario-driven mock LLM (`valid` + `spam_items`,
  `malformed_json`, `huge`, `prompt_echo`, `vague_lesson`, `tool_call_*`,
  `budget_bomb`, `slow`, `empty`); select per-call via a `model@scenario` suffix,
  an `X-Mock-Scenario` header, or the `MOCK_SCENARIO` env var.
- `driver.py` — real turns, a flywheel, forced compression, and the Path-B hooks
  (session-switch / delegation / memory-write) a plain mock turn can't reach.
- `mcp_client.py` — stdlib JSON-RPC client + `--self-check` sweep for the MCP
  surface.
- `golden.py` — tier-agnostic capture→recall→lane-1 behavioral proof.
- `race_dream.py` / `crash_dream.py` — the multi-process lease race and the
  SIGKILL-resume harness.
- `stage.sh` — assembles the live build context (hermes-agent + brain + harness).
- `faults.py` (under `tests/adversarial/`) — the in-process fault-injection
  toolkit for the pytest layer.

## The live phase needs hermes-agent

`p8` (and the `stage.sh` it calls) needs a hermes-agent checkout — set
`HA=/path/to/hermes-agent`, or it is auto-located next to this repo / under
`$HOME`. The pytest layer and phases p1/p4/p5/p6 do **not** need hermes-agent.
