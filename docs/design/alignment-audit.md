# Alignment audit — 2026-07-25

Audit of the shipped product against three authorities:

1. the **live `hermes-agent` source** (`C:\Users\dasbl\hermes-agent`, commit
   `1bf0f8d9f`) — the real `MemoryProvider` ABC, the loader, the setup wizard,
   the dashboard, and the eight bundled providers;
2. the published docs — [memory providers (user
   guide)](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers)
   and [memory provider plugins (developer
   guide)](https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin);
3. `docs/design/` — the normative design record.

Findings are numbered so code comments can cite them the way they cite
`critique.md`. Verdicts: **fixed** / **documented** / **wontfix**.

---

## A. Contract conformance — what was already correct

Recorded so it is not re-litigated. Each was checked against the real source,
not inferred:

- **A1.** All 23 members of `agent/memory_provider.py::MemoryProvider` are
  implemented, and `_compat.py`'s standalone fallback mirrors the surface
  exactly. No renames, no missing abstract methods.
- **A2.** The install path `$HERMES_HOME/plugins/brain/` is correct for a
  **user** plugin (`plugins/memory/__init__.py::_iter_provider_dirs`, step 2).
  The developer guide's `plugins/memory/<name>/` describes the **bundled**
  tree. `docker/Dockerfile.hermes` proves the user path end to end.
- **A3.** `register(ctx)`, `cli.py`'s `register_cli(subparser)` +
  `brain_command(args)`, and the directory-name-is-the-provider-name-is-the-CLI-verb
  rule are all satisfied.
- **A4.** Profile isolation: every path derives from the `hermes_home` kwarg
  the host injects (`memory_manager.initialize_all`). The one exception, the
  ONNX model cache, is deliberate and documented in `backup_paths()`.
- **A5.** `sync_turn()` is non-blocking (queue-only, asserted at <5ms mean over
  50 turns in `tests/test_provider.py`), satisfying the guide's one hard
  threading requirement.
- **A6.** `memory` is a reserved core tool name (F6); the file tool is named
  `memories`. Pinned by `tests/test_hermes_loader.py`.

---

## B. Host surfaces the brain never wired up

### B1. `get_status_config()` was not implemented — **fixed**

`hermes_cli/memory_setup.py:439` calls `provider.get_status_config(provider_config)`
when present (implemented by `openviking` and `supermemory`). The brain did not
have it, and because the brain stores nothing under `memory.brain` in
`config.yaml`, `hermes memory status` printed the provider name and *no
configuration at all*.

Added to `provider.py`: reports resolved tier, lane budgets, `recall_mode`,
schedule, counts, last dream and the DB path over a short-lived read
connection. Never raises — failures are returned as values.

### B2. The dashboard config round-trip was write-only — **fixed**

`hermes_cli/web_server.py:_read_memory_provider_existing_values` (line 4890)
prefills the provider form from `$HERMES_HOME/<name>.json` or
`$HERMES_HOME/<name>/config.json`. The brain wrote only
`$HERMES_HOME/brain/brain.yaml`, so the dashboard rendered **schema defaults**
regardless of actual settings — and saving that form wrote those defaults back
through `save_config`, **silently discarding the user's real configuration**.

This was a data-loss bug, not a cosmetic one. `config.save_config` now also
writes an atomic JSON mirror to `$HERMES_HOME/brain/config.json` — exactly the
second path the reader checks. `brain.yaml` remains the single source of truth;
the mirror is write-only and best-effort (a failed mirror never fails a save).

### B3. No `recall_mode` knob — **fixed**

Honcho exposes `recallMode` and Hindsight `memory_mode`, both
`hybrid | context | tools`. The brain had lane budgets but no way to say
"injection only" or "tools only", so a user migrating from either found nothing
where they expected it.

Added `recall_mode` (default `hybrid`, i.e. today's behavior). Read **once** at
`initialize()` so lane-1 byte-stability is untouched; an unknown value falls
back to `hybrid` with a warning rather than silently disabling memory.

### B4. `plugin.yaml` lacked `hooks:` — **fixed**

Every bundled provider declares `hooks:`; the brain declared only
`provides_hooks:`. Neither gates anything on the memory path (discovery reads
only `description` and `pip_dependencies`), so this is documentary — but the
documented format should be present. Both keys are now carried, and
`tests/test_audit_fixes.py` encodes the sharper rule: `observer/plugin.yaml` is
a *general* plugin where `provides_hooks` is genuinely parsed and a bare
`hooks:` is a trap, so it stays banned there.

### B5. The shared query rewriter was unused — **fixed (opt-in, off by default)**

`plugins/memory/query_rewrite.py` is explicitly provider-agnostic ("any memory
provider can pass `rewrite_memory_query` as its query rewriter") and honcho
wires it at `register()` time. The brain retrieved on raw user text.

Wired behind `query_rewrite` (default **off** — it is one auxiliary LLM call per
turn). It runs on the brain-bg worker, never the turn path, and any failure
falls back to the raw query.

**Note on invariant #5** (`llm.py` is the sole gateway for brain-initiated LLM
calls): the host helper calls `agent.auxiliary_client` directly, which would
route this spend around the daily budget. So it is **wrapped, not bypassed** —
`llm.call_query_rewrite` budget-gates before and meters after. That preserves
the invariant's purpose rather than carving an exception out of it.

---

## C. Design-doc drift

### C1. `dream_schedule` / `dream_time` were dead keys — **fixed**

The setup wizard asked two scheduling questions and **nothing in the codebase
read either answer**. Combined with the (correct) decision never to auto-spawn a
background process, this meant the learning system — the product's headline
feature — only ever ran when a human typed `hermes brain dream-now`.

Design §1.3 specified a `no_agent=True` cron job created at setup. Now:

- `brain_setup._offer_cron_job` creates it via `cron.jobs.create_job`, honouring
  `dream_time`, skipping for `manual`/`on-idle`, and degrading with a useful
  message when `cron` is absent (it is gateway-resident — CLI-only installs have
  no scheduler).
- `dream_schedule: on-idle` is a new opt-in path: `provider._maybe_idle_dream`
  runs a due shift on the **existing** brain-bg worker during its idle tick. No
  process is spawned; the no-background-processes decision stands.

The due-check moved to `dream/lease.py:is_due` so the cron entry point and the
idle path share one definition — a held lease reads as not-due, which is what
makes them mutually exclusive rather than racing.

### C2. The due-check misread its own timestamps — **fixed**

While consolidating C1: the old `cli._iso_add_hours` ran a UTC timestamp
(`db.iso_now`) through `time.mktime`, which reinterprets it as **local** time.
The dream interval was therefore skewed by the machine's UTC offset — firing
hours early or late depending on the timezone. Now uses `calendar.timegm`.
Pinned by `test_due_check_treats_stored_timestamps_as_utc`.

### C3. `doctor` had no config-drift check — **fixed**

Design §4.6 promised `hermes brain doctor` warns on drift; it checked only
brain-side state. Added:

- `host-provider` — **FAIL** when `memory.provider != "brain"`. The plugin is
  installed and reporting health while Hermes routes memory elsewhere, so
  nothing in either lane reaches the model.
- `host-builtins` — WARN only on **partial** adoption of the §4.6 matrix. Keeping
  the built-ins on during the transition is a supported configuration, not a
  defect; a half-applied matrix is the genuinely inconsistent state.
- `dream-freshness` — WARN when no dream has ever completed, or none in 3 days.
  This is what makes C1's failure mode visible instead of silent.

The matrix itself is now one constant (`cli._ADOPT_MATRIX`) shared by
`adopt-memory` and `doctor`.

### C4. The `memories` tool was specified but unbuilt — **fixed**

Design §3.1 tool #5, deferred under critique item 8, with its config key
removed. Now implemented in the `memfs/` subpackage (a subpackage, not a root
module: the loader eagerly imports every root `*.py`). Virtual `/memories/*.md`
views over the same memories table — `profile.md`, read-only `index.md`, and
`topics/<tag>.md` — with the six-command Anthropic grammar.

Every write funnels through `tools._remember`, so the trust cap, scoping,
instruction-shape quarantine, dedup, event seam and embedding behave identically
here and on `brain_remember`. Reads re-apply caller scope; `peer_card` and the
dream-owned internal kinds are unreachable; `delete` is a soft tombstone.
Gated by `memories_tool` at all three surfaces (agent schema, `tools.dispatch`,
MCP). Covered by `tests/test_memories_tool.py` and
`tests/adversarial/test_memfs_scope.py`.

Two behaviors worth recording because they were judgment calls, not
transcriptions of the design:

- **`profile.md` shows the caller's OWN standing facts.** `recall.search` lets
  an owner read every row including peer-scoped ones, which is right for search
  and wrong for a file called "profile": a peer's private preference is not one
  of the owner's standing facts. The owner's view is therefore restricted to
  unscoped rows.
- **Writing a known fact into a new topic file adds the tag.** `brain_remember`
  dedups on content hash and leaves tags alone, which would have made "put this
  fact in another topic" a silent no-op — the file the user just wrote would
  come back empty.

### C5. Remaining doc-vs-code drift — **documented**

Small and non-behavioral; recorded rather than "fixed" by changing working code:

- §5.3 documents `export [--format jsonl|md]`. The shipped `export` takes only
  `--out` and always writes **both** JSONL and a markdown tree. The shipped
  behavior is better (the formats are complementary, not alternatives).
- §1.3 describes CLI-only users getting a detached `hermes brain dream --if-due`
  spawn from `initialize()`/`shutdown()`. That was **deliberately reversed** — the
  brain never spawns processes — and is superseded by C1's `on-idle` path.
  `integration.md` now carries a dated note at §1.3.
- §5.2's "NOTE (audit 2026-07-24): `memories_tool` is NOT currently in
  `config.py:DEFAULTS`" is stale as of C4 and has been removed.

---

## D. Verified-and-correct claims

Checked during the sweep; no action needed. Listed so a later reader knows they
were actually confirmed rather than skipped:

- **D1.** `store/db.py:SCHEMA_VERSION == 3`, and `store/migrations/` holds
  `002`/`003` — matches CLAUDE.md's "schema.sql is law (currently v3)".
- **D2.** `dream/shift.py:PIPELINE` matches CLAUDE.md's documented order
  exactly, and `DEFAULT_MODES` matches the "active by default except `facts`
  and `tune` (shadow)" claim.
- **D3.** Lane 1's section structure (`recall/lane1.py:_SECTION_ORDER`) matches
  design §2.1: warnings → open loops → standing facts → stats.
- **D4.** `bootstrap --daemon <path>` (design §5.3) is implemented and wired
  (`bootstrap/daemon_import.py`).
- **D5.** `brain_context` is advertised on the MCP surface via
  `tools.context_schema()`, matching CLAUDE.md. (It is deliberately absent from
  the agent-facing schema — `on_pre_compress` covers that path in-process.)
- **D6.** All 47 `config.DEFAULTS` keys now have a live read outside
  `config.py`/`brain_setup.py`. Pinned going forward by
  `tests/test_config.py::test_every_default_key_is_actually_read_somewhere`, so
  the C1 class of bug cannot recur silently.

---

## E. Not done

- **E1.** The brain is a third-party provider, so it does not appear in the
  upstream docs' provider table. `README.md` now carries the equivalent row
  (Storage / Cost / Tools / Key feature) for anyone comparing.
- **E2.** No attempt was made to reconcile `docs/research/`. It is a source
  corpus, not a specification of this product.

---

# Pre-install hardening — 2026-07-26

The audit above was conducted against source. This section records what running
the plugin against **real data** found, which was a different and more useful
class of bug. Findings are numbered §F for citation from code.

## F1. Bootstrap imported ~9% of history — **fixed**

Measured on a real `state.db`: 49 sessions, 290 user messages, **26 turns
imported**.

| Stage | Turns |
|---|---|
| As shipped | 26 |
| Ignoring the `ended_at IS NULL` skip | 57 |
| Also ignoring the `active=0` skip | 95 |
| User messages present | 290 |

Three independent losses:

1. **22 of 49 sessions had `ended_at IS NULL` and were skipped permanently**,
   holding 210 of 290 user messages. Hermes stamps `ended_at` only on a clean
   close, so "still running" also meant "abandoned months ago". The original
   skip is correct in principle — watermarking a live session freezes a partial
   transcript forever — but there was no reaper. Added
   `_session_is_abandoned()`: an open session whose newest message is older than
   `bootstrap_stale_days` (default 7) is imported; a genuinely live one is still
   withheld.
2. **Consecutive user messages overwrote each other** in `_pair_turns`. Real
   transcripts are full of follow-ups sent before the assistant replies, and
   every one silently discarded its predecessor. They are now joined.
3. **Trailing user messages with no reply were discarded entirely** — 156 of
   290. Often the most useful rows in the file (the last thing asked for). They
   now emit a turn with an empty assistant side.

`active=0` (compacted) rows are also imported now, behind
`bootstrap_include_compacted`. Skipping them is right for live capture and
backwards for bootstrap: compacted history is exactly what an external memory
system exists to preserve.

**Result: 26 to 52 turns**, plus 47 more in one genuinely-live session that
import when it closes. Note this is NOT the ">200" first predicted — that
target confused user messages with pairable turns. 1006 of 1383 assistant
messages are blank tool-call rows, so ~95 pairs is the true ceiling, and the fix
now reaches all of it that is not still live.

`hermes brain bootstrap` prints coverage, so a thin import is visible instead of
looking identical to a complete one.

## F2. Retrieval legs degraded silently — **fixed**

The rerank models on the test machine were **empty directories**, so
`rerank: auto` resolved to `None` and the stage never ran, with nothing anywhere
saying so. Doctor check 8 only ever validated the *embedding* model.

Added `rerank-model`, `tier-deps` (warns when the tier resolves to `full` but
onnxruntime/tokenizers/numpy/sqlite-vec are missing, i.e. silently fts-only),
and a **`legs`** line in both `doctor` and `status` naming what retrieval will
actually do: `fts=yes vec=no rerank=no graph=no facts=no`. Each leg degrades
independently and correctly; the aggregate was invisible.

## F3. Retrieval had never been measured — **fixed (tooling), open (result)**

Six legs shipped with zero evidence any beat plain BM25. A first label-free
measurement (95 episodes, 71 same-session-proxy queries) gave FTS-only
recall@5 0.755 / MRR 0.918 versus 0.727 / 0.889 with vectors — **87% of queries
identical, 5 wins, 4 losses.** That is noise, and the benchmark favored FTS by
construction (same-session turns share identifiers verbatim).

Two methodology lessons are now enforced in `evalkit/`:

* **Paraphrase queries.** A query lifted verbatim from indexed text hands BM25
  an exact-token match and measures nothing. `generate._too_similar` rejects a
  "paraphrase" that reuses the source's long tokens.
* **Paired counts, not just means.** The "-3.7%" headline was two queries out of
  71. `compare.format_report` prints win/loss/tie and calls out small n.

A configuration whose leg is unavailable is reported **skipped with a reason**,
never scored — an absent reranker quietly scoring like the baseline is how a
stage that never executed got read as "no improvement". That includes an
embedder with an empty vector index.

**The result itself remains open**: nobody has yet run `--generate` against a
real model on a real corpus. Until then, no claim about the vector stack —
positive or negative — is supported.

## F4. Two subsystems could not work by construction — **fixed**

`dream/tune.py` fitted per-leg weights from the injection-to-outcome labels,
wrote them to a proposal, and let you approve it — while `fusion.rrf()` had **no
weight parameter** and `review --approve` set `status='approved'` and returned.
The brain learned what worked, asked permission, and discarded the answer.
`recall/intent.py` was worse: shadow-only, logging to `audit_log`, with no
proposal, no review surface, and no consumer anywhere.

Fixed by finishing them, not by deleting or papering over:

* `fusion.rrf(rankings, weights=...)` — the missing consumer. Uniform weights
  are asserted byte-identical to no weights.
* `recall/weights.py` — validated weights in a meta row, bumping the generation
  counter so `query_cache` cannot serve pre-weight results.
* `review --approve` applies them and prints before/after; a proposal with no
  fitted weights is **refused** rather than reporting a success that does
  nothing. `hermes brain weights show|reset`.
* `intent_weighting: active` applies per-query multipliers on top of the
  approved base weights.

Two translations in `weights.from_proposal` are load-bearing: `fit_weights`
names the graph leg `ppr`, and its weights are **convex** (~0.33 each) — applied
literally that is a uniform down-scale, and RRF ranking is invariant under a
uniform scale, so approving would provably have changed nothing.

**The invariant is intact**: `tune` remains `shadow`, is never auto-applied, and
only an explicit human approve moves retrieval.

## F5. Adoption gaps — **partially fixed**

* **`hermes brain import-provider`** (`bootstrap/providers.py`) with adapters
  for `holographic` (plain SQLite) and `jsonl` (the universal path). Everything
  else — byterover, mem0, openviking, honcho, retaindb, supermemory — is
  **deliberately not adapted**: their data needs a CLI binary, a running server,
  a configured vector store, or cloud credentials. Each refusal names the export
  route in. This is narrower than first planned; a half-working importer that
  silently drops rows is worse than a documented export step. Imports land at
  `agent` trust (never `owner`) with `source=import:<provider>`.
* **`hermes brain why-not <query> <uid>`** — `why` explains a memory you already
  found; this explains the one you did not. Reports structural exclusions first
  (status, scope, kind), then rank in the real search, then the lifecycle
  modulation that demoted it.
* **Cross-profile linking is NOT built.** Attaching a sibling profile's brain.db
  read-only at search time touches the scope-enforcement path, which is the most
  security-sensitive code in the repo, and doing it properly needs its own
  adversarial pass. Deferred rather than rushed.

## F6. `pip install -e .` was broken — **fixed**

No `[tool.setuptools]` section, so flat-layout auto-discovery found a dozen
top-level packages, could not choose, and aborted **before** reading
`optional-dependencies` — making the documented `.[dev]` and `.[full]` extras
unreachable. Declared explicitly that nothing installs: the host loads the
plugin by path, there are no entry points, and a second importable copy would
create the dual-identity problem `_compat.py` exists to avoid.

Side effect worth noting: with `[full]` installable, ~50 previously-skipped
tests now run, and three provider tests had to pin `mode: fts-only` — they were
implicitly relying on no embedder being present, and a real ONNX load blew their
poll deadlines.
