# Live Hermes integration runbook (E)

The build + standalone-test work is done (370+ tests green). This is the
checklist to stand the plugin up inside a REAL Hermes process and validate the
paths the standalone suite can only mock (the live LLM, real lane injection,
the dream mutating live memory, MCP cross-platform recall). Run these in your
Hermes environment — they cannot be exercised headless here.

## 0. Install

```bash
# brain: the repo root IS the provider; the dir name `brain` is load-bearing.
git clone <this repo>  $HERMES_HOME/plugins/brain

# companion observer (B3) is a SEPARATE top-level plugin — it is NOT discovered
# nested under plugins/brain/, so install it on its own and enable it:
cp -r $HERMES_HOME/plugins/brain/observer  $HERMES_HOME/plugins/brain_observer
hermes plugins enable brain_observer

# optional heavy tiers (floor tier is stdlib-only and needs none of this):
pip install -e "$HERMES_HOME/plugins/brain[full]"     # ONNX embeddings + sqlite-vec
pip install -e "$HERMES_HOME/plugins/brain[rerank]"   # ColBERT reranker (A1)
hermes brain models --download                        # fetch embed + rerank models
```

Select the provider and run setup:
```bash
hermes memory setup            # walks brain_setup.post_setup (identity enrollment,
                               # aux-slot registration B1, model download, bootstrap)
# ensure config.yaml has:  memory: { provider: brain }
hermes brain bootstrap         # first-run import (MEMORY.md/USER.md + state.db backfill)
hermes brain doctor            # PASS/WARN/FAIL health checks
```

## 1. Capture + retrieval (turn path)

- Start a session; hold a short conversation with a durable fact ("my staging DB
  is postgres 14 on fly.io").
- `hermes brain status` → capture counts climb; `hermes brain search "staging db"`
  returns it. Confirm `legs:` shows `fts+vec` (and `+rerank` if the model is
  installed) — the A1/A2 engine is live.
- Ask a paraphrased question next turn ("what's my staging database?") and confirm
  lane-2 recall surfaces it — validates the reranker + write-time rewriting (D2).
- Confirm lane-1 is byte-stable within the session (no mid-session churn).

## 2. Real LLM extraction + the dream (learning active by default)

- After a session ends, `hermes brain status` should show the extraction sweep
  ran (real `auxiliary_client` call — this is the path the standalone suite
  mocks). Check `hermes brain insights`.
- `hermes brain dream-now` (manual) → watch the pipeline run. With active-by-default
  (C1), `cases/distill/consolidate/contradict/forget/forge/revise/peers` mutate
  live. Confirm:
  - new `insight`/`strategy`/`guardrail`/`case` rows appear (`hermes brain search`,
    `hermes brain review`);
  - `forget` archives raw text before any purge (A3) — `hermes brain why <id>` on a
    purged row recovers the archived content;
  - `probes` passes (post-shift regression guard);
  - a forged skill lands in `$HERMES_HOME/skills/<name>/SKILL.md` with
    `created_by: hermes-brain` (C2);
  - `hermes brain dream --disable forget` then re-run → forget skips (rollback works).
- `hermes brain review` → approve a `skill_revision`/`skill_retire` proposal and
  confirm the SKILL.md is patched / marked stale (C4 apply path, #18).

## 3. Cost + aux slots (B1/B2)

- After a dream, the `llm_ledger` should carry REAL `est_usd`/token rows (B2), not
  the flat proxy — spot-check via `hermes brain status` / the ledger.
- `hermes model → Configure auxiliary models` should list **Brain: extraction** and
  **Brain: consolidation** (B1 #19); pin a cheap/local model and confirm the next
  dream routes through it. (Runtime routing already works from the config block
  even if the picker entry is absent.)

## 4. Observer plugin (B3)

- With `brain_observer` enabled, exercise some tool calls + a subagent delegation.
- The brain-bg worker should drain `work_queue` (an `audit_log` `actor='observer'`
  `action='drain'` row per batch; an `activity` heartbeat `source='observer'`).
- Kill switch check: `BRAIN_OBSERVER_DISABLE=1` silences it without unregistering.

## 5. MCP cross-platform "money shot"

- Write a memory from one platform (e.g. Telegram gateway session).
- `hermes brain mcp` (stdio server); connect Claude Code / another agent as an MCP
  client and `brain_recall` — it should surface the owner's memory written elsewhere
  (tool-trust reads; writes capped + quarantined). This is the cross-platform payoff.

## 6. Group-chat peer modeling (D3)

- In a group chat (trust-gated; `capture_peers: true`), after a dream the `peers`
  strategy should write a `peer_card` (kind='peer_card') scoped to each observed
  non-owner principal. Confirm the owner's lane-2 in that chat surfaces the peer
  card, and that a non-owner caller can NEVER retrieve another peer's card.

## 7. Commit

Once §1-6 validate in your environment:
```bash
git checkout -b enhancement/engine-connectivity-frontier
git add -A && git commit     # see the prepared message
```
The work is currently uncommitted by prior explicit choice; the commit is the
finish line of E. Nothing here auto-commits — it is your call when §1-6 pass.

## Known host-side gap (B4)

The desktop/web app renders no config panel for `brain` because
`hermes_cli/memory_providers.py` hard-declares only `hindsight`. That is a
hermes-agent PR (add a `brain` entry, or make the host read
`get_config_schema()` dynamically), not a plugin change. CLI `hermes memory
setup` works today regardless.
