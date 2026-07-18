# Docker smoke & live-integration tests

Four images, in two families. Each `docker build` **is** the test — a failed
`RUN` fails the build.

## Standalone smoke (no hermes-agent) — build from the repo root

These validate the plugin in isolation (eager-import invariant, the full suite
in true standalone mode, ruff, the replay hook sequence).

```bash
docker build -f docker/Dockerfile      -t hermes-brain:floor .   # stdlib floor tier
docker build -f docker/Dockerfile.full -t hermes-brain:full  .   # onnx+vec+numpy tier
```

## Live integration (a REAL hermes-agent with the brain installed)

These install a real `hermes-agent`, drop the brain in as its memory provider,
and exercise it through Hermes's own plugin loader — catching bugs the
standalone tests can't (the loader registers the brain as `_hermes_user_memory.brain`,
not `brain`). They need a **staged build context** containing a clean
`hermes-agent` tree and a clean brain tree, because the Dockerfiles `COPY hermes/`
and `COPY brain/`.

Stage and build (bash; `HA` = your hermes-agent checkout):

```bash
HA=/path/to/hermes-agent
B=$(mktemp -d)/hermes-live; mkdir -p "$B/hermes" "$B/brain"
git -C "$HA" archive HEAD | tar -x -C "$B/hermes"          # clean tree, no .git
# brain: use the working tree so local changes are tested
tar -C . --exclude=.git --exclude=.pytest_cache --exclude='__pycache__' -cf - . \
  | tar -C "$B/brain" -xf -
cp docker/Dockerfile.hermes docker/hermes_provider_smoke.py "$B/"
cp docker/Dockerfile.hermes-mock docker/mock_llm.py docker/hermes_turn_driver.py "$B/"
cd "$B"

# Phase 1 — loads under real Hermes; `hermes brain` CLI + provider lifecycle.
docker build -f Dockerfile -t hermes-brain-live:phase1 .          # (Dockerfile == Dockerfile.hermes)

# Phase 2 — a mock OpenAI-compatible LLM drives a REAL agent turn + dream,
# fully offline (no API key), asserting the brain captures + learns.
docker build -f Dockerfile.mock -t hermes-brain-mock:live .       # (== Dockerfile.hermes-mock)
docker run --rm hermes-brain-mock:live                            # re-demonstrates capture+recall
```

### What Phase 2 proves
A real `run_agent.AIAgent` turn (main model streamed from the mock) → the brain
captures the episode → `hermes brain dream-now` extracts it through the mock →
`hermes brain search "staging database"` returns the fact at score 1.000, and a
second run **merges** instead of duplicating. `mock_llm.py` (stdlib only)
discriminates the agent turn (streaming) from the brain's aux extraction calls
(non-streaming JSON) and returns a valid extraction payload; `hermes_turn_driver.py`
drives the headless turn. See `hermes_turn_driver.py` / `mock_llm.py` for the
exact `$HERMES_HOME/config.yaml` needed to point Hermes at a custom endpoint
(inline `api_key`, `context_length >= 64000`).
