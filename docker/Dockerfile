# Floor-tier smoke test: a clean Python with NO hermes-agent and NO optional
# deps. Proves the plugin is eager-loader-safe (stdlib-only root modules),
# degrades to the FTS/stdlib floor, passes its suite in TRUE standalone mode
# (the dev machine can't — hermes-agent is importable there), and drives the
# real provider hook sequence. If `docker build` succeeds, it didn't explode.
FROM python:3.12-slim

WORKDIR /plugin

# The floor tier is pure stdlib; only the dev tools are needed to run the checks.
RUN pip install --no-cache-dir pytest ruff

COPY . /plugin

# 1. Eager-import every root module (+ observer) — invariant #1.
RUN python docker/smoke_imports.py

# 2. The whole suite in true standalone mode. sqlite-vec-gated tests self-skip.
RUN python -m pytest -p no:cacheprovider -q

# 3. Lint clean.
RUN python -m ruff check .

# 4. The real MemoryProvider hook sequence + CI invariants (lane-1 byte
#    stability, lane-2 budget, sync latency).
RUN python replay/run.py --fixture tests/fixtures/session_basic.json \
        --assert-lane1-stable --budget-check

CMD ["python", "-c", "print('hermes-brain floor-tier smoke: OK')"]
