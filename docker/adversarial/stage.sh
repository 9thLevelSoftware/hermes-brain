#!/usr/bin/env bash
# Stage a Docker build context for the live adversarial phases: a clean
# hermes-agent tree + this brain working tree + the adversarial harness.
#
# Usage:
#   docker/adversarial/stage.sh [OUTPUT_DIR]
# Env:
#   HA   path to a hermes-agent checkout (default: ../hermes-agent, ~/hermes-agent)
#
# Emits the staged context path on stdout (the last line) so run-suite.sh can
# capture it. The context layout the live Dockerfiles expect:
#   <ctx>/hermes/   clean hermes-agent tree (git archive HEAD — no .git)
#   <ctx>/brain/    this repo, minus .git/.pytest_cache/__pycache__/.ruff_cache
#   <ctx>/*.py      the adversarial harness (mock_llm, driver, mcp_client, ...)
#   <ctx>/*.sh      per-phase run scripts
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRAIN_ROOT="$(cd "$HERE/../.." && pwd)"

# Locate hermes-agent.
HA="${HA:-}"
if [ -z "$HA" ]; then
  for cand in "$BRAIN_ROOT/../hermes-agent" "$HOME/hermes-agent" \
              "$HOME/Documents/hermes-agent" "/c/Users/$USER/hermes-agent"; do
    if [ -d "$cand/.git" ] || [ -f "$cand/pyproject.toml" ]; then HA="$cand"; break; fi
  done
fi
if [ -z "$HA" ] || [ ! -d "$HA" ]; then
  echo "ERROR: hermes-agent checkout not found. Set HA=/path/to/hermes-agent." >&2
  echo "  (the live phases need it; the pytest phases do not — run those with --quick)" >&2
  exit 3
fi

OUT="${1:-}"
if [ -z "$OUT" ]; then
  OUT="$(mktemp -d)/hermes-adv"
fi
mkdir -p "$OUT/hermes" "$OUT/brain"

echo "staging hermes-agent  <- $HA" >&2
if [ -d "$HA/.git" ]; then
  git -C "$HA" archive HEAD | tar -x -C "$OUT/hermes"
else
  tar -C "$HA" --exclude=.git --exclude='__pycache__' --exclude='.pytest_cache' \
      -cf - . | tar -C "$OUT/hermes" -xf -
fi

echo "staging brain working tree  <- $BRAIN_ROOT" >&2
tar -C "$BRAIN_ROOT" \
    --exclude=.git --exclude=.pytest_cache --exclude='__pycache__' \
    --exclude=.ruff_cache --exclude='*.egg-info' \
    -cf - . | tar -C "$OUT/brain" -xf -

echo "staging adversarial harness" >&2
# Flatten the harness scripts + phase Dockerfiles to the context root so the
# build can find them by name (docker build -f Dockerfile.live from $CTX).
cp "$HERE"/*.py "$OUT"/ 2>/dev/null || true
cp "$HERE"/*.sh "$OUT"/ 2>/dev/null || true
cp "$HERE"/Dockerfile* "$OUT"/ 2>/dev/null || true
# Also carry the base docker mock (some phases reuse the valid-only one).
cp "$BRAIN_ROOT/docker/mock_llm.py" "$OUT/base_mock_llm.py" 2>/dev/null || true

echo "staged context ready:" >&2
echo "$OUT"
