#!/usr/bin/env bash
# Adversarial Docker gauntlet orchestrator for Hermes-Brain.
#
# Builds + runs every phase, prints a per-phase PASS/FAIL table, and exits
# non-zero if ANY invariant broke. A failing assertion means a load-bearing
# invariant did not hold — that's the point of the suite.
#
#   docker/adversarial/run-suite.sh            # full gauntlet (pytest + Docker)
#   docker/adversarial/run-suite.sh --quick    # pytest gauntlet only (no Docker)
#   docker/adversarial/run-suite.sh --phase p5 # one phase (p1..p8 | pytest)
#   docker/adversarial/run-suite.sh --no-live  # skip the heavy live phase (p8)
#   docker/adversarial/run-suite.sh --keep     # keep built images/containers
#
# Env: HA=/path/to/hermes-agent  (only the live phase p8 needs it)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

MODE="full"; ONLY=""; KEEP=0; NO_LIVE=0
for a in "$@"; do
  case "$a" in
    --quick) MODE="quick" ;;
    --full)  MODE="full" ;;
    --no-live) NO_LIVE=1 ;;
    --keep)  KEEP=1 ;;
    --phase) MODE="one" ;;
    p1|p2|p3|p4|p5|p6|p7|p8|pytest) ONLY="$a" ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

FLOOR_IMG="hermes-brain:floor"
FULL_IMG="hermes-brain:full"
declare -a NAMES=() STATUS=()

hr(){ printf '%s\n' "----------------------------------------------------------------"; }
record(){ NAMES+=("$1"); STATUS+=("$2"); }

# run_phase <key> <title> <command...>  — runs, records PASS/FAIL, never aborts.
run_phase(){
  local key="$1" title="$2"; shift 2
  if [ -n "$ONLY" ] && [ "$ONLY" != "$key" ]; then return 0; fi
  hr; echo ">>> [$key] $title"; hr
  local t0 t1; t0=$(date +%s)
  if "$@"; then
    t1=$(date +%s); echo "<<< [$key] PASS (${title})  $((t1-t0))s"; record "$key $title" PASS
  else
    t1=$(date +%s); echo "<<< [$key] FAIL (${title})  $((t1-t0))s"; record "$key $title" FAIL
  fi
}

have_docker(){ command -v docker >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# phase implementations
# ---------------------------------------------------------------------------

# pytest gauntlet — the pure-logic seams, run locally (also runs inside p1 images)
phase_pytest(){
  python -m pytest -p no:cacheprovider tests/adversarial -q
}

# p1 — tier/degradation matrix: the floor AND full standalone images each run
# the WHOLE suite (incl. tests/adversarial) + eager-import + ruff + replay.
phase_p1(){
  have_docker || { echo "docker not available"; return 1; }
  docker build -f docker/Dockerfile      -t "$FLOOR_IMG" . \
    && docker build -f docker/Dockerfile.full -t "$FULL_IMG" . \
    && docker run --rm "$FLOOR_IMG" python docker/adversarial/golden.py \
    && docker run --rm "$FULL_IMG"  python docker/adversarial/golden.py
}

# p4 — resource ceiling: the floor golden under a hard 256MB / 1 CPU cap
# (the Termux-class floor). Degrade-not-crash under real memory pressure.
phase_p4(){
  have_docker || { echo "docker not available"; return 1; }
  docker image inspect "$FLOOR_IMG" >/dev/null 2>&1 || docker build -f docker/Dockerfile -t "$FLOOR_IMG" .
  docker run --rm --memory=256m --memory-swap=256m --cpus=1 "$FLOOR_IMG" \
    python docker/adversarial/golden.py
}

# p5 — multi-PROCESS lease exclusion: 3 contenders, exactly one WON.
phase_p5(){
  have_docker || { echo "docker not available"; return 1; }
  docker image inspect "$FLOOR_IMG" >/dev/null 2>&1 || docker build -f docker/Dockerfile -t "$FLOOR_IMG" .
  docker run --rm "$FLOOR_IMG" bash -lc '
    set -e
    export GOLDEN_HOME=/tmp/racehome PYTHONIOENCODING=utf-8
    python -c "import importlib.util,os,sys; \
      spec=importlib.util.spec_from_file_location(\"brain\",\"/plugin/__init__.py\",submodule_search_locations=[\"/plugin\"]); \
      m=importlib.util.module_from_spec(spec); sys.modules[\"brain\"]=m; spec.loader.exec_module(m); \
      from brain.store import db; db.connect(os.environ[\"GOLDEN_HOME\"]).close()"
    START=$(( $(date +%s) + 2 ))
    OUT=$(mktemp)
    for i in 1 2 3; do START_EPOCH=$START HOLD_SECONDS=3 python docker/adversarial/race_dream.py >>"$OUT" 2>&1 & done
    wait
    cat "$OUT"
    won=$(grep -c "^WON" "$OUT" || true)
    echo "winners=$won (expect exactly 1)"
    [ "$won" -eq 1 ]'
}

# p6 — SIGKILL-mid-dream idempotent resume: crash after 2 phases, resume skips them.
phase_p6(){
  have_docker || { echo "docker not available"; return 1; }
  docker image inspect "$FLOOR_IMG" >/dev/null 2>&1 || docker build -f docker/Dockerfile -t "$FLOOR_IMG" .
  docker run --rm "$FLOOR_IMG" bash -lc '
    export GOLDEN_HOME=/tmp/crashhome PYTHONIOENCODING=utf-8
    python docker/adversarial/crash_dream.py --run --shift s1 --crash-after 2 || true
    python docker/adversarial/crash_dream.py --check --shift s1'
}

# p8 — full live flywheel + lifecycle: real hermes-agent + the adversarial mock.
phase_p8(){
  [ "$NO_LIVE" -eq 1 ] && { echo "skipped (--no-live)"; return 0; }
  have_docker || { echo "docker not available"; return 1; }
  local CTX; CTX="$("$HERE/stage.sh")" || { echo "staging failed (set HA=/path/to/hermes-agent)"; return 1; }
  echo "staged context: $CTX"
  ( cd "$CTX" && docker build -f Dockerfile.live -t hermes-brain-adv:live . )
}

# ---------------------------------------------------------------------------
# drive
# ---------------------------------------------------------------------------
echo "hermes-brain adversarial gauntlet — mode=$MODE ${ONLY:+phase=$ONLY}"

run_phase pytest "pytest gauntlet (never-raise / trust / budget / anti-spam / concurrency / recovery / capability)" phase_pytest
if [ "$MODE" != "quick" ]; then
  run_phase p1 "tier degradation matrix (floor + full images, full suite + golden)" phase_p1
  run_phase p4 "resource ceiling (floor golden under 256MB / 1 CPU)" phase_p4
  run_phase p5 "multi-process lease exclusion (exactly one winner)" phase_p5
  run_phase p6 "SIGKILL-mid-dream idempotent resume" phase_p6
  run_phase p8 "live flywheel + lifecycle (real hermes + adversarial mock)" phase_p8
fi

[ "$KEEP" -eq 0 ] && have_docker && docker image rm "$FLOOR_IMG" "$FULL_IMG" >/dev/null 2>&1 || true

hr; echo "SUMMARY"; hr
fails=0
for i in "${!NAMES[@]}"; do
  printf '  %-4s  %s\n' "${STATUS[$i]}" "${NAMES[$i]}"
  [ "${STATUS[$i]}" = "FAIL" ] && fails=$((fails+1))
done
hr
if [ "$fails" -ne 0 ]; then
  echo "GAUNTLET: $fails phase(s) FAILED — an invariant did not hold."; exit 1
fi
echo "GAUNTLET: all phases PASSED."
