#!/usr/bin/env bash
# Overnight v16 accuracy sweep v3 — rescoped after v2's timeout miscalibration.
#
# Learned the hard way: this model spends ~20k reasoning tokens on hotel-lights
# (v1's 16-min "run" only fit 45 min because 22/30 requests never connected),
# so v2's timeouts killed healthy runs. v3 rescopes for real token economics:
#   - gsm8k: 300-item paired subset (deterministic item order; verify item_id
#     overlap at analysis), FIRST in each battery. 16k token cap. 3h timeout.
#   - hotel-lights: 16 runs, LAST in battery (droppable), 90 min timeout.
#   - estonia + lavd: cells 1-2 only. Cells 3-4: gsm8k only.
#   - rc=124 (timeout kill) is NOT retried — the run was healthy, just slow.
set -uo pipefail

SCRATCH="$(cd "$(dirname "$0")" && pwd)"
BENCH=/home/jugs/git/llm-inference-bench
STEPS="$SCRATCH/overnight-steps.log"
FULL="$SCRATCH/overnight-bench-full.log"
SUMMARIZE="$SCRATCH/summarize_accuracy.py"
HOST_URL=http://192.168.2.45:8000
HEALTH_URL="$HOST_URL/v1/models"
NAME=ds4-v16-tp2
COOLDOWN=300
RETRY_GAP=240
DSPARK_MODEL=DeepSeek-V4-Flash-DSpark
FLASH_MODEL=DeepSeek-V4-Flash

step() { local line="[STEP $(date '+%H:%M:%S')] $*"; echo "$line"; echo "$line" >>"$STEPS"; }

healthy() { curl -sf -m 8 "$HEALTH_URL" >/dev/null 2>&1; }

teardown() {
  step "teardown: stop worker (toby) then head (rusty), -t 60"
  ssh toby  "podman stop -t 60 $NAME >/dev/null 2>&1; podman rm $NAME >/dev/null 2>&1" || true
  ssh rusty "podman stop -t 60 $NAME >/dev/null 2>&1; podman rm $NAME >/dev/null 2>&1" || true
}

launch() { # mode backend
  local mode=$1 backend=$2
  step "launch: MODE=$mode BACKEND=$backend MAX_NUM_SEQS=8"
  if ! ssh toby "MODE=$mode BACKEND=$backend MAX_NUM_SEQS=8 ROLE=worker ~/v16/run-ds4-v16-tp2-node.sh" >>"$FULL" 2>&1; then
    step "FAIL: worker launch ($mode/$backend)"; return 1
  fi
  sleep 5
  if ! ssh rusty "MODE=$mode BACKEND=$backend MAX_NUM_SEQS=8 ROLE=head ~/v16/run-ds4-v16-tp2-node.sh" >>"$FULL" 2>&1; then
    step "FAIL: head launch ($mode/$backend)"; return 1
  fi
  local deadline=$((SECONDS + 1500))
  while (( SECONDS < deadline )); do
    if healthy; then
      step "healthy: $mode/$backend serving on rusty:8000"
      sleep 30
      return 0
    fi
    for node in rusty toby; do
      if [[ -z "$(ssh "$node" "podman ps -q --filter name=$NAME" 2>/dev/null)" ]]; then
        step "FAIL: $node container exited during boot ($mode/$backend)"
        ssh "$node" "podman logs --tail 25 $NAME 2>&1" >>"$FULL" 2>&1 || true
        return 1
      fi
    done
    sleep 20
  done
  step "FAIL: boot timeout 25min ($mode/$backend)"
  return 1
}

# bench LABEL MODEL PROFILE CC TIMEOUT_S [extra args...]
# rc: 0 ok | 1 gave up / too slow, server alive | 2 server dead (abandon cell)
bench() {
  local label=$1 model=$2 profile=$3 cc=$4 tmo=$5; shift 5
  local attempt rc f summary errs att
  for attempt in 1 2 3; do
    step "bench start: $label (profile=$profile cc=$cc attempt=$attempt timeout=${tmo}s)"
    ( cd "$BENCH" && HOST="$HOST_URL" MODEL="$model" LABEL="$label" timeout "$tmo" \
        ./run_bench.sh --test-profile "$profile" --profile-concurrency "$cc" \
        --display-mode plain "$@" ) >>"$FULL" 2>&1
    rc=$?
    f=$(ls -t "$BENCH/benchmark_results-$label"_*.json 2>/dev/null | head -1)
    if (( rc == 0 )) && [[ -n "$f" ]]; then
      summary=$(python3 "$SUMMARIZE" "$f")
      errs=$(sed -n 's/.*errors=\([0-9]*\)\/\([0-9]*\).*/\1 \2/p' <<<"$summary")
      if [[ -n "$errs" ]]; then
        read -r e a <<<"$errs"
        if (( a > 0 && 3 * e > a )); then
          step "RETRYABLE: $label attempt $attempt high error fraction — $summary"
          sleep "$RETRY_GAP"
          continue
        fi
      fi
      step "bench done: $label $summary"
      step "cooldown ${COOLDOWN}s"; sleep "$COOLDOWN"
      return 0
    fi
    if (( rc == 124 )); then
      step "FAIL: bench $label hit ${tmo}s timeout — healthy but too slow, NOT retrying"
      step "cooldown ${COOLDOWN}s"; sleep "$COOLDOWN"
      return 1
    fi
    step "FAIL: bench $label rc=$rc (attempt $attempt)"
    if ! healthy; then
      step "FAIL: server down after $label — abandoning cell"
      return 2
    fi
    sleep "$RETRY_GAP"
  done
  step "FAIL: bench $label gave up after 3 attempts (server alive) — continuing battery"
  step "cooldown ${COOLDOWN}s"; sleep "$COOLDOWN"
  return 1
}

GSM_ARGS=(--profile-runs 300 --max-tokens 16384)
HOTEL_ARGS=(--profile-runs 16)

step "overnight sweep v3 begins (host=$HOST_URL; gsm8k 300-item subset, hotel 16 runs)"

# ---------------------------------------------------------------- cell 1
if healthy; then
  step "cell 1: lucifer-cutlass dspark already serving — reusing"
else
  teardown
  if ! launch dspark lucifer-cutlass; then
    step "FAIL: cell 1 (lucifer) never came up — aborting sweep so this can be fixed"
    exit 1
  fi
fi

step "PHASE 0+1 begin (lucifer-cutlass dspark, seqs=8)"
bench v16-lucifer-gsm8k   "$DSPARK_MODEL" gsm8k 8 10800 "${GSM_ARGS[@]}"; rc=$?
if (( rc != 2 )); then bench v16-lucifer-estonia "$DSPARK_MODEL" estonia 4 7200; rc=$?; fi
if (( rc != 2 )); then bench v16-lucifer-lavd  "$DSPARK_MODEL" lavd-test 8 3600; rc=$?; fi
if (( rc != 2 )); then bench v16-lucifer-hotel "$DSPARK_MODEL" hotel-lights 8 5400 "${HOTEL_ARGS[@]}"; rc=$?; fi
step "cell 1/4 (lucifer dspark) finished"
teardown; sleep "$COOLDOWN"

# ---------------------------------------------------------------- cell 2
if launch mtp0 b12x-a16; then
  bench v16-mtp0-gsm8k   "$FLASH_MODEL" gsm8k 8 10800 "${GSM_ARGS[@]}"; rc=$?
  if (( rc != 2 )); then bench v16-mtp0-estonia "$FLASH_MODEL" estonia 4 7200; rc=$?; fi
  if (( rc != 2 )); then bench v16-mtp0-lavd    "$FLASH_MODEL" lavd-test 8 3600; rc=$?; fi
  if (( rc != 2 )); then bench v16-mtp0-hotel   "$FLASH_MODEL" hotel-lights 8 5400 "${HOTEL_ARGS[@]}"; rc=$?; fi
else
  step "FAIL: cell 2 (mtp0 baseline) skipped — launch failed"
fi
step "cell 2/4 (mtp0 baseline) finished"
teardown; sleep "$COOLDOWN"

# ---------------------------------------------------------------- cell 3 (gsm8k only)
if launch dspark b12x-a16; then
  bench v16-b12xds-gsm8k "$DSPARK_MODEL" gsm8k 8 10800 "${GSM_ARGS[@]}"
else
  step "FAIL: cell 3 (b12x dspark) skipped — launch failed"
fi
step "cell 3/4 (b12x dspark) finished"
teardown; sleep "$COOLDOWN"

# ---------------------------------------------------------------- cell 4 (gsm8k only)
if launch mtp2 b12x-a16; then
  bench v16-mtp2-gsm8k "$FLASH_MODEL" gsm8k 8 10800 "${GSM_ARGS[@]}"
else
  step "FAIL: cell 4 (mtp2) skipped — launch failed"
fi
step "cell 4/4 (mtp2) finished — leaving mtp2 cell serving on rusty:8000"

step "ALL DONE"
