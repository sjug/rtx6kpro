#!/usr/bin/env bash
set -euo pipefail

# Compare r18p's B12X dense FP8 path with the upstream DGLIN launcher mode.
# The six independent engine starts use balanced adjacent pairs: A/B, B/A,
# A/B.  Each adjacent pair shares a deterministic prompt seed.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_ROOT=${BENCH_ROOT:-/home/jugs/git/llm-inference-bench}
BENCH=${BENCH:-${BENCH_ROOT}/llm_decode_bench.py}
PYTHON=${PYTHON:-${BENCH_ROOT}/.venv/bin/python}
HEAD_HOST=${HEAD_HOST:-dusty}
WORKER_HOST=${WORKER_HOST:-kirby}
RUNNER=${RUNNER:-/home/jugs/run-ds4-ii-r10-tp2-node.sh}
RUNNER_SHA256=${RUNNER_SHA256:-ab3abfaaf4ab98b537cdbaa149207b870e88bcbf8d264e3daabb19db3880f3d1}
NAME=${NAME:-ds4-0731-tp2}
PORT=${PORT:-8000}
MODEL=${MODEL:-DeepSeek-V4-Flash-0731}
IMAGE=${IMAGE:-localhost/voipmonitor/vllm:infernal-invocation-r18p-spark-sm121-vllmf560085-b12x07cdf45-fi1ac6942-cu133-torch213-20260820}
IMAGE_ID=${IMAGE_ID:-445f9ac3196dd10a47d0d90441ba43a8417903ba029f59a1a9c7fbef8ecfa4a1}
START_PAIR=${START_PAIR:-1}
SSH=(ssh -F /dev/null)

mkdir -p "${ROOT}/results" "${ROOT}/receipts"
exec > >(tee -a "${ROOT}/dglin-ab.log") 2>&1

timestamp() { date -Is; }

backend_for() {
  case "$1" in
    a) printf '%s\n' b12x-a8 ;;
    b) printf '%s\n' b12x-a8-dglin ;;
    *) printf 'unknown arm: %s\n' "$1" >&2; return 2 ;;
  esac
}

assert_runner() {
  local host=$1 actual
  actual=$("${SSH[@]}" "${host}" "sha256sum ${RUNNER}" | cut -d' ' -f1)
  if [[ "${actual}" != "${RUNNER_SHA256}" ]]; then
    printf '%s runner digest %s, expected %s\n' \
      "${host}" "${actual}" "${RUNNER_SHA256}" >&2
    return 1
  fi
}

stop_pair() {
  printf '[%s] stopping worker then head\n' "$(timestamp)"
  "${SSH[@]}" "${WORKER_HOST}" \
    "if podman container exists ${NAME}; then timeout 120 podman stop -t 90 ${NAME} >/dev/null && podman rm ${NAME} >/dev/null; fi"
  "${SSH[@]}" "${HEAD_HOST}" \
    "if podman container exists ${NAME}; then timeout 120 podman stop -t 90 ${NAME} >/dev/null && podman rm ${NAME} >/dev/null; fi"
}

launch_node() {
  local host=$1 role=$2 backend=$3
  "${SSH[@]}" "${host}" \
    "IMAGE=${IMAGE} ROLE=${role} BACKEND=${backend} DSPARK_TOKENS=5 DRAFT_SAMPLE_METHOD=probabilistic NCCL_MAX_NCHANNELS=4 NCCL_MIN_NCHANNELS=4 SERVED_MODEL_NAME=${MODEL} ${RUNNER}" \
    >/dev/null
}

wait_ready() {
  local deadline=$((SECONDS + 1200))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 5 "http://${HEAD_HOST}:${PORT}/health" >/dev/null; then
      curl -fsS --max-time 5 "http://${HEAD_HOST}:${PORT}/v1/models" \
        | jq -e --arg expected "${MODEL}" \
          '(.data | type == "array") and ([.data[].id] == [$expected])' >/dev/null
      printf '[%s] endpoint ready\n' "$(timestamp)"
      return 0
    fi
    if ! "${SSH[@]}" "${HEAD_HOST}" \
      "podman inspect ${NAME} --format '{{.State.Running}}'" 2>/dev/null \
      | grep -Fxq true; then
      printf 'head container exited before readiness\n' >&2
      return 1
    fi
    printf '[%s] waiting for readiness\n' "$(timestamp)"
    sleep 30
  done
  printf 'endpoint did not become ready within 1200 seconds\n' >&2
  return 1
}

process_environment() {
  local host=$1
  "${SSH[@]}" "${host}" \
    "podman exec ${NAME} bash -lc 'pid=\$(pgrep -f \"/opt/venv/bin/vllm serve\" | head -1); tr \"\\0\" \"\\n\" < /proc/\${pid}/environ'"
}

validate_live() {
  local arm=$1 backend expected_kernel host got_id env_dump process_env command_line
  backend=$(backend_for "${arm}")
  if [[ "${arm}" == a ]]; then
    expected_kernel=B12xFp8BlockScaledMMKernel
  else
    expected_kernel=DeepGemmFp8BlockScaledMMKernel
  fi

  for host in "${HEAD_HOST}" "${WORKER_HOST}"; do
    assert_runner "${host}"
    got_id=$("${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{.Image}}'")
    [[ "${got_id}" == "${IMAGE_ID}" ]]
    env_dump=$("${SSH[@]}" "${host}" \
      "podman inspect ${NAME} --format '{{range .Config.Env}}{{println .}}{{end}}'")
    grep -Fxq "BACKEND=${backend}" <<<"${env_dump}"
    grep -Fxq 'NCCL_MIN_NCHANNELS=4' <<<"${env_dump}"
    grep -Fxq 'NCCL_MAX_NCHANNELS=4' <<<"${env_dump}"
    process_env=$(process_environment "${host}")
    command_line=$("${SSH[@]}" "${host}" \
      "podman exec ${NAME} bash -lc 'ps -eo args | grep \"/opt/venv/bin/vllm serve\" | grep -v grep | head -1'")
    if [[ "${arm}" == a ]]; then
      grep -Fxq 'VLLM_USE_B12X_FP8_GEMM=1' <<<"${process_env}"
      grep -Fq -- '--linear-backend b12x' <<<"${command_line}"
    else
      grep -Fxq 'VLLM_USE_B12X_FP8_GEMM=0' <<<"${process_env}"
      if grep -Fq -- '--linear-backend b12x' <<<"${command_line}"; then
        printf '%s DGLIN arm unexpectedly passed --linear-backend b12x\n' \
          "${host}" >&2
        return 1
      fi
    fi
  done

  "${SSH[@]}" "${HEAD_HOST}" \
    "podman logs ${NAME} 2>&1 | grep -Fq 'Selected ${expected_kernel} for Fp8LinearMethod'"
}

start_arm() {
  local arm=$1 label=$2 backend
  backend=$(backend_for "${arm}")
  stop_pair
  printf '[%s] launching %s arm=%s backend=%s, worker then head\n' \
    "$(timestamp)" "${label}" "${arm}" "${backend}"
  launch_node "${WORKER_HOST}" worker "${backend}"
  launch_node "${HEAD_HOST}" head "${backend}"
  wait_ready
  validate_live "${arm}"
}

remote_telemetry() {
  local label=$1 host
  for host in "${HEAD_HOST}" "${WORKER_HOST}"; do
    {
      printf 'label=%s host=%s timestamp=%s\n' \
        "${label}" "${host}" "$(timestamp)"
      "${SSH[@]}" "${host}" \
        'free -h; nvidia-smi --query-gpu=timestamp,name,temperature.gpu,power.draw,clocks.sm,utilization.gpu --format=csv,noheader,nounits'
    } >>"${ROOT}/receipts/telemetry.log"
  done
}

validate_result() {
  local path=$1
  jq -e '
    (.metadata.concurrency_levels == [1,4]) and
    (.metadata.context_lengths == [0,16384,32768,65536,131072]) and
    ([.results[]] | length == 10) and
    (all(.results[];
      (.concurrency == 1 or .concurrency == 4) and
      .num_errors == 0 and
      .warmup_timed_out == false and
      .capacity_limited == false and
      .queue_fraction == 0 and
      .server_steps_per_s > 0)) and
    (.prefill["8192"].client_tok_per_sec > 0) and
    (.prefill["65536"].client_tok_per_sec > 0)
  ' "${path}" >/dev/null
}

run_sweep() {
  local arm=$1 pair=$2 order=$3 seed output label
  label="pair${pair}-${order}-arm${arm}"
  seed="ds4-dglin-paired-v1-pair${pair}"
  output="${ROOT}/results/${label}.json"
  [[ ! -e "${output}" ]] || {
    printf 'refusing to overwrite %s\n' "${output}" >&2
    return 1
  }

  remote_telemetry "${label}-before"
  printf '[%s] benchmark %s seed=%s\n' "$(timestamp)" "${label}" "${seed}"
  (
    cd "${BENCH_ROOT}"
    PYTHONUNBUFFERED=1 "${PYTHON}" -c \
      'import random, runpy, sys; script = sys.argv.pop(1); random.seed(sys.argv.pop(1)); sys.argv[0] = script; runpy.run_path(script, run_name="__main__")' \
      "${BENCH}" "${seed}" \
      --host "http://${HEAD_HOST}:${PORT}" \
      --model "${MODEL}" \
      --concurrency 1,4 \
      --contexts 0,16384,32768,65536,131072 \
      --duration 30 \
      --max-tokens 8192 \
      --display-mode plain \
      --no-hw-monitor \
      --no-resume \
      --output "${output}"
  )
  validate_result "${output}"
  remote_telemetry "${label}-after"
}

capture_receipt() {
  local label=$1 host
  for host in "${HEAD_HOST}" "${WORKER_HOST}"; do
    "${SSH[@]}" "${host}" "podman inspect ${NAME}" \
      >"${ROOT}/receipts/${label}-${host}-inspect.json"
    "${SSH[@]}" "${host}" "podman logs --timestamps ${NAME} 2>&1" \
      >"${ROOT}/receipts/${label}-${host}.log"
    process_environment "${host}" \
      >"${ROOT}/receipts/${label}-${host}-process-env.txt"
  done
}

measure() {
  local arm=$1 pair=$2 order=$3 label
  label="pair${pair}-${order}-arm${arm}"
  start_arm "${arm}" "${label}"
  run_sweep "${arm}" "${pair}" "${order}"
  capture_receipt "${label}"
}

measure_live() {
  local arm=$1 pair=$2 order=$3 label
  label="pair${pair}-${order}-arm${arm}"
  printf '[%s] reusing live %s\n' "$(timestamp)" "${label}"
  validate_live "${arm}"
  run_sweep "${arm}" "${pair}" "${order}"
  capture_receipt "${label}"
}

if [[ ! -x "${PYTHON}" || ! -f "${BENCH}" ]]; then
  printf 'benchmark runtime is missing: python=%s bench=%s\n' \
    "${PYTHON}" "${BENCH}" >&2
  exit 1
fi
assert_runner "${HEAD_HOST}"
assert_runner "${WORKER_HOST}"
case "${START_PAIR}" in
  1 | 2 | 3) ;;
  *) printf 'START_PAIR must be 1, 2, or 3; got %s\n' "${START_PAIR}" >&2; exit 2 ;;
esac

printf '[%s] beginning balanced DGLIN A/B sequence\n' "$(timestamp)"
if (( START_PAIR <= 1 )); then
  if [[ "${RESUME_LIVE_B:-0}" == 1 ]]; then
    measure_live b 1 second
  elif [[ "${RESUME_LIVE_A:-0}" == 1 ]]; then
    measure_live a 1 first
    measure b 1 second
  else
    measure a 1 first
    measure b 1 second
  fi
  if [[ "${STOP_AFTER_PAIR1:-0}" == 1 ]]; then
    printf '[%s] pair 1 complete; leaving the matched DGLIN arm live\n' \
      "$(timestamp)"
    exit 0
  fi
fi
if (( START_PAIR <= 2 )); then
  measure b 2 first
  measure a 2 second
fi
if (( START_PAIR <= 3 )); then
  measure a 3 first
  measure b 3 second
fi
printf '[%s] DGLIN comparison complete; staging remains on the final DGLIN arm\n' \
  "$(timestamp)"
