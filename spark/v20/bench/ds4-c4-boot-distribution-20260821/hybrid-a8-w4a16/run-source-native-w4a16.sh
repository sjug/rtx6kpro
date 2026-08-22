#!/usr/bin/env bash
set -euo pipefail

# Measure the planner-supported source-native W4A16 layout on r18p, then return
# dusty/kirby to the stock A8 contract. Contemporary A8 controls come from the
# immediately preceding Phase 5 closing boot.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_ROOT=${BENCH_ROOT:-/home/jugs/git/llm-inference-bench}
BENCH=${BENCH:-${BENCH_ROOT}/llm_decode_bench.py}
PYTHON=${PYTHON:-${BENCH_ROOT}/.venv/bin/python}
HEAD_HOST=${HEAD_HOST:-dusty}
WORKER_HOST=${WORKER_HOST:-kirby}
NAME=${NAME:-ds4-0731-tp2}
PORT=${PORT:-8000}
MODEL=${MODEL:-DeepSeek-V4-Flash-0731}
IMAGE=${IMAGE:-localhost/voipmonitor/vllm:infernal-invocation-r18p-spark-sm121-vllmf560085-b12x07cdf45-fi1ac6942-cu133-torch213-20260820}
IMAGE_ID=${IMAGE_ID:-445f9ac3196dd10a47d0d90441ba43a8417903ba029f59a1a9c7fbef8ecfa4a1}
MODULE=/opt/infernal-invocation/vllm/vllm/model_executor/layers/fused_moe/b12x_moe.py
MODULE_SHA=${MODULE_SHA:-b2a37be72c6299d61d764b887f3332c4529f137b3ec75626f716c1e783959e3d}
SSH=(ssh -F /dev/null)
needs_restore=0

mkdir -p "${ROOT}/results" "${ROOT}/receipts"
exec > >(tee -a "${ROOT}/source-native-w4a16.log") 2>&1

timestamp() { date -Is; }

stop_pair() {
  printf '[%s] stopping worker then head\n' "$(timestamp)"
  "${SSH[@]}" "${WORKER_HOST}" "timeout 120 podman stop -t 90 ${NAME} >/dev/null && podman rm ${NAME} >/dev/null"
  "${SSH[@]}" "${HEAD_HOST}" "timeout 120 podman stop -t 90 ${NAME} >/dev/null && podman rm ${NAME} >/dev/null"
}

launch_node() {
  local host=$1 rank=$2 ip=$3 mode=$4 headless='' backend='b12x-a8' layout_env=''
  if [[ "${rank}" == 1 ]]; then
    headless='--headless'
  fi
  if [[ "${mode}" == source-native-w4a16 ]]; then
    backend='b12x-a16'
    layout_env='-e VLLM_B12X_MOE_FORCE_MODELOPT_PREP=1'
  fi

  "${SSH[@]}" "${host}" "podman run -d \
    --name ${NAME} \
    --device nvidia.com/gpu=all \
    --device /dev/infiniband \
    --security-opt label=disable \
    --network host \
    --ipc host \
    --init \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=500000:500000 \
    -v /home/jugs/.cache/huggingface:/root/.cache/huggingface:rw \
    -v /home/jugs/.cache/vllm-ds4-v20:/cache:rw \
    -v /home/jugs/.cache/vllm-ds4-v20/tmp:/container-tmp:rw \
    ${layout_env} \
    -e MODE=dspark \
    -e BACKEND=${backend} \
    -e TP_SIZE=2 \
    -e PORT=${PORT} \
    -e MAX_NUM_SEQS=4 \
    -e MAX_MODEL_LEN=524288 \
    -e GPU_MEMORY_UTILIZATION=0.87 \
    -e ALLREDUCE_MODE=nccl \
    -e 'EXTRA_VLLM_ARGS=--nnodes 2 --node-rank ${rank} --master-addr 10.11.1.1 --master-port 25000 ${headless} ' \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e CUTE_DSL_ARCH=sm_121a \
    -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
    -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_IB_TC=106 \
    -e NCCL_PROTO=LL,Simple \
    -e NCCL_MAX_NCHANNELS=4 \
    -e NCCL_MIN_NCHANNELS=4 \
    -e NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1 \
    -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
    -e VLLM_HOST_IP=${ip} \
    -e HF_HUB_OFFLINE=1 \
    -e TMPDIR=/container-tmp \
    -e XDG_CACHE_HOME=/cache \
    -e CUDAGRAPH_CAPTURE_SIZES=1,2,4,6,8,12,16,18,24 \
    -e DSPARK_TOKENS=5 \
    -e DRAFT_SAMPLE_METHOD=probabilistic \
    -e SERVED_MODEL_NAME=DeepSeek-V4-Flash-0731 \
    -e DSPARK_MODEL=deepseek-ai/DeepSeek-V4-Flash-0731 \
    -e MODEL_REVISION=9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
    -e DSPARK_MODEL_REVISION=9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
    --entrypoint /usr/local/bin/serve-ds4-flash.sh \
    ${IMAGE}" >/dev/null
}

wait_ready() {
  local deadline=$((SECONDS + 1200))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 5 "http://${HEAD_HOST}:${PORT}/health" >/dev/null; then
      curl -fsS --max-time 5 "http://${HEAD_HOST}:${PORT}/v1/models" \
        | jq -e --arg expected "${MODEL}" \
          '(.data | type == "array") and ([.data[].id] == [$expected])' >/dev/null \
        || { printf 'served model identity mismatch; expected only %s\n' "${MODEL}" >&2; return 1; }
      printf '[%s] endpoint ready\n' "$(timestamp)"
      return 0
    fi
    if ! "${SSH[@]}" "${HEAD_HOST}" "podman inspect ${NAME} --format '{{.State.Running}}'" 2>/dev/null | grep -Fxq true; then
      printf 'head container exited before readiness\n' >&2
      return 1
    fi
    printf '[%s] waiting for readiness\n' "$(timestamp)"
    sleep 30
  done
  printf 'endpoint did not become ready within 1200 seconds\n' >&2
  return 1
}

validate_live() {
  local mode=$1 host got_id module_sha env_dump expected_backend='b12x-a8'
  if [[ "${mode}" == source-native-w4a16 ]]; then
    expected_backend='b12x-a16'
  fi
  for host in "${HEAD_HOST}" "${WORKER_HOST}"; do
    got_id=$("${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{.Image}}'")
    [[ "${got_id}" == "${IMAGE_ID}" ]]
    env_dump=$("${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{range .Config.Env}}{{println .}}{{end}}'")
    grep -Fxq "BACKEND=${expected_backend}" <<<"${env_dump}"
    grep -Fxq 'NCCL_MIN_NCHANNELS=4' <<<"${env_dump}"
    grep -Fxq 'NCCL_MAX_NCHANNELS=4' <<<"${env_dump}"
    module_sha=$("${SSH[@]}" "${host}" "podman exec ${NAME} sha256sum ${MODULE}" | cut -d' ' -f1)
    [[ "${module_sha}" == "${MODULE_SHA}" ]]
    if [[ "${mode}" == source-native-w4a16 ]]; then
      grep -Fxq 'VLLM_B12X_MOE_FORCE_MODELOPT_PREP=1' <<<"${env_dump}"
    elif grep -q '^VLLM_B12X_MOE_FORCE_MODELOPT_PREP=' <<<"${env_dump}"; then
      printf '%s stock A8 unexpectedly carries source-native W4A16 policy\n' "${host}" >&2
      return 1
    fi
  done
}

start_mode() {
  local mode=$1
  stop_pair
  needs_restore=1
  printf '[%s] launching %s worker then head\n' "$(timestamp)" "${mode}"
  launch_node "${WORKER_HOST}" 1 10.11.1.2 "${mode}"
  launch_node "${HEAD_HOST}" 0 10.11.1.1 "${mode}"
  wait_ready
  validate_live "${mode}"
}

capture_receipt() {
  local label=$1 host
  for host in "${HEAD_HOST}" "${WORKER_HOST}"; do
    "${SSH[@]}" "${host}" "podman inspect ${NAME}" \
      >"${ROOT}/receipts/${label}-${host}-inspect.json" 2>&1 || true
    "${SSH[@]}" "${host}" "podman logs --timestamps ${NAME} 2>&1" \
      >"${ROOT}/receipts/${label}-${host}.log" 2>&1 || true
  done
}

restore_a8() {
  if (( needs_restore == 0 )); then
    return
  fi
  capture_receipt source-native-w4a16
  printf '[%s] restoring stock r18p A8\n' "$(timestamp)"
  start_mode a8
  needs_restore=0
}

on_exit() {
  local rc=$?
  trap - EXIT
  if (( needs_restore != 0 )); then
    restore_a8 || true
  fi
  exit "${rc}"
}
trap on_exit EXIT

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
    (.prefill["65536"].client_tok_per_sec > 0)
  ' "${path}" >/dev/null
}

run_sweep() {
  local sweep=$1 seed output
  seed="ds4-a8-a16-crossover-v1-sweep${sweep}"
  output="${ROOT}/results/source-native-w4a16-sweep${sweep}.json"
  [[ ! -e "${output}" ]] || {
    printf 'refusing to overwrite %s\n' "${output}" >&2
    return 1
  }
  printf '[%s] benchmark source-native W4A16 sweep %s\n' "$(timestamp)" "${sweep}"
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
}

if [[ ! -x "${PYTHON}" || ! -f "${BENCH}" ]]; then
  printf 'benchmark runtime is missing: python=%s bench=%s\n' "${PYTHON}" "${BENCH}" >&2
  exit 1
fi

validate_live a8
start_mode source-native-w4a16
curl -fsS --max-time 120 "http://${HEAD_HOST}:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4-Flash-0731","messages":[{"role":"user","content":"Return only the integer answer: 19 + 23"}],"temperature":0,"max_tokens":32}' \
  >"${ROOT}/arithmetic-smoke.json"
jq -e '.choices[0].finish_reason != null and (.choices[0].message.content | contains("42"))' \
  "${ROOT}/arithmetic-smoke.json" >/dev/null
run_sweep 1
run_sweep 2
restore_a8
printf '[%s] source-native W4A16 probe complete; final staging state is stock r18p A8\n' "$(timestamp)"
