#!/usr/bin/env bash
set -euo pipefail

# Compare corrected-r34 and II r18p across independent vLLM engine starts on
# the dusty/kirby TP2 pair. Host reboots and cache clearing are intentionally
# out of scope: each sample resets CUDA graphs, NCCL communicators, and the
# model engine while preserving the qualified on-disk caches.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_ROOT=${BENCH_ROOT:-/home/jugs/git/llm-inference-bench}
BENCH=${BENCH:-${BENCH_ROOT}/llm_decode_bench.py}
PYTHON=${PYTHON:-${BENCH_ROOT}/.venv/bin/python}
HEAD_HOST=${HEAD_HOST:-dusty}
WORKER_HOST=${WORKER_HOST:-kirby}
NAME=${NAME:-ds4-0731-tp2}
PORT=${PORT:-8000}
MODEL=${MODEL:-DeepSeek-V4-Flash-0731}
IMAGE_R18P=${IMAGE_R18P:-localhost/voipmonitor/vllm:infernal-invocation-r18p-spark-sm121-vllmf560085-b12x07cdf45-fi1ac6942-cu133-torch213-20260820}
IMAGE_R34=${IMAGE_R34:-localhost/voipmonitor/vllm:gilded-gnosis-v20-r34-spark-sm121-vllm17b78ef-b12xcd3ce19-fi1ac6942-cu132-20260818}
ID_R18P=${ID_R18P:-445f9ac3196dd10a47d0d90441ba43a8417903ba029f59a1a9c7fbef8ecfa4a1}
ID_R34=${ID_R34:-276f00868134ed3116ffaf44db975f1b4d8803c7f528c8f772bdaae43506fbd6}
SSH=(ssh -F /dev/null)

mkdir -p "${ROOT}/results" "${ROOT}/receipts"
exec > >(tee -a "${ROOT}/study.log") 2>&1

timestamp() { date -Is; }

image_for() {
  case "$1" in
    r18p) printf '%s\n' "${IMAGE_R18P}" ;;
    r34) printf '%s\n' "${IMAGE_R34}" ;;
    *) printf 'unknown image arm: %s\n' "$1" >&2; return 2 ;;
  esac
}

id_for() {
  case "$1" in
    r18p) printf '%s\n' "${ID_R18P}" ;;
    r34) printf '%s\n' "${ID_R34}" ;;
    *) printf 'unknown image arm: %s\n' "$1" >&2; return 2 ;;
  esac
}

remote_telemetry() {
  local label=$1 host
  for host in "${HEAD_HOST}" "${WORKER_HOST}"; do
    {
      printf 'label=%s host=%s timestamp=%s\n' "${label}" "${host}" "$(timestamp)"
      "${SSH[@]}" "${host}" 'free -h; nvidia-smi --query-gpu=timestamp,name,temperature.gpu,power.draw,clocks.sm,utilization.gpu --format=csv,noheader,nounits'
    } >>"${ROOT}/receipts/telemetry.log"
  done
}

validate_live_arm() {
  local arm=$1 expected_id host got_id
  expected_id=$(id_for "${arm}")
  for host in "${HEAD_HOST}" "${WORKER_HOST}"; do
    got_id=$("${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{.Image}}'")
    if [[ "${got_id}" != "${expected_id}" ]]; then
      printf '%s has image %s, expected %s for %s\n' \
        "${host}" "${got_id}" "${expected_id}" "${arm}" >&2
      return 1
    fi
    "${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{range .Config.Env}}{{println .}}{{end}}'" \
      | grep -Fxq 'NCCL_PROTO=LL,Simple'
    if [[ "${arm}" == r18p ]]; then
      "${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{range .Config.Env}}{{println .}}{{end}}'" \
        | grep -Fxq 'NCCL_MIN_NCHANNELS=4'
      "${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{range .Config.Env}}{{println .}}{{end}}'" \
        | grep -Fxq 'NCCL_MAX_NCHANNELS=4'
    else
      if "${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{range .Config.Env}}{{println .}}{{end}}'" \
        | grep -Eq '^NCCL_(MIN|MAX)_NCHANNELS='; then
        printf '%s unexpectedly pins NCCL channels for r34\n' "${host}" >&2
        return 1
      fi
    fi
  done
}

stop_pair() {
  printf '[%s] stopping worker then head\n' "$(timestamp)"
  "${SSH[@]}" "${WORKER_HOST}" "timeout 120 podman stop -t 90 ${NAME} >/dev/null && podman rm ${NAME} >/dev/null"
  "${SSH[@]}" "${HEAD_HOST}" "timeout 120 podman stop -t 90 ${NAME} >/dev/null && podman rm ${NAME} >/dev/null"
}

launch_node() {
  local host=$1 rank=$2 ip=$3 image=$4 arm=$5
  local headless='' channel_args=''
  if [[ "${rank}" == 1 ]]; then
    headless='--headless'
  fi
  if [[ "${arm}" == r18p ]]; then
    channel_args='-e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4'
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
    -e MODE=dspark \
    -e BACKEND=b12x-a8 \
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
    ${channel_args} \
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
    ${image}" >/dev/null
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

start_arm() {
  local arm=$1 boot=$2 image
  image=$(image_for "${arm}")
  stop_pair
  printf '[%s] launching %s boot %s worker then head\n' "$(timestamp)" "${arm}" "${boot}"
  launch_node "${WORKER_HOST}" 1 10.11.1.2 "${image}" "${arm}"
  launch_node "${HEAD_HOST}" 0 10.11.1.1 "${image}" "${arm}"
  wait_ready
  validate_live_arm "${arm}"
}

validate_result() {
  local path=$1
  jq -e '
    (.metadata.concurrency_levels == [4]) and
    (.metadata.context_lengths == [0,16384,32768,65536,131072]) and
    ([.results[] | select(.concurrency == 4)] | length == 5) and
    (all(.results[];
      .concurrency == 4 and
      .num_errors == 0 and
      .warmup_timed_out == false and
      .capacity_limited == false and
      .server_steps_per_s > 0)) and
    (.prefill["65536"].client_tok_per_sec > 0)
  ' "${path}" >/dev/null
}

run_sweep() {
  local arm=$1 boot=$2 sweep=$3 seed output
  seed="ds4-c4-boot-distribution-v1-sweep${sweep}"
  output="${ROOT}/results/${arm}-boot${boot}-sweep${sweep}.json"
  if [[ -e "${output}" ]]; then
    printf 'refusing to overwrite %s\n' "${output}" >&2
    return 1
  fi

  remote_telemetry "${arm}-boot${boot}-sweep${sweep}-before"
  printf '[%s] benchmark %s boot %s sweep %s\n' "$(timestamp)" "${arm}" "${boot}" "${sweep}"
  (
    cd "${BENCH_ROOT}"
    PYTHONUNBUFFERED=1 "${PYTHON}" -c \
      'import random, runpy, sys; script = sys.argv.pop(1); random.seed(sys.argv.pop(1)); sys.argv[0] = script; runpy.run_path(script, run_name="__main__")' \
      "${BENCH}" "${seed}" \
      --host "http://${HEAD_HOST}:${PORT}" \
      --model "${MODEL}" \
      --concurrency 4 \
      --contexts 0,16384,32768,65536,131072 \
      --duration 30 \
      --max-tokens 8192 \
      --display-mode plain \
      --no-hw-monitor \
      --no-resume \
      --output "${output}"
  )
  validate_result "${output}"
  remote_telemetry "${arm}-boot${boot}-sweep${sweep}-after"
}

capture_receipt() {
  local arm=$1 boot=$2 host
  for host in "${HEAD_HOST}" "${WORKER_HOST}"; do
    "${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{.ImageName}}|{{.Image}}|{{.State.StartedAt}}|{{json .Config.CreateCommand}}'" \
      >"${ROOT}/receipts/${arm}-boot${boot}-${host}-inspect.txt"
    "${SSH[@]}" "${host}" "podman logs --timestamps ${NAME} 2>&1" \
      >"${ROOT}/receipts/${arm}-boot${boot}-${host}.log"
  done
}

measure_boot() {
  local arm=$1 boot=$2
  validate_live_arm "${arm}"
  run_sweep "${arm}" "${boot}" 1
  run_sweep "${arm}" "${boot}" 2
  capture_receipt "${arm}" "${boot}"
}

if [[ ! -x "${PYTHON}" || ! -f "${BENCH}" ]]; then
  printf 'benchmark runtime is missing: python=%s bench=%s\n' "${PYTHON}" "${BENCH}" >&2
  exit 1
fi

# Current staging state is already a clean r18p engine start and becomes A1.
printf '[%s] beginning A/B/B/A/A/B engine-start sequence\n' "$(timestamp)"
measure_boot r18p 1
start_arm r34 1
measure_boot r34 1
start_arm r34 2
measure_boot r34 2
start_arm r18p 2
measure_boot r18p 2
start_arm r18p 3
measure_boot r18p 3
start_arm r34 3
measure_boot r34 3

printf '[%s] study complete; final staging state is corrected-r34\n' "$(timestamp)"
