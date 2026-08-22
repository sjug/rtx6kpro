#!/usr/bin/env bash
set -euo pipefail

# Complete a contemporary r18p/r34/r18p c1 comparison. The first r18p arm is
# the immediately preceding a8-control-2 boot from the A8/A16/A8 crossover;
# this script runs the middle r34 arm and closing r18p arm with the same two
# prompt seeds and benchmark contract.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STUDY_ROOT=$(cd "${ROOT}/.." && pwd)
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
exec > >(tee -a "${ROOT}/c1-image-aba.log") 2>&1

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

validate_live_arm() {
  local arm=$1 expected_id host got_id
  expected_id=$(id_for "${arm}")
  for host in "${HEAD_HOST}" "${WORKER_HOST}"; do
    got_id=$("${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{.Image}}'")
    [[ "${got_id}" == "${expected_id}" ]] || {
      printf '%s has image %s, expected %s for %s\n' "${host}" "${got_id}" "${expected_id}" "${arm}" >&2
      return 1
    }
    "${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{range .Config.Env}}{{println .}}{{end}}'" \
      | grep -Fxq 'BACKEND=b12x-a8'
    "${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{range .Config.Env}}{{println .}}{{end}}'" \
      | grep -Fxq 'NCCL_PROTO=LL,Simple'
    if [[ "${arm}" == r18p ]]; then
      "${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{range .Config.Env}}{{println .}}{{end}}'" \
        | grep -Fxq 'NCCL_MIN_NCHANNELS=4'
      "${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{range .Config.Env}}{{println .}}{{end}}'" \
        | grep -Fxq 'NCCL_MAX_NCHANNELS=4'
    elif "${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{range .Config.Env}}{{println .}}{{end}}'" \
      | grep -Eq '^NCCL_(MIN|MAX)_NCHANNELS='; then
      printf '%s unexpectedly pins NCCL channels for r34\n' "${host}" >&2
      return 1
    fi
  done
}

start_arm() {
  local arm=$1 image
  image=$(image_for "${arm}")
  stop_pair
  printf '[%s] launching %s worker then head\n' "$(timestamp)" "${arm}"
  launch_node "${WORKER_HOST}" 1 10.11.1.2 "${image}" "${arm}"
  launch_node "${HEAD_HOST}" 0 10.11.1.1 "${image}" "${arm}"
  wait_ready
  validate_live_arm "${arm}"
}

validate_result() {
  local path=$1
  jq -e '
    (.metadata.concurrency_levels == [1]) and
    (.metadata.context_lengths == [0,16384,32768,65536,131072]) and
    ([.results[]] | length == 5) and
    (all(.results[];
      .concurrency == 1 and
      .num_errors == 0 and
      .warmup_timed_out == false and
      .capacity_limited == false and
      .queue_fraction == 0 and
      .server_steps_per_s > 0)) and
    (.prefill["65536"].client_tok_per_sec > 0)
  ' "${path}" >/dev/null
}

validate_opening_result() {
  local path=$1
  jq -e '
    (.metadata.concurrency_levels == [1,4]) and
    (.metadata.context_lengths == [0,16384,32768,65536,131072]) and
    ([.results[] | select(.concurrency == 1)] | length == 5) and
    (all(.results[] | select(.concurrency == 1);
      .num_errors == 0 and
      .warmup_timed_out == false and
      .capacity_limited == false and
      .queue_fraction == 0 and
      .server_steps_per_s > 0)) and
    (.prefill["65536"].client_tok_per_sec > 0)
  ' "${path}" >/dev/null
}

run_sweep() {
  local arm=$1 sweep=$2 seed output
  seed="ds4-a8-a16-crossover-v1-sweep${sweep}"
  output="${ROOT}/results/${arm}-sweep${sweep}.json"
  [[ ! -e "${output}" ]] || {
    printf 'refusing to overwrite %s\n' "${output}" >&2
    return 1
  }

  printf '[%s] benchmark %s sweep %s\n' "$(timestamp)" "${arm}" "${sweep}"
  (
    cd "${BENCH_ROOT}"
    PYTHONUNBUFFERED=1 "${PYTHON}" -c \
      'import random, runpy, sys; script = sys.argv.pop(1); random.seed(sys.argv.pop(1)); sys.argv[0] = script; runpy.run_path(script, run_name="__main__")' \
      "${BENCH}" "${seed}" \
      --host "http://${HEAD_HOST}:${PORT}" \
      --model "${MODEL}" \
      --concurrency 1 \
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

capture_receipt() {
  local arm=$1 host
  for host in "${HEAD_HOST}" "${WORKER_HOST}"; do
    "${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{.ImageName}}|{{.Image}}|{{.State.StartedAt}}|{{json .Config.CreateCommand}}'" \
      >"${ROOT}/receipts/${arm}-${host}-inspect.txt"
    "${SSH[@]}" "${host}" "podman logs --timestamps ${NAME} 2>&1" \
      >"${ROOT}/receipts/${arm}-${host}.log"
  done
}

measure_arm() {
  local arm=$1
  start_arm "${arm}"
  run_sweep "${arm}" 1
  run_sweep "${arm}" 2
  capture_receipt "${arm}"
}

if [[ ! -x "${PYTHON}" || ! -f "${BENCH}" ]]; then
  printf 'benchmark runtime is missing: python=%s bench=%s\n' "${PYTHON}" "${BENCH}" >&2
  exit 1
fi

for sweep in 1 2; do
  validate_opening_result "${STUDY_ROOT}/a8-a16-crossover/results/a8-control-2-sweep${sweep}.json"
done

printf '[%s] using preceding r18p A8 control, beginning r34/r18p closing arms\n' "$(timestamp)"
measure_arm r34
measure_arm r18p
printf '[%s] c1 image A/B/A complete; final staging state is r18p A8\n' "$(timestamp)"
