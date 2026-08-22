#!/usr/bin/env bash
set -euo pipefail

# Run a single-GPU, production-geometry W4A8 policy screen on dusty, then
# restore the exact stock r18p DS4 TP2 service on dusty/kirby.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HEAD_HOST=${HEAD_HOST:-dusty}
WORKER_HOST=${WORKER_HOST:-kirby}
NAME=${NAME:-ds4-0731-tp2}
SCREEN_NAME=${SCREEN_NAME:-ds4-a8-kernel-screen}
PORT=${PORT:-8000}
MODEL=${MODEL:-DeepSeek-V4-Flash-0731}
IMAGE=${IMAGE:-localhost/voipmonitor/vllm:infernal-invocation-r18p-spark-sm121-vllmf560085-b12x07cdf45-fi1ac6942-cu133-torch213-20260820}
IMAGE_ID=${IMAGE_ID:-445f9ac3196dd10a47d0d90441ba43a8417903ba029f59a1a9c7fbef8ecfa4a1}
REMOTE_ROOT=${REMOTE_ROOT:-/tmp/ds4-a8-kernel-screen-20260822}
REMOTE_RESULT=${REMOTE_RESULT:-results.json}
LOCAL_RESULT=${LOCAL_RESULT:-results.json}
REMOTE_B12X=${REMOTE_B12X:-/tmp/b12x-r18p-roundtrip}
B12X_TREE=${B12X_TREE:-07cdf4567b50fa983462f0f0e1bc992de3033adc}
MICRO_ARMS=${MICRO_ARMS:-stock,persistent_grid,no_materialized,no_share_input,no_tiny}
MICRO_M_VALUES=${MICRO_M_VALUES:-1,3,4,5,6,24,256,1024}
MODULE=/opt/infernal-invocation/vllm/vllm/model_executor/layers/fused_moe/b12x_moe.py
MODULE_SHA=${MODULE_SHA:-b2a37be72c6299d61d764b887f3332c4529f137b3ec75626f716c1e783959e3d}
SSH=(ssh -F /dev/null)
needs_restore=0

mkdir -p "${ROOT}/receipts"
exec > >(tee -a "${ROOT}/kernel-screen.log") 2>&1

timestamp() { date -Is; }

stop_pair() {
  printf '[%s] stopping worker then head\n' "$(timestamp)"
  "${SSH[@]}" "${WORKER_HOST}" "timeout 120 podman stop -t 90 ${NAME} >/dev/null && podman rm ${NAME} >/dev/null"
  "${SSH[@]}" "${HEAD_HOST}" "timeout 120 podman stop -t 90 ${NAME} >/dev/null && podman rm ${NAME} >/dev/null"
}

launch_node() {
  local host=$1 rank=$2 ip=$3 headless=''
  if [[ "${rank}" == 1 ]]; then
    headless='--headless'
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
  while ((SECONDS < deadline)); do
    if curl -fsS --max-time 5 "http://${HEAD_HOST}:${PORT}/health" >/dev/null; then
      curl -fsS --max-time 5 "http://${HEAD_HOST}:${PORT}/v1/models" \
        | jq -e --arg expected "${MODEL}" \
          '(.data | type == "array") and ([.data[].id] == [$expected])' >/dev/null \
        || { printf 'served model identity mismatch; expected only %s\n' "${MODEL}" >&2; return 1; }
      printf '[%s] restored endpoint ready\n' "$(timestamp)"
      return 0
    fi
    if ! "${SSH[@]}" "${HEAD_HOST}" "podman inspect ${NAME} --format '{{.State.Running}}'" 2>/dev/null | grep -Fxq true; then
      printf 'restored head container exited before readiness\n' >&2
      return 1
    fi
    printf '[%s] waiting for restored endpoint\n' "$(timestamp)"
    sleep 30
  done
  printf 'restored endpoint did not become ready within 1200 seconds\n' >&2
  return 1
}

validate_stock() {
  local host got_id env_dump module_sha
  for host in "${HEAD_HOST}" "${WORKER_HOST}"; do
    got_id=$("${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{.Image}}'")
    [[ "${got_id}" == "${IMAGE_ID}" ]]
    env_dump=$("${SSH[@]}" "${host}" "podman inspect ${NAME} --format '{{range .Config.Env}}{{println .}}{{end}}'")
    grep -Fxq 'BACKEND=b12x-a8' <<<"${env_dump}"
    grep -Fxq 'NCCL_MIN_NCHANNELS=4' <<<"${env_dump}"
    grep -Fxq 'NCCL_MAX_NCHANNELS=4' <<<"${env_dump}"
    if grep -Eq '^B12X_DYNAMIC_(WORK_SOURCE|W4A8_MATERIALIZED|W4A8_SHARE_INPUT)=' <<<"${env_dump}"; then
      printf '%s restored service carries a kernel-policy override\n' "${host}" >&2
      return 1
    fi
    module_sha=$("${SSH[@]}" "${host}" "podman exec ${NAME} sha256sum ${MODULE}" | cut -d' ' -f1)
    [[ "${module_sha}" == "${MODULE_SHA}" ]]
  done
}

restore_pair() {
  printf '[%s] restoring stock r18p A8 worker then head\n' "$(timestamp)"
  "${SSH[@]}" "${HEAD_HOST}" "podman rm -f ${SCREEN_NAME} >/dev/null 2>&1 || true"
  launch_node "${WORKER_HOST}" 1 10.11.1.2
  launch_node "${HEAD_HOST}" 0 10.11.1.1
  wait_ready
  validate_stock
  needs_restore=0
}

copy_results() {
  rsync -a -e 'ssh -F /dev/null' \
    "${HEAD_HOST}:${REMOTE_ROOT}/${REMOTE_RESULT}" "${ROOT}/${LOCAL_RESULT}" 2>/dev/null || true
}

on_exit() {
  local rc=$?
  trap - EXIT
  copy_results
  if ((needs_restore != 0)); then
    restore_pair || true
  fi
  exit "${rc}"
}
trap on_exit EXIT

printf '[%s] validating remote source and image\n' "$(timestamp)"
remote_tree=$("${SSH[@]}" "${HEAD_HOST}" "git -C ${REMOTE_B12X} write-tree")
[[ "${remote_tree}" == "${B12X_TREE}" ]]
remote_image=$("${SSH[@]}" "${HEAD_HOST}" "podman image inspect ${IMAGE} --format '{{.Id}}'")
[[ "${remote_image}" == "${IMAGE_ID}" ]]
"${SSH[@]}" "${HEAD_HOST}" "mkdir -p ${REMOTE_ROOT}"
rsync -a --checksum -e 'ssh -F /dev/null' "${ROOT}/microbench.py" "${HEAD_HOST}:${REMOTE_ROOT}/microbench.py"

stop_pair
needs_restore=1
printf '[%s] starting single-GPU W4A8 kernel screen on %s\n' "$(timestamp)" "${HEAD_HOST}"
"${SSH[@]}" "${HEAD_HOST}" "podman run --rm \
  --name ${SCREEN_NAME} \
  --device nvidia.com/gpu=all \
  --security-opt label=disable \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ${REMOTE_B12X}:/workspace/b12x:ro \
  -v ${REMOTE_ROOT}:/output:rw \
  -v /home/jugs/.cache/vllm-ds4-v20:/cache:rw \
  -v /home/jugs/.cache/vllm-ds4-v20/tmp:/container-tmp:rw \
  -e PYTHONPATH=/workspace/b12x \
  -e PYTHONUNBUFFERED=1 \
  -e B12X_PRINT_COMPILE_PROGRESS=1 \
  -e CUTE_DSL_ARCH=sm_121a \
  -e TMPDIR=/container-tmp \
  -e XDG_CACHE_HOME=/cache \
  --entrypoint /opt/venv/bin/python \
  ${IMAGE} \
  /output/microbench.py \
  --b12x-root /workspace/b12x \
  --output /output/${REMOTE_RESULT} \
  --arms ${MICRO_ARMS} \
  --m-values ${MICRO_M_VALUES}"
copy_results
restore_pair
printf '[%s] kernel screen complete; final staging state is stock r18p A8\n' "$(timestamp)"
