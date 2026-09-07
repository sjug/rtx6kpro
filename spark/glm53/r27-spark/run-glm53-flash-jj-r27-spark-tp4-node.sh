#!/usr/bin/env bash
set -euo pipefail

# GLM-5.3-Flash NVFP4 JJ r27 Spark profile for the four-node
# sparky/buddy/rocky/lucky DGX Spark cluster. All distributed traffic uses
# the two switched 200G rails; the duplicated back-to-back 10.11.1/2 links
# and the public management interfaces are deliberately excluded.
#
# Start buddy, lucky, and rocky first with ROLE=worker, then sparky with
# ROLE=head. Stop workers first and sparky last with ROLE=stop.

ROLE=${ROLE:?set ROLE=head|worker|stop}
IMAGE=${IMAGE:-localhost/voipmonitor/vllm:glm53-jj-r27-spark-sm121-vllmf3c3fef-b12x95fdcb1-lmcachefe5442f-cu133-torch213-20260906-r1}
EXPECTED_IMAGE_ID=${EXPECTED_IMAGE_ID:-ef669fa1cde3e99936c02575eca8f610990bb6c64dd1bbfcb04fd42dde87afae}
NAME=${NAME:-glm53-flash-nvfp4-jj-r27-spark-tp4}
PORT=${PORT:-8000}
# The r27 image entrypoint is the reviewed launcher (no policy block). The
# optional mount keeps the runner and launcher revisions paired for controls.
runner_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GLM_LAUNCHER_FILE=${GLM_LAUNCHER_FILE:-${runner_dir}/launchers/serve-glm53-flash-jj-r27-spark.sh}

MODEL_REPO_DIR=${MODEL_REPO_DIR:-models--local-inference-lab--GLM-5.3-Flash-NVFP4}
MODEL_REVISION=${MODEL_REVISION:-46aaae8a82032f77100f2f03e9cc11b391df3b4d}
MODEL=${MODEL:-/root/.cache/huggingface/hub/${MODEL_REPO_DIR}/snapshots/${MODEL_REVISION}}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-GLM-5.3-Flash}
EXPECTED_MODEL_BYTES=${EXPECTED_MODEL_BYTES:-198042331512}
EXPECTED_MODEL_SHARDS=${EXPECTED_MODEL_SHARDS:-44}
DFLASH_MODEL_REPO_DIR=${DFLASH_MODEL_REPO_DIR:-models--local-inference-lab--GLM-5.3-Flash-DFlash2}
DFLASH_MODEL_REVISION=${DFLASH_MODEL_REVISION:-aea0ac8a05624512ca9e106c09c16087da998426}
DFLASH_MODEL=${DFLASH_MODEL:-/root/.cache/huggingface/hub/${DFLASH_MODEL_REPO_DIR}/snapshots/${DFLASH_MODEL_REVISION}}
EXPECTED_DFLASH_MODEL_BYTES=${EXPECTED_DFLASH_MODEL_BYTES:-1284719240}
EXPECTED_DFLASH_MODEL_SHA256=${EXPECTED_DFLASH_MODEL_SHA256:-c033e03d47c7d5608596c8fc4e9336a1fe086eb781c08fe031be2bdea1614e58}

HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
CACHE=${CACHE:-$HOME/.cache/vllm-jj-glm53-flash-tp4}
MASTER_ADDR=${MASTER_ADDR:-10.11.11.1}
MASTER_PORT=${MASTER_PORT:-25000}
TP_SIZE=${TP_SIZE:-4}
NNODES=${NNODES:-4}
DCP_SIZE=${DCP_SIZE:-1}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# TP4 leaves enough UMA headroom for vLLM to profile the real runtime and size
# KV from the percentage budget. Do not impose a fixed KV_CACHE_MEMORY_BYTES.
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
SPECULATOR=${SPECULATOR:-mtp}
MTP=${MTP:-3}
VLLM_GLM53_MTP_DRAFT_HEAD=${VLLM_GLM53_MTP_DRAFT_HEAD:-bf16}
MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-'1 2 4 8 12 16 24 32'}
PREFILL_SCHEDULE_INTERVAL=${PREFILL_SCHEDULE_INTERVAL:-8}
FAIRNESS_ENGINE=${FAIRNESS_ENGINE:-none}
PREFILL_COMPUTE_SHARE=${PREFILL_COMPUTE_SHARE:-}
GLM53_KDA_DECODE_BACKEND=${GLM53_KDA_DECODE_BACKEND:-auto}
GLM53_KDA_PREFILL_BACKEND=${GLM53_KDA_PREFILL_BACKEND:-flashkda}
INSTANTTENSOR_BACKEND=${INSTANTTENSOR_BACKEND:-BUFFERED}
INSTANTTENSOR_DEBUG=${INSTANTTENSOR_DEBUG:-0}
VLLM_ENABLE_ROCE_ALLREDUCE=${VLLM_ENABLE_ROCE_ALLREDUCE:-1}
VLLM_ROCE_ALLREDUCE_MAX_SIZE=${VLLM_ROCE_ALLREDUCE_MAX_SIZE:-2MB}
VLLM_ROCE_ALLGATHER_MAX_SIZE=${VLLM_ROCE_ALLGATHER_MAX_SIZE:-16MB}
B12X_ROCE_SPIN_LIMIT=${B12X_ROCE_SPIN_LIMIT:-50000000}
B12X_ROCE_CACHE_DIR=/opt/jovian-judgement/b12x-roce

case "$(hostname -s)" in
  sparky) NODE_RANK=0; HOST_IP=${HOST_IP:-10.11.11.1} ;;
  buddy)  NODE_RANK=1; HOST_IP=${HOST_IP:-10.11.11.2} ;;
  rocky)  NODE_RANK=2; HOST_IP=${HOST_IP:-10.11.11.4} ;;
  lucky)  NODE_RANK=3; HOST_IP=${HOST_IP:-10.11.11.3} ;;
  *)
    NODE_RANK=${NODE_RANK:?unknown host; set NODE_RANK=0..3}
    HOST_IP=${HOST_IP:?unknown host; set the switched 10.11.11.x HOST_IP}
    ;;
esac

case "${ROLE}" in
  head)
    if [[ "${NODE_RANK}" != 0 ]]; then
      echo "ROLE=head is valid only on rank 0 (sparky); got rank ${NODE_RANK}" >&2
      exit 78
    fi
    role_args=()
    ;;
  worker)
    if [[ "${NODE_RANK}" == 0 ]]; then
      echo "ROLE=worker is not valid on rank 0 (sparky)" >&2
      exit 78
    fi
    role_args=(--headless)
    ;;
  stop)
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
      printf 'DRY-RUN: podman stop -t 60 %q\n' "${NAME}"
      printf 'DRY-RUN: podman rm %q\n' "${NAME}"
      exit 0
    fi
    if podman container exists "${NAME}" 2>/dev/null; then
      podman stop -t 60 "${NAME}" >/dev/null
      podman rm "${NAME}" >/dev/null
    fi
    printf '%s stopped and removed on %s\n' "${NAME}" "$(hostname)"
    exit 0
    ;;
  *) echo "ROLE must be head, worker, or stop" >&2; exit 2 ;;
esac

if [[ "${TP_SIZE}" != 4 || "${NNODES}" != 4 || "${DCP_SIZE}" != 1 ]]; then
  echo "This first-admission profile requires TP_SIZE=4, NNODES=4, and DCP_SIZE=1" >&2
  exit 78
fi
if [[ "${MASTER_ADDR}" != 10.11.11.1 ]]; then
  echo "MASTER_ADDR must use sparky's switched 200G address 10.11.11.1" >&2
  exit 78
fi
if [[ "${HOST_IP}" != 10.11.11.* ]]; then
  echo "HOST_IP must use the switched 10.11.11.0/24 fabric; got ${HOST_IP}" >&2
  exit 78
fi
for value in MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS; do
  if [[ ! "${!value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value} must be a positive integer; got ${!value}" >&2
    exit 2
  fi
done
if [[ ! "${PREFILL_SCHEDULE_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PREFILL_SCHEDULE_INTERVAL must be a positive integer; got ${PREFILL_SCHEDULE_INTERVAL}" >&2
  exit 2
fi
case "${FAIRNESS_ENGINE}" in
  none) ;;
  compute_share)
    if [[ "${PREFILL_SCHEDULE_INTERVAL}" != 1 ]]; then
      echo 'PREFILL_SCHEDULE_INTERVAL must be 1 when FAIRNESS_ENGINE is enabled' >&2
      exit 2
    fi
    if ! awk -v value="${PREFILL_COMPUTE_SHARE}" \
      'BEGIN { exit !(value > 0.0 && value < 1.0) }'; then
      echo "PREFILL_COMPUTE_SHARE must be between zero and one; got ${PREFILL_COMPUTE_SHARE}" >&2
      exit 2
    fi
    ;;
  *) echo 'FAIRNESS_ENGINE must be none or compute_share' >&2; exit 2 ;;
esac
case "${GLM53_KDA_DECODE_BACKEND}" in auto|b12x|triton) ;; *) echo 'invalid GLM53_KDA_DECODE_BACKEND' >&2; exit 2 ;; esac
case "${GLM53_KDA_PREFILL_BACKEND}" in auto|b12x|flashkda|triton) ;; *) echo 'invalid GLM53_KDA_PREFILL_BACKEND' >&2; exit 2 ;; esac
case "${SPECULATOR}" in
  mtp)
    NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-${MTP}}
    ;;
  dflash2)
    NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-7}
    ;;
  *)
    echo "SPECULATOR must be mtp or dflash2; got ${SPECULATOR}" >&2
    exit 2
    ;;
esac
if [[ ! "${NUM_SPECULATIVE_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "NUM_SPECULATIVE_TOKENS must be a non-negative integer; got ${NUM_SPECULATIVE_TOKENS}" >&2
  exit 2
fi
case "${VLLM_GLM53_MTP_DRAFT_HEAD}" in
  bf16|nvfp4) ;;
  *) echo 'VLLM_GLM53_MTP_DRAFT_HEAD must be bf16 or nvfp4' >&2; exit 2 ;;
esac
case "${VLLM_ENABLE_ROCE_ALLREDUCE}" in
  0|1) ;;
  *) echo 'VLLM_ENABLE_ROCE_ALLREDUCE must be 0 or 1' >&2; exit 2 ;;
esac
if [[ ! "${B12X_ROCE_SPIN_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "B12X_ROCE_SPIN_LIMIT must be a positive integer; got ${B12X_ROCE_SPIN_LIMIT}" >&2
  exit 2
fi
if [[ -z "${MAX_CUDAGRAPH_CAPTURE_SIZE}" ]]; then
  MAX_CUDAGRAPH_CAPTURE_SIZE=$((MAX_NUM_SEQS * (NUM_SPECULATIVE_TOKENS + 1)))
fi
if [[ ! "${MAX_CUDAGRAPH_CAPTURE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CUDAGRAPH_CAPTURE_SIZE must be a positive integer; got ${MAX_CUDAGRAPH_CAPTURE_SIZE}" >&2
  exit 2
fi
read -r -a capture_size_values <<<"${CUDAGRAPH_CAPTURE_SIZES}"
if ((${#capture_size_values[@]} == 0)); then
  echo 'CUDAGRAPH_CAPTURE_SIZES must not be empty' >&2
  exit 2
fi
previous=0
for size in "${capture_size_values[@]}"; do
  if [[ ! "${size}" =~ ^[1-9][0-9]*$ ]] || ((size <= previous)); then
    echo 'CUDAGRAPH_CAPTURE_SIZES must be strictly increasing positive integers' >&2
    exit 2
  fi
  previous=${size}
done
if ((previous != MAX_CUDAGRAPH_CAPTURE_SIZE)); then
  echo "CUDAGRAPH_CAPTURE_SIZES must end at MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE}; got ${previous}" >&2
  exit 2
fi

launcher_mount=()
if [[ -n "${GLM_LAUNCHER_FILE:-}" ]]; then
  [[ -r "${GLM_LAUNCHER_FILE}" ]] || { echo 'GLM_LAUNCHER_FILE must be readable' >&2; exit 78; }
  launcher_mount=(-v "${GLM_LAUNCHER_FILE}:/usr/local/bin/serve-glm53-flash-jj-r27-spark.sh:ro")
fi
cmd=(
  podman run -d
  --name "${NAME}"
  --device nvidia.com/gpu=all
  --device /dev/infiniband
  --security-opt label=disable
  --network host
  --ipc host
  --init
  --ulimit memlock=-1
  --ulimit stack=67108864
  --ulimit nofile=500000:500000
  -v "${HF_CACHE}:/root/.cache/huggingface:ro"
  -v "${CACHE}:/cache:rw"
  -v "${CACHE}/tmp:/container-tmp:rw"
  "${launcher_mount[@]}"
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -e TMPDIR=/container-tmp
  -e MODEL="${MODEL}"
  -e MODEL_REVISION="${MODEL_REVISION}"
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}"
  -e PORT="${PORT}"
  -e TP="${TP_SIZE}"
  -e DCP="${DCP_SIZE}"
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}"
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}"
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}"
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}"
  -e MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE}"
  -e CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES}"
  -e PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL}"
  -e FAIRNESS_ENGINE="${FAIRNESS_ENGINE}"
  -e PREFILL_COMPUTE_SHARE="${PREFILL_COMPUTE_SHARE}"
  -e GLM53_KDA_DECODE_BACKEND="${GLM53_KDA_DECODE_BACKEND}"
  -e GLM53_KDA_PREFILL_BACKEND="${GLM53_KDA_PREFILL_BACKEND}"
  -e SPECULATOR="${SPECULATOR}"
  -e MTP="${MTP}"
  -e NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS}"
  -e VLLM_GLM53_MTP_DRAFT_HEAD="${VLLM_GLM53_MTP_DRAFT_HEAD}"
  -e DFLASH_MODEL="${DFLASH_MODEL}"
  -e DFLASH_MODEL_REVISION="${DFLASH_MODEL_REVISION}"
  -e LMCACHE_ENABLED=0
  -e INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND}"
  -e INSTANTTENSOR_DEBUG="${INSTANTTENSOR_DEBUG}"
  -e CUTE_DSL_ARCH=sm_121a
  -e TORCH_CUDA_ARCH_LIST=12.1a
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0
  -e VLLM_USE_B12X_PCIE_DMA=0
  -e VLLM_PCIE_DMA_FP8=0
  -e B12X_PCIE_DMA_FP8=0
  -e VLLM_ENABLE_ROCE_ALLREDUCE="${VLLM_ENABLE_ROCE_ALLREDUCE}"
  -e VLLM_ROCE_ALLREDUCE_MAX_SIZE="${VLLM_ROCE_ALLREDUCE_MAX_SIZE}"
  -e VLLM_ROCE_ALLGATHER_MAX_SIZE="${VLLM_ROCE_ALLGATHER_MAX_SIZE}"
  -e B12X_ROCE_SPIN_LIMIT="${B12X_ROCE_SPIN_LIMIT}"
  -e B12X_ROCE_CACHE_DIR="${B12X_ROCE_CACHE_DIR}"
  -e 'B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0'
  -e B12X_ROCE_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
  -e VLLM_HOST_IP="${HOST_IP}"
  -e NCCL_DEBUG="${NCCL_DEBUG}"
  -e NCCL_IB_DISABLE=0
  -e 'NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0'
  -e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
  -e NCCL_IB_TC=106
  -e NCCL_IB_MERGE_NICS=0
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1
  -e 'NCCL_PROTO=LL,Simple'
  -e 'NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0'
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0
  --entrypoint /usr/local/bin/serve-glm53-flash-jj-r27-spark.sh
  "${IMAGE}"
  --nnodes "${NNODES}"
  --node-rank "${NODE_RANK}"
  --master-addr "${MASTER_ADDR}"
  --master-port "${MASTER_PORT}"
  "${role_args[@]}"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY-RUN:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ "$(uname -m)" != aarch64 ]]; then
  echo "This runner requires aarch64" >&2
  exit 78
fi

host_snapshot="${HF_CACHE}/hub/${MODEL_REPO_DIR}/snapshots/${MODEL_REVISION}"
index_file="${host_snapshot}/model.safetensors.index.json"
config_file="${host_snapshot}/config.json"
if [[ ! -r "${index_file}" || ! -r "${config_file}" ]]; then
  echo "Pinned model snapshot is incomplete: ${host_snapshot}" >&2
  exit 78
fi
python3 - "${index_file}" "${config_file}" "${EXPECTED_MODEL_BYTES}" \
  "${EXPECTED_MODEL_SHARDS}" <<'PY'
import json
from pathlib import Path
import sys

index_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
expected_bytes = int(sys.argv[3])
expected_shards = int(sys.argv[4])
index = json.loads(index_path.read_text())
config = json.loads(config_path.read_text())
shards = sorted(set(index["weight_map"].values()))
assert index["metadata"]["total_size"] == expected_bytes
assert len(shards) == expected_shards
assert all((index_path.parent / shard).is_file() for shard in shards)
assert config["architectures"] == ["Glm5NextForConditionalGeneration"]
assert config["model_type"] == "glm5_next"
assert config["quantization_config"]["quant_algo"] == "MIXED_PRECISION"
assert config["quantization_config"]["quant_method"] == "modelopt"
assert config["text_config"]["max_position_embeddings"] == 1048576
assert config["text_config"]["num_nextn_predict_layers"] == 1
PY
broken_links=$(find "${host_snapshot}" -maxdepth 1 -xtype l | wc -l)
if [[ "${broken_links}" != 0 ]]; then
  echo "Pinned model snapshot has ${broken_links} broken links" >&2
  exit 78
fi

if [[ "${SPECULATOR}" == dflash2 ]]; then
  host_dflash_snapshot="${HF_CACHE}/hub/${DFLASH_MODEL_REPO_DIR}/snapshots/${DFLASH_MODEL_REVISION}"
  dflash_weight="${host_dflash_snapshot}/model.safetensors"
  dflash_config="${host_dflash_snapshot}/config.json"
  if [[ ! -r "${dflash_weight}" || ! -r "${dflash_config}" ]]; then
    echo "Pinned DFlash2 snapshot is incomplete: ${host_dflash_snapshot}" >&2
    exit 78
  fi
  python3 - "${dflash_weight}" "${dflash_config}" \
    "${EXPECTED_DFLASH_MODEL_BYTES}" "${EXPECTED_DFLASH_MODEL_SHA256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

weight_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
expected_bytes = int(sys.argv[3])
expected_sha256 = sys.argv[4]
config = json.loads(config_path.read_text())
assert weight_path.stat().st_size == expected_bytes
digest = hashlib.sha256()
with weight_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
assert digest.hexdigest() == expected_sha256
assert config["architectures"] == ["DFlash2DraftModel"]
assert config["num_hidden_layers"] == 5
assert config["quantization_config"]["quant_method"] == "modelopt"
PY
  dflash_broken_links=$(find "${host_dflash_snapshot}" -maxdepth 1 -xtype l | wc -l)
  if [[ "${dflash_broken_links}" != 0 ]]; then
    echo "Pinned DFlash2 snapshot has ${dflash_broken_links} broken links" >&2
    exit 78
  fi
fi

if ! podman image exists "${IMAGE}"; then
  echo "Pinned image is not present locally: ${IMAGE}" >&2
  exit 78
fi
if [[ -z "${EXPECTED_IMAGE_ID}" ]]; then
  echo 'EXPECTED_IMAGE_ID is unset; pin the built r27 Spark image before launch' >&2
  exit 78
fi
actual_image_id=$(podman image inspect "${IMAGE}" --format '{{.Id}}')
actual_image_id=${actual_image_id#sha256:}
if [[ "${actual_image_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "Image ID mismatch: expected ${EXPECTED_IMAGE_ID}, got ${actual_image_id}" >&2
  exit 78
fi
labels=$(podman image inspect "${IMAGE}" --format '{{json .Config.Labels}}')
jq -e '
  ."local-inference.vllm.integration.tree" == "f54cd9ca2b9434727715197d32150b75e82a9ebf" and
  ."local-inference.vllm.package.tree" == "f3c3fef548738807f7d1ef8157df63815c498607" and
  ."local-inference.vllm.spark-overlay.tree" == "8881b7772e1644d63162ad3b7494888ef7ac54fd" and
  ."local-inference.flashkda.commit" == "3b225bf26bb8e218928a1fe14751cb48cf31d11b" and
  ."local-inference.b12x.integration.tree" == "f3cd8a9eb00d3226a1acbbed1efedf10cc1c3e71" and
  ."local-inference.b12x.package.tree" == "95fdcb1cfea380480b8882fa44055cfef358ddbb" and
  ."local-inference.b12x.spark-source-patches" == "none; the r27 B12X integration tree is applied unmodified" and
  ."local-inference.b12x.rocenante.api-version" == "1" and
  ."local-inference.b12x.rocenante.proxy-abi" == "3" and
  ."local-inference.b12x.rocenante.proxy-cache-dir" == "/opt/jovian-judgement/b12x-roce" and
  ."local-inference.lmcache.integration.tree" == "008ac3e09ae5917aa0849147480d7bd5b9f8b37a" and
  ."local-inference.lmcache.package.tree" == "fe5442fbf258accaa7f26d2bbb00d8b7b5c349ca" and
  ."local-inference.lmcache.default" == "disabled"
' <<<"${labels}" >/dev/null || {
  echo "Image provenance does not match the JJ r27 Spark contract" >&2
  exit 78
}

mkdir -p "${CACHE}" "${CACHE}/tmp"
if podman container exists "${NAME}" 2>/dev/null; then
  echo "Container already exists: ${NAME}; stop all four nodes before replacement" >&2
  exit 78
fi

"${cmd[@]}"
printf '%s ROLE=%s rank=%s/%s host_ip=%s model=%s gpu_mem=%s speculator=%s spec_tokens=%s\n' \
  "${NAME}" "${ROLE}" "${NODE_RANK}" "${NNODES}" "${HOST_IP}" \
  "${SERVED_MODEL_NAME}" "${GPU_MEMORY_UTILIZATION}" "${SPECULATOR}" \
  "${NUM_SPECULATIVE_TOKENS}"
