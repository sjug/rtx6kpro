#!/usr/bin/env bash
set -euo pipefail
case ${DRY_RUN:-0} in 0|1) ;; *) echo 'DRY_RUN must be 0 or 1' >&2; exit 2;; esac

# Qwen3.8-Flash-Next NVFP4 QAD on the dusty/kirby DGX Spark pair. Start
# kirby as worker first, then dusty as head. Stop worker first, then head.

ROLE=${ROLE:?set ROLE=head|worker}
[[ $ROLE != stop ]] || { echo "Stop gracefully and retain containers" >&2; exit 78; }
IMAGE=${IMAGE:-localhost/voipmonitor/vllm:karmic-beta-20261001-spark-sm121}
EXPECTED_IMAGE_ID=${EXPECTED_IMAGE_ID:-}
NAME=${NAME:-qwen38-flash-next-nvfp4-karmic-beta-20261001-tp2}
PORT=${PORT:-8000}

MODEL_REPO_DIR=${MODEL_REPO_DIR:-models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4}
MODEL_REVISION=${MODEL_REVISION:-7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd}
MODEL=${MODEL:-/root/.cache/huggingface/hub/${MODEL_REPO_DIR}/snapshots/${MODEL_REVISION}}
# Client-facing alias only; the checkpoint repository and revision stay pinned.
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3.8-Flash-Next}
EXPECTED_MODEL_BYTES=${EXPECTED_MODEL_BYTES:-105798973864}
EXPECTED_MODEL_SHARDS=${EXPECTED_MODEL_SHARDS:-36}

HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
CACHE=${CACHE:-$HOME/.cache/vllm-jj-qwen38-4p89}
MASTER_ADDR=${MASTER_ADDR:-10.11.1.1}
MASTER_PORT=${MASTER_PORT:-25000}
TP_SIZE=${TP_SIZE:-2}
NNODES=${NNODES:-2}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-3}
INSTANTTENSOR_BACKEND=${INSTANTTENSOR_BACKEND:-BUFFERED}
INSTANTTENSOR_DEBUG=${INSTANTTENSOR_DEBUG:-0}
# Carry R29's qualified aligned profile into the Karmic candidate.
# Keep the policy explicit in rank command receipts; Karmic is not qualified yet.
RECURRENT_CHECKPOINT_POLICY=${RECURRENT_CHECKPOINT_POLICY:-aligned}
# Qualified HC-off profile; explicit HC=1 is diagnostic only.
VLLM_QWEN3_8_FLASH_NEXT_HC_TP=${VLLM_QWEN3_8_FLASH_NEXT_HC_TP:-0}
[[ $MAX_NUM_SEQS == 4 && $NUM_SPECULATIVE_TOKENS == 3 ]] || { echo "QSA865 requires four sequences and MTP3" >&2; exit 78; }
hc_args=()
if [[ ${VLLM_QWEN3_8_FLASH_NEXT_HC_TP+x} ]]; then
  case $VLLM_QWEN3_8_FLASH_NEXT_HC_TP in
    0|1) hc_args=(-e "VLLM_QWEN3_8_FLASH_NEXT_HC_TP=$VLLM_QWEN3_8_FLASH_NEXT_HC_TP") ;;
    *) echo 'VLLM_QWEN3_8_FLASH_NEXT_HC_TP must be 0 or 1' >&2; exit 2 ;;
  esac
fi

case "${ROLE}" in
  head)
    NODE_RANK=0
    HOST_IP=${HOST_IP:-10.11.1.1}
    expected_host=dusty
    role_args=()
    ;;
  worker)
    NODE_RANK=1
    HOST_IP=${HOST_IP:-10.11.1.2}
    expected_host=kirby
    role_args=(--headless)
    ;;
  *) echo 'ROLE must be head, worker, or stop' >&2; exit 2 ;;
esac

if [[ "${TP_SIZE}" != 2 || "${NNODES}" != 2 ]]; then
  echo 'This Qwen profile requires TP_SIZE=2 and NNODES=2' >&2
  exit 78
fi
if [[ "${MASTER_ADDR}" != 10.11.1.1 ]]; then
  echo 'MASTER_ADDR must use dusty direct-pair address 10.11.1.1' >&2
  exit 78
fi
case "${HOST_IP}" in 10.11.1.1|10.11.1.2) ;; *) echo "invalid direct-pair HOST_IP=${HOST_IP}" >&2; exit 78 ;; esac
for value in MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS; do
  [[ "${!value}" =~ ^[1-9][0-9]*$ ]] || { echo "${value} must be a positive integer" >&2; exit 2; }
done
[[ "${NUM_SPECULATIVE_TOKENS}" =~ ^[0-9]+$ ]] || { echo 'NUM_SPECULATIVE_TOKENS must be non-negative' >&2; exit 2; }

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
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -e TMPDIR=/container-tmp
  -e MODEL="${MODEL}"
  -e MODEL_REVISION="${MODEL_REVISION}"
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}"
  -e PORT="${PORT}"
  -e TP="${TP_SIZE}"
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}"
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}"
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}"
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}"
  -e NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS}"
  -e LMCACHE_ENABLED=0
  -e INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND}"
  -e INSTANTTENSOR_DEBUG="${INSTANTTENSOR_DEBUG}"
  -e CUTE_DSL_ARCH=sm_121a
  -e TORCH_CUDA_ARCH_LIST=12.1a
  -e VLLM_HOST_IP="${HOST_IP}"
  -e NCCL_DEBUG="${NCCL_DEBUG}"
  -e NCCL_IB_DISABLE=0
  -e 'NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1'
  -e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
  -e NCCL_IB_TC=106
  -e NCCL_IB_MERGE_NICS=0
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1
  -e 'NCCL_PROTO=LL,Simple'
  -e 'NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1'
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1
  "${hc_args[@]}"
  --entrypoint /usr/local/bin/serve-qwen38-flash-next-karmic-spark.sh
  "${IMAGE}"
  --nnodes "${NNODES}"
  --node-rank "${NODE_RANK}"
  --master-addr "${MASTER_ADDR}"
  --master-port "${MASTER_PORT}"
  "${role_args[@]}"
  --recurrent-checkpoint-policy "${RECURRENT_CHECKPOINT_POLICY}"
  --max-cudagraph-capture-size 32
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY-RUN:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ "$(uname -m)" == aarch64 ]] || { echo 'This runner requires aarch64' >&2; exit 78; }
[[ "$(hostname -s)" == "${expected_host}" ]] || {
  echo "ROLE=${ROLE} must run on ${expected_host}, not $(hostname -s)" >&2
  exit 78
}
host_snapshot="${HF_CACHE}/hub/${MODEL_REPO_DIR}/snapshots/${MODEL_REVISION}"
index_file="${host_snapshot}/model.safetensors.index.json"
config_file="${host_snapshot}/config.json"
if [[ ! -r "${index_file}" || ! -r "${config_file}" ]]; then
  echo "Pinned model snapshot is incomplete: ${host_snapshot}" >&2
  exit 78
fi
python3 - "${index_file}" "${config_file}" "${EXPECTED_MODEL_BYTES}" "${EXPECTED_MODEL_SHARDS}" <<'PY'
if not __debug__:
    raise RuntimeError("Karmic verification requires Python assertions enabled")
import json
from pathlib import Path
import sys

index_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
index = json.loads(index_path.read_text())
config = json.loads(config_path.read_text())
shards = sorted(set(index["weight_map"].values()))
assert index["metadata"]["total_size"] == int(sys.argv[3])
assert len(shards) == int(sys.argv[4])
assert all((index_path.parent / shard).is_file() for shard in shards)
assert config["architectures"] == ["Qwen4ExpForConditionalGeneration", "Qwen3_8FlashNextForConditionalGeneration"]
assert config["model_type"] == "qwen4_exp"
assert config["quantization_config"]["quant_algo"] == "MIXED_PRECISION"
assert config["quantization_config"]["quant_method"] == "modelopt"
assert config["text_config"]["max_position_embeddings"] == 262144
PY
[[ "$(find "${host_snapshot}" -maxdepth 1 -xtype l | wc -l)" == 0 ]] || { echo 'Pinned model has broken links' >&2; exit 78; }

podman image exists "${IMAGE}" || { echo "Pinned image is absent: ${IMAGE}" >&2; exit 78; }
[[ -n "${EXPECTED_IMAGE_ID}" ]] || { echo 'EXPECTED_IMAGE_ID is unset; pin the built Karmic beta image (build receipt image.id)' >&2; exit 78; }
actual_image_id=$(podman image inspect "${IMAGE}" --format '{{.Id}}')
[[ "${actual_image_id#sha256:}" == "${EXPECTED_IMAGE_ID}" ]] || { echo 'Image ID mismatch' >&2; exit 78; }
labels=$(podman image inspect "${IMAGE}" --format '{{json .Config.Labels}}')
jq -e '
  ."vllm.source-tree" == "f9af0c3482812649f9591d802500309623383594" and
  ."vllm.source-commit" == "980d84efb8c36894374d8333c4c0b7f92d0fb825" and
  ."b12x.source-tree" == "1cfde5254dbe7613f5a549619e72c3172cbd861e" and
  ."b12x.source-commit" == "914921dad15d71ffc68ea329c344b71bf1ae7fa7" and
  ."local-inference.cutlass-dsl.version" == "4.7.1" and
  ."local-inference.flashinfer.commit" == "dbd6238c6655b98195fdf77f04bba6facf5a38a4" and
  ."lmcache.default" == "disabled" and
  ."local-inference.qwen.launcher.sha256" == "5551db04b4411d7c177466f956ef057ca10427316dba4a426cab41cdf56696fe"
' <<<"${labels}" >/dev/null || { echo 'Karmic provenance mismatch' >&2; exit 78; }


mkdir -p "${CACHE}" "${CACHE}/tmp"
if podman container exists "${NAME}" 2>/dev/null; then
  echo "Container already exists: ${NAME}; stop the pair before replacement" >&2
  exit 78
fi

"${cmd[@]}"
printf '%s ROLE=%s rank=%s/%s model=%s\n' \
  "${NAME}" "${ROLE}" "${NODE_RANK}" "${NNODES}" "${SERVED_MODEL_NAME}"
