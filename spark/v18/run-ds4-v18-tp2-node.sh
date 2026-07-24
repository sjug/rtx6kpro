#!/usr/bin/env bash
set -euo pipefail

# DS4 v18 (Gilded Gnosis) two-node TP2 launcher for the DGX Spark pair
# (rusty + toby).
#
# Identical launch contract to spark/v16/run-ds4-v16-tp2-node.sh: the image
# ships /usr/local/bin/serve-ds4-flash.sh (env-only interface, now from vLLM
# build/gilded-gnosis-v18-final-20260718 @ 264bce1). The only helper change
# since v16 is default HF revision pinning for the two hub checkpoints
# (standard 60d8d70, dspark 62af8ff) — both match the snapshots already
# cached on rusty/toby, so offline boot resolves the same weights as before.
# Override with MODEL_REVISION / STANDARD_MODEL_REVISION /
# DSPARK_MODEL_REVISION if the cache ever diverges.
#
# All Spark deviations from the reference x86 deployment are unchanged from
# the v16 wrapper (podman+CDI, two-node TP2 via EXTRA_VLLM_ARGS, RoCE NCCL
# env, CUTE_DSL_ARCH=sm_121a, ALLREDUCE_MODE=nccl, GPU_MEM=0.85, seqs=4) —
# see spark/v16/run-ds4-v16-tp2-node.sh for the full rationale table.
#
# Usage:
#   toby : ROLE=worker MODE=dspark BACKEND=lucifer-cutlass ./run-ds4-v18-tp2-node.sh
#   rusty: ROLE=head   MODE=dspark BACKEND=lucifer-cutlass ./run-ds4-v18-tp2-node.sh

ROLE=${ROLE:?set ROLE=head|worker}

IMAGE=${IMAGE:-localhost/voipmonitor/vllm:gilded-gnosis-v18-spark-sm121-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718}
NAME=${NAME:-ds4-v18-tp2}
PORT=${PORT:-8000}

HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
CACHE=${CACHE:-$HOME/.cache/vllm-ds4-v18}
mkdir -p "$CACHE" "$CACHE/tmp"

MASTER_ADDR=${MASTER_ADDR:-10.11.1.1}
MASTER_PORT=${MASTER_PORT:-25000}
case "$ROLE" in
  head)
    NODE_RANK=0
    HOST_IP=${HOST_IP:-10.11.1.1}
    ROLE_ARGS=""
    ;;
  worker)
    NODE_RANK=1
    HOST_IP=${HOST_IP:-10.11.1.2}
    ROLE_ARGS="--headless"
    ;;
  *)
    echo "ROLE must be head or worker" >&2
    exit 2
    ;;
esac

MODE=${MODE:-mtp2}
BACKEND=${BACKEND:-b12x-a16}
TP_SIZE=${TP_SIZE:-2}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
GPU_MEM=${GPU_MEM:-0.85}
ALLREDUCE_MODE=${ALLREDUCE_MODE:-nccl}

EXTRA_VLLM_ARGS="--nnodes 2 --node-rank ${NODE_RANK} --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} ${ROLE_ARGS}"

optional_env=(
  GRAPH MAX_CUDAGRAPH_CAPTURE_SIZE CUDAGRAPH_CAPTURE_SIZES
  MAX_NUM_BATCHED_TOKENS PREFIX_CACHE DCP_SIZE LOAD_FORMAT
  INSTANTTENSOR_BACKEND INDEXER_BACKEND KV_CACHE_DTYPE BLOCK_SIZE
  DSPARK_TOKENS DRAFT_SAMPLE_METHOD REJECTION_SAMPLE_METHOD
  B12X_PCIE_DMA ENABLE_FLASHINFER_AUTOTUNE SERVED_MODEL_NAME
  STANDARD_MODEL DSPARK_MODEL DRY_RUN
  MODEL_REVISION STANDARD_MODEL_REVISION DSPARK_MODEL_REVISION
)
opt_args=()
for key in "${optional_env[@]}"; do
  if [[ -n "${!key+x}" ]]; then opt_args+=(-e "${key}=${!key}"); fi
done

# Graceful cleanup of a leftover container: SIGTERM with a long grace period,
# never rm -f (its 10 s window falls back to SIGKILL mid-shutdown). If the old
# container won't die, the podman run below fails loudly on the name conflict.
if podman container exists "$NAME" 2>/dev/null; then
  podman stop -t 60 "$NAME" >/dev/null 2>&1 || true
  podman rm "$NAME" >/dev/null 2>&1 || true
fi

podman run -d \
  --name "$NAME" \
  --device nvidia.com/gpu=all \
  --device /dev/infiniband \
  --security-opt label=disable \
  --network host \
  --ipc host \
  --init \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=500000:500000 \
  -v "$HF_CACHE:/root/.cache/huggingface:rw" \
  -v "$CACHE:/cache:rw" \
  -v "$CACHE/tmp:/container-tmp:rw" \
  -e MODE="$MODE" \
  -e BACKEND="$BACKEND" \
  -e TP_SIZE="$TP_SIZE" \
  -e PORT="$PORT" \
  -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  -e GPU_MEMORY_UTILIZATION="$GPU_MEM" \
  -e ALLREDUCE_MODE="$ALLREDUCE_MODE" \
  -e EXTRA_VLLM_ARGS="$EXTRA_VLLM_ARGS" \
  -e CUTE_DSL_ARCH=sm_121a \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
  -e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}" \
  -e NCCL_IB_TC=106 \
  -e NCCL_PROTO=LL,Simple \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HUB_OFFLINE=1 \
  -e TMPDIR=/container-tmp \
  -e XDG_CACHE_HOME=/cache \
  "${opt_args[@]}" \
  --entrypoint /usr/local/bin/serve-ds4-flash.sh \
  "$IMAGE"

echo "$NAME ROLE=$ROLE rank=$NODE_RANK mode=$MODE backend=$BACKEND allreduce=$ALLREDUCE_MODE tp=$TP_SIZE port=$PORT seqs=$MAX_NUM_SEQS gpu_mem=$GPU_MEM gid=${NCCL_IB_GID_INDEX:-3} extra='$EXTRA_VLLM_ARGS'"
