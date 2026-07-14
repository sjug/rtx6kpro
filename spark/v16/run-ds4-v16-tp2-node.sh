#!/usr/bin/env bash
set -euo pipefail

# DS4 v16 two-node TP2 launcher for the DGX Spark pair (rusty + toby).
#
# Unlike the v10 launcher, this no longer replicates the vllm serve command:
# the v16 image ships the launch contract at /usr/local/bin/serve-ds4-flash.sh
# (env-only interface, from vLLM codex/fathomless-firmament-v16-unified @
# 8f86f42). This wrapper only does container placement and the Spark-specific
# environment, then delegates everything else to the in-image helper.
#
# Deviations from the reference single-host x86 deployment
# (models/ds4dspark-v10.md, v16 revision):
#   1. podman + CDI GPU + /dev/infiniband instead of docker --gpus; no
#      --shm-size (invalid with --ipc host under podman); nofile capped at
#      500000 (rootless hard limit).
#   2. Two-node TP2: one GB10 per node, model (149 GB) > 121 GB node. The
#      helper is single-node; the distributed flags ride its sanctioned
#      EXTRA_VLLM_ARGS escape hatch (--nnodes/--node-rank/--master-addr/
#      --master-port, --headless on the worker).
#   3. NCCL over the 200 GbE CX7 RoCE pair replaces the helper's
#      single-host defaults (it sets NCCL_IB_DISABLE=1): IB enabled, HCA
#      list, RoCE-v2 GID 3, TC 106 (DSCP 26, switch-fabric compatibility),
#      socket ifnames, per-node VLLM_HOST_IP. NCCL_PROTO=LL,Simple keeps
#      the helper's spirit minus LL128 (uncertified off NVLink-class
#      links). NCCL_P2P_LEVEL is left at the helper default — a no-op
#      with one GPU per node.
#   4. CUTE_DSL_ARCH=sm_121a (helper default sm_120a; CuTe/FlashInfer
#      refuse SM120 cubins on SM121).
#   5. ALLREDUCE_MODE=nccl (helper default b12x = intra-node PCIe
#      allreduce; no PCIe peer exists across nodes).
#   6. GPU_MEMORY_UTILIZATION=0.85 (helper defaults 0.91-0.953 assume
#      dedicated VRAM; on 121 GB UMA 0.90 left the host 0 bytes available).
#   7. MAX_NUM_SEQS=4 (helper default 64) to maximize KV headroom; the
#      graph cap stays on the helper's own derivation (mtp0 16 / mtp2-3 32
#      / dspark 24 at seqs=4 — full graphs self-bound at the real decode
#      width, the excess only widens the cheap piecewise ladder).
#
# Usage:
#   toby : ROLE=worker MODE=mtp2 BACKEND=b12x-a16 ./run-ds4-v16-tp2-node.sh
#   rusty: ROLE=head   MODE=mtp2 BACKEND=b12x-a16 ./run-ds4-v16-tp2-node.sh

ROLE=${ROLE:?set ROLE=head|worker}

IMAGE=${IMAGE:-localhost/voipmonitor/vllm:fathomless-firmament-v16-spark-sm121-vllm8f86f42-b12xfe06f49-fi801d57a-cu132-20260714}
NAME=${NAME:-ds4-v16-tp2}
PORT=${PORT:-8000}

HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
CACHE=${CACHE:-$HOME/.cache/vllm-ds4-v16}
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
)
opt_args=()
for key in "${optional_env[@]}"; do
  if [[ -n "${!key+x}" ]]; then opt_args+=(-e "${key}=${!key}"); fi
done

podman rm -f "$NAME" >/dev/null 2>&1 || true

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
