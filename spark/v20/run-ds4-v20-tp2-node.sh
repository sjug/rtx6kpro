#!/usr/bin/env bash
set -euo pipefail

# DS4 v20p0 (GG v20 release-candidate pins, SparkInfer rename) two-node TP2
# launcher for the DGX Spark pair (rusty + toby).
#
# Same launch contract as spark/v19/run-ds4-v19-tp2-node.sh: the image ships
# /usr/local/bin/serve-ds4-flash.sh (env-only interface, from the v20
# release branch @ 2167295c). SparkInfer keeps B12X_* env shims, so the
# BACKEND/env plumbing below is unchanged. No fused-indexer overlay: the
# Int64 fix (upstream 50ae819) is baked into the image.
#
# Luke's recipe translates as (see spark/v19 build script header):
#   VLLM_ENABLE_DSPARK=1                          -> MODE=dspark
#   NUM_SPECULATIVE_TOKENS=N                      -> DSPARK_TOKENS=N
#   DCP_SIZE=1                                    -> helper default (dspark
#                                                    rejects DCP != 1)
#   DSPARK_SPS_CURVE=auto                         -> DSPARK_CAPACITY=1
#                                                    DSPARK_SPS_CURVE=auto
#   VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=1  -> DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=1
#   VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH=1             -> DSPARK_DYNAMIC_DRAFT_DEPTH=1
#   his exported b12x flag set (incl. FORCE_A8)   -> BACKEND=b12x-a8
#
# Usage (safer config, Luke 2026-07-19: depth 6, no dynamic knobs):
#   toby : ROLE=worker MODE=dspark BACKEND=b12x-a8 DSPARK_TOKENS=6 ./run-ds4-v19-tp2-node.sh
#   rusty: ROLE=head   MODE=dspark BACKEND=b12x-a8 DSPARK_TOKENS=6 ./run-ds4-v19-tp2-node.sh
# Experimental config (his current daily driver): add
#   DSPARK_TOKENS=7 DSPARK_CAPACITY=1 DSPARK_SPS_CURVE=auto \
#   DSPARK_DYNAMIC_DRAFT_DEPTH=1 DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=1
#
# Deliberate deviations from his profile, both env-overridable:
#   ALLREDUCE_MODE=nccl (the helper's b12x PCIe-allreduce default is a no-op
#     on multi-node: vllm/config/parallel.py forces
#     disable_custom_all_reduce=True when nnodes>1, so every mode runs NCCL
#     over RoCE anyway — nccl declares the real path)
#   GPU_MEM=0.87 (measured production ceiling on the pair since v16; the
#     0.82 from their spark profile leaves ~250k KV tokens on the table)

ROLE=${ROLE:?set ROLE=head|worker}

IMAGE=${IMAGE:-localhost/voipmonitor/vllm:gilded-gnosis-v20p1-ds4-spark-sm121-vllm2167295-si6a92bcc-fi7ad08da-cu132-20260722}
NAME=${NAME:-ds4-v20-tp2}
PORT=${PORT:-8000}

HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
CACHE=${CACHE:-$HOME/.cache/vllm-ds4-v20}
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

MODE=${MODE:-dspark}
BACKEND=${BACKEND:-b12x-a8}
# Production K. Defaulting it here (not just in launch commands) keeps the
# dense capture-size derivation below active on bare launches; without it
# the helper falls back to K=5 silently.
DSPARK_TOKENS=${DSPARK_TOKENS:-6}
TP_SIZE=${TP_SIZE:-2}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}

# DSpark verify steps are exact multiples of (DSPARK_TOKENS+1) tokens. vLLM's
# default capture ladder tops out below max_num_seqs*(K+1) for K=6 (28 > 24),
# dropping cc=4 verify steps off FULL graphs (measured: cc4 halved, 45 vs 79+
# tok/s). Capture every verify multiple plus the generic rungs unless the
# caller passes an explicit list.
if [[ "$MODE" == dspark && -z "${CUDAGRAPH_CAPTURE_SIZES+x}" && -n "${DSPARK_TOKENS+x}" ]]; then
  k1=$((DSPARK_TOKENS + 1))
  CUDAGRAPH_CAPTURE_SIZES=$(
    { printf '%s\n' 1 2 4 8 16 24; seq "$k1" "$k1" $((MAX_NUM_SEQS * k1)); } \
      | sort -n -u | paste -sd,
  )
fi
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
GPU_MEM=${GPU_MEM:-0.87}
ALLREDUCE_MODE=${ALLREDUCE_MODE:-nccl}

# EXTRA_VLLM_ARGS_APPEND: extra `vllm serve` CLI flags for debug runs (e.g.
# "--enforce-eager" for CUDA_LAUNCH_BLOCKING kernel attribution).
EXTRA_VLLM_ARGS="--nnodes 2 --node-rank ${NODE_RANK} --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} ${ROLE_ARGS} ${EXTRA_VLLM_ARGS_APPEND:-}"

optional_env=(
  GRAPH MAX_CUDAGRAPH_CAPTURE_SIZE CUDAGRAPH_CAPTURE_SIZES
  MAX_NUM_BATCHED_TOKENS PREFIX_CACHE DCP_SIZE LOAD_FORMAT
  INSTANTTENSOR_BACKEND INDEXER_BACKEND KV_CACHE_DTYPE BLOCK_SIZE
  DSPARK_TOKENS DRAFT_SAMPLE_METHOD REJECTION_SAMPLE_METHOD
  DSPARK_CAPACITY DSPARK_CAPACITY_VERIFICATION_MODE
  DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE DSPARK_ONLINE_STS DSPARK_SPS_CURVE
  DSPARK_CONFIDENCE_THRESHOLD DSPARK_BUDGET_FRAC
  DSPARK_CONFIDENCE_TEMPERATURE DSPARK_SPS_OVERHEAD_MS
  DSPARK_FP8_DRAFT_HEAD DSPARK_DYNAMIC_DRAFT_DEPTH
  DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW DSPARK_DRAFT_ATTENTION_BACKEND
  VLLM_USE_B12X_MHC B12X_MHC_MAX_TOKENS
  B12X_PCIE_DMA ENABLE_FLASHINFER_AUTOTUNE SERVED_MODEL_NAME
  STANDARD_MODEL DSPARK_MODEL DRY_RUN
  MODEL_REVISION STANDARD_MODEL_REVISION DSPARK_MODEL_REVISION
)
opt_args=()
for key in "${optional_env[@]}"; do
  if [[ -n "${!key+x}" ]]; then opt_args+=(-e "${key}=${!key}"); fi
done

# Optional extra podman-run args, e.g. a read-only bind mount overlaying a
# python-only hotfix onto the image (label any results accordingly — the
# running code then differs from the image digest):
#   EXTRA_PODMAN_ARGS="-v $HOME/v19/kernel-fixed.py:/opt/venv/.../kernel.py:ro"
# shellcheck disable=SC2206
extra_podman_args=( ${EXTRA_PODMAN_ARGS:-} )

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
  -e VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}" \
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
  "${extra_podman_args[@]}" \
  --entrypoint /usr/local/bin/serve-ds4-flash.sh \
  "$IMAGE"

echo "$NAME ROLE=$ROLE rank=$NODE_RANK mode=$MODE backend=$BACKEND allreduce=$ALLREDUCE_MODE tp=$TP_SIZE port=$PORT seqs=$MAX_NUM_SEQS gpu_mem=$GPU_MEM gid=${NCCL_IB_GID_INDEX:-3} extra='$EXTRA_VLLM_ARGS'"
