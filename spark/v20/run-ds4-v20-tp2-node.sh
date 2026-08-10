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

# Production image since 2026-08-09: r33-spark (upstream r33 composition +
# SM121 overlay; our #234 q_len guard now rides UPSTREAM in the manifest;
# adds #245 indexer query-split fix, #251 B12X graph channels + DSpark
# context-KV FULL graph, #252/#254 offload ordering, FlashInfer 0.6.18
# with the #3932 quantfix mainlined - the sjug FlashInfer pin is retired).
# The #235 reasoning contract behavior (default effort=high is real) is
# unchanged from r28-spark. Rollback: r28-spark (vllm47d1950-...-20260804).
IMAGE=${IMAGE:-localhost/voipmonitor/vllm:gilded-gnosis-v20-r33-spark-sm121-vllm28e8eaf-b12x06db0f4-fi1ac6942-cu132-20260808}
NAME=${NAME:-ds4-0731-tp2}
PORT=${PORT:-8000}

# Production model since 2026-07-31: DeepSeek-V4-Flash-0731 (official
# release; DSpark module attached). Revision = the staged local snapshot.
DSPARK_MODEL=${DSPARK_MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}
MODEL_REVISION=${MODEL_REVISION:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb}
DSPARK_MODEL_REVISION=${DSPARK_MODEL_REVISION:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}
# Alias (2026-08-04): also serve the bare upstream name so clients using the
# GG default ("DeepSeek-V4-Flash") resolve. Appended via EXTRA_VLLM_ARGS;
# argparse last-wins over the helper's single-name flag, serving BOTH names.
SERVED_MODEL_ALIAS=${SERVED_MODEL_ALIAS:-DeepSeek-V4-Flash}

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
# dense capture-size derivation below active on bare launches.
# K=5 supersedes the 0731 card's K=7: same-pair A/B on GB10 (2026-08-03,
# v20p3, probabilistic both arms) measured K5 +10.2% decode geomean over
# 15 cells (14/15 cells) and +8.8% coding-peak mean — acceptance decays
# to ~4-8% by draft positions 6-7, so K7's tail is wasted verify width.
# Matches upstream r24's RTX PRO 6000 data (217.8 vs 192.1 tok/s) and
# avoids their still-open K7 long-context quality flag. JSONs:
# benchmark_results-ds4-0731-v20p3-k{5,7}prob_*.json.
DSPARK_TOKENS=${DSPARK_TOKENS:-5}
# Deliberate deviation from the 0731 card's draft_sample_method=greedy:
# A/B on this pair (2026-07-31, K=7, identical stack) measured
# probabilistic faster on BOTH coding-peak (59.2 vs 52.8 tok/s mean of 8)
# and the 15-cell decode sweep (+9.9% geomean), and it keeps the lossless
# speculative-sampling guarantee. Greedy's one +38% probe was a single
# boilerplate-code anecdote that did not survive repetition. JSONs:
# benchmark_results-ds4-0731-{v20p2-k7*,codingpeak-k7*}.
DRAFT_SAMPLE_METHOD=${DRAFT_SAMPLE_METHOD:-probabilistic}
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
EXTRA_VLLM_ARGS="--nnodes 2 --node-rank ${NODE_RANK} --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} ${ROLE_ARGS} --served-model-name ${SERVED_MODEL_NAME} ${SERVED_MODEL_ALIAS} ${EXTRA_VLLM_ARGS_APPEND:-}"

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
