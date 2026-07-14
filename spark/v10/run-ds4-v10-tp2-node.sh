#!/usr/bin/env bash
set -euo pipefail

# DS4 v10 (Fathomless Firmament) two-node TP2 launcher for DGX Spark.
#
# Faithful port of the v10 runtime contract from models/ds4dspark-v10.md:
# serve logic and defaults follow rtx6kpro/scripts/run-ds4-v9-server.sh (the
# contract the v10 wrappers inherit), backend rows and B12X env follow the
# doc tables. v10 validates the STANDARD checkpoint only (standard-mtp0 /
# standard-mtp2); the DSpark checkpoint is not validated on Fathomless and
# is intentionally not offered here — use the v9 deployment for DSpark.
#
# Deviations from the reference, strictly limited to SM121 / GB10 / 2-node:
#   1. podman rootless + CDI (--device nvidia.com/gpu=all) instead of
#      docker --gpus; no --shm-size with --ipc host (host /dev/shm is used);
#      ulimit nofile capped at the host hard limit 500000.
#   2. TP=2 across rusty+toby via --nnodes/--node-rank/--master-addr/
#      --headless: one GB10 (121 GB) cannot hold the 149 GB checkpoint.
#   3. NCCL over the back-to-back 200G CX7 RoCE links (reference is a
#      single host and sets NCCL_IB_DISABLE=1): NCCL_IB_HCA, RoCE-v2 GID
#      auto-detect, NCCL_IB_TC=106 (site requirement, switch-fabric class),
#      socket ifnames, VLLM_HOST_IP per node.
#   4. CUTE_DSL_ARCH=sm_121a (reference hardcodes sm_120a; SM121 cannot run
#      sm_120a cubins).
#   5. VLLM_ENABLE_PCIE_ALLREDUCE=0 (doc sets 1: single-node PCIe one-shot
#      allreduce, inapplicable across nodes).
#   6. Model snapshot resolved from the local HF cache (latest local
#      revision; per site policy the snapshot is not pinned). Both nodes
#      must have the same snapshot.
#   7. GPUS/topo-pin logic from the 16-GPU host is dropped (single GPU,
#      single socket per node), as are that host's NCCL tuning knobs
#      NCCL_P2P_LEVEL=SYS (no intra-node GPU pairs exist here) and
#      NCCL_PROTO=LL,LL128,Simple (LL128 is not certified off NVLink-class
#      links; NCCL picks protocols itself on RoCE).
#   8. GB10-scaled scheduler defaults: MAX_NUM_SEQS=4 (reference: 64) to
#      maximize KV-cache headroom, and GRAPH auto = 4 for mtp0 / 16 for
#      mtp2 (reference: 256/512). Graph capture beyond the max decode
#      batch (seqs, or seqs*(1+draft) with MTP) is wasted memory on a
#      121 GB UMA node. The doc's reduced validation cell
#      (MAX_NUM_SEQS=1 GRAPH=6) is env-selectable.
#   9. GPU_MEM default 0.85 (reference: 0.90). On UMA the GPU budget and
#      system RAM are one pool; 0.90 left the host with ~0 available
#      (measured 120/121 GB used). 0.85 keeps ~6 GB host headroom and
#      still yields ~26 GiB KV (~890k fp8 tokens, >3 full 262k contexts).

ROLE=${ROLE:?set ROLE=head|worker}

IMAGE=${IMAGE:-localhost/voipmonitor/vllm:fathomless-firmament-dspark-spark-sm121-vf5f4af3-b12x90172a5-cu132-20260709}
NAME=${NAME:-ds4-v10-tp2}
PORT=${PORT:-8000}

HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
STANDARD_REPO_DIR=models--deepseek-ai--DeepSeek-V4-Flash
SNAPSHOT=${SNAPSHOT:-$(ls "$HF_CACHE/hub/$STANDARD_REPO_DIR/snapshots/" | head -1)}
STANDARD_MODEL=${STANDARD_MODEL:-/root/.cache/huggingface/hub/$STANDARD_REPO_DIR/snapshots/$SNAPSHOT}
CACHE=${CACHE:-$HOME/.cache/vllm-ds4-v10}
CONTAINER_TMP=${CONTAINER_TMP:-$CACHE/tmp}

# --- Cluster networking (SM121 2-node deltas 2 & 3) ---
MASTER_ADDR=${MASTER_ADDR:-10.11.1.1}
NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1,enP2p1s0f1np1}
CONTROL_IF=${CONTROL_IF:-enp1s0f1np1}
NCCL_IB_TC=${NCCL_IB_TC:-106}

case "$ROLE" in
  head)   NODE_RANK=0; VLLM_HOST_IP=${VLLM_HOST_IP:-10.11.1.1}; HEADLESS_ARGS=() ;;
  worker) NODE_RANK=1; VLLM_HOST_IP=${VLLM_HOST_IP:-10.11.1.2}; HEADLESS_ARGS=(--headless) ;;
  *) echo "Unknown ROLE=$ROLE (head|worker)" >&2; exit 2 ;;
esac

NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-}
if [[ -z "$NCCL_IB_GID_INDEX" ]]; then
  first_hca=${NCCL_IB_HCA%%,*}
  for i in $(seq 0 15); do
    t=$(cat "/sys/class/infiniband/$first_hca/ports/1/gid_attrs/types/$i" 2>/dev/null || true)
    g=$(cat "/sys/class/infiniband/$first_hca/ports/1/gids/$i" 2>/dev/null || true)
    if [[ "$t" == *"RoCE v2"* && "$g" == *0000:0000:0000:0000:0000:ffff:* ]]; then
      NCCL_IB_GID_INDEX=$i
      break
    fi
  done
fi

# --- Reference contract defaults (run-ds4-v9-server.sh) ---
TP=${TP:-2}
BACKEND=${BACKEND:-b12x-a16}        # b12x-a16|b12x-a8|b12x-a8-dglin (doc rows)
MODE=${MODE:-standard-mtp0}         # standard-mtp0 | standard-mtp2
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_BATCHED=${MAX_BATCHED:-8192}
GPU_MEM=${GPU_MEM:-0.85}
GRAPH=${GRAPH:-auto}
PREFIX_CACHE=${PREFIX_CACHE:-1}
MTP_TOKENS_WAS_SET=0
if [[ -n "${MTP_TOKENS+x}" ]]; then MTP_TOKENS_WAS_SET=1; fi
MTP_TOKENS=${MTP_TOKENS:-2}
SAMPLE=${SAMPLE:-probabilistic}

case "$MODE" in
  standard-mtp0)
    SERVED_MODEL=DeepSeek-V4-Flash
    SPEC_ARGS=()
    if [[ "$GRAPH" == "auto" ]]; then GRAPH=4; fi
    ;;
  standard-mtp2|standard-mtp3)
    SERVED_MODEL=DeepSeek-V4-Flash
    if [[ "$MODE" == "standard-mtp3" && "$MTP_TOKENS_WAS_SET" == "0" ]]; then
      MTP_TOKENS=3
    fi
    if [[ "$BACKEND" == b12x* ]]; then
      SPEC_JSON=$(printf '{"method":"mtp","num_speculative_tokens":%s,"draft_sample_method":"%s","moe_backend":"b12x"}' "$MTP_TOKENS" "$SAMPLE")
    else
      SPEC_JSON=$(printf '{"method":"mtp","num_speculative_tokens":%s,"draft_sample_method":"%s"}' "$MTP_TOKENS" "$SAMPLE")
    fi
    SPEC_ARGS=(--speculative-config "$SPEC_JSON")
    if [[ "$GRAPH" == "auto" ]]; then GRAPH=16; fi
    ;;
  *)
    echo "Unknown MODE=$MODE (v10 validates standard-mtp0/standard-mtp2)" >&2
    exit 2
    ;;
esac

MODEL=$STANDARD_MODEL
test -f "$HF_CACHE/hub/$STANDARD_REPO_DIR/snapshots/$SNAPSHOT/config.json"

# Doc "B12X common env"; PCIe allreduce flipped off per deviation 5.
b12x_common_env=(
  -e VLLM_USE_B12X_WO_PROJECTION=1
  -e VLLM_USE_B12X_MHC=1
  -e VLLM_USE_B12X_MOE=1
  -e VLLM_USE_B12X_SPARSE_INDEXER=1
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0
  -e B12X_MLA_SM120_UNIFIED=1
  -e B12X_MHC_MAX_TOKENS=16384
  -e B12X_DENSE_SPLITK_TURBO=1
  -e B12X_W4A16_TC_DECODE=1
)

case "$BACKEND" in
  b12x|b12x-a16)
    BACKEND_ARGS=(--attention-backend B12X_MLA_SPARSE --moe-backend b12x --linear-backend b12x)
    BACKEND_ENV=("${b12x_common_env[@]}"
      -e VLLM_USE_B12X_FP8_GEMM=1 -e B12X_MOE_FORCE_A8=0 -e B12X_MOE_FORCE_A16=1)
    ;;
  b12x-a8)
    BACKEND_ARGS=(--attention-backend B12X_MLA_SPARSE --moe-backend b12x --linear-backend b12x)
    BACKEND_ENV=("${b12x_common_env[@]}"
      -e VLLM_USE_B12X_FP8_GEMM=1 -e B12X_MOE_FORCE_A8=1 -e B12X_MOE_FORCE_A16=0)
    ;;
  b12x-a8-dglin)
    BACKEND_ARGS=(--attention-backend B12X_MLA_SPARSE --moe-backend b12x)
    BACKEND_ENV=("${b12x_common_env[@]}"
      -e VLLM_USE_B12X_FP8_GEMM=0 -e B12X_MOE_FORCE_A8=1 -e B12X_MOE_FORCE_A16=0)
    ;;
  *)
    echo "Unknown BACKEND=$BACKEND (v10 validates b12x-a16|b12x-a8|b12x-a8-dglin)" >&2
    exit 2
    ;;
esac

PREFIX_ARGS=(--enable-prefix-caching)
if [[ "$PREFIX_CACHE" != "1" ]]; then
  PREFIX_ARGS=(--no-enable-prefix-caching)
fi

mkdir -p \
  "$CACHE/vllm" "$CACHE/tilelang/tmp" "$CACHE/tvm" "$CACHE/triton" \
  "$CACHE/torchinductor" "$CACHE/torch_extensions" "$CACHE/flashinfer" \
  "$CONTAINER_TMP"

podman rm -f "$NAME" >/dev/null 2>&1 || true

podman run -d \
  --name "$NAME" \
  --device nvidia.com/gpu=all \
  --device /dev/infiniband \
  --ipc host \
  --network host \
  --init \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --ulimit nofile=500000:500000 \
  -v "$HF_CACHE:/root/.cache/huggingface:ro" \
  -v "$CACHE:/cache:rw" \
  -v "$CONTAINER_TMP:/container-tmp:rw" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e VLLM_HOST_IP="$VLLM_HOST_IP" \
  -e GLOO_SOCKET_IFNAME="$CONTROL_IF" \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e CUTE_DSL_ARCH=sm_121a \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$NCCL_IB_HCA" \
  ${NCCL_IB_GID_INDEX:+-e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX"} \
  -e NCCL_IB_TC="$NCCL_IB_TC" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TMPDIR=/container-tmp \
  -e XDG_CACHE_HOME=/cache \
  -e VLLM_CACHE_DIR=/cache/vllm \
  -e TILELANG_CACHE_DIR=/cache/tilelang \
  -e TILELANG_TMP_DIR=/cache/tilelang/tmp \
  -e TVM_CACHE_DIR=/cache/tvm \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  "${BACKEND_ENV[@]}" \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; exec vllm serve "$@"' \
  -- "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --load-format auto \
  --tensor-parallel-size "$TP" \
  --decode-context-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --max-cudagraph-capture-size "$GRAPH" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  --enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --enable-prompt-tokens-details \
  --enable-force-include-usage \
  --enable-request-id-headers \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort=high \
  --nnodes 2 \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port 25000 \
  "${HEADLESS_ARGS[@]}" \
  "${SPEC_ARGS[@]}" \
  "${BACKEND_ARGS[@]}" \
  "${PREFIX_ARGS[@]}"

echo "$NAME ROLE=$ROLE rank=$NODE_RANK $BACKEND $MODE TP=$TP port=$PORT graph=$GRAPH seqs=$MAX_NUM_SEQS gid=${NCCL_IB_GID_INDEX:-auto} model=$SERVED_MODEL"
