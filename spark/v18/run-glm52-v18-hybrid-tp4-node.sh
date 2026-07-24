#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2 v18 (Gilded Gnosis) hybrid four-node TP4 launcher for the DGX Spark cluster
# (sparky + buddy + rocky + lucky, one GB10 per node).
#
# Serves madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid per models/glm5.2_v17.md.
# Unlike the DS4 v16 launcher, this REPLICATES the vllm serve command
# (v10-launcher style) instead of delegating to the in-image helper:
# serve-glm52-hybrid-v18.sh -> serve-glm52-v18.sh -> serve-glm52-v16.sh have no EXTRA_VLLM_ARGS
# escape hatch and unconditionally export single-node x86 values
# (CUTE_DSL_ARCH=sm_120a, NCCL_IB_DISABLE=1, VLLM_ENABLE_PCIE_ALLREDUCE=1)
# that would override anything passed with -e. Env block and argv below are
# a faithful copy of the v18 launcher chain (blackwell-llm-docker @ 7f3cbc6) with the hybrid
# preset applied; keep them in sync if the pin moves. Upstream candidate:
# add EXTRA_VLLM_ARGS + default-if-unset env to serve-glm52-v16.sh.
#
# Deviations from the x86 v17 reference, limited to SM121 / GB10 / 4-node:
#   1. podman rootless + CDI (--device nvidia.com/gpu=all), /dev/infiniband
#      for RoCE; no --shm-size with --ipc host; nofile capped at 500000.
#   2. TP4 across four nodes via --nnodes/--node-rank/--master-addr/
#      --headless (torch-distributed, no Ray): one GB10 (121 GB) cannot
#      hold the ~TP4 checkpoint shard plus peers.
#   3. DCP pinned to 1: the B12X DCP pool exchanges CUDA IPC handles,
#      which do not cross nodes. DCP>1 is refused.
#   4. NCCL over the two SWITCHED 200 GbE CX7 rails (helper sets
#      NCCL_IB_DISABLE=1): IB on, HCA list, RoCE-v2 GID auto-detect,
#      TC 106, socket ifnames, per-node VLLM_HOST_IP. NCCL_PROTO=LL,Simple
#      (LL128 uncertified off NVLink-class links). NCCL_P2P_LEVEL dropped
#      (no intra-node GPU pairs). The f1 back-to-back rails are excluded —
#      only pairwise-reachable (see the networking block below).
#   5. CUTE_DSL_ARCH=sm_121a, TORCH_CUDA_ARCH_LIST=12.1a (helper hardcodes
#      sm_120a/12.0a; SM121 refuses SM120 cubins).
#   6. VLLM_ENABLE_PCIE_ALLREDUCE=0 and VLLM_USE_B12X_PCIE_DMA=0 (helper:
#      1/1) — no PCIe peers exist across nodes.
#   7. GPU_MEMORY_UTILIZATION=0.85 (helper 0.90, hybrid preset 0.96 —
#      dedicated-VRAM assumptions; on 121 GB UMA, 0.90 starves the host).
#   8. MAX_MODEL_LEN=131072 (hybrid preset 262144) for first bring-up;
#      raise after KV headroom is measured. MAX_NUM_SEQS=8 and
#      GRAPH=seqs*4 follow the helper's own derivation.
#   9. MTP default 0 (the only v17-validated mode).
#
# The checkpoint (revision 68babde) must be staged in the HF cache on ALL
# four nodes before launch (HF_HUB_OFFLINE=1), or pass MODEL=/path.
#
# Usage (workers first, head last; same command everywhere — the rank is
# derived from the hostname, MASTER_ADDR defaults to sparky's switch IP):
#   buddy : ./run-glm52-v17-hybrid-tp4-node.sh
#   rocky : ./run-glm52-v17-hybrid-tp4-node.sh
#   lucky : ./run-glm52-v17-hybrid-tp4-node.sh
#   sparky: ./run-glm52-v17-hybrid-tp4-node.sh
#   endpoint: sparky:8000, served model GLM-5.2-MXFP8-NVFP4-NF3-Hybrid

IMAGE=${IMAGE:-localhost/voipmonitor/vllm:gilded-gnosis-v18-spark-sm121-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718}
NAME=${NAME:-glm52-v18-tp4}
PORT=${PORT:-8000}
NNODES=${NNODES:-4}

if [[ -z "${NODE_RANK:-}" ]]; then
  case "$(hostname -s)" in
    sparky) NODE_RANK=0 ;;
    buddy)  NODE_RANK=1 ;;
    rocky)  NODE_RANK=2 ;;
    lucky)  NODE_RANK=3 ;;
    *) echo "Unknown host $(hostname -s): set NODE_RANK=0..$((NNODES - 1)) (0 = head)" >&2; exit 2 ;;
  esac
fi
HEADLESS_ARGS=()
if [[ "$NODE_RANK" != "0" ]]; then HEADLESS_ARGS=(--headless); fi

MASTER_ADDR=${MASTER_ADDR:-10.11.11.1}
MASTER_PORT=${MASTER_PORT:-25000}

DCP=${DCP:-1}
if [[ "$DCP" != "1" ]]; then
  echo "DCP=$DCP unsupported: B12X DCP pool is CUDA-IPC (single-node only); cross-node TP4 requires DCP=1" >&2
  exit 2
fi

# --- Cluster networking (deviations 2 & 4) ---
# Surveyed 2026-07-14: each node has two SWITCHED 200G rails on the f0
# ports — 10.11.11.{1,2,3,4} on enp1s0f0np0 (rocep1s0f0) and
# 10.11.12.{1,2,3,4} on enP2p1s0f0np0 (roceP2p1s0f0); node order
# sparky/buddy/lucky/rocky. The f1 ports are BACK-TO-BACK pairs
# (sparky<->buddy, lucky<->rocky on 10.11.1.x/10.11.2.x) and are NOT
# all-to-all — never put them in NCCL_IB_HCA for 4-node TP.
NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f0,roceP2p1s0f0}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f0np0,enP2p1s0f0np0}
CONTROL_IF=${CONTROL_IF:-enp1s0f0np0}
NCCL_IB_TC=${NCCL_IB_TC:-106}

VLLM_HOST_IP=${VLLM_HOST_IP:-$(ip -4 -o addr show dev "$CONTROL_IF" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)}
if [[ -z "$VLLM_HOST_IP" ]]; then
  echo "Could not derive VLLM_HOST_IP from $CONTROL_IF; set VLLM_HOST_IP or CONTROL_IF" >&2
  exit 2
fi

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

# --- Model (v17 pinned revision), resolved from the local HF cache ---
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
HYBRID_REPO_DIR=models--madeby561--GLM-5.2-MXFP8-NVFP4-NF3-Hybrid
SNAPSHOT=${SNAPSHOT:-68babde27a97a4c980c2494e830dd424975cd5a3}
MODEL=${MODEL:-/root/.cache/huggingface/hub/$HYBRID_REPO_DIR/snapshots/$SNAPSHOT}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-GLM-5.2-MXFP8-NVFP4-NF3-Hybrid}
if [[ "$MODEL" == /root/.cache/huggingface/* ]] \
  && ! test -f "$HF_CACHE/hub/$HYBRID_REPO_DIR/snapshots/$SNAPSHOT/config.json"; then
  echo "Checkpoint not staged: $HF_CACHE/hub/$HYBRID_REPO_DIR/snapshots/$SNAPSHOT" >&2
  echo "Stage madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid @ $SNAPSHOT on all $NNODES nodes (or pass MODEL=/path)." >&2
  exit 2
fi

CACHE=${CACHE:-$HOME/.cache/vllm-glm52-v18}
CONTAINER_TMP=${CONTAINER_TMP:-$CACHE/tmp}

# --- Serving profile (helper defaults + hybrid preset + deviations 7-9) ---
TP=${TP:-4}
MTP=${MTP:-0}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
GRAPH=${GRAPH:-$((MAX_NUM_SEQS * 4))}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-2048}
GPU_MEM=${GPU_MEM:-0.85}
# 78-char pattern, verbatim from the helper; the serving contract depends on
# the exact string.
INDEX_TOPK_PATTERN=${INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}
[[ "${#INDEX_TOPK_PATTERN}" -eq 78 ]] || { echo "INDEX_TOPK_PATTERN must be 78 chars" >&2; exit 2; }

# TP4:DCP1 resolves the helper's DCP_PREFILL_WORKSPACE=auto gate to 0.
DCP_PROJECT_MIN_PREFILL_TOKENS=1024
if ((GRAPH > DCP_PROJECT_MIN_PREFILL_TOKENS)); then
  DCP_PROJECT_MIN_PREFILL_TOKENS=$GRAPH
fi

SPEC_ARGS=()
if [[ "$MTP" != "0" ]]; then
  SPEC_JSON=$(printf '{"model":"%s","method":"mtp","num_speculative_tokens":%s,"moe_backend":"b12x","draft_sample_method":"probabilistic"}' "$MODEL" "$MTP")
  SPEC_ARGS=(--speculative-config "$SPEC_JSON")
fi

HF_OVERRIDES=$(printf '{"use_index_cache":true,"index_topk_pattern":"%s"}' "$INDEX_TOPK_PATTERN")
QUANT_CONFIG='{"linear":{"weight":"mxfp8"},"shared_experts":{"weight":"mxfp8"},"ignore":["re:^model\\.layers\\.0\\.","re:.*\\.self_attn\\.indexer\\.","re:.*\\.mlp\\.gate$","model.layers.78.eh_proj","lm_head"]}'

mkdir -p \
  "$CACHE/vllm" "$CACHE/tilelang/tmp" "$CACHE/tvm" "$CACHE/triton" \
  "$CACHE/torchinductor" "$CACHE/torch_extensions" "$CACHE/flashinfer" \
  "$CONTAINER_TMP"

# Graceful teardown of a stale instance: never rm -f a running server.
if podman container exists "$NAME" 2>/dev/null; then
  podman stop -t 60 "$NAME" >/dev/null 2>&1 || true
  podman rm "$NAME" >/dev/null 2>&1 || true
fi

podman run -d \
  --name "$NAME" \
  --device nvidia.com/gpu=all \
  --device /dev/infiniband \
  --security-opt label=disable \
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
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_121a \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e SAFETENSORS_FAST_GPU=1 \
  -e INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-BUFFERED}" \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e VLLM_USE_B12X_MHC=1 \
  -e VLLM_NF3_GRID188_DECODE="${NF3_GRID188:-1}" \
  -e VLLM_DCP_QUERY_SPLIT=0 \
  -e VLLM_B12X_MLA_CKV_GATHER=0 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_USE_B12X_DCP_A2A=1 \
  -e VLLM_DCP_A2A_MAX_TOKENS=16 \
  -e VLLM_DCP_A2A_LARGE_BACKEND=ag_rs \
  -e VLLM_DCP_PROJECT_BEFORE_MERGE=0 \
  -e VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS="$DCP_PROJECT_MIN_PREFILL_TOKENS" \
  -e VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e VLLM_USE_B12X_PCIE_DMA=0 \
  -e VLLM_PCIE_DMA_FP8=0 \
  -e B12X_PCIE_DMA_FP8=0 \
  -e VLLM_DCP_GLOBAL_TOPK=1 \
  -e VLLM_DCP_SHARD_DRAFT=1 \
  -e B12X_MLA_SM120_UNIFIED=1 \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_W4A8_TINY_DECODE=1 \
  -e B12X_MOE_FORCE_A8=0 \
  -e B12X_MOE_FORCE_A16=1 \
  -e NCCL_PROTO=LL,Simple \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$NCCL_IB_HCA" \
  ${NCCL_IB_GID_INDEX:+-e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX"} \
  -e NCCL_IB_TC="$NCCL_IB_TC" \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e GLOO_SOCKET_IFNAME="$CONTROL_IF" \
  -e VLLM_HOST_IP="$VLLM_HOST_IP" \
  -e LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4 \
  -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4 \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
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
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; exec vllm serve "$@"' \
  -- "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size "$TP" \
  --decode-context-parallel-size "$DCP" \
  --kv-cache-dtype nvfp4_ds_mla \
  --attention-backend B12X_MLA_SPARSE \
  --moe-backend b12x \
  --quantization nvfp4_nf3_hybrid \
  --quantization-config "$QUANT_CONFIG" \
  --load-format instanttensor \
  -cc.pass_config.fuse_allreduce_rms=True \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --max-cudagraph-capture-size "$GRAPH" \
  --async-scheduling \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
  --enable-prompt-tokens-details \
  --enable-force-include-usage \
  --enable-request-id-headers \
  --hf-overrides "$HF_OVERRIDES" \
  --nnodes "$NNODES" \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  "${HEADLESS_ARGS[@]}" \
  "${SPEC_ARGS[@]}"

echo "$NAME rank=$NODE_RANK/$NNODES master=$MASTER_ADDR host_ip=$VLLM_HOST_IP tp=$TP dcp=$DCP mtp=$MTP port=$PORT seqs=$MAX_NUM_SEQS graph=$GRAPH len=$MAX_MODEL_LEN gpu_mem=$GPU_MEM gid=${NCCL_IB_GID_INDEX:-auto}"
