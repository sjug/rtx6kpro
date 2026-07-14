#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4-Flash-DSpark TP2 across two DGX Sparks (rusty + toby),
# Ray-free: vLLM native --nnodes/--node-rank/--headless (torch distributed).
#
# Run on each node with ROLE=head (rusty, rank 0) or ROLE=worker (toby,
# rank 1). Start the WORKER first, then the head.
#
# Serve command follows rtx6kpro/scripts/run-ds4-v9-server.sh (same vLLM
# fork); multi-node/RoCE handling follows the Aiden GB10 serving stack
# (spark-vllm-docker/DeepSeek-v4-DSpark-Aidendle94-GB10-ServingStack).

ROLE=${ROLE:?set ROLE=head|worker}

IMAGE=${IMAGE:-localhost/voipmonitor/vllm:eldritch-enlightenment-dspark-spark-sm121-v45c1582-b12xf3686b5-cu132-20260708}
NAME=${NAME:-ds4-dspark-tp2}
PORT=${PORT:-8000}

HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
MODEL_REPO_DIR=models--deepseek-ai--DeepSeek-V4-Flash-DSpark
# Resolve the locally cached snapshot (must be identical on both nodes).
SNAPSHOT=${SNAPSHOT:-$(ls "$HF_CACHE/hub/$MODEL_REPO_DIR/snapshots/" | head -1)}
MODEL=${MODEL:-/root/.cache/huggingface/hub/$MODEL_REPO_DIR/snapshots/$SNAPSHOT}
SERVED_MODEL=${SERVED_MODEL:-DeepSeek-V4-Flash-DSpark}
CACHE=${CACHE:-$HOME/.cache/vllm-ds4-tp2}

# Cluster networking: dual direct 200G CX7 links between rusty and toby.
MASTER_ADDR=${MASTER_ADDR:-10.11.1.1}          # rusty on the first 200G link
NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1,enP2p1s0f1np1}
CONTROL_IF=${CONTROL_IF:-enp1s0f1np1}
# Traffic class DSCP 26 (lossless RoCE queue) for switch-fabric compatibility;
# harmless on the direct back-to-back links.
NCCL_IB_TC=${NCCL_IB_TC:-106}

case "$ROLE" in
  head)
    NODE_RANK=0
    VLLM_HOST_IP=${VLLM_HOST_IP:-10.11.1.1}
    HEADLESS_ARGS=()
    ;;
  worker)
    NODE_RANK=1
    VLLM_HOST_IP=${VLLM_HOST_IP:-10.11.1.2}
    HEADLESS_ARGS=(--headless)
    ;;
  *)
    echo "Unknown ROLE=$ROLE (head|worker)" >&2
    exit 2
    ;;
esac

TP=${TP:-2}
GPU_MEM=${GPU_MEM:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-524288}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_BATCHED=${MAX_BATCHED:-8192}
GRAPH=${GRAPH:-256}
DSPARK_TOKENS=${DSPARK_TOKENS:-5}
SAMPLE=${SAMPLE:-probabilistic}

SPEC_JSON=$(printf '{"model":"%s","method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s"}' \
  "$MODEL" "$DSPARK_TOKENS" "$SAMPLE")

# RoCE-v2 IPv4 GID index re-numbers across reboots; detect it on the host.
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
if [[ -z "$NCCL_IB_GID_INDEX" ]]; then
  echo "WARNING: no RoCE v2 IPv4 GID found on ${NCCL_IB_HCA%%,*}; leaving NCCL to autodetect" >&2
fi

mkdir -p "$CACHE"/{vllm,tilelang/tmp,tvm,triton,torchinductor,torch_extensions,flashinfer,tmp}

test -f "$HF_CACHE/hub/$MODEL_REPO_DIR/snapshots/$SNAPSHOT/config.json"

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
  -v "$CACHE/tmp:/container-tmp:rw" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_121a \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_HOST_IP="$VLLM_HOST_IP" \
  -e GLOO_SOCKET_IFNAME="$CONTROL_IF" \
  -e TP_SOCKET_IFNAME="$CONTROL_IF" \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e NCCL_NET=IB \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$NCCL_IB_HCA" \
  ${NCCL_IB_GID_INDEX:+-e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX"} \
  -e NCCL_IB_TC="$NCCL_IB_TC" \
  -e NCCL_CROSS_NIC=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_DEBUG=WARN \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=0 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_MOE=0 \
  -e VLLM_DSPARK_REPLICATE_MARKOV_W1=1 \
  -e SAFETENSORS_FAST_GPU=1 \
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
  --enable-prefix-caching \
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
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --moe-backend b12x \
  --disable-custom-all-reduce \
  --speculative-config "$SPEC_JSON" \
  --nnodes 2 \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port 25000 \
  "${HEADLESS_ARGS[@]}"

echo "$NAME ROLE=$ROLE rank=$NODE_RANK master=$MASTER_ADDR gid=${NCCL_IB_GID_INDEX:-auto} port=$PORT model=$SERVED_MODEL spec=dspark/$DSPARK_TOKENS"
