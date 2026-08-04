#!/usr/bin/env bash
set -euo pipefail

# Laguna-S-2.1-FP8 two-node TP2 launcher for a DGX Spark pair
# (dusty + kirby by default; same 10.11.1.x back-to-back convention as
# rusty/toby, so HOST_IP/MASTER_ADDR defaults transfer).
#
# Adapted from the GG bare-metal launcher (vllm tree
# serve-laguna-s21-nvfp4.sh) for the official FP8 checkpoint
# (poolside/Laguna-S-2.1-FP8, promoted RC2, published 2026-08-01):
#   - 117.6B MoE (256 experts, top-10), GQA 8 KV heads, laguna model_type
#     (supported by the GG vLLM in our v20p3 image, incl. poolside_v1
#     parsers). FP8 weights ~131 GB total -> ~66 GB/node at TP2.
#   - Quantization auto-detected from quantization_config; the FP8 card
#     requires VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 under vLLM.
#   - DFlash speculative decoding with the quantization-matched draft
#     (Laguna-S-2.1-DFlash-FP8). Card recommends K=15; env-overridable
#     (DFLASH_TOKENS) — depth deserves its own A/B before blessing.
#   - Containerized + cross-node deltas vs the bare-metal launcher:
#     sm_121a (not sm_120a), NCCL over the pair RoCE rails instead of
#     PCIe allreduce, no venv/nvcc. GPU_MEM 0.85: FP8 weights are only
#     ~66G/node at TP2, so even 0.85 leaves a >1.8M-token KV pool (0.87
#     measured 1.93M on first boot 2026-08-03) while returning ~2.5G/node
#     of UMA host margin vs the DS4 pair's 0.87 ceiling.
#   - MoE/linear backends left on vLLM auto-selection for FP8 (the GG
#     launcher's b12x pins are NVFP4-path choices); override with
#     MOE_BACKEND/LINEAR_BACKEND envs to experiment.
#   - Sampling defers to the checkpoint's generation_config.json (temp 1.0,
#     top_p 1.0, top_k 20) per the RC2 card; the GG launcher's 0.7/0.95
#     override was dropped 2026-08-03 (user decision). NOTE: the K5-vs-K7
#     A/B ran at 0.7/0.95 — re-validate acceptance under checkpoint
#     sampling before leaning on those absolute numbers.
#
# Usage (cold start, nothing running on either node):
#   kirby: ROLE=worker ./run-laguna-s21-fp8-tp2-node.sh
#   dusty: ROLE=head   ./run-laguna-s21-fp8-tp2-node.sh
#
# REPLACING A LIVE PAIR — teardown BOTH nodes before launching EITHER role:
#   kirby: ROLE=stop ./run-laguna-s21-fp8-tp2-node.sh   # worker first
#   dusty: ROLE=stop ./run-laguna-s21-fp8-tp2-node.sh   # head last
#   then the cold-start sequence above (worker, then head).
# The inline stop below is NOT sufficient for rolling replacement: launching
# the new worker while the OLD head still listens on MASTER_PORT lets the
# worker rendezvous with the dying master and exit 1 mid-broadcast when that
# head is stopped, leaving the new head waiting on a dead rank (hit
# 2026-08-03 during the K5/0.85 rollover).

ROLE=${ROLE:?set ROLE=head|worker|stop}

# r28-spark (2026-08-04): upstream r28 composition + SM121 overlay with the
# PR#234-refined q_len guard. Qualified on this pair 2026-08-04: both
# frozen-wrapper reproducers token-exact vs PIECEWISE refs, 61 FULL replays,
# clean error scan. Rollback: gilded-gnosis-v20p3p1-...-20260803 (staged on
# all 8 nodes).
IMAGE=${IMAGE:-localhost/voipmonitor/vllm:gilded-gnosis-v20-r28-spark-sm121-vllm47d1950-si200c1db-fi7ad08da-cu132-20260804}
NAME=${NAME:-laguna-s21-fp8-tp2}
PORT=${PORT:-8000}

MODEL=${MODEL:-poolside/Laguna-S-2.1-FP8}
# Deliberately tracks main (user decision 2026-08-03: no snapshot pins).
# HF_HUB_OFFLINE=1 below means updates only land via an explicit re-download.
MODEL_REVISION=${MODEL_REVISION:-main}
DFLASH_MODEL=${DFLASH_MODEL:-poolside/Laguna-S-2.1-DFlash-FP8}
# K=5 won the pair A/B on 2026-08-03 (same verdict as DS4): coding-peak
# +10.1% over K=7 (36.9 vs 33.5 tok/s, 8-run means), 15-cell decode
# geomean +2.6%, and K7's positions 5-6 accept only 22%/19% — 0.26 extra
# tokens/step for two extra verify positions every step. Overrides both
# poolside card values (text says 7, benchmark tables used 15; K15 arm
# dropped as hopeless — community measured K6-15 accept ~0% on GB10).
# JSONs: benchmark_results-laguna-s21-fp8-k{5,7}_*.json.
DFLASH_TOKENS=${DFLASH_TOKENS:-5}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-laguna-s-2.1}

HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
CACHE=${CACHE:-$HOME/.cache/vllm-laguna-s21}
mkdir -p "$CACHE" "$CACHE/tmp"

MASTER_ADDR=${MASTER_ADDR:-10.11.1.1}
MASTER_PORT=${MASTER_PORT:-25000}
case "$ROLE" in
  head)   NODE_RANK=0; HOST_IP=${HOST_IP:-10.11.1.1}; ROLE_ARGS="" ;;
  worker) NODE_RANK=1; HOST_IP=${HOST_IP:-10.11.1.2}; ROLE_ARGS="--headless" ;;
  stop)
    if podman container exists "$NAME" 2>/dev/null; then
      podman stop -t 60 "$NAME" >/dev/null 2>&1 || true
      podman rm "$NAME" >/dev/null 2>&1 || true
    fi
    echo "$NAME stopped and removed on $(hostname)"
    exit 0 ;;
  *) echo "ROLE must be head, worker, or stop" >&2; exit 2 ;;
esac

TP_SIZE=${TP_SIZE:-2}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
# 8224 (not 8192): the scheduler reserves max_num_seqs*(K-1) slots for
# DFlash drafts, which at K=5/seqs=8 clamped an 8192 budget to 8160
# schedulable tokens. 8224 restores a clean 8192.
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8224}
# FULL_AND_PIECEWISE requires the v20p3.1+ image. v20p3 had a frozen-q_len
# bug in the FULL-graph decode path (lone steps != 1+K tokens — fused
# prefill tails, spec truncation near max_tokens — hit a captured wrapper's
# frozen shape: EngineDeadError "planned with 6, got 8|5"). Fixed in
# v20p3p1 (persistent_decode_wrapper_eligible predicate); validated
# 2026-08-03 via 4-gate ceremony: both reproducers clean on FULL,
# token-exact vs PIECEWISE, 163 FULL replays observed, bench sweep
# survived. FULL-96 vs PIECEWISE at same sampling: +28.8% decode geomean,
# +49% coding-peak (32.8 vs 22.0). If ever rolled back to a pre-fix
# image, set CUDAGRAPH_MODE=PIECEWISE.
GPU_MEM=${GPU_MEM:-0.85}

SPEC_CONFIG=${SPEC_CONFIG:-"{\"model\":\"${DFLASH_MODEL}\",\"num_speculative_tokens\":${DFLASH_TOKENS},\"method\":\"dflash\",\"attention_backend\":\"FLASHINFER\"}"}

backend_args=()
if [[ -n "${MOE_BACKEND:-}" ]]; then backend_args+=(--moe-backend "$MOE_BACKEND"); fi
if [[ -n "${LINEAR_BACKEND:-}" ]]; then backend_args+=(--linear-backend "$LINEAR_BACKEND"); fi
MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-96}
backend_args+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")

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
  -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 \
  -e VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}" \
  -e VLLM_DEBUG_B12X_MINIMAX_M3_MSA="${VLLM_DEBUG_B12X_MINIMAX_M3_MSA:-0}" \
  -e VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}" \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
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
  --entrypoint /opt/venv/bin/python \
  "$IMAGE" \
  -m vllm.entrypoints.cli.main serve \
  "$MODEL" \
  --revision "$MODEL_REVISION" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size "$TP_SIZE" \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --attention-backend FLASHINFER \
  --speculative-config "$SPEC_CONFIG" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}\",\"custom_ops\":[\"all\"]}" \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --disable-custom-all-reduce \
  --nnodes 2 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  $ROLE_ARGS \
  "${backend_args[@]}"

echo "$NAME ROLE=$ROLE rank=$NODE_RANK model=$MODEL dflash_k=$DFLASH_TOKENS tp=$TP_SIZE port=$PORT seqs=$MAX_NUM_SEQS gpu_mem=$GPU_MEM"
