#!/usr/bin/env bash
set -euo pipefail
case ${DRY_RUN:-0} in 0|1) ;; *) echo 'DRY_RUN must be 0 or 1' >&2; exit 2;; esac

# JJ r28 Spark profile for Qwen3.8-Flash-Next NVFP4-4p89. The host runner
# supplies the two-node topology. This deliberately preserves the qualified
# r15p Qwen execution envelope so the r28 comparison changes the source line,
# not the serving configuration.

model=${MODEL:-local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89}
if (($# > 0)) && [[ "$1" != -* ]]; then
  model=$1
  shift
fi
model_revision=${MODEL_REVISION-c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d}

served_model_name=${SERVED_MODEL_NAME:-Qwen3.8-Flash-Next-NVFP4-4p89}
host=${HOST:-0.0.0.0}
port=${PORT:-8000}
tp=${TP:-2}
max_model_len=${MAX_MODEL_LEN:-262144}
max_num_seqs=${MAX_NUM_SEQS:-4}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-4096}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.85}
# r27 removed the InstantTensor copy option (vLLM 6575b5ac8) but still routes
# --load-format instanttensor through the default loader with the buffered
# INSTANTTENSOR_* streaming environment. fastsafetensors, which upstream r28
# used on its x86 launchers, exhausted GB10 host memory here on 2026-09-06
# (dusty unresponsive, kirby NCCL watchdog); keep the streamed loader.
load_format=${LOAD_FORMAT:-instanttensor}
# R27 qualification found that the default recurrent checkpoint policy (auto, resolving to
# request_boundaries) schedules an exact-repeat prompt as the only work in its
# step, which serialized identical concurrent requests and blocked fresh
# arrivals on 2026-09-06. aligned (block-aligned checkpoints, the r26
# behavior) removed both effects under otherwise matched settings and passed
# the R27 correctness set. Both R28 profiles retain aligned pending qualification.
recurrent_checkpoint_policy=${RECURRENT_CHECKPOINT_POLICY:-aligned}
num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-3}
b12x_policy_mode=${B12X_POLICY_MODE:-auto}
compilation_config=${VLLM_QWEN38_COMPILATION_CONFIG:-}
lmcache_enabled=${LMCACHE_ENABLED:-0}
if [[ -z "${compilation_config}" ]]; then
  compilation_config='{"pass_config":{"fuse_act_quant":true}}'
fi

if [[ "${lmcache_enabled}" != 0 ]]; then
  echo 'LMCache is packaged but not qualified on DGX Spark; set LMCACHE_ENABLED=0.' >&2
  exit 78
fi
unset LMCACHE_CONFIG_FILE

if [[ "${tp}" != 2 ]]; then
  echo "The Qwen Spark profile is qualified only for TP=2; got ${tp}" >&2
  exit 2
fi
if [[ ! "${num_speculative_tokens}" =~ ^[0-9]+$ ]]; then
  echo 'NUM_SPECULATIVE_TOKENS must be a non-negative integer' >&2
  exit 2
fi
case "${b12x_policy_mode}" in
  auto | heuristic-only | preplanned-only) ;;
  *) echo "Invalid B12X_POLICY_MODE: ${b12x_policy_mode}" >&2; exit 2 ;;
esac

export CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_PLUGINS=
export VLLM_SSM_CONV_STATE_LAYOUT="${VLLM_SSM_CONV_STATE_LAYOUT:-DS}"
export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-1}"
export VLLM_USE_MEGA_AOT_ARTIFACT="${VLLM_USE_MEGA_AOT_ARTIFACT:-1}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE=0
export VLLM_MXFP8_LM_HEAD="${VLLM_MXFP8_LM_HEAD:-1}"
export VLLM_LM_HEAD_A16="${VLLM_LM_HEAD_A16:-1}"
export VLLM_MTP_NVFP4_LM_HEAD="${VLLM_MTP_NVFP4_LM_HEAD:-1}"
export VLLM_QWEN3_8_FLASH_NEXT_OVERLAP="${VLLM_QWEN3_8_FLASH_NEXT_OVERLAP:-1}"
export VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT="${VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT:-1}"
export VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH="${VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH:-1}"
export B12X_POLICY_MODE="${b12x_policy_mode}"
export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-BUFFERED}"
export INSTANTTENSOR_BUFFER_SIZE="${INSTANTTENSOR_BUFFER_SIZE:-1342177280}"
export INSTANTTENSOR_CONCURRENCY="${INSTANTTENSOR_CONCURRENCY:-1}"
export INSTANTTENSOR_IO_DEPTH="${INSTANTTENSOR_IO_DEPTH:-3}"

revision_args=()
if [[ -n "${model_revision}" && "${model}" != /* ]]; then
  revision_args=(--revision "${model_revision}")
fi

speculative_args=()
if ((num_speculative_tokens > 0)); then
  speculative_args=(
    --speculative-config
    "{\"method\":\"mtp\",\"num_speculative_tokens\":${num_speculative_tokens}}"
  )
fi

loader_args=(--load-format "${load_format}")

cmd=(
  /opt/venv/bin/vllm serve "${model}"
  "${revision_args[@]}"
  --served-model-name "${served_model_name}"
  --host "${host}"
  --port "${port}"
  --trust-remote-code
  --tensor-parallel-size "${tp}"
  --pipeline-parallel-size 1
  --disable-custom-all-reduce
  --mamba-cache-mode align
  --enable-prefix-caching
  --recurrent-checkpoint-policy "${recurrent_checkpoint_policy}"
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype fp8
  --quantization modelopt_mixed
  --block-size 16
  "${loader_args[@]}"
  --gpu-memory-utilization "${gpu_memory_utilization}"
  --max-model-len "${max_model_len}"
  --max-num-seqs "${max_num_seqs}"
  --max-num-batched-tokens "${max_num_batched_tokens}"
  "${speculative_args[@]}"
  --gdn-decode-kernel b12x
  --linear-backend b12x
  --moe-backend b12x
  --no-enable-flashinfer-autotune
  --mm-encoder-tp-mode data
  --mm-processor-cache-gb 0
  --limit-mm-per-prompt '{"image":1}'
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --compilation-config "${compilation_config}"
)
cmd+=("$@")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'VLLM_MXFP8_LM_HEAD=%q VLLM_LM_HEAD_A16=%q ' \
    "${VLLM_MXFP8_LM_HEAD}" "${VLLM_LM_HEAD_A16}"
  printf 'VLLM_MTP_NVFP4_LM_HEAD=%q ' "${VLLM_MTP_NVFP4_LM_HEAD}"
  printf 'VLLM_QWEN3_8_FLASH_NEXT_OVERLAP=%q ' \
    "${VLLM_QWEN3_8_FLASH_NEXT_OVERLAP}"
  printf 'VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT=%q ' \
    "${VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT}"
  printf 'VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH=%q\n' \
    "${VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH}"
  printf 'Qwen3.8 Flash Next NVFP4 JJ r28 Spark launch:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if ! /opt/venv/bin/python -c '
if not __debug__:
    raise RuntimeError("R28 verification requires Python assertions enabled")
from pathlib import Path
import b12x, vllm
assert Path(vllm.__file__).resolve().is_relative_to(Path("/opt/jovian-judgement/vllm"))
assert Path(b12x.__file__).resolve().is_relative_to(Path("/opt/jovian-judgement/b12x"))
'; then
  printf 'JJ r28 Spark runtime source-path preflight failed; refusing launch.\n' >&2
  exit 78
fi

exec "${cmd[@]}"
