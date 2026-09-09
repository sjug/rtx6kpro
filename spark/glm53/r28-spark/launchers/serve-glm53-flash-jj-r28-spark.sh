#!/usr/bin/env bash
set -euo pipefail
case ${DRY_RUN:-0} in 0|1) ;; *) echo 'DRY_RUN must be 0 or 1' >&2; exit 2;; esac

# JJ r28 Spark launcher for GLM-5.3-Flash. The host runner owns topology and
# transport. In particular, this launcher never passes
# --disable-custom-all-reduce: doing so would disable RoCEnante as well as the
# cross-node CUDA-IPC implementation that we actually intend to bypass.
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

model=${MODEL:-local-inference-lab/GLM-5.3-Flash-NVFP4}
if (($# > 0)) && [[ "$1" != -* ]]; then
  model=$1
  shift
fi

model_revision=${MODEL_REVISION-46aaae8a82032f77100f2f03e9cc11b391df3b4d}
served_model_name=${SERVED_MODEL_NAME:-GLM-5.3-Flash}
host=${HOST:-0.0.0.0}
port=${PORT:-8000}
tp=${TP:-4}
dcp=${DCP:-1}
cp_kv_cache_interleave_size=${CP_KV_CACHE_INTERLEAVE_SIZE:-4}
max_num_seqs=${MAX_NUM_SEQS:-8}
max_model_len=${MAX_MODEL_LEN:-1048576}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-4096}
prefill_schedule_interval=${PREFILL_SCHEDULE_INTERVAL:-8}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.85}
# Keep the buffered InstantTensor loader; fastsafetensors exhausted GB10 host
# memory on the Qwen TP2 pair on 2026-09-06.
load_format=${LOAD_FORMAT:-instanttensor}
speculator=${SPECULATOR:-mtp}
dflash_model=${DFLASH_MODEL:-local-inference-lab/GLM-5.3-Flash-DFlash2}
dflash_revision=${DFLASH_MODEL_REVISION-}
dflash_kv_cache_dtype=${DFLASH_KV_CACHE_DTYPE:-auto}
dflash_attention_backend=${DFLASH_ATTENTION_BACKEND:-FLASH_ATTN}
kv_cache_dtype=${KV_CACHE_DTYPE:-fp8}
mtp_attention_backend=${MTP_ATTENTION_BACKEND:-B12X}
mtp_moe_backend=${MTP_MOE_BACKEND:-marlin}
kda_decode_backend=${GLM53_KDA_DECODE_BACKEND:-auto}
kda_prefill_backend=${GLM53_KDA_PREFILL_BACKEND:-flashkda}
cudagraph_mode=${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}
capture_sizes=${CUDAGRAPH_CAPTURE_SIZES:-'1 2 4 8 12 16 24 32'}
fairness_engine=${FAIRNESS_ENGINE:-none}
prefill_compute_share=${PREFILL_COMPUTE_SHARE:-}
lmcache_enabled=${LMCACHE_ENABLED:-0}
draft_head_mode=${VLLM_GLM53_MTP_DRAFT_HEAD:-bf16}

fail() {
  printf '%s\n' "$1" >&2
  exit 2
}

if [[ "${lmcache_enabled}" != 0 ]]; then
  printf 'LMCache is packaged but not qualified on DGX Spark; set LMCACHE_ENABLED=0.\n' >&2
  exit 78
fi
unset LMCACHE_CONFIG_FILE

if [[ ! "${tp}" =~ ^[1-9][0-9]*$ || ! "${dcp}" =~ ^[1-9][0-9]*$ ]] \
    || ((tp % dcp != 0)); then
  fail "TP and DCP must be positive integers with TP divisible by DCP; got TP=${tp} DCP=${dcp}"
fi
for pair in \
  "CP_KV_CACHE_INTERLEAVE_SIZE:${cp_kv_cache_interleave_size}" \
  "MAX_NUM_SEQS:${max_num_seqs}" \
  "MAX_MODEL_LEN:${max_model_len}" \
  "MAX_NUM_BATCHED_TOKENS:${max_num_batched_tokens}" \
  "PREFILL_SCHEDULE_INTERVAL:${prefill_schedule_interval}"; do
  name=${pair%%:*}
  value=${pair#*:}
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer; got ${value}"
done

case "${speculator}" in
  mtp) num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-${MTP:-3}} ;;
  dflash | dflash2) num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-7} ;;
  *) fail "SPECULATOR must be mtp, dflash, or dflash2; got ${speculator}" ;;
esac
[[ "${num_speculative_tokens}" =~ ^[0-9]+$ ]] \
  || fail "NUM_SPECULATIVE_TOKENS/MTP must be a non-negative integer; got ${num_speculative_tokens}"

case "${kda_decode_backend}" in auto|b12x|triton) ;; *) fail "invalid GLM53_KDA_DECODE_BACKEND=${kda_decode_backend}" ;; esac
case "${kda_prefill_backend}" in auto|b12x|flashkda|triton) ;; *) fail "invalid GLM53_KDA_PREFILL_BACKEND=${kda_prefill_backend}" ;; esac
case "${dflash_attention_backend}" in FLASHINFER|FLASH_ATTN) ;; *) fail "invalid DFLASH_ATTENTION_BACKEND=${dflash_attention_backend}" ;; esac
case "${kv_cache_dtype}" in fp8|fp8_e4m3|fp8_ds_mla|nvfp4_ds_mla) ;; *) fail "invalid KV_CACHE_DTYPE=${kv_cache_dtype}" ;; esac
case "${draft_head_mode}" in bf16|nvfp4) ;; *) fail "invalid VLLM_GLM53_MTP_DRAFT_HEAD=${draft_head_mode}" ;; esac

if [[ "${fairness_engine}" != none && "${prefill_schedule_interval}" != 1 ]]; then
  fail 'PREFILL_SCHEDULE_INTERVAL must be 1 when FAIRNESS_ENGINE is enabled'
fi
fairness_args=()
case "${fairness_engine}" in
  none) ;;
  compute_share)
    [[ -n "${prefill_compute_share}" ]] || fail 'PREFILL_COMPUTE_SHARE is required for compute_share fairness'
    awk -v value="${prefill_compute_share}" 'BEGIN { exit !(value > 0.0 && value < 1.0) }' \
      || fail "PREFILL_COMPUTE_SHARE must be between zero and one; got ${prefill_compute_share}"
    fairness_args=(--fairness-engine compute_share --prefill-compute-share "${prefill_compute_share}")
    ;;
  *) fail "FAIRNESS_ENGINE must be none or compute_share; got ${fairness_engine}" ;;
esac

read -r -a capture_size_args <<<"${capture_sizes}"
((${#capture_size_args[@]} > 0)) || fail 'CUDAGRAPH_CAPTURE_SIZES must not be empty'
previous=0
for size in "${capture_size_args[@]}"; do
  [[ "${size}" =~ ^[1-9][0-9]*$ ]] || fail "invalid CUDA graph capture size: ${size}"
  ((size > previous)) || fail 'CUDAGRAPH_CAPTURE_SIZES must be strictly increasing'
  previous=${size}
done
max_cudagraph_capture_size=${MAX_CUDAGRAPH_CAPTURE_SIZE:-${previous}}
((previous <= max_cudagraph_capture_size)) \
  || fail "capture size ${previous} exceeds MAX_CUDAGRAPH_CAPTURE_SIZE=${max_cudagraph_capture_size}"

# Preserve the R22 Spark environment for release comparisons. R15 explicitly
# disabled L2 prefetch on GB10; the source default also disables it on SM121.
# Do not force RTX L2 or experimental dynamic-MoE policies into this profile.
export VLLM_B12X_MOE_FP4_FORCE_A16="${VLLM_B12X_MOE_FP4_FORCE_A16:-0}"
export VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE="${VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE:-2048}"
export VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE="${VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE:-auto}"
export VLLM_PLUGINS=

revision_args=()
if [[ -n "${model_revision}" && "${model}" != /* ]]; then
  revision_args=(--revision "${model_revision}")
fi

cmd=(
  /opt/venv/bin/vllm serve "${model}"
  "${revision_args[@]}"
  --served-model-name "${served_model_name}"
  --host "${host}"
  --port "${port}"
  --tensor-parallel-size "${tp}"
  --pipeline-parallel-size 1
  --decode-context-parallel-size "${dcp}"
  --cp-kv-cache-interleave-size "${cp_kv_cache_interleave_size}"
  --dcp-kv-cache-interleave-size "${cp_kv_cache_interleave_size}"
  --max-num-seqs "${max_num_seqs}"
  --max-model-len "${max_model_len}"
  --max-num-batched-tokens "${max_num_batched_tokens}"
  --prefill-schedule-interval "${prefill_schedule_interval}"
  --max-cudagraph-capture-size "${max_cudagraph_capture_size}"
  --cudagraph-capture-sizes "${capture_size_args[@]}"
  --gpu-memory-utilization "${gpu_memory_utilization}"
  --mamba-cache-mode align
  --enable-prefix-caching
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype "${kv_cache_dtype}"
  --quantization modelopt_mixed
  --block-size 256
  --load-format "${load_format}"
  --attention-backend B12X
  --moe-backend b12x
  --linear-backend b12x
  --no-enable-flashinfer-autotune
  --additional-config "{\"glm53_kda_decode_backend\":\"${kda_decode_backend}\",\"kda_prefill_backend\":\"${kda_prefill_backend}\"}"
  --compilation-config "{\"cudagraph_mode\":\"${cudagraph_mode}\"}"
  --enable-auto-tool-choice
  --tool-call-parser glm47
  --reasoning-parser glm45
  "${fairness_args[@]}"
)

if ((num_speculative_tokens > 0)); then
  case "${speculator}" in
    mtp)
      revision_json=
      if [[ -n "${model_revision}" && "${model}" != /* ]]; then
        revision_json=",\"revision\":\"${model_revision}\""
      fi
      cmd+=(--speculative-config "{\"method\":\"mtp\"${revision_json},\"num_speculative_tokens\":${num_speculative_tokens},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"moe_backend\":\"${mtp_moe_backend}\",\"attention_backend\":\"${mtp_attention_backend}\"}")
      ;;
    dflash | dflash2)
      revision_json=
      if [[ -n "${dflash_revision}" && "${dflash_model}" != /* ]]; then
        revision_json=",\"revision\":\"${dflash_revision}\""
      fi
      cmd+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"${dflash_model}\"${revision_json},\"num_speculative_tokens\":${num_speculative_tokens},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"kv_cache_dtype\":\"${dflash_kv_cache_dtype}\",\"attention_backend\":\"${dflash_attention_backend}\"}")
      ;;
  esac
fi
# Next rebuild owns the same qualified default as the host runner. A runner
# supplying an explicit control policy must not create duplicate policy flags.
checkpoint_policy=${RECURRENT_CHECKPOINT_POLICY:-aligned}
case "${checkpoint_policy}" in aligned|auto) ;; *) fail 'invalid RECURRENT_CHECKPOINT_POLICY' ;; esac
policy_args=(--recurrent-checkpoint-policy "${checkpoint_policy}")
for arg in "$@"; do
  case "${arg}" in --recurrent-checkpoint-policy|--recurrent-checkpoint-policy=*) policy_args=() ;; esac
done
cmd+=("${policy_args[@]}" "$@")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'VLLM_GLM53_MTP_DRAFT_HEAD=%q\n' "${draft_head_mode}"
  printf 'GLM-5.3-Flash JJ r28 Spark launch:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

printf 'GLM Spark policy: L2_PREFETCH=%s WORK_SOURCE=%s DRAFT_HEAD=%s\n' \
  "${VLLM_GLM53_L2_PREFETCH:-platform-default-off-sm121}" \
  "${B12X_DYNAMIC_WORK_SOURCE:-materialized_queue}" "${draft_head_mode}"

if ! /opt/venv/bin/python -c '
if not __debug__:
    raise RuntimeError("R28 verification requires Python assertions enabled")
from pathlib import Path
import os
import hashlib
launcher = Path("/usr/local/bin/serve-glm53-flash-jj-r28-spark.sh")
assert hashlib.sha256(launcher.read_bytes()).hexdigest() == os.environ["GLM_LAUNCHER_SHA256"]
import b12x, vllm
import vllm._flashkda_C
from b12x.comm import roce
from b12x.comm.roce._proxy import load
assert Path(vllm.__file__).resolve().is_relative_to(Path("/opt/jovian-judgement/vllm"))
assert Path(b12x.__file__).resolve().is_relative_to(Path("/opt/jovian-judgement/b12x"))
assert roce.API_VERSION == 1
assert os.environ["B12X_ROCE_CACHE_DIR"] == "/opt/jovian-judgement/b12x-roce"
assert load().roce_abi_version() == 3
'; then
  printf 'JJ r28 Spark runtime preflight failed; refusing launch.\n' >&2
  exit 78
fi

exec "${cmd[@]}"
