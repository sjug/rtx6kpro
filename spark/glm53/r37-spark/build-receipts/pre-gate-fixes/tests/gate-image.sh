#!/usr/bin/env bash
set -euo pipefail
image=$1 lock_sha=$2
value() { jq -er "$1" source.lock.json; }
[[ $(podman image inspect "$image" --format '{{index .Config.Labels "local-inference.runtime.source-lock.sha256"}}') == "$lock_sha" ]]
for family in glm qwen; do
  if [[ $family == glm ]]; then launcher=serve-glm53-flash-jj-r37-spark.sh; else launcher=serve-qwen38-flash-next-jj-r37-spark.sh; fi
  digest=$(value ".launchers[\"$launcher\"]")
  [[ $(podman image inspect "$image" --format "{{index .Config.Labels \"local-inference.launcher.$family.sha256\"}}") == "$digest" ]]
  actual=$(podman run --pull=never --rm --entrypoint sha256sum "$image" "/usr/local/bin/$launcher")
  [[ ${actual%% *} == "$digest" ]]
  render=$(podman run --pull=never --rm -e DRY_RUN=1 --entrypoint "/usr/local/bin/$launcher" "$image")
  grep -Fq -- '--recurrent-checkpoint-policy aligned' <<<"$render"
  grep -Fq -- '--load-format instanttensor' <<<"$render"
  grep -Fq -- '--gpu-memory-utilization 0.85' <<<"$render"
  if podman run --pull=never --rm -e DRY_RUN=1 -e LMCACHE_ENABLED=1 --entrypoint "/usr/local/bin/$launcher" "$image"; then
    echo 'Enabled LMCache must fail' >&2; exit 1
  fi
  status=0
  output=$(podman run --pull=never --rm -e PYTHONPATH= --entrypoint "/usr/local/bin/$launcher" "$image" 2>&1) || status=$?
  [[ $status == 78 ]]
  grep -Fq 'preflight failed; refusing launch.' <<<"$output"
  status=0
  output=$(podman run --pull=never --rm --entrypoint bash "$image" -c '
    set -euo pipefail
    printf "\n# checksum-negative-control\n" >> "$1"
    exec "$1"
  ' bash "/usr/local/bin/$launcher" 2>&1) || status=$?
  [[ $status == 78 ]]
  grep -Fq 'preflight failed' <<<"$output"
done
gpu=(podman run --pull=never --rm --device nvidia.com/gpu=all --ipc=host \
  -e PYTHONUNBUFFERED=1 -e B12X_PRINT_COMPILE_PROGRESS=1 \
  -e PYTHONPATH=/build/r37:/opt/jovian-judgement/vllm:/opt/jovian-judgement/b12x)
"${gpu[@]}" --entrypoint python "$image" /opt/local-inference/r37-tests/verify_jj_r37_runtime.py \
  --vllm-version "$(value '.vllm.version')" --vllm-spark-tree "$(value '.vllm.tree')" \
  --vllm-subtree "$(value '.vllm.package_tree')" --b12x-tree "$(value '.b12x.tree')" \
  --b12x-subtree "$(value '.b12x.package_tree')" --lmcache-version "$(value '.lmcache.version')" \
  --flashkda-sha256 "$(value '.flashkda.spark_extension_sha256')" \
  --stable-native-sha256 "$(value '.stable_native_sha256')" --source-lock-sha256 "$lock_sha"
# Variables in this program belong to the container shell.
# shellcheck disable=SC2016
"${gpu[@]}" --entrypoint bash "$image" -c '
  set -euo pipefail
  gates=/opt/local-inference/r37-tests
  required() { python "$gates/run_required.py" "$@"; }
  python "$gates/verify_dependencies.py"
  cd /opt/jovian-judgement/vllm
  python "$gates/run_r37_regressions.py"
  required --confcutdir=tests/v1/worker tests/v1/worker/test_boundary_checkpoint_restore_scalars.py tests/v1/worker/test_b12x_roce_health.py
  required --confcutdir=tests/v1 tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py tests/v1/core/test_mamba_align_chunk_split.py tests/v1/worker/test_mamba_prefix_state_index.py
  required --count 1 --confcutdir=tests/v1/core tests/v1/core/test_scheduler.py -k full_boundary_hit_is_admitted_while_another_request_decodes
  required --confcutdir=tests/kernels/attention tests/kernels/attention/test_mla_bmm_storage.py
  required --confcutdir=tests/model_executor/layers tests/model_executor/layers/test_shared_experts_stream.py
  required --confcutdir=tests/models tests/models/test_glm5next_model.py -k "mtp_draft_head_mode or mtp_nvfp4_draft_head_capability"
  required --confcutdir=tests/models/kimi_k3 tests/models/kimi_k3/test_kda.py -k near_collinear_keys_remain_finite
  required --confcutdir=/opt/jovian-judgement/b12x/tests/gemm /opt/jovian-judgement/b12x/tests/gemm/test_mxfp8_linear.py -k persistent_ctas_complete_single_stage_epilogue_stores
  python "$gates/verify_glm53_nvfp4_draft_head_sm121.py"
  python "$gates/run_flashkda_gate.py"
  cd /opt/jovian-judgement/b12x
  export PYTHONPATH=/build/r37:/opt/jovian-judgement/b12x:/opt/jovian-judgement/vllm
  required --count 10 --confcutdir=tests/moe tests/moe/test_dynamic_fc1_fragment_rebind.py
  required --count 3 --confcutdir=tests/moe tests/moe/test_w4a8_dynamic_kernel.py -k small_tile_parallel_regime_matches_oracle
  required --count 4 --confcutdir=/build/r37/inputs/upstream/tests /build/r37/inputs/upstream/tests/test_dependency_python_patches.py
  required --count 16 --confcutdir="$gates" "$gates/test_sparse_mla_sm120.py" -k prefill_dsv4_dual
'
