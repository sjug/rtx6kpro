#!/usr/bin/env bash
# Separate process: errexit must not be suppressed by a parent conditional.
set -euo pipefail
image=$1 version=$2 flash_sha=$3 lock_sha=$4 native_sha=$5
value() { jq -er "$1" source.lock.json; }
bash tests/check-glm-launcher-identity.sh "$image" \
  launchers/serve-glm53-flash-jj-r32-spark.sh \
  "$(value '.launchers["serve-glm53-flash-jj-r32-spark.sh"]')"
[[ $(podman image inspect "$image" --format '{{index .Config.Labels "local-inference.runtime.source-lock.sha256"}}') == "$lock_sha" ]]
[[ $(podman image inspect "$image" --format '{{index .Config.Labels "local-inference.flashkda.sha256"}}') == "$flash_sha" ]]
[[ $(podman image inspect "$image" --format '{{index .Config.Labels "local-inference.vllm.stable-native.sha256"}}') == "$native_sha" ]]
podman run --pull=never --rm --device nvidia.com/gpu=all --ipc=host --entrypoint python "$image" \
  /opt/local-inference/verify_r32_runtime.py \
  --vllm-version "$version" --vllm-spark-tree "$(value '.vllm.tree')" \
  --vllm-subtree "$(value '.vllm.package_tree')" --b12x-tree "$(value '.b12x.tree')" \
  --b12x-subtree "$(value '.b12x.package_tree')" --lmcache-version "$(value '.lmcache.version')" \
  --flashkda-sha256 "$flash_sha" --source-lock-sha256 "$lock_sha" --stable-native-sha256 "$native_sha"
podman run --pull=never --rm --device nvidia.com/gpu=all --ipc=host -e PYTHONUNBUFFERED=1 \
  -e B12X_PRINT_COMPILE_PROGRESS=1 --entrypoint bash "$image" -c '
    set -euo pipefail
    cd /opt/jovian-judgement/vllm
    python /opt/local-inference/run_r32_regressions.py
    python -m pytest -s -vv --confcutdir=tests/v1/worker \
      tests/v1/worker/test_boundary_checkpoint_restore_scalars.py \
      tests/v1/worker/test_b12x_roce_health.py
    python -m pytest -s -vv --confcutdir=tests/v1 \
      tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py \
      tests/v1/core/test_mamba_align_chunk_split.py \
      tests/v1/worker/test_mamba_prefix_state_index.py
    python -m pytest -s -vv --confcutdir=tests/v1/core tests/v1/core/test_scheduler.py \
      -k full_boundary_hit_is_admitted_while_another_request_decodes
    python -m pytest -s -vv --confcutdir=tests/kernels/attention tests/kernels/attention/test_mla_bmm_storage.py
    python -m pytest -s -vv --confcutdir=tests/model_executor/layers tests/model_executor/layers/test_shared_experts_stream.py
    python -m pytest -s -vv --confcutdir=tests/models tests/models/test_glm5next_model.py \
      -k "mtp_draft_head_mode or mtp_nvfp4_draft_head_capability"
    python -m pytest -s -vv --confcutdir=tests/models/kimi_k3 tests/models/kimi_k3/test_kda.py \
      -k near_collinear_keys_remain_finite
    python -m pytest -s -vv --confcutdir=/opt/jovian-judgement/b12x/tests/gemm \
      /opt/jovian-judgement/b12x/tests/gemm/test_mxfp8_linear.py \
      -k persistent_ctas_complete_single_stage_epilogue_stores
    python /opt/local-inference/verify_glm53_nvfp4_draft_head_sm121.py
    python /opt/local-inference/run_flashkda_gate.py
  '
for launcher in serve-glm53-flash-jj-r32-spark.sh serve-qwen38-flash-next-jj-r32-spark.sh; do
  render=$(podman run --pull=never --rm -e DRY_RUN=1 --entrypoint "/usr/local/bin/$launcher" "$image")
  grep -Fq -- '--recurrent-checkpoint-policy aligned' <<<"$render"
  grep -Fq -- '--load-format instanttensor' <<<"$render"
  if podman run --pull=never --rm -e DRY_RUN=1 -e LMCACHE_ENABLED=1 --entrypoint "/usr/local/bin/$launcher" "$image"; then
    echo 'LMCache enabled render must fail' >&2; exit 1
  fi
  # A hollow site-packages launch must fail before vllm serve is executed.
  status=0
  output=$(podman run --pull=never --rm -e PYTHONPATH= --entrypoint "/usr/local/bin/$launcher" "$image" 2>&1) || status=$?
  [[ $status == 78 ]]
  grep -Fq 'preflight failed; refusing launch.' <<<"$output"
done
# Alter only an ephemeral container copy. The launcher must reject its own
# changed bytes against the image-owned expected hash before importing vLLM.
status=0
output=$(podman run --pull=never --rm --entrypoint bash "$image" -c '
  set -euo pipefail
  launcher=/usr/local/bin/serve-glm53-flash-jj-r32-spark.sh
  printf "\n# checksum-negative-control\n" >> "$launcher"
  exec "$launcher"
' 2>&1) || status=$?
[[ $status == 78 ]]
grep -Fq 'runtime preflight failed' <<<"$output"
