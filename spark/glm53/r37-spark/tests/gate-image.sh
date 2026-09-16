#!/usr/bin/env bash
set -euo pipefail
image=${1:?image ID required} lock_sha=${2:?source lock digest required}
trap 'echo "R37 image gate failed at line $LINENO: $BASH_COMMAND" >&2' ERR
value() { jq -er "$1" source.lock.json; }
expect_equal() {
  [[ $2 == "$3" ]] || { printf '%s: expected=%s actual=%s\n' "$1" "$3" "$2" >&2; return 1; }
}
label() { podman image inspect "$image" --format "{{index .Config.Labels \"$1\"}}"; }
expect_equal 'Source lock label' "$(label local-inference.runtime.source-lock.sha256)" "$lock_sha"
for family in glm qwen; do
  if [[ $family == glm ]]; then launcher=serve-glm53-flash-jj-r37-spark.sh; else launcher=serve-qwen38-flash-next-jj-r37-spark.sh; fi
  digest=$(value ".launchers[\"$launcher\"]")
  expect_equal "$family launcher label" "$(label "local-inference.launcher.$family.sha256")" "$digest"
  actual=$(podman run --pull=never --rm --entrypoint sha256sum "$image" "/usr/local/bin/$launcher")
  expect_equal "$family launcher file" "${actual%% *}" "$digest"
  render=$(podman run --pull=never --rm -e DRY_RUN=1 --entrypoint "/usr/local/bin/$launcher" "$image")
  for flag in '--recurrent-checkpoint-policy aligned' '--load-format instanttensor' '--gpu-memory-utilization 0.85'; do
    grep -Fq -- "$flag" <<<"$render" || { echo "$family render is missing $flag" >&2; exit 1; }
  done
  if podman run --pull=never --rm -e DRY_RUN=1 -e LMCACHE_ENABLED=1 --entrypoint "/usr/local/bin/$launcher" "$image"; then
    echo 'Enabled LMCache must fail' >&2; exit 1
  fi
  status=0
  output=$(podman run --pull=never --rm -e PYTHONPATH= --entrypoint "/usr/local/bin/$launcher" "$image" 2>&1) || status=$?
  expect_equal "$family hollow-install preflight status" "$status" 78
  grep -Fq 'preflight failed; refusing launch.' <<<"$output"
  status=0
  output=$(podman run --pull=never --rm --entrypoint bash "$image" -c '
    set -euo pipefail
    printf "\n# checksum-negative-control\n" >> "$1"
    exec "$1"
  ' bash "/usr/local/bin/$launcher" 2>&1) || status=$?
  expect_equal "$family altered-launcher preflight status" "$status" 78
  grep -Fq 'preflight failed' <<<"$output"
done
gpu=(podman run --pull=never --rm --device nvidia.com/gpu=all --ipc=host \
  -e PYTHONUNBUFFERED=1 -e B12X_PRINT_COMPILE_PROGRESS=1 \
  -e PYTHONPATH=/build/r37:/opt/jovian-judgement/vllm:/opt/jovian-judgement/b12x)
# Some modules query CUDA at import. Collection needs CDI, but executes no tests.
"${gpu[@]}" --entrypoint bash "$image" /opt/local-inference/r37-tests/gate-regressions.sh collect
"${gpu[@]}" --entrypoint python "$image" /opt/local-inference/r37-tests/verify_dependencies.py
"${gpu[@]}" --entrypoint python "$image" /opt/local-inference/r37-tests/verify_jj_r37_runtime.py \
  --vllm-version "$(value '.vllm.version')" --vllm-spark-tree "$(value '.vllm.tree')" \
  --vllm-subtree "$(value '.vllm.package_tree')" --b12x-tree "$(value '.b12x.tree')" \
  --b12x-subtree "$(value '.b12x.package_tree')" --lmcache-version "$(value '.lmcache.version')" \
  --flashkda-sha256 "$(value '.flashkda.spark_extension_sha256')" \
  --stable-native-sha256 "$(value '.stable_native_sha256')" --source-lock-sha256 "$lock_sha"
"${gpu[@]}" --entrypoint bash "$image" /opt/local-inference/r37-tests/gate-regressions.sh run
