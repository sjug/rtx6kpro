#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
image=${1:?exact image ID}
receipt=${2:?receipt directory}
gpu=(podman run --rm --device nvidia.com/gpu=all --security-opt label=disable --ipc=host
  -e PYTHONUNBUFFERED=1 -e B12X_PRINT_COMPILE_PROGRESS=1 -e B12X_GLM53_GPU_TEST=1
  -v "$PWD:/gate:ro")
"${gpu[@]}" "$image" python /gate/tests/gate_cache_agreement.py --collect-only | tee "$receipt/cache-agreement-collection.log"
"${gpu[@]}" "$image" python /gate/tests/gate_cache_agreement.py | tee "$receipt/cache-agreement-tests.log"
"${gpu[@]}" -e CC=/bin/false "$image" python /gate/verify_runtime.py | tee "$receipt/native-smoke.log"
"${gpu[@]}" "$image" python /gate/tests/probe_flashkda_stack.py | tee "$receipt/flashkda-stack.json"
"${gpu[@]}" "$image" python /gate/verify_launch_arguments.py | tee "$receipt/launch-arguments.log"
for model in qwen38-flash-next glm53-flash; do
  launcher="/usr/local/bin/serve-$model-jj-main-spark.sh"
  podman run --rm -e DRY_RUN=1 "$image" bash "$launcher" > "$receipt/$model-render.txt"
  grep -q -- '--recurrent-checkpoint-policy aligned' "$receipt/$model-render.txt" || { echo "Missing aligned policy: $model"; exit 78; }
  # No DRY_RUN here: exercise real source-path, launcher-hash and proxy preflights.
  "${gpu[@]}" "$image" bash "$launcher" --help > "$receipt/$model-live-preflight.log" 2>&1
done
for mode in --collect-only ''; do
  options=()
  [[ -z $mode ]] || options+=("$mode")
  "${gpu[@]}" "$image" python /gate/gate_flashkda.py \
    /opt/jj-main-build/components/vllm/flashkda-tests "${options[@]}" | tee "$receipt/flashkda${mode}.log"
done
"${gpu[@]}" "$image" python /gate/tests/run_regressions.py --collect-only-count | tee "$receipt/regression-collection.log"
"${gpu[@]}" "$image" python /gate/tests/run_regressions.py | tee "$receipt/regressions.log"
"${gpu[@]}" "$image" python /gate/tests/verify_glm53_nvfp4_draft_head_sm121.py | tee "$receipt/draft-head.log"
