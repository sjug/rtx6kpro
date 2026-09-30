#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
image=${1:?image ID required}
receipt=${2:?receipt directory required}
run() {
  local log=$1
  shift
  podman run --rm --pull=never --device nvidia.com/gpu=all \
    --security-opt label=disable --ipc=host \
    -e PYTHONUNBUFFERED=1 -e PYTHONOPTIMIZE=0 -e B12X_PRINT_COMPILE_PROGRESS=1 \
    -e B12X_GLM53_GPU_TEST=1 -e "CC=${CC:-cc}" \
    -v "$PWD:/gate:ro" "$image" python "$@" 2>&1 | tee "$receipt/$log.log"
}
CC=/bin/false run identity /gate/verify.py
CC=/bin/false run native /gate/inherited/verify_runtime.py
run flashkda-collection /gate/inherited/gate_flashkda.py /gate/inherited/flashkda-tests --collect-only
run flashkda /gate/inherited/gate_flashkda.py /gate/inherited/flashkda-tests
run cli /gate/inherited/verify_launch_arguments.py
for model in qwen38-flash-next glm53-flash; do
  podman run --rm --pull=never --device nvidia.com/gpu=all \
    --security-opt label=disable --ipc=host -e CC=/bin/false -e PYTHONOPTIMIZE=0 \
    --entrypoint /bin/bash "$image" "/usr/local/bin/serve-$model-karmic-spark.sh" --help \
    2>&1 | tee "$receipt/$model-live-preflight.log"
done
# Collection is GPU-visible and must match exactly before executing matrices.
run inherited-collection /gate/inherited/tests/run_r38_regressions.py --collect-only-count
run memory-collection /gate/run_memory_gates.py --collect-only-count
run inherited-regressions /gate/inherited/tests/run_r38_regressions.py
run memory-regressions /gate/run_memory_gates.py
run draft-head /gate/inherited/tests/verify_glm53_nvfp4_draft_head_sm121.py
printf 'KARMIC-MAIN-BUILD-GATES-PASS\n' | tee "$receipt/GATES-OK"
