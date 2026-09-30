#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
runtime=${1:?Usage: build-serving.sh exact-runtime-image-id}
[[ $runtime =~ ^(sha256:)?[0-9a-f]{64}$ ]] || { echo 'Exact runtime image ID required' >&2; exit 78; }
sha256sum -c kit.sha256
case ${DRY_RUN:-0} in 0|1) ;; *) echo 'DRY_RUN must be 0 or 1' >&2; exit 78 ;; esac
if [[ ${DRY_RUN:-0} == 1 ]]; then
  echo "DRY_RUN: package $runtime, native gates, CLI preflight, regression matrices; no node operations"
  exit 0
fi
[[ $(hostname -s) == dusty && $(uname -m) == aarch64 ]] || exit 78
exec 9>.native-build.lock
flock -n 9 || exit 78
require_idle() {
  local idle peer
  idle=$(podman ps -q) || { echo 'Dusty probe failed' >&2; return 78; }
  peer=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || { echo 'Kirby probe failed' >&2; return 78; }
  [[ -z $idle && -z $peer ]] || { echo 'Pair must be idle' >&2; return 78; }
}
require_idle
labels=$(podman image inspect "$runtime" --format '{{json .Labels}}')
jq -e '."vllm.source-tree" == "182f7f4ac37b980a3bb81e74797a78e9dbf822cb" and ."b12x.source-tree" == "ff2218d710ca9cae71459e1b9eb7ca3fe10ae08e"' <<<"$labels" >/dev/null || { echo 'Wrong runtime sources' >&2; exit 78; }
receipt="build-receipts/serving-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
exec > >(tee "$receipt/build.log") 2>&1
trap 'printf "exit_status=%s\n" "$?" > "$receipt/status.txt"' EXIT
cp Dockerfile.serving build-serving.sh verify_runtime.py verify_launch_arguments.py gate_flashkda.py build.lock.json source-selection.json kit.sha256 "$receipt/"
cp -a launchers tests "$receipt/"
printf '%s\n' "$runtime" > "$receipt/runtime.id"
qwen=$(sha256sum launchers/serve-qwen38-flash-next-jj-main-spark.sh | cut -d' ' -f1)
glm=$(sha256sum launchers/serve-glm53-flash-jj-main-spark.sh | cut -d' ' -f1)
fingerprint=$(printf '%s' "${runtime#sha256:}" | sha256sum | cut -d' ' -f1)
retention="localhost/voipmonitor/build-components:jj-main-${receipt##*/}"
podman build --pull=never --format docker --layers --tag "$retention" --iidfile "$receipt/image.id" \
  -f Dockerfile.serving --build-arg "RUNTIME_IMAGE=$runtime" \
  --build-arg "QWEN_LAUNCHER_INPUT_SHA256=$qwen" --build-arg "GLM_LAUNCHER_INPUT_SHA256=$glm" \
  --build-arg "CACHE_INPUT_SHA256=$fingerprint" \
  --build-arg "RECIPE_INPUT_SHA256=$(sha256sum Dockerfile.serving | cut -d' ' -f1)" .
image=$(<"$receipt/image.id")
podman image inspect "$image" > "$receipt/image-inspect.json"
require_idle
gpu=(podman run --rm --device nvidia.com/gpu=all --security-opt label=disable --ipc=host
  -e PYTHONUNBUFFERED=1 -e B12X_PRINT_COMPILE_PROGRESS=1 -e B12X_GLM53_GPU_TEST=1
  -v "$PWD:/gate:ro")
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
printf 'BUILD-GATES-PASS id=%s; model qualification pending\n' "$image" | tee "$receipt/BUILD-OK"
# Retention only. Node runners must first be bound to this exact image and reviewed.
