#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
component=${1:?Usage: build-component.sh vllm|runtime [vllm-component-image-id]}
case "$component" in vllm|runtime) ;; *) echo 'Unknown component' >&2; exit 78 ;; esac
args=()
if [[ $component == runtime ]]; then
  vllm_image=${2:?Provide the newly built JJ-main vLLM component image ID}
  [[ $vllm_image =~ ^(sha256:)?[0-9a-f]{64}$ ]] || { echo 'An exact image ID is required' >&2; exit 78; }
  args+=(--build-arg "VLLM_COMPONENT_IMAGE=$vllm_image")
fi
sha256sum -c kit.sha256
case ${DRY_RUN:-0} in 0|1) ;; *) echo 'DRY_RUN must be 0 or 1' >&2; exit 78 ;; esac
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'DRY_RUN component=%s MAX_JOBS=20 NVCC_THREADS=1; no node or container operations\n' "$component"
  exit 0
fi
[[ $(hostname -s) == dusty && $(uname -m) == aarch64 ]] || { echo 'Build requires dusty ARM64' >&2; exit 78; }
exec 9>.native-build.lock
flock -n 9 || { echo 'Another build owns this kit' >&2; exit 78; }
idle=$(podman ps -q) || { echo 'Dusty serving probe failed' >&2; exit 78; }
peer=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || { echo 'Kirby serving probe failed' >&2; exit 78; }
[[ -z $idle && -z $peer ]] || { echo 'Pair must be idle in an approved build window' >&2; exit 78; }
if [[ $component == runtime ]]; then
  actual=$(podman image inspect "$vllm_image" --format '{{index .Labels "local-inference.vllm.commit"}}')
  [[ $actual == 8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368 ]] || { echo 'Wrong vLLM component source' >&2; exit 78; }
  actual=$(podman image inspect "$vllm_image" --format '{{index .Labels "local-inference.vllm.recipe.sha256"}}')
  expected=$(sha256sum Dockerfile.vllm | cut -d' ' -f1)
  [[ $actual == "$expected" ]] || { echo 'Wrong vLLM component recipe' >&2; exit 78; }
fi
receipt="build-receipts/${component}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
exec > >(tee -a "$receipt/build.log") 2>&1
trap 'printf "exit_status=%s\n" "$?" > "$receipt/status.txt"' EXIT
tag="localhost/voipmonitor/build-components:jj-main-${receipt##*/}"
printf '%s\n' "$tag" > "$receipt/retained-tag.txt"
cp "Dockerfile.$component" build-component.sh build.lock.json source-selection.json \
  verify_sources.py prepare_overlay.py spark-overlay.patch vllm-build-requirements.lock \
  install_runtime_payload.py prepare_flashkda_tests.py kit.sha256 "$receipt/"
cp inspect_shared_components.py "$receipt/"
python3 inspect_shared_components.py > "$receipt/shared-component-inspect.json"
find "$receipt" -type f ! -name build.log -print0 | sort -z | xargs -0 sha256sum > "$receipt/inputs.sha256"
podman build --pull=never --format docker --layers --jobs=1 --tag "$tag" \
  --iidfile "$receipt/image.id" -f "Dockerfile.$component" \
  --build-arg "RECIPE_SHA256=$(sha256sum "Dockerfile.$component" | cut -d' ' -f1)" "${args[@]}" .
image=$(<"$receipt/image.id")
podman image inspect "$image" > "$receipt/image-inspect.json"
printf 'COMPONENT-BUILT component=%s id=%s; GPU and model gates pending\n' "$component" "$image" | tee "$receipt/COMPONENT-BUILT"
