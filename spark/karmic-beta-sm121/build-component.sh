#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
component=${1:?Usage: build-component.sh nccl|vllm|lmcache|instanttensor|runtime-deps|runtime}
case "$component" in nccl|vllm|lmcache|instanttensor|runtime-deps|runtime) ;; *) echo 'Unknown component' >&2; exit 78 ;; esac
[[ $(hostname -s) == dusty && $(uname -m) == aarch64 ]] || exit 78
exec 9>.native-build.lock
flock -n 9 || { echo 'Another native component build owns the workspace.' >&2; exit 78; }
if systemctl --user is-active --quiet karmic-flashinfer-build.service; then
  echo 'FlashInfer is still building; refusing concurrent native build.' >&2; exit 78
fi
idle=$(podman ps -q) || exit 78
[[ -z $idle ]] || { echo 'Dusty is serving; refusing build.' >&2; exit 78; }
peer=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || exit 78
[[ -z $peer ]] || { echo 'Kirby is serving; refusing build.' >&2; exit 78; }
receipt="build-receipts/${component}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
tag="localhost/voipmonitor/build-components:karmic-${receipt##*/}"
exec > >(tee -a "$receipt/build.log") 2>&1
finish_receipt() {
  local status=$?
  printf 'exit_status=%s\n' "$status" > "$receipt/status.txt"
}
trap finish_receipt EXIT
printf '%s\n' "$tag" > "$receipt/retained-tag.txt"
cp "Dockerfile.$component" build-component.sh source-selection.json "$receipt/"
if [[ $component == vllm || $component == lmcache ]]; then cp vllm-build-requirements.lock "$receipt/"; fi
if [[ $component == vllm ]]; then cp spark-overlay.patch "$receipt/"; fi
if [[ $component == runtime-deps ]]; then
  cp runtime-arm64.lock runtime-arm64.in "$receipt/"
  cp -a runtime-inputs "$receipt/"
fi
if [[ $component == runtime ]]; then cp install_runtime_payload.py "$receipt/"; fi
inputs=("$receipt"/Dockerfile.* "$receipt"/*.sh "$receipt"/*.json)
if [[ $component == vllm || $component == lmcache ]]; then inputs+=("$receipt/vllm-build-requirements.lock"); fi
if [[ $component == vllm ]]; then inputs+=("$receipt/spark-overlay.patch"); fi
if [[ $component == runtime-deps ]]; then
  inputs+=("$receipt/runtime-arm64.lock" "$receipt/runtime-arm64.in")
  while IFS= read -r -d '' input; do inputs+=("$input"); done < <(find "$receipt/runtime-inputs" -type f -print0 | sort -z)
fi
if [[ $component == runtime ]]; then inputs+=("$receipt/install_runtime_payload.py"); fi
sha256sum "${inputs[@]}" > "$receipt/recipe.sha256"
podman build --pull=never --format docker --layers --jobs=1 --tag "$tag" \
  --iidfile "$receipt/image.id" -f "Dockerfile.$component" \
  --build-arg "RECIPE_SHA256=$(sha256sum "Dockerfile.$component" | cut -d' ' -f1)" .
image=$(<"$receipt/image.id")
podman image inspect "$image" > "$receipt/image-inspect.json"
printf 'COMPONENT-BUILT component=%s id=%s\n' "$component" "$image" | tee "$receipt/BUILD-OK"
