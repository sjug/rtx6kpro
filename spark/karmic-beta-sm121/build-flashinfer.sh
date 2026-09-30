#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ $(hostname -s) == dusty && $(uname -m) == aarch64 ]] || {
  echo 'Build requires dusty ARM64.' >&2; exit 78;
}
idle=$(podman ps -q) || exit 78
[[ -z $idle ]] || { echo 'Dusty is serving; refusing build.' >&2; exit 78; }
peer=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || exit 78
[[ -z $peer ]] || { echo 'Kirby is serving; refusing build.' >&2; exit 78; }
base=0abf0782ad35f4d9ef6497c8442fd9c59a91ba64c0044a645283d95533b3887f
[[ $(podman image inspect "$base" --format '{{.Id}}') == "$base" ]] || exit 78
receipt="build-receipts/flashinfer-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
tag="localhost/voipmonitor/build-components:karmic-${receipt##*/}"
exec > >(tee -a "$receipt/build.log") 2>&1
finish_receipt() {
  local status=$?
  printf 'exit_status=%s\n' "$status" > "$receipt/status.txt"
}
trap finish_receipt EXIT
printf '%s\n' "$tag" > "$receipt/retained-tag.txt"
cp Dockerfile.flashinfer build-flashinfer.sh verify_flashinfer_component.py "$receipt/"
sha256sum Dockerfile.flashinfer build-flashinfer.sh verify_flashinfer_component.py > "$receipt/recipe.sha256"
podman build --pull=never --format docker --layers --jobs=1 --tag "$tag" \
  --iidfile "$receipt/image.id" -f Dockerfile.flashinfer \
  --build-arg "RECIPE_SHA256=$(sha256sum Dockerfile.flashinfer | cut -d' ' -f1)" .
image=$(<"$receipt/image.id")
podman image inspect "$image" > "$receipt/image-inspect.json"
printf 'FLASHINFER-COMPONENT-BUILT id=%s\n' "$image" | tee "$receipt/BUILD-OK"
