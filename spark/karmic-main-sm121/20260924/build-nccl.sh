#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 check_nccl.py
case ${DRY_RUN:-0} in
  1) echo 'DRY-RUN: stock NCCL 2.30.7; offline ARM64/SM121 build; 20 jobs; idle dusty/kirby required'; exit 0 ;;
  0) ;;
  *) echo 'DRY_RUN must be 0 or 1' >&2; exit 78 ;;
esac
[[ $(hostname -s) == dusty ]] || { echo 'Build requires dusty' >&2; exit 78; }
podman image exists nvcr.io/nvidia/pytorch@sha256:237ecf9ac7373daf91b31bb4f86651ce1ce57b676366ed435aa1aba61dad81d5 || {
  echo 'Pinned NGC foundation is missing; refuse build' >&2; exit 78;
}
exec 9>../.build.lock
flock -n 9 || { echo 'Another build owns the pair lock' >&2; exit 78; }
head=$(podman ps -q) || { echo 'Dusty idle probe failed' >&2; exit 78; }
worker=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || {
  echo 'Kirby idle probe failed' >&2; exit 78;
}
[[ -z $head && -z $worker ]] || { echo 'Pair is serving; refuse build' >&2; exit 78; }
receipt="$PWD/receipts/nccl-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
exec > >(tee "$receipt/build.log") 2>&1
evidence_container=
finish() {
  local result=$?
  if [[ -n $evidence_container ]]; then podman rm "$evidence_container" || true; fi
  printf 'exit_status=%s\n' "$result" > "$receipt/status.txt"
}
trap finish EXIT
cp Dockerfile.nccl nccl.lock.json build-nccl.sh check_nccl.py prepare_nccl.py "$receipt/"
sha256sum Dockerfile.nccl nccl.lock.json build-nccl.sh check_nccl.py prepare_nccl.py > "$receipt/recipe.sha256"
python3 check_nccl.py --values > "$receipt/values.txt"
readarray -t values < "$receipt/values.txt"
[[ ${#values[@]} == 2 ]] || { echo 'Malformed build values' >&2; exit 78; }
# A non-serving component tag protects the expensive object from daily dangling prune.
podman build --pull=never --network=none --format docker --layers \
  --tag "localhost/voipmonitor/build-components:${receipt##*/}" \
  --iidfile "$receipt/image.id" -f Dockerfile.nccl \
  --build-arg "SOURCE_SHA256=${values[0]}" --build-arg "RECIPE_SHA256=${values[1]}" .
image=$(<"$receipt/image.id")
podman image inspect "$image" > "$receipt/image-inspect.json"
evidence_container=$(podman create --pull=never --network=none "$image" /bin/true)
for file in SHA256SUMS ldd.txt cubins.txt elf.txt compile.log; do
  podman cp "$evidence_container:/artifacts/nccl/$file" "$receipt/$file"
done
podman rm "$evidence_container"
evidence_container=
podman run --rm --pull=never --network=none "$image" \
  bash -c 'cd /artifacts/nccl && sha256sum -c SHA256SUMS'
printf 'BUILD-OK image=%s distributed_qualification=pending\n' "$image" | tee "$receipt/BUILD-OK"
