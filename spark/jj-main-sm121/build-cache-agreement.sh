#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
sha256sum -c cache-agreement-inputs.sha256
case ${DRY_RUN:-0} in
  1) echo 'DRY_RUN: pinned Python-only cache handshake backport, unchanged natives, full gates'; exit 0;;
  0) ;;
  *) exit 78;;
esac
[[ $(hostname -s) == dusty && $(uname -m) == aarch64 ]] || exit 78
exec 9>.native-build.lock
flock -n 9 || exit 78
require_idle() {
  local local_ids peer_ids
  local_ids=$(podman ps -q) || { echo 'Dusty probe failed'; return 78; }
  peer_ids=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || { echo 'Kirby probe failed'; return 78; }
  [[ -z $local_ids && -z $peer_ids ]] || { echo 'Pair must be idle'; return 78; }
}
require_idle
parent=e7926f763859ba5800cc24198ef297a12322fdaac93b55ac89b88abe02c1b1fb
[[ $(podman image inspect "$parent" --format '{{.Id}}') == "$parent" ]] || exit 78
receipt="build-receipts/cache-agreement-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
exec > >(tee "$receipt/build.log") 2>&1
trap 'printf "exit_status=%s\n" "$?" > "$receipt/status.txt"' EXIT
cp Dockerfile.cache-agreement cache-agreement.patch cache-agreement.lock.json \
  cache-agreement-inputs.sha256 install_cache_backport.py build-cache-agreement.sh gate-existing-image.sh "$receipt/"
cp -a tests "$receipt/"
podman build --pull=never --format docker --layers \
  --tag "localhost/voipmonitor/build-components:jj-main-${receipt##*/}" \
  --iidfile "$receipt/image.id" -f Dockerfile.cache-agreement .
image=$(<"$receipt/image.id")
podman image inspect "$image" > "$receipt/image-inspect.json"
require_idle
bash gate-existing-image.sh "$image" "$receipt"
podman tag "$image" localhost/voipmonitor/vllm:jj-main-spark-sm121
printf 'BUILD-GATES-PASS id=%s; model qualification pending\n' "$image" | tee "$receipt/BUILD-OK"
