#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# SM121 foundation tree: base image, shared contracts, gates and the dusty/kirby pair lock.
FOUNDATION=$(cd ../../karmic-main-sm121/20260922 && pwd)
python3 preflight.py
case ${DRY_RUN:-0} in
  1) echo 'DRY-RUN: offline tracked refresh of base 1a7a8acf; idle dusty/kirby; parent, PR865 and beta gates; tag only after gates'; exit 0 ;;
  0) ;;
  *) echo 'DRY_RUN must be 0 or 1' >&2; exit 78 ;;
esac
[[ $(hostname -s) == dusty ]] || { echo 'Build requires dusty' >&2; exit 78; }
exec 9>"$FOUNDATION/../.build.lock"
flock -n 9 || { echo 'Another build owns the pair lock' >&2; exit 78; }
idle_pair() {
  local head worker
  head=$(podman ps -q) || { echo 'Dusty probe failed' >&2; return 78; }
  worker=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || {
    echo 'Kirby probe failed' >&2; return 78;
  }
  [[ -z $head && -z $worker ]] || { echo 'Pair is serving; refuse build' >&2; return 78; }
}
idle_pair
podman image exists 1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc || {
  echo 'Pinned base image is absent on dusty' >&2; exit 78;
}
receipt="$PWD/receipts/build-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
exec > >(tee "$receipt/build.log") 2>&1
finish() { local result=$?; printf 'exit_status=%s\n' "$result" > "$receipt/status.txt"; }
trap finish EXIT
python3 preflight.py --values > "$receipt/values.txt"
readarray -t values < "$receipt/values.txt"
[[ ${#values[@]} == 6 ]] || { echo 'Malformed preflight values' >&2; exit 78; }
python3 preflight.py --manifest > "$receipt/recipe-inputs.json"
cp build.lock.json Dockerfile install.py build.sh "$receipt/"
podman build --pull=never --network=none --format docker --layers \
  --tag "localhost/voipmonitor/build-components:${receipt##*/}" --iidfile "$receipt/image.id" \
  --build-arg "BUILD_LOCK_INPUT=${values[0]}" --build-arg "SOURCE_LOCK_INPUT=${values[1]}" \
  --build-arg "CACHE_INPUT=${values[2]}" --build-arg "VLLM_TREE_INPUT=${values[3]}" \
  --build-arg "B12X_TREE_INPUT=${values[4]}" --build-arg "RECIPE_INPUT=${values[5]}" .
image=$(<"$receipt/image.id")
podman image inspect "$image" > "$receipt/image-inspect.json"
idle_pair
bash "$FOUNDATION/gate.sh" "$image" "$receipt"
gate() {
  local log=$1 script=$2
  podman run --rm --pull=never --device nvidia.com/gpu=all --security-opt label=disable --ipc=host \
    -e PYTHONUNBUFFERED=1 -e PYTHONOPTIMIZE=0 -e B12X_PRINT_COMPILE_PROGRESS=1 \
    -v "$FOUNDATION:/gate:ro" -v "$PWD:/kit:ro" "$image" python "$script" 2>&1 | tee "$receipt/$log.log"
}
gate qsa865 /gate/qsa865/gate.py
gate karmic-beta /kit/gate_beta.py
grep -q '^KARMIC-BETA-GATE-PASS ' "$receipt/karmic-beta.log" || { echo 'Beta gate did not pass' >&2; exit 1; }
podman tag "$image" localhost/voipmonitor/vllm:karmic-beta-20260929-spark-sm121
printf 'BUILD-OK image=%s model_qualification=pending\n' "$image" | tee "$receipt/BUILD-OK"
