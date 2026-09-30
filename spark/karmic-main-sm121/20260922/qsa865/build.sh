#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 preflight.py
case ${DRY_RUN:-0} in
  1) echo 'DRY-RUN: image-ID-pinned offline derivative; idle pair; all parent GPU gates plus PR865; publish after gates'; exit 0 ;;
  0) ;;
  *) echo 'DRY_RUN must be 0 or 1' >&2; exit 78 ;;
esac
[[ $(hostname -s) == dusty ]] || { echo 'Build requires dusty' >&2; exit 78; }
exec 9>../../.build.lock
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
receipt="$PWD/../build-receipts/qsa865-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
exec > >(tee "$receipt/build.log") 2>&1
finish() {
  local result=$?
  printf 'exit_status=%s\n' "$result" > "$receipt/status.txt"
}
trap finish EXIT
python3 preflight.py --values > "$receipt/values.txt"
readarray -t values < "$receipt/values.txt"
[[ ${#values[@]} == 6 ]] || { echo 'Malformed preflight values' >&2; exit 78; }
python3 preflight.py --manifest > "$receipt/recipe-inputs.json"
cp backport.lock.json runtime.lock.json "$receipt/"
podman build --pull=never --network=none --format docker --layers \
  --tag "localhost/voipmonitor/build-components:${receipt##*/}" --iidfile "$receipt/image.id" \
  --build-arg "BACKPORT_LOCK_INPUT=${values[0]}" --build-arg "RUNTIME_LOCK_INPUT=${values[1]}" \
  --build-arg "CACHE_INPUT=${values[2]}" --build-arg "TREE_INPUT=${values[3]}" \
  --build-arg "RECIPE_INPUT=${values[4]}" --build-arg "B12X_TREE_INPUT=${values[5]}" .
image=$(<"$receipt/image.id")
podman image inspect "$image" > "$receipt/image-inspect.json"
idle_pair
bash ../gate.sh "$image" "$receipt"
podman run --rm --pull=never --device nvidia.com/gpu=all --security-opt label=disable --ipc=host \
  -e PYTHONUNBUFFERED=1 -e PYTHONOPTIMIZE=0 -e B12X_PRINT_COMPILE_PROGRESS=1 \
  -v "$PWD/..:/gate:ro" "$image" python /gate/qsa865/gate.py 2>&1 | tee "$receipt/qsa865.log"
podman tag "$image" localhost/voipmonitor/vllm:karmic-main-qsa865-spark-sm121
printf 'BUILD-OK image=%s model_qualification=pending\n' "$image" | tee "$receipt/BUILD-OK"
