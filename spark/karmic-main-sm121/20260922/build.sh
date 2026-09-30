#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 preflight.py
if [[ ${DRY_RUN:-0} == 1 ]]; then
  echo 'DRY-RUN: offline source refresh; idle dusty/kirby required; publish after GPU gates only'
  exit 0
fi
[[ $(hostname -s) == dusty ]] || { echo 'Build host must be dusty'; exit 78; }
exec 9>../.build.lock
flock -n 9 || { echo 'Another build owns the lock'; exit 78; }
idle_pair() {
  local head worker
  head=$(podman ps -q) || { echo 'Dusty serving probe failed'; return 78; }
  worker=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || {
    echo 'Kirby serving probe failed'; return 78;
  }
  [[ -z $head && -z $worker ]] || { echo 'Pair must be idle'; return 78; }
}
idle_pair
receipt="$PWD/build-receipts/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
exec > >(tee "$receipt/build.log") 2>&1
finish() {
  local result=$?
  printf 'exit_status=%s\n' "$result" > "$receipt/status.txt"
}
trap finish EXIT
readarray -t values < <(python3 preflight.py --values)
[[ ${#values[@]} == 7 ]] || { echo 'Invalid build input list'; exit 78; }
lock=${values[0]} recipe=${values[1]} cache=${values[2]}
qwen=${values[3]} glm=${values[4]} vtree=${values[5]} btree=${values[6]}
cp source.lock.json "$receipt/"
python3 preflight.py --manifest > "$receipt/recipe-inputs.json"
cp recipe-origin.json "$receipt/"
retention="localhost/voipmonitor/build-components:karmic-main-${receipt##*/}"
podman build --pull=never --network=none --format docker --layers \
  --tag "$retention" --iidfile "$receipt/image.id" \
  --build-arg "LOCK_INPUT_SHA256=$lock" --build-arg "RECIPE_INPUT_SHA256=$recipe" \
  --build-arg "CACHE_INPUT=$cache" --build-arg "QWEN_INPUT_SHA256=$qwen" \
  --build-arg "GLM_INPUT_SHA256=$glm" --build-arg "VLLM_TREE_INPUT=$vtree" \
  --build-arg "B12X_TREE_INPUT=$btree" .
image=$(<"$receipt/image.id")
podman image inspect "$image" > "$receipt/image-inspect.json"
idle_pair
bash gate.sh "$image" "$receipt"
podman tag "$image" localhost/voipmonitor/vllm:karmic-main-spark-sm121
printf 'BUILD-OK image=%s model_qualification=pending\n' "$image" | tee "$receipt/BUILD-OK"
