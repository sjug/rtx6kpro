#!/usr/bin/env bash
# Same boot, unchanged gates; retain the first semantic failure independently.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
out=$kit/qualification/glm-allocator-confirmation-20260921
mkdir "$out"
exec > >(tee "$out/execution.log") 2>&1
image=$(jq -er .serving_image_id "$kit/glm-serving.lock.json")
name=glm53-flash-nvfp4-karmic-spark-tp4
nodes=(sparky buddy rocky lucky)
observers=()
finish() {
  status=$?
  trap - EXIT
  for pid in "${observers[@]}"; do kill "$pid" 2>/dev/null || true; done
  for node in "${nodes[@]}"; do
    boot=$(ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" | jq -er '.[0].State.StartedAt')
    ssh -n -o BatchMode=yes "$node" "journalctl -k --since '$boot' --no-pager" > "$out/$node-kernel.log" 2>&1 || true
  done
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$out/status.txt"
  exit "$status"
}
trap finish EXIT
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" > "$out/$node-before.json"
  jq -e --arg image "$image" '.[0] | .Image==$image and .State.Running and (.Config.Env | index("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")!=null)' "$out/$node-before.json"
  ssh -n -o BatchMode=yes "$node" 'while :; do date -Is; grep -E "MemAvailable|MemFree|SwapFree" /proc/meminfo; cat /proc/buddyinfo; grep -E "allocstall|compact_stall|pgscan_direct" /proc/vmstat; sleep 1; done' > "$out/$node-memory.log" 2>&1 & observers+=("$!")
done
GLM_RECEIPT_DIR=$out EXPECTED_IMAGE_ID=$image GLM_GRID_VARIANT=karmic-allocator-aligned-sm121-tp4-dcp1-mtp3-native1m bash "$kit/qualify-glm.sh"
