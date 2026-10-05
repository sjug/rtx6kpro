#!/usr/bin/env bash
# Approved Qwen pair window only. Never touches GLM, DS4 or benchmark source.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$base/../../.." && pwd)
hcbase="$repo/spark/karmic-main-sm121/20260922/qsa865-hcbase"
image=${EXPECTED_IMAGE_ID:?built image ID required}
remote=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121/20261005
name=qwen38-flash-next-nvfp4-karmic-beta-20261005-tp2
qualification=${QUALIFICATION_ROOT:-$base/qualification/qad-7c4f1bc1}
out="$qualification/mtp3"
window="$qualification/window"
[[ $image =~ ^[0-9a-f]{64}$ && ! -e $out && ! -e $window ]] || exit 78
for node in dusty kirby; do
  running=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" 'podman ps -q') || exit 78
  [[ -z $running ]] || { echo "$node is busy"; exit 78; }
  actual=$(ssh -n "$node" 'podman image inspect localhost/voipmonitor/vllm:karmic-beta-20261005-spark-sm121 --format "{{.Id}}"')
  [[ ${actual#sha256:} == "$image" ]] || exit 78
  staged=$(ssh -n "$node" "sha256sum '$remote/run-qwen.sh'" | cut -d' ' -f1)
  [[ $staged == "$(sha256sum "$base/run-qwen.sh" | cut -d' ' -f1)" ]] || { echo "$node runner drift"; exit 78; }
done
mkdir -p "$window"
exec > >(tee "$window/execution.log") 2>&1
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$since" > "$window/start.txt"
observers=()
finish() {
  local result=$?
  trap - EXIT
  for pid in "${observers[@]}"; do kill "$pid" 2>/dev/null || true; done
  for node in dusty kirby; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman logs --timestamps '$name'" > "$window/$node-final.log" 2>&1 || true
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "journalctl -k --since '$since' --no-pager -o short-iso" > "$window/$node-kernel.log" 2>&1 || true
  done
  printf 'exit_status=%s\n' "$result" > "$window/status.txt"
  if (( result != 0 )); then
    echo 'QUALIFICATION STOPPED: preserve candidate logs, inspect pair, restore the retained karmic-beta-20260929 Qwen containers (kirby worker, then dusty head) if unsafe. No benchmark or promotion.'
  fi
  exit "$result"
}
trap finish EXIT
for node in dusty kirby; do
  bash "$repo/spark/jj-main-sm121/observe-qwen.sh" "$node" > "$window/$node-telemetry.log" 2>&1 &
  observers+=("$!")
done
for node in kirby dusty; do
  role=worker
  [[ $node != dusty ]] || role='head'
  ssh -n "$node" "cd '$remote' && ROLE=$role EXPECTED_IMAGE_ID=$image bash run-qwen.sh"
  ssh -n "$node" "podman inspect '$name'" > "$window/$node-container.json"
  jq -e --arg image "$image" '.[0] | .State.Running and .Image==$image and
    (.Config.Env | index("VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0") != null) and
    (.Config.Env | index("NUM_SPECULATIVE_TOKENS=3") != null) and
    (.Args | index("--max-cudagraph-capture-size") as $i | $i != null and .[$i+1]=="32")' "$window/$node-container.json" >/dev/null
done
python3 -u "$repo/spark/karmic-main-sm121/20260922/qualify.py" --out "$out"
for node in dusty kirby; do
  ssh -n "$node" "podman logs '$name'" > "$window/$node-boot.log" 2>&1
  grep -q 'Model loading took' "$window/$node-boot.log"
  if grep -q 'Sharding HyperConnection projections' "$window/$node-boot.log"; then
    echo 'Unexpected HC sharding'; exit 78
  fi
done
python3 -u "$hcbase/probe-counting.py" --out "$out/counting"
python3 -u "$repo/spark/glm53/r38-spark/qualification/probes/probe-qwen-head-of-line.py" --policy aligned --receipt-file "$out/head-of-line.jsonl"
python3 -u "$repo/spark/glm53/r38-spark/qualification/probes/probe-concurrency-identical-vs-distinct.py" | tee "$out/identical-vs-distinct.log"
# Deliberately stop before timing until the private replay and health are reviewed.
echo 'KARMIC-BETA-QWEN-SERVING-GATES-COMPLETE: private replay and health review required before benchmark.'
