#!/usr/bin/env bash
# Candidate-only correctness then the unchanged standard grid. No node restart.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$base/../.." && pwd)
out="$base/qualification/qwen-mtp3-cache-agreement"
window="$base/qualification/window-cache-agreement"
mkdir -p "$window"
[[ ! -e $window/execution.log ]] || { echo 'Execution receipt exists'; exit 78; }
exec > >(tee "$window/execution.log") 2>&1
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$since" > "$window/observer-start.txt"
observers=()
finish() {
  status=$?
  trap - EXIT
  for pid in "${observers[@]}"; do kill "$pid" 2>/dev/null || true; done
  for node in dusty kirby; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" \
      'podman logs --timestamps qwen38-flash-next-nvfp4-jj-main-tp2' > "$window/$node-final.log" 2>&1 || true
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" \
      "journalctl -k --since '$since' --no-pager -o short-iso" > "$window/$node-kernel.log" 2>&1 || true
  done
  printf 'exit_status=%s\n' "$status" > "$window/status.txt"
  exit "$status"
}
trap finish EXIT
for node in dusty kirby; do
  bash "$base/observe-qwen.sh" "$node" > "$window/$node-telemetry.log" 2>&1 &
  observers+=("$!")
done
python3 -u "$base/qualify_qwen.py" --out "$out"
python3 -u "$repo/spark/glm53/r38-spark/qualification/probes/probe-qwen-head-of-line.py" \
  --policy aligned --receipt-file "$out/head-of-line.jsonl"
python3 -u "$repo/spark/glm53/r38-spark/qualification/probes/probe-concurrency-identical-vs-distinct.py" \
  | tee "$out/identical-vs-distinct.log"
bash "$base/benchmark-qwen.sh" 1
echo 'QWEN-JJ-MAIN-EXECUTION-COMPLETE: inspect correctness, grid and health before promotion'
