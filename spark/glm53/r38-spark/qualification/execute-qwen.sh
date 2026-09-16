#!/usr/bin/env bash
# Authorized Qwen rollout. Preserve R32 containers and every model/JIT cache.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
root=/home/jugs/git/bld-jj-r38-spark
export EXPECTED_IMAGE_ID=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
old=qwen38-flash-next-nvfp4-jj-r32-tp2
new=qwen38-flash-next-nvfp4-jj-r38-tp2
[[ ! -e $base/execution.log ]] || { echo 'Execution exists; inspect before retry'; exit 78; }
exec > >(tee "$base/execution.log") 2>&1
finish() {
  local status=$?
  if [[ -n ${telemetry_pid:-} ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$base/execution-status.txt"
}
trap finish EXIT
mkdir -p "$base/previous-qwen-r32"
for node in dusty kirby; do
  actual=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman image inspect localhost/voipmonitor/vllm:jj-r38-spark-sm121 --format '{{.Id}}'")
  [[ $actual == "$EXPECTED_IMAGE_ID" ]] || exit 78
  ssh -n -o BatchMode=yes "$node" "podman inspect '$old'" > "$base/previous-qwen-r32/$node.json"
  jq -e '.[0] | .State.Running and (.Image|ltrimstr("sha256:"))=="74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c"' "$base/previous-qwen-r32/$node.json"
  ssh -n -o BatchMode=yes "$node" "! podman container exists '$new'"
  role=worker; [[ $node != dusty ]] || role='head'
  ssh -n -o BatchMode=yes "$node" "DRY_RUN=1 ROLE=$role EXPECTED_IMAGE_ID=$EXPECTED_IMAGE_ID bash '$root/run-qwen38-flash-next-jj-r38-spark-tp2-node.sh'" > "$base/previous-qwen-r32/$node-render.txt"
done
echo "CUTOVER $(date -Is)"
bash "$base/sample-qwen.sh" "$base/qwen-20260915/telemetry" & telemetry_pid=$!
for node in kirby dusty; do
  ssh -n -o BatchMode=yes "$node" "podman stop -t 60 '$old'"
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps '$old'" > "$base/previous-qwen-r32/$node.log" 2>&1
done
bash "$base/launch-qwen.sh" initial
bash "$base/qualify-qwen.sh" initial
bash "$base/benchmark-qwen.sh" initial
bash "$base/launch-qwen.sh" aligned-mtp0
bash "$base/qualify-qwen.sh" aligned-mtp0
bash "$base/launch-qwen.sh" aligned-mtp3-restored
bash "$base/qualify-qwen.sh" aligned-mtp3-restored
REPETITION=2 bash "$base/benchmark-qwen.sh" aligned-mtp3-restored
results=/home/jugs/git/llm-inference-bench/results/runs/qwen3.8-flash-next/nvfp4-4p89
baseline=$results/2026-09-jj-r32-vs-r29/throughput/20260910T120516-0400__jj-r32-aligned-sm121-tp2-mtp3__r02.json
for repetition in 01 02; do
  mapfile -t paths < <(find "$results/2026-09-jj-r38-vs-r32/throughput" -maxdepth 1 -name "*__jj-r38-aligned-sm121-tp2-mtp3__r$repetition.json")
  [[ ${#paths[@]} == 1 ]] || exit 78
  python3 "$base/compare-qwen-grids.py" "$baseline" "${paths[0]}" > "$base/r32-vs-r38-qwen-r$repetition.json"
done
echo "QWEN-R38-BATTERIES-COMPLETE $(date -Is)"
echo 'Aligned MTP3 remains serving. Review measured deltas before GLM cutover.'
