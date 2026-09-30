#!/usr/bin/env bash
# Approved single-variable loading retry. Preserve both previous containers.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
out=$kit/qualification/glm-allocator-20260921
remote=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121
image=$(jq -er .serving_image_id "$kit/glm-serving.lock.json")
old=glm53-flash-nvfp4-jj-r38-spark-tp4
new=glm53-flash-nvfp4-karmic-spark-tp4
nodes=(sparky buddy rocky lucky)
mkdir "$out"
exec > >(tee "$out/execution.log") 2>&1
started=$(date -Is)
observers=()
finish() {
  status=$?
  trap - EXIT
  for pid in "${observers[@]}"; do kill "$pid" 2>/dev/null || true; done
  for node in "${nodes[@]}"; do
    ssh -n -o BatchMode=yes "$node" "journalctl -k --since '$started' --no-pager" > "$out/$node-kernel.log" 2>&1 || true
    ssh -n -o BatchMode=yes "$node" "podman logs --timestamps --since '$started' '$new'" > "$out/$node-container.log" 2>&1 || true
    ssh -n -o BatchMode=yes "$node" "podman inspect '$new'" > "$out/$node-final.json" 2>&1 || true
  done
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$out/status.txt"
  exit "$status"
}
trap finish EXIT
curl -fsS --max-time 10 http://sparky:8000/metrics > "$out/before.metrics"
python3 - "$out/before.metrics" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
for key in ('running', 'waiting'):
    values = re.findall(r'^vllm:num_requests_' + key + r'\{[^\n]*\} ([\d.e+-]+)$', text, re.M)
    if not values or any(float(v) != 0 for v in values):
        raise SystemExit('GLM not idle: ' + key)
PY
for node in "${nodes[@]}"; do
  running=$(ssh -n -o BatchMode=yes "$node" 'podman ps --format "{{.Names}}"')
  [[ $running == "$old" ]] || exit 78
  ssh -n -o BatchMode=yes "$node" "podman inspect '$old'" > "$out/$node-r38-before.json"
  jq -e '.[0] | .State.Running and .Image=="ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5"' "$out/$node-r38-before.json"
  ssh -n -o BatchMode=yes "$node" "podman inspect '$new'" > "$out/$node-failed-before.json"
  jq -e --arg image "$image" '.[0] | (.State.Running|not) and .Image==$image' "$out/$node-failed-before.json"
  ssh -n -o BatchMode=yes "$node" "! podman container exists '$new-load-failed-20260921' && podman image exists '$image'"
  scp -q "$kit/run-glm-tp4-node.sh" "$node:$remote/"
done
for node in buddy rocky lucky sparky; do
  ssh -n -o BatchMode=yes "$node" "podman stop -t 60 '$old' && podman rename '$new' '$new-load-failed-20260921'"
done
for node in "${nodes[@]}"; do
  idle=$(ssh -n -o BatchMode=yes "$node" 'podman ps -q')
  [[ -z $idle ]] || exit 78
  ssh -n -o BatchMode=yes "$node" 'while :; do date -Is; grep -E "MemAvailable|MemFree|SwapFree" /proc/meminfo; cat /proc/buddyinfo; grep -E "allocstall|compact_stall|pgscan_direct" /proc/vmstat; sleep 1; done' > "$out/$node-memory.log" 2>&1 & observers+=("$!")
done
for node in buddy rocky lucky sparky; do
  role=worker; [[ $node != sparky ]] || role='head'
  ssh -n -o BatchMode=yes "$node" "ROLE=$role EXPECTED_IMAGE_ID=$image PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash '$remote/run-glm-tp4-node.sh'"
  ssh -n -o BatchMode=yes "$node" "podman inspect '$new'" > "$out/$node-started.json"
done
python3 - "$out" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for node in ('sparky', 'buddy', 'rocky', 'lucky'):
    before = json.loads((root / f'{node}-failed-before.json').read_text())[0]
    after = json.loads((root / f'{node}-started.json').read_text())[0]
    if before['Image'] != after['Image'] or before['Config']['Cmd'] != after['Config']['Cmd']:
        raise SystemExit(f'{node}: image/command drift')
    old, new = set(before['Config']['Env']), set(after['Config']['Env'])
    if new - old != {'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'} or old - new:
        raise SystemExit(f'{node}: environment drift')
print('ALLOCATOR-ONLY-IDENTITY-PASS', flush=True)
PY
GLM_RECEIPT_DIR=$out EXPECTED_IMAGE_ID=$image GLM_GRID_VARIANT=karmic-allocator-aligned-sm121-tp4-dcp1-mtp3-native1m bash "$kit/qualify-glm.sh"
