#!/usr/bin/env bash
# Authorized GLM-only Karmic qualification. Retain R38 unchanged for rollback.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
out=$kit/qualification/glm-20260921
remote=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121
old=glm53-flash-nvfp4-jj-r38-spark-tp4
new=glm53-flash-nvfp4-karmic-spark-tp4
image=f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233
nodes=(sparky buddy rocky lucky)
[[ ! -e $out/execution.log ]] || exit 78
mkdir -p "$out"
exec > >(tee "$out/execution.log") 2>&1
started=$(date -Is)
observers=()
finish() {
  status=$?
  trap - EXIT
  for pid in "${observers[@]}"; do kill "$pid" 2>/dev/null || true; done
  for node in "${nodes[@]}"; do
    ssh -n -o BatchMode=yes "$node" "journalctl -k --since '$started' --no-pager" > "$out/$node-kernel.log" 2>&1 || true
    ssh -n -o BatchMode=yes "$node" "podman inspect '$new'" > "$out/$node-final.json" 2>&1 || true
  done
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$out/status.txt"
  if ((status != 0)); then
    echo 'STOPPED: qualification failed. Inspect receipts; R38 containers remain retained. No promotion.'
  fi
  exit "$status"
}
trap finish EXIT
curl -fsS --max-time 10 http://sparky:8000/metrics > "$out/preflight.metrics"
python3 - "$out/preflight.metrics" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
for metric in ('num_requests_running', 'num_requests_waiting'):
    values = re.findall(r'^vllm:' + metric + r'\{[^\n]*\} ([\d.e+-]+)$', text, re.M)
    if not values or any(float(v) != 0 for v in values):
        raise SystemExit('GLM is not verified idle: ' + metric)
PY
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$old'" > "$out/$node-r38-before.json"
  jq -e '.[0] | .State.Running and (.Image == "ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5")' "$out/$node-r38-before.json"
  ssh -n -o BatchMode=yes "$node" "! podman container exists '$new'"
  ssh -n -o BatchMode=yes "$node" "mkdir -p '$remote/launchers'"
  scp -q "$kit/run-glm-tp4-node.sh" "$node:$remote/"
  scp -q "$kit/launchers/serve-glm53-flash-karmic-spark.sh" "$node:$remote/launchers/"
  role=worker; [[ $node != sparky ]] || role='head'
  ssh -n -o BatchMode=yes "$node" "ROLE=$role DRY_RUN=1 bash '$remote/run-glm-tp4-node.sh'" > "$out/$node-render.txt"
  ssh -n -o BatchMode=yes "$node" 'while :; do date -Is; grep -E "MemAvailable|MemFree|SwapFree" /proc/meminfo; cat /proc/buddyinfo; grep -E "allocstall|compact_stall|pgscan_direct" /proc/vmstat; sleep 1; done' > "$out/$node-memory.log" 2>&1 & observers+=("$!")
done
scp -q "$kit/distribute-clusters.sh" "dusty:$remote/"
for node in buddy rocky lucky sparky; do
  ssh -n -o BatchMode=yes "$node" "podman stop -t 60 '$old'"
done
ssh -n -o BatchMode=yes dusty "QUALIFICATION_WINDOW=glm bash '$remote/distribute-clusters.sh' sparky buddy rocky lucky"
echo 'TRANSFER-COMPLETE. Waiting for reviewed launch approval receipt.'
while [[ ! -f $out/REVIEW-OK ]]; do sleep 5; done
for node in buddy rocky lucky sparky; do
  role=worker; [[ $node != sparky ]] || role='head'
  ssh -n -o BatchMode=yes "$node" "ROLE=$role EXPECTED_IMAGE_ID=$image bash '$remote/run-glm-tp4-node.sh'"
done
GLM_RECEIPT_DIR=$out EXPECTED_IMAGE_ID=$image bash "$kit/qualify-glm.sh"
echo "GLM-KARMIC-QUALIFICATION-COMPLETE $(date -Is)"
