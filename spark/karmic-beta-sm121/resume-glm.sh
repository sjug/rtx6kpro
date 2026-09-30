#!/usr/bin/env bash
# Resume the authorized window after stopped R38 and reviewed serving repair.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
out=$kit/qualification/glm-20260921
remote=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121
image=$(jq -er .serving_image_id "$kit/glm-serving.lock.json")
old=glm53-flash-nvfp4-jj-r38-spark-tp4
new=glm53-flash-nvfp4-karmic-spark-tp4
nodes=(sparky buddy rocky lucky)
[[ -f $out/REVIEW-OK && ! -e $out/resume.log ]] || exit 78
exec > >(tee "$out/resume.log") 2>&1
started=$(date -Is)
observers=()
finish() {
  status=$?
  trap - EXIT
  for pid in "${observers[@]}"; do kill "$pid" 2>/dev/null || true; done
  for node in "${nodes[@]}"; do
    ssh -n -o BatchMode=yes "$node" "journalctl -k --since '$started' --no-pager" > "$out/$node-resume-kernel.log" 2>&1 || true
    ssh -n -o BatchMode=yes "$node" "podman logs --timestamps --since '$started' '$new'" > "$out/$node-resume-container.log" 2>&1 || true
    ssh -n -o BatchMode=yes "$node" "podman inspect '$new'" > "$out/$node-resume-final.json" 2>&1 || true
  done
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$out/resume-status.txt"
  exit "$status"
}
trap finish EXIT
for node in "${nodes[@]}"; do
  idle=$(ssh -n -o BatchMode=yes "$node" 'podman ps -q') || exit 78
  [[ -z $idle ]] || exit 78
  actual=$(ssh -n -o BatchMode=yes "$node" 'podman image inspect localhost/voipmonitor/vllm:karmic-spark-sm121 --format "{{.Id}}"')
  [[ $actual == "$image" ]] || exit 78
  ssh -n -o BatchMode=yes "$node" "podman inspect '$old'" > "$out/$node-r38-retained.json"
  jq -e '.[0] | (.State.Running|not) and (.Image == "ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5")' "$out/$node-r38-retained.json"
  ssh -n -o BatchMode=yes "$node" "! podman container exists '$new'"
  scp -q "$kit/run-glm-tp4-node.sh" "$node:$remote/"
  scp -q "$kit/launchers/serve-glm53-flash-karmic-spark.sh" "$node:$remote/launchers/"
  ssh -n -o BatchMode=yes "$node" 'while :; do date -Is; grep -E "MemAvailable|MemFree|SwapFree" /proc/meminfo; cat /proc/buddyinfo; grep -E "allocstall|compact_stall|pgscan_direct" /proc/vmstat; sleep 1; done' > "$out/$node-resume-memory.log" 2>&1 & observers+=("$!")
done
for node in buddy rocky lucky sparky; do
  role=worker; [[ $node != sparky ]] || role='head'
  ssh -n -o BatchMode=yes "$node" "ROLE=$role EXPECTED_IMAGE_ID=$image bash '$remote/run-glm-tp4-node.sh'"
done
GLM_RECEIPT_DIR=$out EXPECTED_IMAGE_ID=$image bash "$kit/qualify-glm.sh"
