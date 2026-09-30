#!/usr/bin/env bash
# Authorized GLM qualification of the September 29 Karmic beta image on sparky/buddy/rocky/lucky.
# Production JJ R38 containers are stopped worker first and retained for rollback.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
out=$kit/../qualification/glm
remote=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121/20260929/glm
old=glm53-flash-nvfp4-jj-r38-spark-tp4
new=glm53-flash-nvfp4-karmic-beta-20260929-tp4
image=500ae05b98da0658c1a5e1820387f96f2121c5659bd7954ad1c5861f20934f05
r38=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
tag=localhost/voipmonitor/vllm:karmic-beta-20260929-spark-sm121
nodes=(sparky buddy rocky lucky)
declare -A ip=([sparky]=10.11.11.1 [buddy]=10.11.11.2 [lucky]=10.11.11.3 [rocky]=10.11.11.4)
[[ ${GLM_KARMIC_BETA_APPROVED:-0} == 1 ]] || exit 78
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
    echo 'STOPPED: qualification failed. R38 containers remain retained; restore with podman start, workers then sparky.'
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
  jq -e --arg id "$r38" '.[0] | .State.Running and (.Image == $id)' "$out/$node-r38-before.json"
  ssh -n -o BatchMode=yes "$node" "! podman container exists '$new'"
  ssh -n -o BatchMode=yes "$node" "mkdir -p '$remote/templates'"
  scp -q "$kit/run-glm-tp4-node.sh" "$node:$remote/"
  scp -q "$kit/templates/glm53-flash.jinja" "$node:$remote/templates/"
  role=worker; [[ $node != sparky ]] || role='head'
  ssh -n -o BatchMode=yes "$node" "ROLE=$role DRY_RUN=1 bash '$remote/run-glm-tp4-node.sh'" > "$out/$node-render.txt"
  ssh -n -o BatchMode=yes "$node" 'while :; do date -Is; grep -E "MemAvailable|MemFree|SwapFree" /proc/meminfo; cat /proc/buddyinfo; grep -E "allocstall|compact_stall|pgscan_direct" /proc/vmstat; sleep 1; done' > "$out/$node-memory.log" 2>&1 & observers+=("$!")
done
echo "STOP-R38 $(date -Is)"
for node in buddy rocky lucky sparky; do
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps '$old' > ~/logs/$old-$node-\$(date -u +%Y%m%dT%H%M%SZ).log 2>&1; podman stop -t 60 '$old'"
done
for node in "${nodes[@]}"; do
  [[ -z $(ssh -n -o BatchMode=yes "$node" 'podman ps -q') ]] || { echo "$node not idle after stop"; exit 78; }
done
# dusty serves Qwen: exactly one niced save and one send to idle sparky; sparky fans out over 10.11.11.x.
echo "TRANSFER $(date -Is)"
xfer=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121/20260929/receipts/glm-xfer
ssh -n -o BatchMode=yes dusty "set -e; mkdir -p '$xfer'; nice -n 10 podman save --format docker-archive -o '$xfer/beta.docker.tar' '$tag'; sha256sum '$xfer/beta.docker.tar' | cut -d' ' -f1 > '$xfer/sha'; ssh -o BatchMode=yes 10.11.11.1 'mkdir -p $xfer'; rsync -a --whole-file -e 'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com' '$xfer/beta.docker.tar' '$xfer/sha' 10.11.11.1:$xfer/; rm -rf '$xfer'"
ssh -n -o BatchMode=yes sparky "set -e; A=$xfer/beta.docker.tar; [ \"\$(sha256sum \$A | cut -d' ' -f1)\" = \"\$(cat $xfer/sha)\" ]; podman load -q -i \$A >/dev/null
for peer in ${ip[buddy]} ${ip[rocky]} ${ip[lucky]}; do
  ( ssh -o BatchMode=yes \$peer 'mkdir -p $xfer' && rsync -a --whole-file -e 'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com' \$A $xfer/sha \$peer:$xfer/ && ssh -o BatchMode=yes \$peer \"set -e; [ \\\$(sha256sum $xfer/beta.docker.tar | cut -d' ' -f1) = \\\$(cat $xfer/sha) ]; podman load -q -i $xfer/beta.docker.tar >/dev/null; rm -rf $xfer\" ) &
done; wait; rm -rf $xfer"
for node in "${nodes[@]}"; do
  actual=$(ssh -n -o BatchMode=yes "$node" "podman image inspect '$tag' --format '{{.Id}}'")
  [[ ${actual#sha256:} == "$image" ]] || { echo "$node image mismatch: $actual"; exit 78; }
  echo "TRANSFER-VERIFIED $node $image"
done
for node in buddy rocky lucky sparky; do
  role=worker; [[ $node != sparky ]] || role='head'
  ssh -n -o BatchMode=yes "$node" "ROLE=$role EXPECTED_IMAGE_ID=$image bash '$remote/run-glm-tp4-node.sh'"
done
GLM_RECEIPT_DIR=$out EXPECTED_IMAGE_ID=$image bash "$kit/qualify-glm.sh"
echo "GLM-KARMIC-BETA-QUALIFICATION-COMPLETE $(date -Is)"
