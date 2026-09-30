#!/usr/bin/env bash
# User-approved (2026-09-30): qualify the QAD-distilled GLM checkpoint 175ae8ce on the production
# Karmic beta profile. The previous-weights production container is stopped worker first and
# retained as $name-46aaae8a-20260930 for rollback (podman start, workers then sparky).
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
out=$kit/../qualification/glm-qad-175ae8ce
remote=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121/20260929/glm
name=glm53-flash-nvfp4-karmic-beta-20260929-tp4
kept=$name-46aaae8a-20260930
image=500ae05b98da0658c1a5e1820387f96f2121c5659bd7954ad1c5861f20934f05
revision=175ae8ce3b5af842b0d0140dbeb43e9cfc557c49
nodes=(sparky buddy rocky lucky)
[[ ! -e $out ]] || { echo "receipt directory exists: $out"; exit 78; }
mkdir -p "$out"
exec > >(tee "$out/execution.log") 2>&1
echo "START $(date -Is)"
curl -fsS --max-time 10 http://sparky:8000/metrics | python3 -c '
import re, sys
t = sys.stdin.read()
for m in ("num_requests_running", "num_requests_waiting"):
    v = re.findall(r"^vllm:" + m + r"\{[^\n]*\} ([\d.e+-]+)$", t, re.M)
    if not v or any(float(x) for x in v): raise SystemExit("GLM not idle: " + m)'
for node in "${nodes[@]}"; do
  scp -q "$kit/run-glm-tp4-node.sh" "$node:$remote/"
  ssh -n -o BatchMode=yes "$node" "! podman container exists '$kept'"
  role=worker; [[ $node != sparky ]] || role=head
  ssh -n -o BatchMode=yes "$node" "ROLE=$role DRY_RUN=1 bash '$remote/run-glm-tp4-node.sh'" > "$out/$node-render.txt"
  grep -qF "MODEL_REVISION=$revision" "$out/$node-render.txt"
  grep -qF "snapshots/$revision" "$out/$node-render.txt"
  ! grep -q -- '--profiler-config' "$out/$node-render.txt"
done
echo "STOP-OLD-WEIGHTS $(date -Is)"
for node in buddy rocky lucky sparky; do
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps '$name' > ~/logs/$name-46aaae8a-$node-\$(date -u +%Y%m%dT%H%M%SZ).log 2>&1; podman stop -t 60 '$name' >/dev/null && podman rename '$name' '$kept'"
done
for node in "${nodes[@]}"; do
  [[ -z $(ssh -n -o BatchMode=yes "$node" 'podman ps -q') ]] || { echo "$node not idle after stop"; exit 78; }
done
echo "START-QAD $(date -Is)"
for node in buddy rocky lucky sparky; do
  role=worker; [[ $node != sparky ]] || role=head
  ssh -n -o BatchMode=yes "$node" "ROLE=$role EXPECTED_IMAGE_ID=$image bash '$remote/run-glm-tp4-node.sh'"
done
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" | jq -e --arg rev "$revision" \
    '.[0] | (.Config.Env|index("MODEL_REVISION="+$rev)!=null) and (.Args|index("--profiler-config")==null)' >/dev/null \
    || { echo "$node: revision or profiler contract failed"; exit 1; }
done
GLM_RECEIPT_DIR=$out EXPECTED_IMAGE_ID=$image GLM_CAMPAIGN=2026-09-qad-175ae8ce-qualification \
  GLM_CHECKPOINT_REVISION=$revision GLM_SKIP_R38_COMPARE=1 \
  GLM_GRID_VARIANT=karmic-beta-20260929-qad-175ae8ce-max-topp095-seq4-tp4-mtp3-native1m \
  bash "$kit/qualify-glm.sh"
echo "GLM-QAD-QUALIFICATION-COMPLETE $(date -Is)"
