#!/usr/bin/env bash
# Authorized GLM-only cutover. Preserve stopped R27 containers and all caches.
set -euo pipefail
repo=/home/jugs/git/rtx6kpro
out=$repo/spark/glm53/r28-spark/qualification/glm-20260908
root=/home/jugs/git/bld-jj-r28-spark
old=glm53-flash-nvfp4-jj-r27-spark-tp4
new=glm53-flash-nvfp4-jj-r28-spark-tp4
tag=localhost/voipmonitor/vllm:jj-r28-spark-sm121
image=cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1
nodes=(sparky buddy rocky lucky)
exec > >(tee "$out/cutover.log") 2>&1
echo "CUTOVER-START $(date -Is)"
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$old'; podman logs --since 15m '$old'" > "$out/$node-before-stop.txt" 2>&1
  ssh -n -o BatchMode=yes "$node" "test -s '$root/jj-r28-spark-sm121.docker.tar'; ! podman container exists '$new'"
done
for node in buddy rocky lucky sparky; do
  ssh -n -o BatchMode=yes "$node" "podman stop -t 60 '$old'"
done
echo "R27-STOPPED $(date -Is)"
pids=()
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "podman load -i '$root/jj-r28-spark-sm121.docker.tar'" > "$out/$node-load.log" 2>&1 & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || { echo 'Image load failed; R27 containers retained'; exit 1; }
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "podman image inspect '$tag'" > "$out/$node-image.json"
  jq -e --arg image "$image" '.[0] | (.Id|ltrimstr("sha256:"))==$image' "$out/$node-image.json"
  cmp <(jq -S '.[0]|{Id,ManifestType,RootFS,labels:.Config.Labels}' "$out/source-image.json") \
      <(jq -S '.[0]|{Id,ManifestType,RootFS,labels:.Config.Labels}' "$out/$node-image.json")
  ssh -n -o BatchMode=yes "$node" 'date -Is; free -b' > "$out/$node-before-launch.txt"
done
echo "ALL-IMAGE-IDENTITIES-VERIFIED $(date -Is)"
for node in buddy rocky lucky; do
  ssh -n -o BatchMode=yes "$node" "EXPECTED_IMAGE_ID=$image RECURRENT_CHECKPOINT_POLICY=aligned ROLE=worker bash '$root/run-glm53-flash-jj-r28-spark-tp4-node.sh'"
done
ssh -n -o BatchMode=yes sparky "EXPECTED_IMAGE_ID=$image RECURRENT_CHECKPOINT_POLICY=aligned ROLE=head bash '$root/run-glm53-flash-jj-r28-spark-tp4-node.sh'"
echo "R28-LAUNCHED $(date -Is)"
bash "$repo/spark/glm53/r28-spark/tests/qualify-glm-r28-aligned.sh"
