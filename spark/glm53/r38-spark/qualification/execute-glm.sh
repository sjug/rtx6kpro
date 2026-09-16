#!/usr/bin/env bash
# Authorized four-node R38 test. Qwen and DS4 are out of scope.
set -euo pipefail
[[ ${GLM_CUTOVER_APPROVED:-0} == 1 ]] || { echo "A separately approved GLM cutover is required"; exit 78; }
base=$(cd "$(dirname "$0")" && pwd)
root=/home/jugs/git/bld-jj-r38-spark
out=${GLM_RECEIPT_DIR:?new receipt directory required}
mkdir -p "$out"
[[ ! -e $out/execution.log ]] || exit 78
exec > >(tee "$out/execution.log") 2>&1
finish() { local status=$?; printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$out/status.txt"; }
trap finish EXIT
image=${EXPECTED_IMAGE_ID:?gated image ID required}
[[ $image =~ ^[0-9a-f]{64}$ ]] || exit 78
old=glm53-flash-nvfp4-jj-r32-spark-tp4
new=glm53-flash-nvfp4-jj-r38-spark-tp4
nodes=(sparky buddy rocky lucky)
echo '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  /home/jugs/git/llm-inference-bench/llm_decode_bench.py' | sha256sum -c -
echo "VERIFY-STAGED-IMAGES $(date -Is)"
for node in "${nodes[@]}"; do
  actual=$(ssh -n -o BatchMode=yes "$node" "podman image inspect localhost/voipmonitor/vllm:jj-r38-spark-sm121 --format '{{.Id}}'")
  [[ $actual == "$image" ]] || exit 78
  expected_launcher=$(jq -er '.launchers["serve-glm53-flash-jj-r38-spark.sh"]' "$base/../source.lock.json")
  ssh -n -o BatchMode=yes "$node" "bash '$root/tests/check-glm-launcher-identity.sh' '$image' '$root/launchers/serve-glm53-flash-jj-r38-spark.sh' '$expected_launcher'"
  ssh -n -o BatchMode=yes "$node" "podman inspect '$old'" > "$out/$node-r32.json"
  jq -e '.[0] | .State.Running and (.Image|ltrimstr("sha256:"))=="74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c"' "$out/$node-r32.json"
  ssh -n -o BatchMode=yes "$node" "! podman container exists '$new'"
  ssh -n -o BatchMode=yes "$node" "ROLE=$([[ $node == sparky ]] && echo head || echo worker) DRY_RUN=1 EXPECTED_IMAGE_ID=$image bash '$root/run-glm53-flash-jj-r38-spark-tp4-node.sh'" > "$out/$node-render.txt"
done
echo "CUTOVER $(date -Is)"
for node in buddy rocky lucky sparky; do
  ssh -n -o BatchMode=yes "$node" "podman stop -t 60 '$old'"
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps '$old'" > "$out/$node-r32.log" 2>&1
done
for node in buddy rocky lucky sparky; do
  role=worker; [[ $node != sparky ]] || role='head'
  ssh -n -o BatchMode=yes "$node" "ROLE=$role EXPECTED_IMAGE_ID=$image bash '$root/run-glm53-flash-jj-r38-spark-tp4-node.sh'"
done
bash "$base/qualify-glm.sh"
results=/home/jugs/git/llm-inference-bench/results/runs/glm-5.3-flash/nvfp4
mapfile -t baseline < <(find "$results/2026-09-jj-r32-vs-r29/throughput" -maxdepth 1 -name '20260910T160142-0400__jj-r32-aligned-sm121-tp4-dcp1-mtp3-native1m__r02.json')
mapfile -t candidate < <(find "$results/2026-09-jj-r38-vs-r32/throughput" -maxdepth 1 -name '*__jj-r38-aligned-sm121-tp4-dcp1-mtp3-native1m__r01.json')
[[ ${#baseline[@]} == 1 && ${#candidate[@]} == 1 ]] || exit 78
python3 "$base/compare-qwen-grids.py" "${baseline[0]}" "${candidate[0]}" > "$out/r32-vs-r38.json"
echo "GLM-R38-COMPLETE $(date -Is)"
echo 'R38 aligned MTP3 remains serving. R32 preserved; results await review.'
