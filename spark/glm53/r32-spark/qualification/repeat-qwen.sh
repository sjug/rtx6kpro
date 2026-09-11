#!/usr/bin/env bash
# Matched second-boot R32 repetition. No relaunch, image change or auto arm.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
[[ ! -e "$base/repeat.log" ]] || { echo 'Repeat receipt exists; inspect before retry'; exit 78; }
exec > >(tee "$base/repeat.log") 2>&1
finish() {
  local status=$?
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$base/repeat-status.txt"
}
trap finish EXIT
export EXPECTED_IMAGE_ID=74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c
for host in dusty kirby; do
  state=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    'podman inspect qwen38-flash-next-nvfp4-jj-r32-tp2')
  jq -e --arg image "$EXPECTED_IMAGE_ID" '
    .[0] | .State.Running and (.Image | ltrimstr("sha256:")) == $image and
    (.Args | index("--recurrent-checkpoint-policy") as $i | $i != null and .[$i+1] == "aligned") and
    (.Config.Env | index("NUM_SPECULATIVE_TOKENS=3") != null)
  ' <<<"$state"
done
echo "REPEAT-START $(date -Is)"
# Validate the final boot and seed the same shapes before a measured window.
bash "$base/qualify-qwen.sh" aligned-mtp3-restored
REPETITION=2 bash "$base/benchmark-qwen.sh" aligned-mtp3-restored
results=/home/jugs/git/llm-inference-bench/results/runs/qwen3.8-flash-next/nvfp4-4p89
baseline=$results/2026-09-jj-r29-vs-r28/throughput/20260909T092714-0400__jj-r29-aligned-sm121-tp2-mtp3__r01.json
shopt -s nullglob
repeats=("$results"/2026-09-jj-r32-vs-r29/throughput/*__jj-r32-aligned-sm121-tp2-mtp3__r02.json)
[[ ${#repeats[@]} == 1 ]] || { echo 'Expected one completed R32 repetition 2'; exit 78; }
python3 "$base/compare-qwen-grids.py" "$baseline" "${repeats[0]}" > "$base/r29-vs-r32-repeat.json"
jq '{summary,prefill}' "$base/r29-vs-r32-repeat.json"
echo "REPEAT-AND-COMPARISON-COMPLETE $(date -Is)"
