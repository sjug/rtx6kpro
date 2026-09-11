#!/usr/bin/env bash
# Contemporary R29 C4 control on the staging pair; preserve both containers.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
out=$base/qwen-20260910/r29-c4-control
mkdir -p "$out"
[[ ! -e "$out/execution.log" ]] || { echo 'Control receipt exists'; exit 78; }
exec > >(tee "$out/execution.log") 2>&1
finish() { local status=$?; printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$out/status.txt"; }
trap finish EXIT
r29=ee996ef8e531eb6e9c3208ebe30b141dc41702da1466e6ab2c47daa666d5694f
r32=74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c
old=qwen38-flash-next-nvfp4-jj-r29-tp2
new=qwen38-flash-next-nvfp4-jj-r32-tp2
inspect_pair() {
  local name=$1 image=$2 running=$3 label=$4
  for host in dusty kirby; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$host" "podman inspect '$name'" > "$out/$host-$label.json"
    jq -e --arg image "$image" --argjson running "$running" '
      .[0] | .State.Running == $running and (.Image | ltrimstr("sha256:")) == $image and
      (.Args | index("--recurrent-checkpoint-policy") as $i | $i != null and .[$i+1] == "aligned") and
      (.Config.Env | index("NUM_SPECULATIVE_TOKENS=3") != null and index("GPU_MEMORY_UTILIZATION=0.85") != null and
        index("MAX_MODEL_LEN=262144") != null and index("MAX_NUM_SEQS=4") != null and index("MAX_NUM_BATCHED_TOKENS=4096") != null)
    ' "$out/$host-$label.json"
  done
}
ready() {
  local name=$1 receipt=$2 deadline=$(( $(date +%s) + 1200 ))
  while (( $(date +%s) < deadline )); do
    for host in dusty kirby; do
      [[ $(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$host" "podman inspect '$name' --format '{{.State.Running}}'") == true ]] || return 1
    done
    if curl --fail --silent --show-error --connect-timeout 3 --max-time 60 http://dusty:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"Qwen3.8-Flash-Next-NVFP4-4p89","messages":[{"role":"user","content":"Calculate 17 times 23 minus 58. Reply with only the resulting integer."}],"chat_template_kwargs":{"enable_thinking":false},"temperature":0,"max_tokens":32}' > "$receipt"; then
      jq -e '.choices[0] | .finish_reason == "stop" and (.message.content | gsub("^\\s+|\\s+$"; "")) == "333"' "$receipt"
      return
    fi
    sleep 10
  done
  echo 'No correct completion within 20 minutes'; return 1
}
inspect_pair "$new" "$r32" true r32-before
inspect_pair "$old" "$r29" false r29-before
echo "CONTROL-BOOT $(date -Is)"
for host in kirby dusty; do
  ssh -n -o BatchMode=yes "$host" "podman logs --timestamps '$new'" > "$out/$host-r32-final.log" 2>&1
  ssh -n -o BatchMode=yes "$host" "podman stop -t 60 '$new'"
done
for host in kirby dusty; do ssh -n -o BatchMode=yes "$host" "podman start '$old'"; done
ready "$old" "$out/first-completion.json"
inspect_pair "$old" "$r29" true r29-running
for host in dusty kirby; do ssh -n -o BatchMode=yes "$host" "podman logs --timestamps '$old'" > "$out/$host-boot.log" 2>&1; done
RELEASE_LABEL=r29 bash "$base/qualify-qwen.sh" r29-c4-control
echo '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  /home/jugs/git/llm-inference-bench/llm_decode_bench.py' | sha256sum -c -
echo "CONTROL-BENCHMARK-START $(date -Is)"
HOST=http://dusty:8000 MODEL=Qwen3.8-Flash-Next-NVFP4-4p89 \
  MODEL_FAMILY=qwen3.8-flash-next MODEL_VARIANT=nvfp4-4p89 \
  CAMPAIGN=2026-09-jj-r32-vs-r29 VARIANT=jj-r29-aligned-sm121-tp2-mtp3-c4-control \
  CONCURRENCY=4 /home/jugs/git/llm-inference-bench/run_bench.sh \
  --metadata image_id="$r29" --metadata checkpoint_revision=c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d \
  --metadata recurrent_checkpoint_policy=aligned \
  --metadata harness_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3 \
  | tee "$out/benchmark.log"
echo "CONTROL-BENCHMARK-END $(date -Is)"
for host in dusty kirby; do ssh -n -o BatchMode=yes "$host" "podman logs --timestamps '$old'" > "$out/$host-after-grid.log" 2>&1; done
results=/home/jugs/git/llm-inference-bench/results/runs/qwen3.8-flash-next/nvfp4-4p89/2026-09-jj-r32-vs-r29/throughput
shopt -s nullglob
controls=("$results"/*__jj-r29-aligned-sm121-tp2-mtp3-c4-control__r01.json)
[[ ${#controls[@]} == 1 ]] || exit 78
python3 "$base/compare-qwen-grids.py" "${controls[0]}" \
  "$results/20260910T120516-0400__jj-r32-aligned-sm121-tp2-mtp3__r02.json" --concurrency 4 > "$out/comparison.json"
jq '{summary,prefill}' "$out/comparison.json"
for host in kirby dusty; do ssh -n -o BatchMode=yes "$host" "podman stop -t 60 '$old'"; done
for host in kirby dusty; do ssh -n -o BatchMode=yes "$host" "podman start '$new'"; done
ready "$new" "$out/r32-restored-completion.json"
inspect_pair "$new" "$r32" true r32-restored
echo "CONTROL-COMPLETE-R32-RESTORED $(date -Is)"
