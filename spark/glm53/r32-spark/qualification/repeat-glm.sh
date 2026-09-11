#!/usr/bin/env bash
# Quiet matched repeat after the user paused competing Pi traffic. No restart.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
out=$base/glm-20260910/quiet-repeat
[[ ! -e $out ]] || { echo 'Repeat receipts exist; inspect before retry'; exit 78; }
mkdir -p "$out"
exec > >(tee "$out/driver.log") 2>&1
export PYTHONUNBUFFERED=1
image=74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c
name=glm53-flash-nvfp4-jj-r32-spark-tp4
nodes=(sparky buddy rocky lucky)
telemetry=()
finish() {
  local status=$?
  for pid in "${telemetry[@]}"; do kill "$pid" 2>/dev/null || true; done
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$out/status.txt"
}
trap finish EXIT
echo "QUIET-REPEAT-START $(date -Is)"
echo '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  /home/jugs/git/llm-inference-bench/llm_decode_bench.py' | sha256sum -c -
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman inspect '$name'" > "$out/$node-before.json"
  jq -e --arg image "$image" '.[0] | .State.Running and (.State.OOMKilled|not) and
    (.Image|ltrimstr("sha256:"))==$image and
    (.Args|index("--recurrent-checkpoint-policy") as $i|$i!=null and .[$i+1]=="aligned") and
    (.Config.Env|index("NUM_SPECULATIVE_TOKENS=3")!=null) and
    (.Config.Env|index("DCP=1")!=null) and
    (.Config.Env|index("VLLM_GLM53_MTP_DRAFT_HEAD=bf16")!=null)' "$out/$node-before.json"
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps -f '$name'" > "$out/$node-container.log" 2>&1 & telemetry+=("$!")
  ssh -n -o BatchMode=yes "$node" 'nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_throttle_reasons.active,temperature.gpu,power.draw --format=csv -l 1' > "$out/$node-gpu.log" 2>&1 & telemetry+=("$!")
done
curl --fail --silent --show-error --max-time 15 http://sparky:8000/metrics > "$out/before.metrics"
awk '/^vllm:num_requests_(running|waiting)[{ ]/ {found++; if ($NF != 0) busy=1} END {exit !(found>=2 && !busy)}' "$out/before.metrics" || {
  echo 'Server has active/queued requests or lacks request metrics; no benchmark started'; exit 78;
}
echo "BENCHMARK-START $(date -Is)"
HOST=http://sparky:8000 MODEL=GLM-5.3-Flash MODEL_FAMILY=glm-5.3-flash MODEL_VARIANT=nvfp4 \
  CAMPAIGN=2026-09-jj-r32-vs-r29 VARIANT=jj-r32-aligned-sm121-tp4-dcp1-mtp3-native1m \
  REPETITION=2 CONCURRENCY=1,2,4 /home/jugs/git/llm-inference-bench/run_bench.sh \
  --duration 30 --max-total-tokens 6412288 \
  --metadata image_id="$image" --metadata checkpoint_revision=46aaae8a82032f77100f2f03e9cc11b391df3b4d \
  --metadata recurrent_checkpoint_policy=aligned --metadata quiet_repeat=user_paused_pi \
  --metadata harness_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3 \
  | tee "$out/benchmark.log"
echo "BENCHMARK-END $(date -Is)"
curl --fail --silent --show-error --max-time 15 http://sparky:8000/metrics > "$out/after.metrics"
curl --fail --silent --show-error --connect-timeout 3 --max-time 90 http://sparky:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"What is 17 * 23 - 58? Reply with the number only."}],"max_tokens":128,"temperature":0,"reasoning_effort":"low"}' > "$out/final-completion.json"
jq -e '.choices[0] | .finish_reason=="stop" and (.message.content|gsub("^\\s+|\\s+$";""))=="333"' "$out/final-completion.json"
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman inspect '$name'" > "$out/$node-after.json"
  jq -e --arg image "$image" '.[0] | .State.Running and (.State.OOMKilled|not) and (.Image|ltrimstr("sha256:"))==$image' "$out/$node-after.json"
done
results=/home/jugs/git/llm-inference-bench/results/runs/glm-5.3-flash/nvfp4
baseline=$results/2026-09-jj-r29-vs-r28/throughput/20260909T142658-0400__jj-r29-aligned-sm121-tp4-dcp1-mtp3-native1m__r01.json
c1=$results/2026-09-jj-r29-vs-r28/throughput/20260909T164052-0400__jj-r29-aligned-sm121-tp4-dcp1-mtp3-native1m__r02.json
shopt -s nullglob
repeats=("$results"/2026-09-jj-r32-vs-r29/throughput/*__jj-r32-aligned-sm121-tp4-dcp1-mtp3-native1m__r02.json)
[[ ${#repeats[@]} == 1 ]] || { echo 'Expected exactly one repetition 2'; exit 78; }
python3 "$base/compare-qwen-grids.py" "$baseline" "${repeats[0]}" > "$out/r29-vs-r32.json"
python3 "$base/compare-qwen-grids.py" "$c1" "${repeats[0]}" --concurrency 1 > "$out/r29-c1-repeat-vs-r32.json"
jq '{summary,prefill}' "$out/r29-vs-r32.json"
echo "QUIET-REPEAT-AND-COMPARISON-PASS $(date -Is)"
