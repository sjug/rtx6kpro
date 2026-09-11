#!/usr/bin/env bash
# Authorized GLM R32 aligned qualification; does not restart any service.
set -euo pipefail
repo=/home/jugs/git/rtx6kpro
r27=$repo/spark/glm53/r27-spark
out=${GLM_RECEIPT_DIR:?new receipt directory required}/aligned-mtp3
base=http://sparky:8000
model=GLM-5.3-Flash
name=glm53-flash-nvfp4-jj-r32-spark-tp4
image=${EXPECTED_IMAGE_ID:?gated image ID required}
[[ $image =~ ^[0-9a-f]{64}$ ]] || exit 78
nodes=(sparky buddy rocky lucky)
resume=${1:-full}
[[ $resume == full || $resume == --from-native ]] || exit 2
mkdir -p "$out"
exec > >(tee -a "$out/driver.log") 2>&1
export PYTHONUNBUFFERED=1
echo "START $(date -Is)"
telemetry=()
cleanup() { for pid in "${telemetry[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps -f '$name'" > "$out/$node-container.log" 2>&1 & telemetry+=("$!")
  ssh -n -o BatchMode=yes "$node" 'nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_throttle_reasons.active,temperature.gpu,power.draw --format=csv -l 1' > "$out/$node-gpu.log" 2>&1 & telemetry+=("$!")
done
completion() {
  curl --fail --silent --show-error --connect-timeout 3 --max-time 90 "$base/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"What is 17 * 23 - 58? Reply with the number only."}],"max_tokens":128,"temperature":0,"reasoning_effort":"low"}'
}
valid_completion() {
  jq -e '.choices[0] | .finish_reason=="stop" and (.message.content|gsub("^\\s+|\\s+$";""))=="333"' "$1"
}
deadline=$(( $(date +%s) + 1800 ))
ready=0
while (( $(date +%s) < deadline )); do
  if completion > "$out/first-completion.json"; then
    valid_completion "$out/first-completion.json"
    ready=1; break
  fi
  sleep 10
done
[[ $ready == 1 ]] || { echo 'No correct completion within 30 minutes'; exit 1; }
echo "READY $(date -Is)"
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" > "$out/$node-container.json"
  jq -e --arg image "$image" '.[0] | (.Image|ltrimstr("sha256:"))==$image and .State.Running and
    (.Args|index("--recurrent-checkpoint-policy") as $i|$i!=null and .[$i+1]=="aligned") and
    (.Config.Env|index("NUM_SPECULATIVE_TOKENS=3")!=null) and
    (.Config.Env|index("DCP=1")!=null) and (.Config.Env|index("VLLM_GLM53_MTP_DRAFT_HEAD=bf16")!=null)' "$out/$node-container.json"
done
grep -Fq "['B12X_ROCENANTE', 'PYNCCL']" "$out/sparky-container.log" || {
  echo 'Exact RoCEnante/PyNCCL backend list not observed'; exit 1;
}
if [[ $resume == full ]]; then
echo "== short-prompt pool tails $(date -Is) =="
python3 "$repo/spark/glm53/r32-spark/qualification/verify-glm-short-pool.py" --base-url "$base" --model "$model" --receipt-file "$out/short-pool.jsonl"
echo "== semantic x3 $(date -Is) =="
python3 "$repo/spark/glm53/verify-semantic-admission.py" --base-url "$base" --model "$model" --runs 3 --receipt-file "$out/semantic.jsonl"
echo "== concurrency $(date -Is) =="
BASE_URL=$base MODEL=$model python3 "$r27/tests/glm/probe-concurrency-glm.py" | tee "$out/concurrency.txt"
echo "== frozen prefix pairs $(date -Is) =="
cp "$r27/qualification/glm-aligned-20260906/prefix-matrix-inputs.json" "$out/prefix-matrix-inputs.json"
[[ $(sha256sum "$out/prefix-matrix-inputs.json" | cut -d' ' -f1) == 31580c2d9f3e9d6c219d495969cae45b15c45bf8ae88fafd00dd39aa4c5f89b9 ]] || exit 1
OUT_DIR=$out BASE_URL=$base MODEL=$model python3 "$r27/tests/glm/prefix-matrix.py" r32-aligned
echo "== prefix triples $(date -Is) =="
OUT_DIR=$out BASE_URL=$base MODEL=$model python3 "$r27/tests/glm/prefix-triples.py" short
OUT_DIR=$out BASE_URL=$base MODEL=$model python3 "$r27/tests/glm/prefix-triples.py" long
fi
echo "== native context $(date -Is) =="
python3 "$repo/spark/glm53/r28-spark/tests/verify-glm-native-context.py" --base-url "$base" --model "$model" \
  --receipt-file "$out/native-context-glm.jsonl"
jq -se 'length==4 and all(.[];.valid and .finish_reason=="stop" and (.text|gsub("^\\s+|\\s+$";""))==.needle)' "$out/native-context-glm.jsonl"
echo "== standard grid $(date -Is) =="
echo '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  /home/jugs/git/llm-inference-bench/llm_decode_bench.py' | sha256sum -c -
HOST=$base MODEL=$model MODEL_FAMILY=glm-5.3-flash MODEL_VARIANT=nvfp4 \
  CAMPAIGN=2026-09-jj-r32-vs-r29 VARIANT=jj-r32-aligned-sm121-tp4-dcp1-mtp3-native1m CONCURRENCY=1,2,4 \
  /home/jugs/git/llm-inference-bench/run_bench.sh --duration 30 --max-total-tokens 6412288 \
  --metadata image_id="$image" --metadata checkpoint_revision=46aaae8a82032f77100f2f03e9cc11b391df3b4d \
  --metadata recurrent_checkpoint_policy=aligned \
  --metadata harness_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3 | tee "$out/benchmark.log"
completion > "$out/post-benchmark-completion.json"
valid_completion "$out/post-benchmark-completion.json"
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" > "$out/$node-final-container.json"
  jq -e '.[0].State | .Running and (.OOMKilled|not)' "$out/$node-final-container.json"
done
echo "QUALIFICATION-PASS $(date -Is)"
