#!/usr/bin/env bash
# User-approved quiet-window supplement. Never overwrite the original grid.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
old=$base/receipts/qualification
out=$base/receipts/c4-16k-recheck
[[ ! -e $out ]] || { echo 'Recheck receipts already exist'; exit 78; }
mkdir -p "$out"
exec > >(tee "$out/driver.log") 2>&1
export RESULTS_REPO=/home/jugs/git/rtx6kpro PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
started=$(date -Is)
name=ds4-vision-jj-r38p-tp2
expected=ab3ed5285a81c9fd4df3c9001021c420368e7667177470aa2e055420f89ecdcf
pids=()
finish() {
  local status=$?
  for node in rusty toby; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "journalctl -k --since '$started' --no-pager; podman logs --timestamps --since '$started' '$name'" > "$out/$node-window.log" 2>&1 || true
  done
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$out/status.txt"
}
trap finish EXIT
grep -qx VISION-QUALIFICATION-PASS "$old/execution.log"
grep -qx 'STRUCTURED-STRESS-PASS 32' "$old/execution.log"
for node in rusty toby; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman inspect '$name'" > "$out/$node-before.json"
  jq -e --arg id "$expected" '.[0] | (.Image|ltrimstr("sha256:"))==$id and .State.Running and (.State.OOMKilled|not)' "$out/$node-before.json"
  [[ $(jq -r '.[0].Id' "$out/$node-before.json") == $(jq -r '.[0].Id' "$old/$node-container.json") ]]
  ssh -n -o BatchMode=yes "$node" 'nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_event_reasons.active,temperature.gpu,power.draw --format=csv -l 1' > "$out/$node-gpu.csv" 2>&1 & pids+=("$!")
done
curl -fsS --max-time 10 http://rusty:8000/metrics > "$out/metrics-before.txt"
python3 -c 'import re,sys; s=open(sys.argv[1]).read(); v=re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+(\S+)",s,re.M); assert len(v)>=2 and all(float(x)==0 for x in v), "Endpoint not idle"' "$out/metrics-before.txt"
echo '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  /home/jugs/git/llm-inference-bench/llm_decode_bench.py' | sha256sum -c -
echo '5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2  /home/jugs/git/llm-inference-bench/run_bench.sh' | sha256sum -c -
printf 'n\n' | env HOST=http://rusty:8000 MODEL=DeepSeek-V4-Flash-Vision-Exp \
  MODEL_FAMILY=deepseek-v4-flash MODEL_VARIANT=vision-exp CAMPAIGN=2026-09-r38p-grammar-qualification \
  VARIANT=jj-r38p-tp2-dspark-k3-dglin-524k-c4-16k-recheck CONCURRENCY=4 REPETITION=2 \
  /home/jugs/git/llm-inference-bench/run_bench.sh --contexts 16k --skip-prefill \
  --calibration-cache "$old/token-calibration.json" --display-mode plain --duration 30 --max-total-tokens 6412288 \
  --metadata checkpoint_revision=6821d6ad3681a4b137b066b76094fa82ebd0a380 \
  --metadata image_id="$expected" --metadata nodes=rusty,toby \
  --metadata harness_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3
results=$RESULTS_REPO/runs/deepseek-v4-flash/vision-exp/2026-09-r38p-grammar-qualification/throughput
mapfile -t grids < <(find "$results" -maxdepth 1 -name '*__jj-r38p-tp2-dspark-k3-dglin-524k-c4-16k-recheck__r02.json')
[[ ${#grids[@]} == 1 ]]
python3 "$base/validate-recheck.py" "${grids[0]}" > "$out/validation.json"
ssh -n -o BatchMode=yes rusty "podman logs --timestamps --since '$started' '$name'" > "$out/timed-server.log" 2>&1
if grep 'POST /v1/' "$out/timed-server.log" | grep -v '192.168.2.2:'; then
  echo 'Foreign inference overlapped the repeat'; exit 1
fi
python3 -u "$base/../../qualify.py" --out "$out/post-grid-semantic"
for node in rusty toby; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" > "$out/$node-final.json"
  jq -e --arg id "$expected" '.[0] | (.Image|ltrimstr("sha256:"))==$id and .State.Running and (.State.OOMKilled|not)' "$out/$node-final.json"
done
echo "DS4-R38P-SUPPLEMENT-COMPLETE $(date -Is)"
