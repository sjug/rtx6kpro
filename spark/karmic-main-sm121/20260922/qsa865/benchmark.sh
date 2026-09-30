#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$base/../../../.." && pwd)
qualified="$base/qualification/mtp3"
out=${BENCHMARK_OUTPUT:-$qualified}
image=${EXPECTED_IMAGE_ID:?built image required}
name=qwen38-flash-next-nvfp4-karmic-main-qsa865-tp2
[[ -f $qualified/CORRECTNESS-OK && -f $qualified/counting/COUNTING-OK && -f $qualified/REPLAY-HEALTH-REVIEW-OK && ! -e $out/benchmark.log ]] || exit 78
mkdir -p "$out"
for node in dusty kirby; do
  ssh -n "$node" "podman inspect '$name'" > "$out/$node-benchmark.json"
  jq -e --arg image "$image" '.[0] | .State.Running and .Image==$image and
    (.Config.Env | index("VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0") != null) and
    (.Config.Env | index("NUM_SPECULATIVE_TOKENS=3") != null) and
    (.Args | index("--recurrent-checkpoint-policy") as $i | $i != null and .[$i+1]=="aligned")' "$out/$node-benchmark.json" >/dev/null
  [[ $(jq -r '.[0].Id' "$out/$node-benchmark.json") == $(jq -r '.[0].Id' "$base/qualification/window/$node-container.json") ]] || exit 78
done
health="$out/benchmark-health"
mkdir "$health"
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$since" > "$health/start.txt"
observers=()
finish() {
  local result=$?
  trap - EXIT
  for pid in "${observers[@]}"; do kill "$pid" 2>/dev/null || true; done
  for node in dusty kirby; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman logs --timestamps '$name'" > "$health/$node-final.log" 2>&1 || true
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "journalctl -k --since '$since' --no-pager -o short-iso" > "$health/$node-kernel.log" 2>&1 || true
  done
  printf 'exit_status=%s\n' "$result" > "$health/status.txt"
  exit "$result"
}
trap finish EXIT
for node in dusty kirby; do
  bash "$repo/spark/jj-main-sm121/observe-qwen.sh" "$node" > "$health/$node-telemetry.log" 2>&1 &
  observers+=("$!")
done
exec > >(tee "$out/benchmark.log") 2>&1
bench=/home/jugs/git/llm-inference-bench
echo "5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2  $bench/run_bench.sh" | sha256sum -c -
echo "2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  $bench/llm_decode_bench.py" | sha256sum -c -
export RESULTS_REPO="$repo" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HOST=http://dusty:8000 MODEL=Qwen3.8-Flash-Next CONCURRENCY=${CONCURRENCY:-1,2,4}
export MODEL_FAMILY=qwen3.8-flash-next MODEL_VARIANT=nvfp4-4p89
export CAMPAIGN=2026-09-karmic-main-sm121-qualification VARIANT=karmic-main-qsa865-hc-off-mtp3 REPETITION=${REPETITION:-1}
printf 'n\n' | "$bench/run_bench.sh" --duration 30 --contexts "${CONTEXTS:-0,16k,32k,64k,128k}" \
  --display-mode plain --calibration-cache "$qualified/token-calibration.json" \
  --metadata "image_id=$image" --metadata checkpoint_revision=c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d \
  --metadata recurrent_checkpoint_policy=aligned --metadata hc_tp=0 --metadata qsa865=26b42cf9eb715b118e55f91f53fa0ab412a79c09 \
  --metadata run_bench_sha256=5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2 \
  --metadata llm_decode_bench_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3
