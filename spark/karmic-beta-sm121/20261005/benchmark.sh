#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$base/../../.." && pwd)
qualification=${QUALIFICATION_ROOT:-$base/qualification/qad-7c4f1bc1}
qualified="$qualification/mtp3"
out=${BENCHMARK_OUTPUT:-$qualified}
image=${EXPECTED_IMAGE_ID:?built image required}
baseline=${BASELINE_GRID:?matched production QAD grid required; use capture-baseline.sh first}
python3 "$base/compare-production.py" --model qwen --harness-sha256 cc9bb06a4f1142d2bf08afbcfc0e57d8a465a0bd180eee74e276dea70a82071c --validate-baseline "$baseline"
budget=$(python3 - "$baseline" <<'PY'
import json
import sys
budget = json.load(open(sys.argv[1]))['metadata']['max_total_tokens']
if type(budget) is not int or budget <= 0:
    raise SystemExit('Production baseline needs a positive integer token budget')
print(budget)
PY
)
name=qwen38-flash-next-nvfp4-karmic-beta-20261005-tp2
[[ -f $qualified/CORRECTNESS-OK && -f $qualified/counting/COUNTING-OK && -f $qualified/REPLAY-HEALTH-REVIEW-OK && ! -e $out/benchmark.log ]] || exit 78
mkdir -p "$out"
for node in dusty kirby; do
  ssh -n "$node" "podman inspect '$name'" > "$out/$node-benchmark.json"
  jq -e --arg image "$image" '.[0] | .State.Running and .Image==$image and
    (.Config.Env | index("VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0") != null) and
    (.Config.Env | index("NUM_SPECULATIVE_TOKENS=3") != null) and
    (.Config.Env | index("MODEL_REVISION=7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd") != null) and
    (.Args | index("--recurrent-checkpoint-policy") as $i | $i != null and .[$i+1]=="aligned")' "$out/$node-benchmark.json" >/dev/null
  [[ $(jq -r '.[0].Id' "$out/$node-benchmark.json") == $(jq -r '.[0].Id' "$qualification/window/$node-container.json") ]] || exit 78
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
bench=$(bash "$base/freeze_harness.sh")
echo "5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2  $bench/run_bench.sh" | sha256sum -c -
echo "cc9bb06a4f1142d2bf08afbcfc0e57d8a465a0bd180eee74e276dea70a82071c  $bench/llm_decode_bench.py" | sha256sum -c -
export RESULTS_REPO="$repo" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HOST=http://dusty:8000 MODEL=Qwen3.8-Flash-Next CONCURRENCY=${CONCURRENCY:-1,2,4}
export MODEL_FAMILY=qwen3.8-flash-next MODEL_VARIANT=nvfp4
export CAMPAIGN=2026-10-karmic-beta-20261005-sm121-qualification VARIANT=karmic-beta-20261005-qad-7c4f1bc1-hc-off-mtp3 REPETITION=${REPETITION:-1}
printf 'n\n' | "$bench/run_bench.sh" --duration 30 --contexts "${CONTEXTS:-0,16k,32k,64k,128k}" \
  --coding-peak --display-mode plain --kv-budget "$budget" --calibration-cache "$qualified/token-calibration.json" \
  --metadata "image_id=$image" --metadata checkpoint_revision=7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd \
  --metadata recurrent_checkpoint_policy=aligned --metadata hc_tp=0 --metadata vllm_commit=f1c2508f1018f47ee8a78478a819d202f359887b --metadata b12x_commit=52640cb15d4ad1c7230f747e72c45d45dc2da681 \
  --metadata run_bench_sha256=5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2 \
  --metadata llm_decode_bench_sha256=cc9bb06a4f1142d2bf08afbcfc0e57d8a465a0bd180eee74e276dea70a82071c

raw=$(sed -n 's/^Results saved to: //p' "$out/benchmark.log" | tail -n 1)
python3 "$base/compare-production.py" --model qwen --harness-sha256 cc9bb06a4f1142d2bf08afbcfc0e57d8a465a0bd180eee74e276dea70a82071c "$baseline" "$raw" > "$out/production-vs-candidate.json"
