#!/usr/bin/env bash
# Existing standard harness, with all results and calibration kept in this repo.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$base/../.." && pwd)
repetition=1
out="$base/qualification/qwen-r32-return"
admission="$base/qualification/qwen-r32-restored-20260921"
image=74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c
jq -e '.status=="R32-RESTORED"' "$admission/status.json" >/dev/null || { echo 'Correctness battery has not passed'; exit 78; }
mkdir -p "$out"
[[ ! -e $out/benchmark.log ]] || { echo 'Benchmark receipt already exists'; exit 78; }
for node in dusty kirby; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" \
    'podman inspect qwen38-flash-next-nvfp4-jj-r32-tp2' > "$out/$node-benchmark-inspect.json"
  jq -e --arg image "$image" '.[0] | .State.Running and .Image==$image and
    (.Config.Env | index("NUM_SPECULATIVE_TOKENS=3") != null) and
    (.Args | index("--recurrent-checkpoint-policy") as $i | $i != null and .[$i+1]=="aligned")' \
    "$out/$node-benchmark-inspect.json" >/dev/null
  [[ $(jq -r '.[0].Id' "$out/$node-benchmark-inspect.json") == \
     $(jq -r '.Id' "$admission/$node-r32-final.json") ]] || {
    echo 'Container differs from the correctness-qualified boot'; exit 78;
  }
done
started=$(date -Is)
telemetry=()
finish() {
  status=$?
  for node in dusty kirby; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "journalctl -k --since '$started' --no-pager" > "$out/$node-kernel.log" 2>&1 || true
  done
  for pid in "${telemetry[@]}"; do kill "$pid" 2>/dev/null || true; done
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$out/status.txt"
}
trap finish EXIT
for node in dusty kirby; do
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps --since '$started' -f qwen38-flash-next-nvfp4-jj-r32-tp2" > "$out/$node-live.log" 2>&1 & telemetry+=("$!")
  ssh -n -o BatchMode=yes "$node" 'nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_event_reasons.active,temperature.gpu,power.draw --format=csv -l 1' > "$out/$node-gpu.csv" 2>&1 & telemetry+=("$!")
done
exec > >(tee "$out/benchmark.log") 2>&1
bench=/home/jugs/git/llm-inference-bench
echo "5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2  $bench/run_bench.sh" | sha256sum -c -
echo "2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  $bench/llm_decode_bench.py" | sha256sum -c -
export RESULTS_REPO="$repo" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HOST=http://dusty:8000 MODEL=Qwen3.8-Flash-Next CONCURRENCY=1,2,4
export MODEL_FAMILY=qwen3.8-flash-next MODEL_VARIANT=nvfp4-4p89
export CAMPAIGN=2026-09-karmic-sm121-qualification VARIANT=r32-return-aligned-mtp3
export REPETITION=$repetition
echo "BENCHMARK-START $(date -Is)"
printf 'n\n' | "$bench/run_bench.sh" --duration 30 --contexts 0,16k,32k,64k,128k \
  --display-mode plain --calibration-cache "$out/token-calibration.json" \
  --metadata "image_id=$image" \
  --metadata checkpoint_revision=c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d \
  --metadata recurrent_checkpoint_policy=aligned \
  --metadata run_bench_sha256=5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2 \
  --metadata llm_decode_bench_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3
echo "BENCHMARK-END $(date -Is)"
for node in dusty kirby; do
  ssh -n -o BatchMode=yes "$node" 'podman logs --timestamps qwen38-flash-next-nvfp4-jj-r32-tp2' \
    > "$out/$node-after-grid.log" 2>&1
done
