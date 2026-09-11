#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
bench=/home/jugs/git/llm-inference-bench
out=$root/receipts/benchmark
[[ -f "$root/receipts/QUALIFICATION-PASS" ]] || { echo 'Meaningful output must pass first'; exit 78; }
[[ ! -e "$out" ]] || { echo 'Benchmark receipts already exist'; exit 78; }
[[ $(sha256sum "$bench/llm_decode_bench.py" | cut -d' ' -f1) == 2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3 ]] || exit 78
mkdir -p "$out"
exec > >(tee "$out/driver.log") 2>&1
start=$(date -Is)
for node in rusty toby; do
  ssh -o BatchMode=yes "$node" 'podman inspect ds4-vision-jj-r32-tp2 --format "{{.Image}} {{.State.Running}} {{.State.OOMKilled}}"' \
    | tee "$out/$node-before.txt" \
    | grep -qx '74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c true false'
done
curl -fsS --max-time 10 http://rusty:8000/metrics > "$out/before.metrics"
export HOST=http://rusty:8000 MODEL=DeepSeek-V4-Flash-Vision-Exp
export MODEL_FAMILY=deepseek-v4-flash MODEL_VARIANT=vision-exp
export CAMPAIGN=2026-09-jj-r32-vision-spark-admission
export VARIANT=jj-r32-sm121-tp2-dspark-k3-dglin-524k
export CONCURRENCY=1,2,4 REPETITION=1 PYTHONUNBUFFERED=1
bash "$bench/run_bench.sh" --duration 30 --max-total-tokens 6412288 \
  --metadata checkpoint_revision=6821d6ad3681a4b137b066b76094fa82ebd0a380 \
  --metadata image_id=74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c \
  --metadata nodes=rusty,toby | tee "$out/benchmark.log"
curl -fsS --max-time 10 http://rusty:8000/metrics > "$out/after.metrics"
for node in rusty toby; do
  ssh -o BatchMode=yes "$node" "podman logs --since '$start' ds4-vision-jj-r32-tp2 2>&1" > "$out/$node.log"
  ssh -o BatchMode=yes "$node" 'podman inspect ds4-vision-jj-r32-tp2 --format "{{.Image}} {{.State.Running}} {{.State.OOMKilled}}"' \
    | tee "$out/$node-after.txt" \
    | grep -qx '74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c true false'
done
echo "BENCHMARK-COMPLETE $(date -Is)"
