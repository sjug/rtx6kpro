#!/usr/bin/env bash
# Resume timing only after the same image passed meaningful output admission.
set -euo pipefail
export RESULTS_REPO=/home/jugs/git/rtx6kpro PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
started=$(date -Is)
base=$(cd "$(dirname "$0")" && pwd)
out=$base/../qualification/ds4-vision
expected=f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233
name=ds4-vision-karmic-tp2
grep -qx VISION-QUALIFICATION-PASS "$out/execution.log"
[[ ! -e $out/benchmark-driver.log ]] || exit 78
exec > >(tee "$out/benchmark-driver.log") 2>&1
telemetry=()
finish() {
  local status=$?
  for node in rusty toby; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "journalctl -k --since '$started' --no-pager" > "$out/$node-kernel.log" 2>&1 || true
  done
  for pid in "${telemetry[@]}"; do kill "$pid" 2>/dev/null || true; done
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$out/benchmark-status.txt"
}
trap finish EXIT
for node in rusty toby; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" > "$out/$node-pre-benchmark.json"
  jq -e --arg id "$expected" '.[0] | (.Image|ltrimstr("sha256:"))==$id and .State.Running and (.State.OOMKilled|not)' "$out/$node-pre-benchmark.json"
  # Rank-zero logging is intentional. Both actual startup commands disable
  # custom all-reduce; only the head emits the selected backend summary.
  grep -Fq -- '--disable-custom-all-reduce' "$out/$node-before-grid.log"
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps -f '$name'" > "$out/$node-benchmark-live.log" 2>&1 & telemetry+=("$!")
  ssh -n -o BatchMode=yes "$node" 'nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_event_reasons.active,temperature.gpu,power.draw --format=csv -l 1' > "$out/$node-benchmark-gpu.csv" 2>&1 & telemetry+=("$!")
done
grep -Fq "['PYNCCL']" "$out/rusty-before-grid.log"
echo '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  /home/jugs/git/llm-inference-bench/llm_decode_bench.py' | sha256sum -c -
echo "BENCHMARK-START $(date -Is)"
printf 'n\n' | env HOST=http://rusty:8000 MODEL=DeepSeek-V4-Flash-Vision-Exp \
  MODEL_FAMILY=deepseek-v4-flash MODEL_VARIANT=vision-exp \
  CAMPAIGN=2026-09-karmic-sm121-qualification VARIANT=karmic-sm121-tp2-dspark-k3-dglin-524k \
  CONCURRENCY=1,2,4 REPETITION=1 PYTHONUNBUFFERED=1 \
  /home/jugs/git/llm-inference-bench/run_bench.sh --calibration-cache "$out/token-calibration.json" --display-mode plain --duration 30 --max-total-tokens 6412288 \
  --metadata checkpoint_revision=6821d6ad3681a4b137b066b76094fa82ebd0a380 \
  --metadata image_id="$expected" --metadata nodes=rusty,toby \
  --metadata harness_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3
echo "BENCHMARK-END $(date -Is)"
results=/home/jugs/git/rtx6kpro/runs/deepseek-v4-flash/vision-exp/2026-09-karmic-sm121-qualification/throughput
mapfile -t grids < <(find "$results" -maxdepth 1 -name '*__karmic-sm121-tp2-dspark-k3-dglin-524k__r01.json')
[[ ${#grids[@]} == 1 ]] || exit 78
python3 "$base/../../glm53/r38-spark/qualification/validate-grid.py" "${grids[0]}" > "$out/grid-validation.json"
python3 -u "$base/../../ds4-vision/qualify.py" --out "$out/post-grid-semantic"
for node in rusty toby; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" > "$out/$node-final.json"
  jq -e '.[0].State | .Running and (.OOMKilled|not)' "$out/$node-final.json"
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps '$name'" > "$out/$node-after-grid.log" 2>&1
done
echo "DS4-KARMIC-QUALIFICATION-COMPLETE $(date -Is)"
