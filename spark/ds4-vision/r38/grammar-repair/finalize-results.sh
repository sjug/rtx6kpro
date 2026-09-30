#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out=$PWD/receipts/qualification
grep -qx 'exit_status=0' "$out/status.txt"
grep -qx 'exit_status=0' "$out/benchmark-status.txt"
grep -q '^DS4-R38P-QUALIFICATION-COMPLETE ' "$out/benchmark-driver.log"
baseline=/home/jugs/git/llm-inference-bench/results/runs/deepseek-v4-flash/vision-exp/2026-09-jj-r38-vs-r32/throughput/20260915T152722-0400__jj-r38-sm121-tp2-dspark-k3-dglin-524k__r01.json
results=/home/jugs/git/rtx6kpro/runs/deepseek-v4-flash/vision-exp/2026-09-r38p-grammar-qualification/throughput
mapfile -t grids < <(find "$results" -maxdepth 1 -name '*__jj-r38p-tp2-dspark-k3-dglin-524k__r01.json')
[[ ${#grids[@]} == 1 ]] || exit 78
python3 compare.py "$baseline" "${grids[0]}" > "$out/comparison-r38.json"
python3 summarize-health.py > "$out/health-summary.json"
cat "$out/comparison-r38.json"
