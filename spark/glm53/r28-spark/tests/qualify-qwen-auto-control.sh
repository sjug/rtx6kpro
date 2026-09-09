#!/usr/bin/env bash
set -euo pipefail
repo=/home/jugs/git/rtx6kpro
r28=$repo/spark/glm53/r28-spark
out=$r28/qualification/qwen-20260908/auto-mtp3
bash "$r28/tests/switch-qwen-profile.sh" auto-mtp3
exec > >(tee "$out/control.log") 2>&1
export PYTHONUNBUFFERED=1
python3 "$repo/spark/qwen38-flash-next/verify-semantic-admission.py" --base-url http://dusty:8000 --model Qwen3.8-Flash-Next-NVFP4-4p89 --runs 1 --receipt-file "$out/semantic.jsonl"
bash "$r28/tests/qualify-qwen-concurrency-grid.sh" auto
python3 "$repo/spark/glm53/r27-spark/tests/probe-prefix-reuse.py" --base-url http://dusty:8000 --model Qwen3.8-Flash-Next-NVFP4-4p89 --label r28-auto-mtp3 --receipt-file "$out/prefix-reuse.jsonl"
jq -se 'length == 4 and all(.[]; .answer | contains("739184"))' "$out/prefix-reuse.jsonl"
echo "AUTO-CONTROL-COMPLETE $(date -Is)"
