#!/usr/bin/env bash
# Focused experiment only. Does not promote auto or replace the aligned grid.
set -euo pipefail
repo=/home/jugs/git/rtx6kpro
base=$(cd "$(dirname "$0")" && pwd)
out=$base/qwen-20260909/auto-mtp3
bash "$base/launch-qwen.sh" auto-mtp3
exec > >(tee "$out/control.log") 2>&1
export PYTHONUNBUFFERED=1
common=(--base-url http://dusty:8000 --model Qwen3.8-Flash-Next-NVFP4-4p89)
python3 "$repo/spark/qwen38-flash-next/verify-semantic-admission.py" "${common[@]}" --runs 1 --receipt-file "$out/semantic.jsonl"
legacy=$repo/spark/glm53/r27-spark/qualification/qwen-20260906
python3 "$legacy/probe-concurrency-identical-vs-distinct.py" | tee "$out/identical-vs-distinct.txt"
python3 "$legacy/probe-concurrency-fresh-extension-repeat.py" | tee "$out/fresh-extension-repeat.txt"
python3 "$repo/spark/glm53/r28-spark/tests/probe-qwen-head-of-line.py" --policy auto --receipt-file "$out/head-of-line.jsonl"
python3 "$repo/spark/glm53/r27-spark/tests/probe-prefix-reuse.py" "${common[@]}" --label r29-auto-mtp3 --receipt-file "$out/prefix-reuse.jsonl"
jq -se 'length == 4 and all(.[]; .answer | contains("739184"))' "$out/prefix-reuse.jsonl"
echo "AUTO-CONTROL-COMPLETE $(date -Is)"
