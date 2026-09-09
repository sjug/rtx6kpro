#!/usr/bin/env bash
set -euo pipefail
repo=/home/jugs/git/rtx6kpro
r28=$repo/spark/glm53/r28-spark
out=$r28/qualification/qwen-20260908/aligned-mtp0
bash "$r28/tests/switch-qwen-profile.sh" aligned-mtp0
exec > >(tee "$out/control.log") 2>&1
export PYTHONUNBUFFERED=1
python3 "$repo/spark/qwen38-flash-next/verify-native-context.py" --base-url http://dusty:8000 --model Qwen3.8-Flash-Next-NVFP4-4p89 --needle 739184 --lengths 2784 2785 131072 --max-tokens 64 --receipt-file "$out/native-context.jsonl"
jq -se 'all(.[] | select(.kind == "context_needle"); .valid and .finish_reason == "stop" and .text == .needle)' "$out/native-context.jsonl"
python3 "$repo/spark/glm53/r27-spark/tests/probe-prefix-reuse.py" --base-url http://dusty:8000 --model Qwen3.8-Flash-Next-NVFP4-4p89 --label r28-aligned-mtp0 --receipt-file "$out/prefix-reuse.jsonl"
jq -se 'length == 4 and all(.[]; .answer | contains("739184"))' "$out/prefix-reuse.jsonl"
echo "MTP0-CONTROL-COMPLETE $(date -Is)"
