#!/usr/bin/env bash
set -euo pipefail
repo=/home/jugs/git/rtx6kpro
qwen=$repo/spark/qwen38-flash-next
r27=$repo/spark/glm53/r27-spark
out=$repo/spark/glm53/r28-spark/qualification/qwen-20260908/aligned-mtp3
mkdir -p "$out"
exec > >(tee "$out/correctness.log") 2>&1
export PYTHONUNBUFFERED=1
common=(--base-url http://dusty:8000 --model Qwen3.8-Flash-Next-NVFP4-4p89)
echo "START $(date -Is)"
python3 "$qwen/verify-semantic-admission.py" "${common[@]}" --runs 3 --receipt-file "$out/semantic.jsonl"
python3 "$qwen/verify-native-context.py" "${common[@]}" --needle 739184 --lengths 2848 2849 131072 262000 --max-tokens 64 --receipt-file "$out/native-context.jsonl"
python3 "$qwen/benchmark-fixed-token-acceptance.py" "${common[@]}" --corpus-file "$qwen/fixed-token-acceptance-corpus.json" --waves 3 --max-tokens 512 --output "$out/fixed-token-acceptance.json"
python3 "$r27/tests/probe-spec-padded-transition.py" "${common[@]}" --phases 1,3,1,2,4,1 --rounds 3 --max-tokens 256 --spec-tokens 3 --capture-sizes 1,2,4,8,16,24,32 --receipt-file "$out/padded-transition.jsonl"
python3 "$r27/tests/probe-prefix-reuse.py" "${common[@]}" --label r28-aligned-mtp3 --receipt-file "$out/prefix-reuse.jsonl"
echo "CORRECTNESS-BATTERY-OK $(date -Is)"
