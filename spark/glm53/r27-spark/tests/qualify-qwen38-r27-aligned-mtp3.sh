#!/usr/bin/env bash
# Step 2 of the aligned-policy control: aligned + MTP3, everything else fixed.
# Identical-prompt burst and head-of-line reproducers first, then the
# correctness set, the prefix-reuse probe, and the unchanged standard grid.
set -euo pipefail
repo=/home/jugs/git/rtx6kpro; qwen=${repo}/spark/qwen38-flash-next; r27=${repo}/spark/glm53/r27-spark
bench=/home/jugs/git/llm-inference-bench
base_url=${BASE_URL:-http://dusty:8000}; model=${MODEL:-Qwen3.8-Flash-Next-NVFP4-4p89}
date_tag=${DATE_TAG:-20260906}; out=${OUT_DIR:-${r27}/qualification/qwen-${date_tag}}
exec > >(tee -a "${out}/driver-aligned-mtp3.log") 2>&1
echo "START $(date -Is) aligned+MTP3"
completion() { curl -s --max-time 60 "${base_url}/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23 - 58? Reply with the number only.\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"; }
deadline=$(( $(date +%s) + 1200 )); ready=0
while [ "$(date +%s)" -lt "${deadline}" ]; do
  reply=$(completion || true)
  if grep -q '"content"' <<<"${reply}"; then printf '%s\n' "${reply}" > "${out}/first-completion-aligned-mtp3.json"; grep -q 333 <<<"${reply}" && { echo "READY $(date -Is)"; ready=1; break; }; echo "bad completion: ${reply:0:200}"; exit 1; fi
  sleep 10
done
[ "${ready}" = 1 ] || { echo "no completion within 20 minutes"; exit 1; }

echo "== identical-prompt burst vs distinct (aligned) =="
python3 "${out}/probe-concurrency-identical-vs-distinct.py" | tee "${out}/jj-r27-aligned-mtp3-concurrency-identical-vs-distinct-${date_tag}.txt"
echo "== fresh / extension / verbatim-repeat (aligned) =="
python3 "${out}/probe-concurrency-fresh-extension-repeat.py" | tee "${out}/jj-r27-aligned-mtp3-concurrency-fresh-extension-repeat-${date_tag}.txt"
echo "== head-of-line reproducer (aligned) =="
python3 "${out}/probe-repeat-head-of-line.py" | tee "${out}/jj-r27-aligned-mtp3-repeat-head-of-line-${date_tag}.txt"

echo "== semantic admission x3 (aligned) =="
python3 "${qwen}/verify-semantic-admission.py" --base-url "${base_url}" --model "${model}" --runs 3 \
  --receipt-file "${out}/jj-r27-aligned-mtp3-semantic-admission-3x-${date_tag}.jsonl"
echo "== MTP3 boundary + native context (aligned) =="
python3 "${qwen}/verify-native-context.py" --base-url "${base_url}" --model "${model}" --needle 739184 \
  --lengths 2848 2849 131072 262000 --max-tokens 64 \
  --receipt-file "${out}/jj-r27-aligned-mtp3-boundary-native-context-${date_tag}.jsonl"
echo "== fixed-token MTP3 acceptance (aligned) =="
python3 "${qwen}/benchmark-fixed-token-acceptance.py" --base-url "${base_url}" --model "${model}" \
  --corpus-file "${qwen}/fixed-token-acceptance-corpus.json" --waves 3 --max-tokens 512 \
  --output "${out}/jj-r27-aligned-mtp3-fixed-token-acceptance-${date_tag}.json"
echo "== PR 667 transition probe (aligned) =="
python3 "${r27}/tests/probe-spec-padded-transition.py" --base-url "${base_url}" --model "${model}" \
  --phases 1,3,1,2,4,1 --rounds 3 --max-tokens 256 --spec-tokens 3 --capture-sizes 1,2,4,8,16,24,32 \
  --receipt-file "${out}/jj-r27-aligned-mtp3-padded-transition-${date_tag}.jsonl"
echo "== prefix reuse (aligned, MTP3) =="
python3 "${r27}/tests/probe-prefix-reuse.py" --base-url "${base_url}" --model "${model}" --label aligned-mtp3 \
  --receipt-file "${out}/jj-r27-prefix-reuse-aligned-mtp3-${date_tag}.jsonl"

echo "== standard 15-cell grid, unchanged harness (variant jj-r27-aligned-sm121-tp2-mtp3) =="
( cd "${bench}" && HOST="${base_url}" MODEL="${model}" MODEL_FAMILY=qwen3.8-flash-next \
    MODEL_VARIANT=nvfp4-4p89 CAMPAIGN=2026-09-jj-r27-vs-r26 VARIANT=jj-r27-aligned-sm121-tp2-mtp3 \
    CONCURRENCY=1,2,4 ./run_bench.sh )
echo "== post-benchmark completion =="
completion | tee "${out}/post-benchmark-completion-aligned-mtp3.json" | grep -q 333
echo "DONE $(date -Is)"
