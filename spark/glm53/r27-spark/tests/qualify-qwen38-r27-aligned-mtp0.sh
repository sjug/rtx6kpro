#!/usr/bin/env bash
# Step 1 of the aligned-policy control: MTP0 boundary/native-context receipts
# plus the prefix-reuse probe under --recurrent-checkpoint-policy aligned.
set -euo pipefail
repo=/home/jugs/git/rtx6kpro; qwen=${repo}/spark/qwen38-flash-next; r27=${repo}/spark/glm53/r27-spark
base_url=${BASE_URL:-http://dusty:8000}; model=${MODEL:-Qwen3.8-Flash-Next-NVFP4-4p89}
date_tag=${DATE_TAG:-20260906}; out=${OUT_DIR:-${r27}/qualification/qwen-${date_tag}}
exec > >(tee -a "${out}/driver-aligned-mtp0.log") 2>&1
echo "START $(date -Is) aligned+MTP0"
completion() { curl -s --max-time 60 "${base_url}/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23 - 58? Reply with the number only.\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"; }
deadline=$(( $(date +%s) + 1200 )); ready=0
while [ "$(date +%s)" -lt "${deadline}" ]; do
  reply=$(completion || true)
  if grep -q '"content"' <<<"${reply}"; then printf '%s\n' "${reply}" > "${out}/first-completion-aligned-mtp0.json"; grep -q 333 <<<"${reply}" && { echo "READY $(date -Is)"; ready=1; break; }; echo "bad completion: ${reply:0:200}"; exit 1; fi
  sleep 10
done
[ "${ready}" = 1 ] || { echo "no completion within 20 minutes"; exit 1; }
echo "== MTP0 boundary + native context (aligned) =="
python3 "${qwen}/verify-native-context.py" --base-url "${base_url}" --model "${model}" --needle 739184 --lengths 2784 2785 131072 --max-tokens 64 \
  --receipt-file "${out}/jj-r27-aligned-mtp0-boundary-native-context-${date_tag}.jsonl"
echo "== prefix reuse (aligned, MTP0) =="
python3 "${r27}/tests/probe-prefix-reuse.py" --base-url "${base_url}" --model "${model}" --label aligned-mtp0 --receipt-file "${out}/jj-r27-prefix-reuse-aligned-mtp0-${date_tag}.jsonl"
echo "DONE $(date -Is)"
