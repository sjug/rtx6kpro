#!/usr/bin/env bash
# Qwen3.8-Flash-Next JJ r27 Spark qualification driver (workstation side).
# Runs against the live dusty/kirby TP2 pair and mirrors the r26 receipt set:
# readiness by completion, semantic admission x3, MTP3 boundary and native
# context retrieval, four greedy chat responses, fixed-token MTP3 acceptance,
# the PR 667 padded-replay transition probe, and the standard 15-cell grid.
# MTP0 boundary retrieval needs a relaunch and is run separately.
set -euo pipefail

repo=/home/jugs/git/rtx6kpro
qwen=${repo}/spark/qwen38-flash-next
r27=${repo}/spark/glm53/r27-spark
bench=/home/jugs/git/llm-inference-bench
base_url=${BASE_URL:-http://dusty:8000}
model=${MODEL:-Qwen3.8-Flash-Next-NVFP4-4p89}
date_tag=${DATE_TAG:-$(date +%Y%m%d)}
out=${OUT_DIR:-${r27}/qualification/qwen-${date_tag}}
mkdir -p "${out}"
log=${out}/driver.log
exec > >(tee -a "${log}") 2>&1
echo "START $(date -Is) base_url=${base_url} model=${model}"

completion() {
  curl -s --max-time 60 "${base_url}/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23 - 58? Reply with the number only.\"}],
    \"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
}
abort_pair() {
  echo "ABORT $(date -Is): $1"
  timeout 120 ssh -n kirby 'ROLE=stop ~/git/bld-jj-r27-spark/run-qwen38-flash-next-jj-r27-spark-tp2-node.sh' 2>&1 | tail -1
  timeout 120 ssh -n dusty 'ROLE=stop ~/git/bld-jj-r27-spark/run-qwen38-flash-next-jj-r27-spark-tp2-node.sh' 2>&1 | tail -1
  exit 1
}

echo "== readiness: 20-minute wall clock, 60 s per completion probe (never /v1/models) =="
deadline=$(( $(date +%s) + 1200 ))
while [ "$(date +%s)" -lt "${deadline}" ]; do
  reply=$(completion || true)
  if grep -q '"content"' <<<"${reply}"; then
    printf '%s\n' "${reply}" > "${out}/first-completion.json"
    grep -q '333' <<<"${reply}" && { echo "READY $(date -Is): 333 correct"; break; }
    echo "completion returned without 333: ${reply:0:200}"; exit 1
  fi
  sleep 10
done
test -f "${out}/first-completion.json" || abort_pair "no completion within the 20-minute window"

echo "== semantic admission x3 =="
python3 "${qwen}/verify-semantic-admission.py" --base-url "${base_url}" --model "${model}" --runs 3 \
  --receipt-file "${out}/jj-r27-mtp3-semantic-admission-3x-${date_tag}.jsonl"

echo "== MTP3 boundary + native context =="
python3 "${qwen}/verify-native-context.py" --base-url "${base_url}" --model "${model}" --needle 739184 \
  --lengths 2848 2849 131072 262000 --max-tokens 64 \
  --receipt-file "${out}/jj-r27-mtp3-boundary-native-context-${date_tag}.jsonl"

echo "== four greedy chat responses (collapse checks) =="
python3 - "${base_url}" "${model}" "${out}/jj-r27-mtp3-chat-greedy-${date_tag}.jsonl" <<'PY'
import hashlib, json, re, sys, urllib.request
base, model, receipt = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = [
    "Explain how a CUDA graph differs from stream capture replay, in three paragraphs.",
    "Describe the failure modes of speculative decoding with a draft head in about 200 words.",
    "Write a Python function that merges overlapping intervals and explain its complexity.",
    "Summarize how RoCE differs from InfiniBand for collective communication in a data center.",
]
records = []
with open(receipt, "w") as out:
    for prompt in prompts:
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 256, "temperature": 0,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.load(resp)
        text = payload["choices"][0]["message"].get("content") or ""
        words = text.split()
        grams = [" ".join(words[i:i + 6]) for i in range(max(0, len(words) - 5))]
        repeats = len(grams) - len(set(grams)) if grams else 0
        non_ascii = sum(ord(c) > 127 for c in text)
        valid = len(words) >= 40 and repeats < 5 and non_ascii < len(text) * 0.05
        rec = {"prompt": prompt, "completion_tokens": payload["usage"]["completion_tokens"],
               "finish_reason": payload["choices"][0]["finish_reason"], "words": len(words),
               "repeated_6grams": repeats, "sha256": hashlib.sha256(text.encode()).hexdigest(), "valid": valid}
        records.append(rec); out.write(json.dumps(rec, sort_keys=True) + "\n")
        print(json.dumps(rec, sort_keys=True))
    summary = {"kind": "summary", "model": model, "valid": all(r["valid"] for r in records)}
    out.write(json.dumps(summary, sort_keys=True) + "\n"); print(summary)
    sys.exit(0 if summary["valid"] else 1)
PY

echo "== fixed-token MTP3 acceptance =="
python3 "${qwen}/benchmark-fixed-token-acceptance.py" --base-url "${base_url}" --model "${model}" \
  --corpus-file "${qwen}/fixed-token-acceptance-corpus.json" --waves 3 --max-tokens 512 \
  --output "${out}/jj-r27-fixed-token-acceptance-${date_tag}.json"

echo "== PR 667 padded-replay transition probe =="
python3 "${r27}/tests/probe-spec-padded-transition.py" --base-url "${base_url}" --model "${model}" \
  --phases 1,3,1,2,4,1 --rounds 3 --max-tokens 256 --spec-tokens 3 --capture-sizes 1,2,4,8,16,24,32 \
  --receipt-file "${out}/jj-r27-mtp3-padded-transition-${date_tag}.jsonl"

echo "== standard 15-cell grid (campaign 2026-09-jj-r27-vs-r26) =="
( cd "${bench}" && HOST="${base_url}" MODEL="${model}" MODEL_FAMILY=qwen3.8-flash-next \
    MODEL_VARIANT=nvfp4-4p89 CAMPAIGN=2026-09-jj-r27-vs-r26 VARIANT=jj-r27-sm121-tp2-mtp3 \
    CONCURRENCY=1,2,4 ./run_bench.sh )

echo "== post-benchmark completion =="
completion | tee "${out}/post-benchmark-completion.json" | grep -q 333
echo "DONE $(date -Is)"
