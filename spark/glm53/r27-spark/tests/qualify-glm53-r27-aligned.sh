#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 JJ r27 Spark ALIGNED CONTROL driver (workstation side).
# Matched to the executed auto window: same image, runner, model revision,
# corpus, probes, and benchmark; the only change is the four ranks launched
# with RECURRENT_CHECKPOINT_POLICY=aligned. Not a promotion.
# Runs only against an already launched r27 TP4 cluster on
# sparky/buddy/rocky/lucky whose four command lines carry
# --recurrent-checkpoint-policy aligned; the identity gate refuses any rank
# without it.
# Sequence: bounded readiness by completion, all-rank identity gates, semantic
# x3, concurrency reproducers (identical burst, multi-turn extension, repeat
# head-of-line), frozen prefix-cache pair matrix, short and long triples,
# native context 2048/2049/262000/1048000 (262000 warms before 1M as in r26),
# the standard 15-cell grid against corrected R26, and a post-benchmark
# completion. Per-node container and GPU telemetry logs run alongside.
set -euo pipefail
repo=/home/jugs/git/rtx6kpro; r27=${repo}/spark/glm53/r27-spark; bench=/home/jugs/git/llm-inference-bench
archive=${ARCHIVE:-/home/jugs/git/rtx6kpro-artifacts/20260906-pre-r27/spark/glm53/r26-spark/control-r22-env-20260906}
base_url=${BASE_URL:-http://sparky:8000}; model=${MODEL:-GLM-5.3-Flash}
nodes=(sparky buddy rocky lucky); name=glm53-flash-nvfp4-jj-r27-spark-tp4
expected_image=${EXPECTED_IMAGE_ID:-ef669fa1cde3e99936c02575eca8f610990bb6c64dd1bbfcb04fd42dde87afae}
date_tag=${DATE_TAG:-$(date +%Y%m%d)}; out=${OUT_DIR:-${r27}/qualification/glm-aligned-${date_tag}}
mkdir -p "${out}"; exec > >(tee -a "${out}/driver.log") 2>&1
echo "START $(date -Is) base_url=${base_url} model=${model} out=${out}"

completion() { curl -s --max-time 120 "${base_url}/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23 - 58? Reply with the number only.\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"; }
telemetry=()
cleanup() { for p in "${telemetry[@]}"; do kill "${p}" 2>/dev/null || true; done; }
trap cleanup EXIT
for n in "${nodes[@]}"; do
  ssh -n "${n}" "podman logs --timestamps -f ${name}" >"${out}/${n}-container.log" 2>&1 & telemetry+=($!)
  ssh -n "${n}" 'nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_throttle_reasons.active,temperature.gpu,power.draw --format=csv -l 1' >"${out}/${n}-gpu.log" 2>&1 & telemetry+=($!)
done

echo "== readiness: 30-minute wall clock, completion probes only (never /v1/models) =="
deadline=$(( $(date +%s) + 1800 )); ready=0
while [ "$(date +%s)" -lt "${deadline}" ]; do
  reply=$(completion || true)
  if grep -q '"content"' <<<"${reply}"; then printf '%s\n' "${reply}" > "${out}/first-completion.json"; grep -q 333 <<<"${reply}" && { echo "READY $(date -Is)"; ready=1; break; }; echo "bad completion: ${reply:0:200}"; exit 1; fi
  sleep 15
done
[ "${ready}" = 1 ] || { echo "no completion within 30 minutes"; exit 1; }

echo "== all-rank identity gates: image, running, aligned policy on every rank, backends =="
for n in "${nodes[@]}"; do
  img=$(ssh -n "${n}" "podman inspect --format '{{.Image}}' ${name}")
  [[ "${img}" == "${expected_image}" ]] || { echo "${n}: image ${img} != ${expected_image}"; exit 1; }
  cmdline=$(ssh -n "${n}" "podman exec ${name} cat /proc/1/cmdline | tr '\\0' ' '; podman inspect --format '{{join .Args \" \"}}' ${name}")
  if ! grep -q -- '--recurrent-checkpoint-policy aligned' <<<"${cmdline}"; then echo "${n}: GLM command line does not set the aligned policy; this is the aligned control"; exit 1; fi
  printf '%s\n' "${cmdline}" > "${out}/${n}-cmdline.txt"
done
grep -q "B12X_ROCENANTE" "${out}/sparky-container.log" || { echo "RoCEnante backend not reported on sparky"; exit 1; }
echo "identity gates passed on ${nodes[*]}"

echo "== semantic admission x3 =="
python3 "${repo}/spark/glm53/verify-semantic-admission.py" --base-url "${base_url}" --model "${model}" --runs 3 --receipt-file "${out}/jj-r27-aligned-glm-semantic-admission-3x-${date_tag}.jsonl"

echo "== concurrency reproducers under aligned =="
BASE_URL="${base_url}" MODEL="${model}" python3 "${r27}/tests/glm/probe-concurrency-glm.py" | tee "${out}/jj-r27-aligned-glm-concurrency-${date_tag}.txt"

echo "== frozen prefix-cache pair matrix (r26/r22 corpus) =="
cp -n "${archive}/prefix-matrix-inputs.json" "${out}/"
sha256sum "${out}/prefix-matrix-inputs.json" | grep -q '^31580c2d9f3e9d6c219d495969cae45b15c45bf8ae88fafd00dd39aa4c5f89b9 ' || { echo "prefix corpus digest mismatch"; exit 1; }
OUT_DIR="${out}" BASE_URL="${base_url}" MODEL="${model}" python3 "${r27}/tests/glm/prefix-matrix.py" r27-aligned
echo "== prefix-cache triples: short, then long (262000/262000/1048000) =="
OUT_DIR="${out}" BASE_URL="${base_url}" MODEL="${model}" python3 "${r27}/tests/glm/prefix-triples.py" short
OUT_DIR="${out}" BASE_URL="${base_url}" MODEL="${model}" python3 "${r27}/tests/glm/prefix-triples.py" long

echo "== boundary + native context (262000 warms before 1048000) =="
python3 "${repo}/spark/qwen38-flash-next/verify-native-context.py" --base-url "${base_url}" --model "${model}" --needle 739526 --max-tokens 128 \
  --lengths 2048 2049 262000 1048000 --receipt-file "${out}/jj-r27-aligned-glm-native-context-needle-${date_tag}.jsonl"

echo "== standard 15-cell grid (campaign 2026-09-jj-r27-sm121-qualification) =="
( cd "${bench}" && HOST="${base_url}" MODEL="${model}" MODEL_FAMILY=glm-5.3-flash MODEL_VARIANT=nvfp4 \
    CAMPAIGN=2026-09-jj-r27-sm121-qualification VARIANT=jj-r27-aligned-sm121-tp4-dcp1-mtp3-native1m CONCURRENCY=1,2,4 \
    ./run_bench.sh --duration 30 --max-total-tokens 6412288 ) | tee "${out}/benchmark.log"
mkdir -p "${out}/llm-inference-bench" && cp "${bench}"/results/runs/glm-5.3-flash/nvfp4/2026-09-jj-r27-sm121-qualification/throughput/*jj-r27-aligned-sm121-tp4-dcp1-mtp3-native1m* "${out}/llm-inference-bench/" 2>/dev/null || true

echo "== post-benchmark completion =="
completion | tee "${out}/post-benchmark-completion.json" | grep -q 333
for n in "${nodes[@]}"; do ssh -n "${n}" "podman inspect --format '${n} running={{.State.Running}} oom={{.State.OOMKilled}}' ${name}"; done
echo "DONE $(date -Is)"
