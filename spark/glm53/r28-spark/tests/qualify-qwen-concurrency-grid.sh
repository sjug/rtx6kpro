#!/usr/bin/env bash
set -euo pipefail
repo=/home/jugs/git/rtx6kpro
r28=$repo/spark/glm53/r28-spark
legacy=$repo/spark/glm53/r27-spark/qualification/qwen-20260906
policy=${1:?aligned|auto}
case $policy in aligned|auto) ;; *) exit 2;; esac
out=$r28/qualification/qwen-20260908/$policy-mtp3
mkdir -p "$out"
exec > >(tee "$out/concurrency-grid.log") 2>&1
export PYTHONUNBUFFERED=1
echo "START $policy $(date -Is)"
for host in dusty kirby; do
  ssh -o BatchMode=yes "$host" 'podman inspect qwen38-flash-next-nvfp4-jj-r28-tp2' > "$out/$host-concurrency-container.json"
  jq -e --arg policy "$policy" '
    .[0] | .State.Running and
    (.Image | ltrimstr("sha256:")) == "cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1" and
    (.Args | index("--recurrent-checkpoint-policy") as $i | $i != null and .[$i+1] == $policy) and
    (.Config.Env | index("NUM_SPECULATIVE_TOKENS=3") != null)
  ' "$out/$host-concurrency-container.json"
done
python3 "$legacy/probe-concurrency-identical-vs-distinct.py" | tee "$out/identical-vs-distinct.txt"
python3 "$legacy/probe-concurrency-fresh-extension-repeat.py" | tee "$out/fresh-extension-repeat.txt"
python3 "$r28/tests/probe-qwen-head-of-line.py" --policy "$policy" --receipt-file "$out/head-of-line.jsonl"
if [[ $policy == aligned ]]; then
  jq -se 'all(.[] | select(.kind == "context_needle"); .valid and .finish_reason == "stop" and .text == .needle)' "$out/native-context.jsonl"
  echo "BENCHMARK-START $(date -Is)"
  HOST=http://dusty:8000 MODEL=Qwen3.8-Flash-Next-NVFP4-4p89 \
    MODEL_FAMILY=qwen3.8-flash-next MODEL_VARIANT=nvfp4-4p89 \
    CAMPAIGN=2026-09-jj-r28-vs-r27 VARIANT=jj-r28-aligned-sm121-tp2-mtp3 \
    CONCURRENCY=1,2,4 /home/jugs/git/llm-inference-bench/run_bench.sh \
    --metadata image_id=cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1 \
    --metadata checkpoint_revision=c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d \
    --metadata recurrent_checkpoint_policy=aligned \
    --metadata harness_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3
  echo "BENCHMARK-END $(date -Is)"
fi
echo "DONE $policy $(date -Is)"
