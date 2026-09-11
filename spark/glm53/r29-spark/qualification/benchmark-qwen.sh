#!/usr/bin/env bash
# Matched R28 protocol, current image identity required, no custom output filenames.
set -euo pipefail
image=${EXPECTED_IMAGE_ID:?Pin the gated R29 image ID}
[[ "$image" =~ ^[0-9a-f]{64}$ ]] || exit 2
repo=/home/jugs/git/rtx6kpro
base=$(cd "$(dirname "$0")" && pwd)
out="$base/qwen-20260909/initial"
grep -q '^CORRECTNESS-BATTERY-OK ' "$out/correctness.log"
[[ ! -e "$out/benchmark.log" ]] || { echo 'Receipt already exists'; exit 78; }
exec > >(tee "$out/benchmark.log") 2>&1
export PYTHONUNBUFFERED=1
for host in dusty kirby; do
  ssh -n -o BatchMode=yes "$host" 'podman inspect qwen38-flash-next-nvfp4-jj-r29-tp2' > "$out/$host-benchmark-container.json"
  jq -e --arg image "$image" '
    .[0] | .State.Running and (.Image | ltrimstr("sha256:")) == $image and
    (.Args | index("--recurrent-checkpoint-policy") as $i | $i != null and .[$i+1] == "aligned") and
    (.Config.Env | index("NUM_SPECULATIVE_TOKENS=3") != null)
  ' "$out/$host-benchmark-container.json"
done
legacy=$repo/spark/glm53/r27-spark/qualification/qwen-20260906
python3 "$legacy/probe-concurrency-identical-vs-distinct.py" | tee "$out/identical-vs-distinct.txt"
python3 "$legacy/probe-concurrency-fresh-extension-repeat.py" | tee "$out/fresh-extension-repeat.txt"
python3 "$repo/spark/glm53/r28-spark/tests/probe-qwen-head-of-line.py" --policy aligned --receipt-file "$out/head-of-line.jsonl"
echo '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  /home/jugs/git/llm-inference-bench/llm_decode_bench.py' | sha256sum -c -
echo "BENCHMARK-START $(date -Is)"
HOST=http://dusty:8000 MODEL=Qwen3.8-Flash-Next-NVFP4-4p89 \
  MODEL_FAMILY=qwen3.8-flash-next MODEL_VARIANT=nvfp4-4p89 \
  CAMPAIGN=2026-09-jj-r29-vs-r28 VARIANT=jj-r29-aligned-sm121-tp2-mtp3 \
  CONCURRENCY=1,2,4 /home/jugs/git/llm-inference-bench/run_bench.sh \
  --metadata "image_id=$image" \
  --metadata checkpoint_revision=c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d \
  --metadata recurrent_checkpoint_policy=aligned \
  --metadata harness_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3
echo "BENCHMARK-END $(date -Is)"
for host in dusty kirby; do
  ssh -n -o BatchMode=yes "$host" 'podman logs --timestamps qwen38-flash-next-nvfp4-jj-r29-tp2' > "$out/$host-after-grid.log" 2>&1
done
