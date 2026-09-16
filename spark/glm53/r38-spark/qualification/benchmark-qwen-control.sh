#!/usr/bin/env bash
# Matched R32 protocol, current image identity required, no custom output filenames.
set -euo pipefail
image=${EXPECTED_IMAGE_ID:?Pin the arm image ID}
[[ "$image" =~ ^[0-9a-f]{64}$ ]] || exit 2
base=$(cd "$(dirname "$0")" && pwd)
profile=${1:?contrast-r32|contrast-r38-return}
case "$profile" in
  contrast-r32) release=r32;;
  contrast-r38-return) release=r38;;
  *) exit 2;;
esac
out="$base/$profile"
grep -q '^CORRECTNESS-BATTERY-OK ' "$out/correctness.log"
[[ ! -e "$out/benchmark.log" ]] || { echo 'Receipt already exists'; exit 78; }
exec > >(tee "$out/benchmark.log") 2>&1
export PYTHONUNBUFFERED=1
for host in dusty kirby; do
  ssh -n -o BatchMode=yes "$host" "podman inspect qwen38-flash-next-nvfp4-jj-$release-tp2" > "$out/$host-benchmark-container.json"
  jq -e --arg image "$image" '
    .[0] | .State.Running and (.Image | ltrimstr("sha256:")) == $image and
    (.Args | index("--recurrent-checkpoint-policy") as $i | $i != null and .[$i+1] == "aligned") and
    (.Config.Env | index("NUM_SPECULATIVE_TOKENS=3") != null)
  ' "$out/$host-benchmark-container.json"
done
legacy=$base/probes
python3 "$legacy/probe-concurrency-identical-vs-distinct.py" | tee "$out/identical-vs-distinct.txt"
python3 "$legacy/probe-concurrency-fresh-extension-repeat.py" | tee "$out/fresh-extension-repeat.txt"
python3 "$base/probes/probe-qwen-head-of-line.py" --policy aligned --receipt-file "$out/head-of-line.jsonl"
echo '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  /home/jugs/git/llm-inference-bench/llm_decode_bench.py' | sha256sum -c -
echo "BENCHMARK-START $(date -Is)"
HOST=http://dusty:8000 MODEL=Qwen3.8-Flash-Next \
  MODEL_FAMILY=qwen3.8-flash-next MODEL_VARIANT=nvfp4-4p89 \
  CAMPAIGN=2026-09-jj-r38-vs-r32 VARIANT=jj-$release-aligned-sm121-tp2-mtp3 \
  CONCURRENCY=1,2,4 /home/jugs/git/llm-inference-bench/run_bench.sh \
  --metadata "image_id=$image" \
  --metadata checkpoint_revision=c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d \
  --metadata recurrent_checkpoint_policy=aligned \
  --metadata harness_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3
echo "BENCHMARK-END $(date -Is)"
printf -v repetition '%02d' "${REPETITION:-1}"
results=/home/jugs/git/llm-inference-bench/results/runs/qwen3.8-flash-next/nvfp4-4p89/2026-09-jj-r38-vs-r32/throughput
mapfile -t paths < <(find "$results" -maxdepth 1 -name "*__jj-$release-aligned-sm121-tp2-mtp3__r$repetition.json")
[[ ${#paths[@]} == 1 ]] || exit 78
python3 "$base/validate-grid.py" "${paths[0]}" > "$out/grid-validation.json"
for host in dusty kirby; do
  ssh -n -o BatchMode=yes "$host" "podman logs --timestamps qwen38-flash-next-nvfp4-jj-$release-tp2" > "$out/$host-after-grid.log" 2>&1
done
