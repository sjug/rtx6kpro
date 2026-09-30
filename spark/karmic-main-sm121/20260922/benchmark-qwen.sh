#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$base/../../.." && pwd)
image=${EXPECTED_IMAGE_ID:?pin the built image}
case ${QUAL_ARM:-baseline} in
  baseline) suffix=''; variant=karmic-main-aligned-mtp3; hc=default ;;
  hc-off) suffix=-hc-off; variant=karmic-main-hc-off-aligned-mtp3; hc=0 ;;
  hc-on-return) suffix=-hc-on-return; variant=karmic-main-hc-on-return-aligned-mtp3; hc=1 ;;
  *) echo 'Unknown qualification arm'; exit 78 ;;
esac
out="$base/qualification/qwen-mtp3$suffix"
name=qwen38-flash-next-nvfp4-karmic-main-tp2
[[ -f $out/CORRECTNESS-OK ]] || { echo 'Correctness gate has not passed'; exit 78; }
[[ ! -e $out/benchmark.log ]] || { echo 'Refusing benchmark overwrite'; exit 78; }
for node in dusty kirby; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman inspect '$name'" > "$out/$node-benchmark.json"
  jq -e --arg image "$image" '.[0] | .State.Running and .Image==$image and
    (.Config.Env | index("NUM_SPECULATIVE_TOKENS=3") != null) and
    (.Args | index("--recurrent-checkpoint-policy") as $i | $i != null and .[$i+1]=="aligned")' \
    "$out/$node-benchmark.json" >/dev/null
  [[ $(jq -r '.[0].Id' "$out/$node-benchmark.json") == \
     $(jq -r '.[0].Id' "$out/$node-container.json") ]] || { echo 'Correctness boot changed'; exit 78; }
  if [[ $hc != default ]]; then
    jq -e --arg setting "VLLM_QWEN3_8_FLASH_NEXT_HC_TP=$hc" \
      '.[0].Config.Env | index($setting) != null' "$out/$node-benchmark.json" >/dev/null
  fi
done
exec > >(tee "$out/benchmark.log") 2>&1
bench=/home/jugs/git/llm-inference-bench
echo "5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2  $bench/run_bench.sh" | sha256sum -c -
echo "2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  $bench/llm_decode_bench.py" | sha256sum -c -
export RESULTS_REPO="$repo" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HOST=http://dusty:8000 MODEL=Qwen3.8-Flash-Next CONCURRENCY=1,2,4
export MODEL_FAMILY=qwen3.8-flash-next MODEL_VARIANT=nvfp4-4p89
export CAMPAIGN=2026-09-karmic-main-sm121-qualification VARIANT=$variant REPETITION=1
echo "BENCHMARK-START $(date -Is)"
printf 'n\n' | "$bench/run_bench.sh" --duration 30 --contexts 0,16k,32k,64k,128k \
  --display-mode plain --calibration-cache "$out/token-calibration.json" \
  --metadata "image_id=$image" --metadata checkpoint_revision=c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d \
  --metadata recurrent_checkpoint_policy=aligned \
  --metadata "hc_tp=$hc" \
  --metadata run_bench_sha256=5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2 \
  --metadata llm_decode_bench_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3
echo "BENCHMARK-END $(date -Is)"
