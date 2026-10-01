#!/usr/bin/env bash
# Run manually in an approved benchmark window before replacing current production.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
profile=${1:?qwen or glm required}
out=${2:?new task receipt directory required}
out=$(realpath -m "$out")
[[ $out == "$kit"/qualification/* && ! -e $out ]] || exit 78
image=500ae05b98da0658c1a5e1820387f96f2121c5659bd7954ad1c5861f20934f05
case $profile in
  qwen) nodes=(dusty kirby); name=qwen38-flash-next-nvfp4-karmic-beta-20260929-tp2
    model=Qwen3.8-Flash-Next; revision=7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
    host=http://dusty:8000; family=qwen3.8-flash-next; extra=(--contexts '0,16k,32k,64k,128k');;
  glm) nodes=(sparky buddy rocky lucky); name=glm53-flash-nvfp4-karmic-beta-20260929-tp4
    model=GLM-5.3-Flash; revision=175ae8ce3b5af842b0d0140dbeb43e9cfc557c49
    host=http://sparky:8000; family=glm-5.3-flash; extra=(--max-total-tokens 6412288);;
  *) exit 78;;
esac
mkdir -p "$out"
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" > "$out/$node-container.json"
  jq -e --arg image "$image" --arg revision "$revision" '.[0] | (.Image|ltrimstr("sha256:"))==$image and .State.Running and (.State.OOMKilled|not) and (.Config.Env|index("MODEL_REVISION="+$revision)!=null) and (.Args|index("--recurrent-checkpoint-policy") as $i|$i!=null and .[$i+1]=="aligned")' "$out/$node-container.json" >/dev/null
  if [[ $profile == qwen ]]; then
    jq -e '.[0].Config.Env | index("VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0")!=null and index("NUM_SPECULATIVE_TOKENS=3")!=null' "$out/$node-container.json" >/dev/null
  else
    jq -e '.[0].Config.Env | index("NUM_SPECULATIVE_TOKENS=3")!=null and index("DCP=1")!=null and index("MAX_NUM_SEQS=4")!=null and index("VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4")!=null' "$out/$node-container.json" >/dev/null
  fi
done
bench=/home/jugs/git/llm-inference-bench
harness=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3
echo "$harness  $bench/llm_decode_bench.py" | sha256sum -c -
echo "5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2  $bench/run_bench.sh" | sha256sum -c -
export RESULTS_REPO=/home/jugs/git/rtx6kpro HOST=$host MODEL=$model MODEL_FAMILY=$family MODEL_VARIANT=nvfp4
export CONCURRENCY=1,2,4 CAMPAIGN=2026-10-karmic-beta-production-baseline VARIANT=production-20260929-qad-$profile
printf 'n\n' | "$bench/run_bench.sh" --duration 30 --display-mode plain --calibration-cache "$out/token-calibration.json" "${extra[@]}" \
  --metadata image_id="$image" --metadata checkpoint_revision="$revision" --metadata recurrent_checkpoint_policy=aligned \
  --metadata hc_tp=0 --metadata harness_sha256="$harness" --metadata llm_decode_bench_sha256="$harness" | tee "$out/benchmark.log"
raw=$(sed -n 's/^Results saved to: //p' "$out/benchmark.log" | tail -n 1)
python3 "$kit/compare-production.py" --model "$profile" --validate-baseline "$raw"
cp "$raw" "$out/grid.json"
sha256sum "$out/grid.json" > "$out/grid.sha256"
echo "BASELINE_GRID=$(realpath "$out/grid.json")"
