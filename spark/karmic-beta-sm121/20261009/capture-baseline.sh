#!/usr/bin/env bash
# Run manually in an approved benchmark window before replacing current production.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
profile=${1:?qwen, glm or ds4-vision required}
variant=nvfp4
out=${2:?new task receipt directory required}
out=$(realpath -m "$out")
[[ $out == "$kit"/qualification/* && ! -e $out ]] || exit 78
image=500ae05b98da0658c1a5e1820387f96f2121c5659bd7954ad1c5861f20934f05
case $profile in
  qwen) nodes=(dusty kirby); name=qwen38-flash-next-nvfp4-karmic-beta-20260929-tp2-7c4f1bc1-qualified-20261001
    model=Qwen3.8-Flash-Next; revision=7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
    host=http://dusty:8000; family=qwen3.8-flash-next; extra=(--contexts '0,16k,32k,64k,128k');;
  glm) nodes=(sparky buddy rocky lucky); name=glm53-flash-nvfp4-karmic-beta-20260929-tp4
    model=GLM-5.3-Flash; revision=175ae8ce3b5af842b0d0140dbeb43e9cfc557c49
    host=http://sparky:8000; family=glm-5.3-flash; extra=(--max-total-tokens 6412288);;
  ds4-vision) nodes=(rusty toby); name=ds4-vision-karmic-beta-20260929-tp2
    model=DeepSeek-V4-Flash-Vision-Exp; revision=6821d6ad3681a4b137b066b76094fa82ebd0a380
    host=http://rusty:8000; family=deepseek-v4-flash; variant=vision-exp; extra=(--max-total-tokens 6412288);;
  *) exit 78;;
esac
mkdir -p "$out"
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" > "$out/$node-container.json"
  if [[ $profile == ds4-vision ]]; then
    jq -e --arg image "$image" --arg revision "$revision" '.[0] | (.Image|ltrimstr("sha256:"))==$image and .State.Running and (.State.OOMKilled|not) and (.Config.Env|index("MODEL_REVISION="+$revision)!=null)' "$out/$node-container.json" >/dev/null
  else
    jq -e --arg image "$image" --arg revision "$revision" '.[0] | (.Image|ltrimstr("sha256:"))==$image and .State.Running and (.State.OOMKilled|not) and (.Config.Env|index("MODEL_REVISION="+$revision)!=null) and (.Args|index("--recurrent-checkpoint-policy") as $i|$i!=null and .[$i+1]=="aligned")' "$out/$node-container.json" >/dev/null
  fi
  if [[ $profile == qwen ]]; then
    jq -e '.[0].Config.Env | index("VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0")!=null and index("NUM_SPECULATIVE_TOKENS=3")!=null' "$out/$node-container.json" >/dev/null
  elif [[ $profile == ds4-vision ]]; then
    jq -e '.[0].Config.Env | index("DSPARK_TOKENS=3")!=null and index("MAX_NUM_SEQS=4")!=null and index("BACKEND=b12x-a8")!=null and index("MAX_MODEL_LEN=1048576")!=null' "$out/$node-container.json" >/dev/null
  else
    jq -e '.[0].Config.Env | index("NUM_SPECULATIVE_TOKENS=3")!=null and index("DCP=1")!=null and index("MAX_NUM_SEQS=4")!=null and index("VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4")!=null' "$out/$node-container.json" >/dev/null
  fi
done
# The frozen harness copy (freeze_harness.sh): llm_decode_bench.py cc9bb06a, run_bench.sh 5c79b976, the
# same bytes the candidate grid runs, so a same-window fresh-boot production grid is directly comparable.
bench=$(bash "$kit/freeze_harness.sh")
harness=cc9bb06a4f1142d2bf08afbcfc0e57d8a465a0bd180eee74e276dea70a82071c
echo "$harness  $bench/llm_decode_bench.py" | sha256sum -c -
echo "5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2  $bench/run_bench.sh" | sha256sum -c -
export RESULTS_REPO=/home/jugs/git/rtx6kpro HOST=$host MODEL=$model MODEL_FAMILY=$family MODEL_VARIANT=$variant
export CONCURRENCY=1,2,4 CAMPAIGN=2026-10-karmic-beta-20261009-sm121-qualification VARIANT=production-20260929-qad-$profile-fresh-boot
printf 'n\n' | "$bench/run_bench.sh" --duration 30 --display-mode plain --calibration-cache "$out/token-calibration.json" "${extra[@]}" \
  --metadata image_id="$image" --metadata checkpoint_revision="$revision" --metadata recurrent_checkpoint_policy=aligned \
  --metadata hc_tp=0 --metadata nodes="$(IFS=,; echo "${nodes[*]}")" --metadata harness_sha256="$harness" --metadata llm_decode_bench_sha256="$harness" | tee "$out/benchmark.log"
raw=$(sed -n 's/^Results saved to: //p' "$out/benchmark.log" | tail -n 1)
[[ $profile == ds4-vision ]] || python3 "$kit/compare-production.py" --model "$profile" --harness-sha256 "$harness" --validate-baseline "$raw"
cp "$raw" "$out/grid.json"
sha256sum "$out/grid.json" > "$out/grid.sha256"
echo "BASELINE_GRID=$(realpath "$out/grid.json")"
