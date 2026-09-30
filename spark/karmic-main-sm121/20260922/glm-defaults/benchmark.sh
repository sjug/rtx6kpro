#!/usr/bin/env bash
# Run only after the correctness driver and manual concurrency/health review.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$kit/../../../.." && pwd)
out=${GLM_RECEIPT_DIR:?completed correctness receipt required}
qualified=${GLM_CORRECTNESS_RECEIPT_DIR:-$out}
[[ ${GLM_CORRECTNESS_REVIEWED:-} == 1 ]] || { echo 'Explicit correctness review required'; exit 78; }
arm=${GLM_EXPERIMENT_ARM:-defaults}
scope=${GLM_QUALIFICATION_SCOPE:-full}
[[ $scope == full || $scope == screen ]] || { echo 'Unknown qualification scope'; exit 78; }
if [[ $arm == no-prefetch-mxfp8-output-a16 && $scope != full ]]; then
  echo 'Selective MXFP8 requires full native-context qualification before timing'; exit 78
fi
marker=CORRECTNESS-COMPLETE
[[ $scope != screen ]] || marker=SCREEN-COMPLETE
grep -q "^$marker:" "$qualified/driver.log"
python "$kit/experiment.py" "$arm" | cmp - "$qualified/effective-profile.json"
cmp "$qualified/effective-profile.json" "$out/effective-profile.json"
effective=$(jq -er .effective_profile_sha256 "$out/effective-profile.json")
speculative_tokens=$(jq -er '.argv as $a | ($a[($a|index("--speculative-config"))+1]|fromjson).num_speculative_tokens' "$out/effective-profile.json")
rejection_method=$(jq -er '.argv as $a | ($a[($a|index("--speculative-config"))+1]|fromjson).rejection_sample_method' "$out/effective-profile.json")
name=$(python "$kit/experiment.py" "$arm" --name)
[[ ! -e $out/benchmark.log ]] || { echo 'Refuse benchmark receipt overwrite'; exit 78; }
echo '5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2  /home/jugs/git/llm-inference-bench/run_bench.sh' | sha256sum -c -
echo '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  /home/jugs/git/llm-inference-bench/llm_decode_bench.py' | sha256sum -c -
image=$(jq -er .image_id "$kit/profile.lock.json")
digest=$(sha256sum "$kit/profile.lock.json" | cut -d' ' -f1)
for node in sparky buddy rocky lucky; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman inspect '$name'" > "$out/$node-pre-grid-container.json"
  jq -e --arg image "$image" --arg effective "$effective" '.[0] |
    .State.Running and (.State.OOMKilled|not) and .Image==$image and
    .Config.Labels["local-inference.glm-effective-profile.sha256"]==$effective' "$out/$node-pre-grid-container.json" >/dev/null
  # Reuse a completed screen only for the exact same container and boot.
  jq -e --slurpfile qualified "$qualified/$node-container.json" '
    .[0].Id==$qualified[0][0].Id and
    .[0].State.StartedAt==$qualified[0][0].State.StartedAt' \
    "$out/$node-pre-grid-container.json" >/dev/null || {
      echo "Correctness receipt is from another container or boot: $node" >&2; exit 78;
    }
done
curl -fsS --max-time 10 http://sparky:8000/metrics > "$out/pre-grid.metrics"
python - "$out/pre-grid.metrics" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
for key in ('running', 'waiting'):
    values = re.findall(r'^vllm:num_requests_' + key + r'\{[^\n]*\} ([\d.e+-]+)$', text, re.M)
    if not values or any(float(v) for v in values):
        raise SystemExit('Endpoint is not idle: ' + key)
PY
export RESULTS_REPO="$repo" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
HOST=http://sparky:8000 MODEL=GLM-5.3-Flash MODEL_FAMILY=glm-5.3-flash MODEL_VARIANT=nvfp4 \
  CAMPAIGN=2026-09-karmic-sm121-qualification VARIANT="karmic-main-${arm}-sm121-tp4-dcp1-mtp${speculative_tokens}" CONCURRENCY=1,2,4 \
  /home/jugs/git/llm-inference-bench/run_bench.sh --no-resume --duration 30 --max-total-tokens 6412288 --display-mode plain --calibration-cache "$out/token-calibration.json" \
  --metadata image_id="$image" --metadata checkpoint_revision=46aaae8a82032f77100f2f03e9cc11b391df3b4d \
  --metadata recurrent_checkpoint_policy=request_boundaries --metadata profile_sha256="$digest" \
  --metadata experiment_arm="$arm" --metadata effective_profile_sha256="$effective" --metadata qualification_scope="$scope" \
  --metadata speculative_tokens="$speculative_tokens" \
  --metadata rejection_sample_method="$rejection_method" \
  --metadata clear_thinking=false --metadata max_num_seqs=32 --metadata capture_size=256 \
  --metadata comparison_scope=whole_profile_not_engine_only \
  --metadata correctness_receipt="$qualified" \
  --metadata harness_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3 \
  < <(printf 'n\n') | tee "$out/benchmark.log"
raw=$(sed -n 's/^Results saved to: //p' "$out/benchmark.log" | tail -n 1)
[[ -f $raw ]] || { echo 'Missing raw benchmark receipt'; exit 1; }
python "$repo/spark/glm53/r38-spark/qualification/validate-grid.py" "$raw" > "$out/grid-validation.json"
echo 'GRID-COMPLETE: requires health and whole-profile comparison review'
