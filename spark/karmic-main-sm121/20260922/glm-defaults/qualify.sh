#!/usr/bin/env bash
# Correctness only. No container lifecycle changes, no benchmark on a failed gate.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$kit/../../../.." && pwd)
out=${GLM_RECEIPT_DIR:?set a fresh qualification receipt directory}
[[ ! -e $out ]] || { echo 'Receipt directory already exists' >&2; exit 78; }
mkdir -p "$out"
exec > >(tee "$out/driver.log") 2>&1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export BASE_URL=http://sparky:8000 MODEL=GLM-5.3-Flash OUT_DIR="$out"
arm=${GLM_EXPERIMENT_ARM:-defaults}
scope=${GLM_QUALIFICATION_SCOPE:-full}
[[ $scope == full || $scope == screen || $scope == boot ]] || { echo 'Unknown qualification scope'; exit 78; }
name=$(python "$kit/experiment.py" "$arm" --name)
python "$kit/experiment.py" "$arm" > "$out/effective-profile.json"
effective=$(jq -er .effective_profile_sha256 "$out/effective-profile.json")
image=$(jq -er .image_id "$kit/profile.lock.json")
digest=$(sha256sum "$kit/profile.lock.json" | cut -d' ' -f1)
launch_digest=$(sha256sum "$kit/launch.py" | cut -d' ' -f1)
experiment_digest=$(sha256sum "$kit/experiment.py" | cut -d' ' -f1)
for node in sparky buddy rocky lucky; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" > "$out/$node-container.json"
  jq -e --arg image "$image" --arg digest "$digest" --arg effective "$effective" --arg arm "$arm" --arg launcher "$launch_digest" --arg experiment "$experiment_digest" '.[0] | .State.Running and (.State.OOMKilled|not) and
    .Image==$image and .Config.Labels["local-inference.glm-profile.sha256"]==$digest and
    .Config.Labels["local-inference.glm-effective-profile.sha256"]==$effective and
    .Config.Labels["local-inference.glm-experiment.arm"]==$arm and
    (.Config.Env | index("GLM_LAUNCH_SHA256="+$launcher) != null) and
    (.Config.Env | index("GLM_EXPERIMENT_SHA256="+$experiment) != null)' "$out/$node-container.json" >/dev/null
done
# Bound startup without treating /v1/models as a completion gate.
deadline=$(( $(date +%s) + 1800 ))
ready=0
while (( $(date +%s) < deadline )); do
  for node in sparky buddy rocky lucky; do
    state=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman inspect '$name' --format '{{.State.Running}} {{.State.OOMKilled}}'")
    [[ $state == 'true false' ]] || { echo "Startup failed on $node: $state"; exit 1; }
  done
  if curl -fsS --connect-timeout 3 --max-time 60 "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"What is 17 * 23 - 58? Reply with the number only."}],"max_tokens":256,"temperature":0,"reasoning_effort":"low"}' > "$out/first-completion.json"; then
    jq -e '.choices[0] | .finish_reason=="stop" and (.message.content|gsub("^\\s+|\\s+$";""))=="333"' "$out/first-completion.json" >/dev/null || { echo 'First completion returned an incorrect or incomplete answer'; exit 1; }
    ready=1; break
  fi
  sleep 10
done
[[ $ready == 1 ]] || { echo 'First completion failed'; exit 1; }
for node in sparky buddy rocky lucky; do
  # Retained containers keep logs from earlier starts. Verify this boot only.
  started=$(jq -er '.[0].State.StartedAt' "$out/$node-container.json")
  ssh -n -o BatchMode=yes "$node" "podman logs --since '$started' '$name' 2>&1" > "$out/$node-boot.log"
done
python "$kit/verify_boot.py" "$out" "$arm"
if [[ $scope == boot ]]; then
  echo 'BOOT-ONLY-COMPLETE: diagnostic readiness only; no qualification or timing approval'
  exit 0
fi
python "$repo/spark/glm53/r38-spark/qualification/verify-glm-short-pool.py" --base-url "$BASE_URL" --model "$MODEL" --receipt-file "$out/short-pool.jsonl"
python "$repo/spark/glm53/verify-semantic-admission.py" --base-url "$BASE_URL" --model "$MODEL" --runs 3 --receipt-file "$out/semantic.jsonl"
python "$repo/spark/glm53/r27-spark/tests/glm/probe-concurrency-glm.py" | tee "$out/concurrency.txt"
cp "$repo/spark/glm53/r27-spark/qualification/glm-aligned-20260906/prefix-matrix-inputs.json" "$out/prefix-matrix-inputs.json"
echo "31580c2d9f3e9d6c219d495969cae45b15c45bf8ae88fafd00dd39aa4c5f89b9  $out/prefix-matrix-inputs.json" | sha256sum -c -
python "$repo/spark/glm53/r27-spark/tests/glm/prefix-matrix.py" karmic-main-request-boundaries
if [[ $scope == screen ]]; then
  echo 'SCREEN-COMPLETE: diagnostic only; no native-1M qualification or promotion'
  exit 0
fi
python "$repo/spark/glm53/r27-spark/tests/glm/prefix-triples.py" short
python "$repo/spark/glm53/r28-spark/tests/verify-glm-native-context.py" --base-url "$BASE_URL" --model "$MODEL" --receipt-file "$out/native-context.jsonl"
jq -se 'length==4 and all(.[];.valid and .finish_reason=="stop" and (.text|gsub("^\\s+|\\s+$";""))==.needle)' "$out/native-context.jsonl"
echo 'CORRECTNESS-COMPLETE: review concurrency, cache, logs and memory before timing; no automatic promotion'
