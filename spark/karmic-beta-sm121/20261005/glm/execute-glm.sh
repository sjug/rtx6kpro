#!/usr/bin/env bash
# Start only an explicitly approved, already distributed candidate on idle nodes.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
[[ ${GLM_KARMIC_BETA_APPROVED:-0} == 1 ]] || exit 78
image=${EXPECTED_IMAGE_ID:?built image ID required}
[[ $image =~ ^[0-9a-f]{64}$ ]] || exit 78
tag=localhost/voipmonitor/vllm:karmic-beta-20261005-spark-sm121
remote=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121/20261005/glm
expected=$(sha256sum "$kit/run-glm-tp4-node.sh" | cut -d' ' -f1)
baseline=${BASELINE_GRID:?matched production QAD grid required before launch}
python3 "$kit/../compare-production.py" --model glm --validate-baseline "$baseline"
for node in sparky buddy rocky lucky; do
  [[ -z $(ssh -n -o BatchMode=yes "$node" 'podman ps -q') ]] || exit 78
  actual=$(ssh -n -o BatchMode=yes "$node" "podman image inspect '$tag' --format '{{.Id}}'")
  [[ ${actual#sha256:} == "$image" ]] || exit 78
  actual=$(ssh -n -o BatchMode=yes "$node" "sha256sum '$remote/run-glm-tp4-node.sh'" | cut -d' ' -f1)
  [[ $actual == "$expected" ]] || exit 78
done
for node in buddy rocky lucky sparky; do
  role=worker; [[ $node != sparky ]] || role='head'
  ssh -n -o BatchMode=yes "$node" "ROLE=$role EXPECTED_IMAGE_ID=$image bash '$remote/run-glm-tp4-node.sh'"
done
export GLM_RECEIPT_DIR=${GLM_RECEIPT_DIR:-$kit/../qualification/glm}
bash "$kit/qualify-glm.sh"
