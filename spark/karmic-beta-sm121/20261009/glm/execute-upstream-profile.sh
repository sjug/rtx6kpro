#!/usr/bin/env bash
# Start the upstream-profile GLM candidate (run-glm-tp4-node-upstream.sh) on idle nodes and run the
# gate battery and grid. No production comparison: the comparison of record is the same image's
# current-config grid (compare-grids.py, run by hand).
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
[[ ${GLM_UPSTREAM_PROFILE_APPROVED:-0} == 1 ]] || exit 78
image=${EXPECTED_IMAGE_ID:?built image ID required}
[[ $image =~ ^[0-9a-f]{64}$ ]] || exit 78
tag=localhost/voipmonitor/vllm:karmic-beta-20261009-spark-sm121
remote=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121/20261009/glm
runner=run-glm-tp4-node-upstream.sh
expected=$(sha256sum "$kit/$runner" | cut -d' ' -f1)
repo=models--local-inference-lab--GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD
revision=
for node in sparky buddy rocky lucky; do
  running=$(ssh -n -o BatchMode=yes "$node" 'podman ps -q') || { echo "podman ps failed on $node"; exit 78; }
  [[ -z $running ]] || { echo "$node is not idle"; exit 78; }
  actual=$(ssh -n -o BatchMode=yes "$node" "podman image inspect '$tag' --format '{{.Id}}'")
  [[ ${actual#sha256:} == "$image" ]] || exit 78
  actual=$(ssh -n -o BatchMode=yes "$node" "sha256sum '$remote/$runner'" | cut -d' ' -f1)
  [[ $actual == "$expected" ]] || exit 78
  # Every node serves the same refs/main revision of the CSF checkpoint.
  node_revision=$(ssh -n -o BatchMode=yes "$node" "cat ~/.cache/huggingface/hub/$repo/refs/main")
  [[ $node_revision =~ ^[0-9a-f]{40}$ ]] || exit 78
  [[ -z $revision || $revision == "$node_revision" ]] || { echo "refs/main differs on $node"; exit 78; }
  revision=$node_revision
done
echo "CSF checkpoint revision $revision"
[[ $revision == dec48abd33efa73c3bb7c95b74eee10cad34f9be ]] || echo "NOTE: refs/main differs from the recipe 05eaafa profile pin dec48abd (serving refs/main per the download rule)"
for node in buddy rocky lucky sparky; do
  role=worker; [[ $node != sparky ]] || role='head'
  ssh -n -o BatchMode=yes "$node" "ROLE=$role EXPECTED_IMAGE_ID=$image MODEL_REVISION=$revision bash '$remote/$runner'"
done
export GLM_RECEIPT_DIR=${GLM_RECEIPT_DIR:-$kit/../qualification/glm-upstream-profile}
export GLM_NAME=glm53-flash-csf-qad-upstream-profile-karmic-beta-20261009-tp4
export GLM_POLICY=request_boundaries GLM_SKIP_PRODUCTION_COMPARE=1
export GLM_CHECKPOINT_REVISION=$revision
export GLM_GRID_VARIANT=karmic-beta-20261009-upstream-profile-csf-qad-sm121-tp4-dcp1-mtp3-seq4-native1m
bash "$kit/qualify-glm.sh"
