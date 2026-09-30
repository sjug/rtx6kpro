#!/usr/bin/env bash
# Actual Podman layer-boundary regression, using the retained pre-overlay layer.
set -euo pipefail
image=${1:?Pass the pre-overlay image ID containing /build/vllm and the patch}
podman run --rm --pull=never --network=none --entrypoint bash "$image" -euc '
  cd /build/vllm
  git diff --exit-code HEAD
  git update-index --refresh
  git apply --check --index /build/spark-overlay.patch
  git apply --index /build/spark-overlay.patch
  test "$(git write-tree)" = 02a457e2e933d0fb20d5786835110546acf7d8f2
  echo OVERLAY_LAYER_REGRESSION_PASS
'
