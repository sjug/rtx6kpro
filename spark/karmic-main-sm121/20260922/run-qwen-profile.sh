#!/usr/bin/env bash
# Saved HC-off profile; preserve the frozen image/build recipe and control runner.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
image_id=88867036403cb3c9bc3026eb8c907ad64bc1af9c28d316d22b670dde4f561152
if [[ ${EXPECTED_IMAGE_ID:-$image_id} != "$image_id" ]]; then
  echo 'Saved Qwen profile requires its qualified image ID' >&2
  exit 78
fi
if [[ ${VLLM_QWEN3_8_FLASH_NEXT_HC_TP:-0} != 0 ]]; then
  echo 'Saved Qwen profile requires HC_TP=0; use the diagnostic runner for controls' >&2
  exit 78
fi
echo "f77081a9dba1cc2ea36fddb40d3fd6e73e855732657f29c98edf101d8640036d  $base/run-qwen-hc-diagnostic.sh" | sha256sum --check --status
export EXPECTED_IMAGE_ID=$image_id
export VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0
exec bash "$base/run-qwen-hc-diagnostic.sh"
