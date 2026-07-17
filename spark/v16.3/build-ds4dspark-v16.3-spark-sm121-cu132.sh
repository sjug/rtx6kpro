#!/usr/bin/env bash
set -euo pipefail

# DS4/DSpark + GLM unified v16.3 image for DGX Spark (GB10, SM121,
# aarch64). This is the original v16 recipe with only the paired
# FlashInfer PR #3932 and vLLM PR #48536 branches substituted.
#
# | Component  | v16 base | v16.3 pin |
# |------------|----------|-----------|
# | FlashInfer | voipmonitor codex/sm120-dspark-stack-20260711 @ 801d57a | sjug codex/sm120-dspark-stack-pr3932-20260716 @ c0700ff |
# | vLLM       | local-inference-lab codex/fathomless-firmament-v16-unified-20260712 @ 8f86f42 | sjug codex/fathomless-firmament-v16-unified-pr48536-20260716 @ 6a5a106 |
#
# All other source pins are inherited unchanged from ../v16/, including
# B12X fe06f49. The historical v16 recipe remains untouched.

export DATE_TAG="${DATE_TAG:-20260716}"
export IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v16p3-spark-sm121-vllm6a5a106-b12xfe06f49-fic0700ff-cu132-${DATE_TAG}}"

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/sjug/flashinfer.git}"
export FLASHINFER_REF="${FLASHINFER_REF:-codex/sm120-dspark-stack-pr3932-20260716}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-c0700ff92ddbc32bd452d9d03cf75d9a8f780fe5}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/sjug/vllm.git}"
export VLLM_REF="${VLLM_REF:-codex/fathomless-firmament-v16-unified-pr48536-20260716}"
export VLLM_COMMIT="${VLLM_COMMIT:-6a5a106b3f783be87e01e15ad99e590b740f3945}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+fathomless.firmament.v16p3.vllm6a5a106.b12xfe06f49.fic0700ff.sm121.cu132.${DATE_TAG}}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"

exec "$(dirname "$0")/../v16/build-ds4dspark-v16-spark-sm121-cu132.sh" "$@"
