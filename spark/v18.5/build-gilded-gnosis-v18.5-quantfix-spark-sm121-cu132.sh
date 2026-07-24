#!/usr/bin/env bash
# GG v18.5 for DGX Spark: v18.4 (Grid48) + the FP4 quant-numerics fixes.
#   FlashInfer: PR #3932 (4 commits, rebased 2026-07-18: subnormal e4m3 decode,
#     precise-path multiplier, bench selector, input_global_scale + review fixes)
#     as sjug feat/sm12x-dspark-pr3932-20260718 on the 801d57a pin.
#   vLLM: PR #48536 (stop baking weight_scale_2 into b12x MoE block scales)
#     as sjug feat/gg-v18p4-pr48536-20260718 on the Grid48 tree df7a0b7.
#   B12X unchanged at 6b10833. Affects the FlashInfer b12x MoE path (quality:
#   ~17% avg dequant-weight distortion removed); throughput-neutral expected.
set -euo pipefail
cd "$(dirname "$0")"

export DATE_TAG=20260718
export IMAGE="voipmonitor/vllm:gilded-gnosis-v18p5-quantfix-spark-sm121-vllm4966149-b12x6b10833-fi7ad08da-cu132-${DATE_TAG}"
export FLASHINFER_REF=feat/sm12x-dspark-pr3932-20260718
export FLASHINFER_COMMIT=7ad08da11eb5ba3fc92f576905dce3e2cec03313
export VLLM_REF=feat/gg-v18p4-pr48536-20260718
export VLLM_COMMIT=4966149d5e693d58b152e5cca8021e63e6be4169
export VLLM_BUILD_VERSION="0.11.2.dev280+gilded.gnosis.v18p5.quantfix.vllm4966149.b12x6b10833.fi7ad08da.sm121.cu132.${DATE_TAG}"

exec ../v18.4/build-gilded-gnosis-v18.4-grid48-spark-sm121-cu132.sh "$@"
