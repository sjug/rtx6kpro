#!/usr/bin/env bash
set -euo pipefail

# GG v18.2 for DGX Spark (GB10, SM121, aarch64): DS4-capable v18 via a B12X
# revert instead of the v18.1 vLLM forward-port.
#
# Failure chain this resolves (logs in ~/logs on rusty, bootfail 103450 and
# 105351): B12X e71a090 ("Optimize broadcast mHC pre kernel") changed
# b12x_mhc_pre to a 4-tuple/2-D first-layer contract. The immutable GG v18
# vLLM release (264bce1) predates it (3-tuple unpack -> UNPACK_SEQUENCE
# crash), and cherry-picking the rebase-branch fix ff03fd654 alone (v18.1)
# still fails because the 2-D first-layer scaffolding (broadcast weight
# finalization, embedding-expansion removal, mhc_pre_broadcast_tilelang
# fallback, draft-model equivalents in 533b037f3) lives in unrebased drift.
# v18.2 therefore keeps the release vLLM as-is and reverts e71a090 at build
# time, restoring the exact 3-tuple mHC contract the release tree was written
# against (the combination proven by all v16.x Spark serving). Grid188
# (PR #36) is disjoint from the revert and is retained. GLM paths never call
# b12x_mhc_pre and are unaffected. The proper forward-port follows later via
# signed sjug branches once upstream's rebase line stabilizes (Luke is
# actively landing DS4-on-Spark work there, incl. serve-ds4-flash-spark.sh).
#
# Sources build from the sjug mirrors of the exact v18 pins:
#   sjug/vllm       build/gilded-gnosis-v18-final-20260718   @ 264bce1
#   sjug/b12x       codex/nf3-grid188-decode-20260717        @ bc85ef3 (+revert patch)
#   sjug/flashinfer codex/sm120-dspark-stack-20260711        @ 801d57a

export DATE_TAG="${DATE_TAG:-20260718}"
export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v18p2-spark-sm121-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-${DATE_TAG}}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/sjug/vllm.git}"
export VLLM_REF="${VLLM_REF:-build/gilded-gnosis-v18-final-20260718}"

export B12X_REPO="${B12X_REPO:-https://github.com/sjug/b12x.git}"
export B12X_PATCH_FILE="${B12X_PATCH_FILE:-b12x-revert-e71a090-broadcast-mhc-pre-20260718.patch}"

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/sjug/flashinfer.git}"

export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v18p2.vllm264bce1.b12xbc85ef3.reve71a090.fi801d57a.sm121.cu132.${DATE_TAG}}"

exec "$(dirname "$0")/../v18/build-gilded-gnosis-v18-spark-sm121-cu132.sh" "$@"
