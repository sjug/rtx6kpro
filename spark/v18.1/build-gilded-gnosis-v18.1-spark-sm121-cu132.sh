#!/usr/bin/env bash
set -euo pipefail

# GG v18.1 for DGX Spark (GB10, SM121, aarch64): the v18 recipe plus one
# upstream fix. B12X e71a090 ("Optimize broadcast mHC pre kernel", contained
# in the v18 pin bc85ef3) changed b12x_mhc_pre from a 3-tuple to a 4-tuple
# return, but the immutable GG v18 vLLM release (264bce1) still unpacks 3 in
# deepseek_v4/nvidia/model.py — DS4 boot dies in the profile_run dummy pass
# (dynamo UNPACK_SEQUENCE, expected 3 got 4). The v18 x86 release campaign
# validated GLM only, which never touches that model file, so the break
# shipped unnoticed; our first v18 DS4 TP2 boot on rusty/toby hit it
# (~/logs/ds4-v18-tp2-b12x-a16-mtp0-bootfail-20260718_103450.log on rusty).
#
# The matching vLLM-side fix exists upstream but only on the post-release
# branch: local-inference-lab/vllm dev/gilded-gnosis-rebase @ ff03fd654
# "fix(b12x): use broadcast mHC pre kernel". v18.1 applies exactly that
# commit as a sha256-pinned patch on top of the unchanged 264bce1 pin.
# Verified: applies cleanly after the CUDA13 supported-archs patch; no file
# overlap. GLM code paths are untouched by both the bug and the fix.
#
# Everything else (B12X bc85ef3, FlashInfer 801d57a, DeepGEMM, NCCL,
# InstantTensor, CUTLASS, launchers, validation) is inherited from
# ../v18/build-gilded-gnosis-v18-spark-sm121-cu132.sh.

export DATE_TAG="${DATE_TAG:-20260718}"
export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v18p1-spark-sm121-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-${DATE_TAG}}"

export VLLM_PATCH_URL="${VLLM_PATCH_URL:-https://github.com/local-inference-lab/vllm/commit/ff03fd654050b6ef34ae247fa397c11c63aae004.patch}"
export VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-a6931c736abc8cf376cbd0192759e5f774b47e287b890e98508f3c7e0ef3bf4e}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v18p1.vllm264bce1.mhcfixff03fd6.b12xbc85ef3.fi801d57a.sm121.cu132.${DATE_TAG}}"

exec "$(dirname "$0")/../v18/build-gilded-gnosis-v18-spark-sm121-cu132.sh" "$@"
