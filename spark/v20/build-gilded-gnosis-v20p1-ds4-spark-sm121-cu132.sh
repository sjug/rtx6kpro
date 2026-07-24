#!/usr/bin/env bash
set -euo pipefail

# GG v20p0 "ds4" image for DGX Spark (GB10, SM121, aarch64).
#
# Wiki-conformant port of the GLM-5.2 v20 release stack (models/glm5.2_v20.md,
# "Gilded Gnosis Safety And TP6") to our DS4-Flash serving pair. Pins mirror
# the wiki's release-candidate branches, hosted on our forks for provenance
# (pushed 2026-07-22):
#
# | Component     | Ref |
# |---------------|-----|
# | vLLM          | sjug/vllm build/gilded-gnosis-v20-final-candidate-20260721 |
# |               | @ 2167295c (= voipmonitor release branch; canonical GG    |
# |               | base b07bef75 + release PRs #145/#149/#150/#153/#155/     |
# |               | #156/#162)                                                |
# | SparkInfer    | sjug/b12x build/sparkinfer-v20-final-candidate-20260721   |
# |               | @ 6a92bcc0 (renamed b12x: namespaced sparkinfer/ tree,    |
# |               | dist name "sparkinfer" — needs Dockerfile.vllm-sparkinfer |
# |               | sibling for the verify import)                            |
# | FlashInfer    | sjug/flashinfer @ 801d57a0 (wiki pin, version.txt 0.6.15; |
# |               | restores the topk=256 DSV4 dispatches that stock 0.6.15   |
# |               | release lacks -> lucifer-cutlass dspark boots again).     |
# |               | PR#3932 quantfix deliberately NOT included (user call     |
# |               | 2026-07-22; revisit only if the cutlass fallback is       |
# |               | promoted beyond emergency use)                            |
# | DeepGEMM      | a6b593d (SM121 MQA-logits patch still applies) |
# | CUTLASS C++   | e6233cba (wiki v20 pin; v19 used d80a4e5) |
# | NCCL          | nccl-canonical canonical/cu132-nccl2304-amd-noxml @ dfab7c1 |
# | InstantTensor | 85e7c5f (wiki-identical) |
#
# Patch status (verified against these exact pins, 2026-07-22):
#   - VLLM_PATCH_FILE = vllm-sm121-cuda13-supported-archs: STILL REQUIRED.
#     The CUDA>=13 CUDA_SUPPORTED_ARCHS branch lacks 12.1 at BOTH 2167295c
#     and canonical b07bef75 (the visible 12.1 is the CUDA-12.8 branch).
#     Applies at hunk offset +9.
#   - i64 fused-indexer patch: DROPPED — superseded by upstream 50ae819
#     ("Fix Int32 page-offset overflow in the direct-K indexer score"),
#     in the release pin together with 1012199 (top-k candidate clamp) and
#     4812f46 (serial-merge fallback). Our PR#45 closed as superseded.
#   - PR#39 a16 ultra-tile patch: NOT carried. The force_tile validation was
#     rewritten upstream (_candidate_tile_fits + allow_logical_tail); a
#     BACKEND=b12x-a16 boot test in post-build validation decides whether a
#     port is needed.
#
# Run on a DGX Spark node; executes inside ../blackwell-llm-docker.

cd "$(dirname "$0")/../blackwell-llm-docker"

export CONTAINER_ENGINE="${CONTAINER_ENGINE:-podman}"
# SparkInfer rename: the sibling Dockerfile imports/verifies dist "sparkinfer".
export DOCKERFILE="${DOCKERFILE:-Dockerfile.vllm-sparkinfer-cu132}"

export DATE_TAG="${DATE_TAG:-20260722}"
export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20p1-ds4-spark-sm121-vllm2167295-si6a92bcc-fi7ad08da-cu132-${DATE_TAG}}"
export SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:spark-sm121-cu132-system-base-20260708}"
export BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:spark-sm121-cu132-build-base-20260708}"
export BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-0}"
export PUSH_BASE_IMAGE="${PUSH_BASE_IMAGE:-0}"

# SM121 / GB10 arch targets.
export TORCH_CUDA_ARCH_LIST_ARG="${TORCH_CUDA_ARCH_LIST_ARG:-12.1a}"
export CMAKE_CUDA_ARCHITECTURES_ARG="${CMAKE_CUDA_ARCHITECTURES_ARG:-121a}"
export FLASHINFER_CUDA_ARCH_LIST_ARG="${FLASHINFER_CUDA_ARCH_LIST_ARG:-12.1a}"

# GB10: 20 cores, 121 GB unified memory shared with the GPU.
export MAX_JOBS="${MAX_JOBS:-12}"
export VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-12}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
export PIN_SOURCE_COMMITS="${PIN_SOURCE_COMMITS:-1}"

export NCCL_REPO="${NCCL_REPO:-https://github.com/local-inference-lab/nccl-canonical.git}"
export NCCL_REF="${NCCL_REF:-canonical/cu132-nccl2304-amd-noxml}"
export NCCL_COMMIT="${NCCL_COMMIT:-dfab7c1ace32da250ba97757879429c341b7bcf9}"

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/sjug/flashinfer.git}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-7ad08da11eb5ba3fc92f576905dce3e2cec03313}"
export FLASHINFER_REF="${FLASHINFER_REF:-${FLASHINFER_COMMIT}}"
export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-${DEEPGEMM_COMMIT}}"
export DEEPGEMM_PATCH_FILE="${DEEPGEMM_PATCH_FILE-deepgemm-sm121-mqa-logits-arch-number-20260708.patch}"

export B12X_REPO="${B12X_REPO:-https://github.com/sjug/b12x.git}"
export B12X_COMMIT="${B12X_COMMIT:-6a92bcc0f2bf03b13dd03dbc7ce97e26133c580e}"
export B12X_REF="${B12X_REF:-build/sparkinfer-v20-final-candidate-20260721}"
export B12X_PATCH_FILE="${B12X_PATCH_FILE-sparkinfer-w4a16-ultratile-repin-20260722.patch}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/sjug/vllm.git}"
export VLLM_COMMIT="${VLLM_COMMIT:-2167295cd3e133d38ab22a67a42b0004db65d0a6}"
export VLLM_REF="${VLLM_REF:-build/gilded-gnosis-v20-final-candidate-20260721}"
export VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
export VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
# Spark-only: vLLM's CUDA>=13 CUDA_SUPPORTED_ARCHS lacks 12.1 (see header).
# Verified to apply at 2167295c (hunk offset +9). Disable with
# VLLM_PATCH_FILE="".
export VLLM_PATCH_FILE="${VLLM_PATCH_FILE-vllm-v20p1-ds4-fixes-20260722.patch}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev281+gilded.gnosis.v20p1.ds4.vllm2167295.si6a92bcc.fi7ad08da.sm121.cu132.${DATE_TAG}}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"
# All four exist at 2167295c (verified via ls-tree 2026-07-22).
export VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS:-serve-ds4-flash.sh serve-ds4-flash-spark.sh serve-glm51.sh serve-glm52.sh}"

export CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
export CUTLASS_REF="${CUTLASS_REF:-e6233cbac5d7c7a865c19c91cd684ceece19513c}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-e6233cbac5d7c7a865c19c91cd684ceece19513c}"
export TRITON_KERNELS_REF=
export TRITON_KERNELS_COMMIT=

export INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO:-https://github.com/scitix/InstantTensor.git}"
export INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
export INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.10}"
export VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES:-nvtx==0.2.15 PyNvVideoCodec==2.0.4 nccl4py==0.3.1}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: this is an aarch64-native (DGX Spark) build; run on a Spark node." >&2
  exit 1
fi

# Cached FlashInfer wheels on the build node carry unknown provenance (the
# v18 line built at 801d57a but v18.5 rebuilt at 7ad08da, and the cache is
# not keyed by commit). Force a source build at the pinned 801d57a so the
# fi801d57a label is truthful; the stash is restored on exit.
FLASHINFER_WHEEL_DIR=".tmp-flashinfer-wheels"
FLASHINFER_WHEEL_STASH=".tmp-flashinfer-wheels.disabled-gg-v20p1-spark-${DATE_TAG}"
if [[ "${FORCE_FLASHINFER_SOURCE:-1}" == "1" ]] \
  && compgen -G "${FLASHINFER_WHEEL_DIR}/flashinfer_*.whl" >/dev/null; then
  rm -rf "${FLASHINFER_WHEEL_STASH}"
  mkdir -p "${FLASHINFER_WHEEL_STASH}"
  mv "${FLASHINFER_WHEEL_DIR}"/flashinfer_*.whl "${FLASHINFER_WHEEL_STASH}"/
  restore_flashinfer_wheels() {
    if compgen -G "${FLASHINFER_WHEEL_STASH}/flashinfer_*.whl" >/dev/null; then
      mv "${FLASHINFER_WHEEL_STASH}"/flashinfer_*.whl "${FLASHINFER_WHEEL_DIR}"/
    fi
    rmdir "${FLASHINFER_WHEEL_STASH}" 2>/dev/null || true
  }
  trap restore_flashinfer_wheels EXIT
fi

./build-vllm-b12x-cu132.sh "$@"

# ---------------------------------------------------------------------------
# Post-build validation: source-provenance labels, a GPU stack smoke (torch,
# FlashInfer, the renamed sparkinfer package, dspark engine envs, GB10 device
# identity), and DS4 helper DRY_RUN wiring for dspark/mtp2/cutlass. All
# DRY_RUN captures use 2>&1 (helper prints to stderr).
# ---------------------------------------------------------------------------

for label_check in \
  "local-inference.vllm.commit=${VLLM_COMMIT}" \
  "local-inference.b12x.commit=${B12X_COMMIT}" \
  "local-inference.flashinfer.commit=${FLASHINFER_COMMIT}"; do
  label_key="${label_check%%=*}"
  label_want="${label_check#*=}"
  label_have="$(podman image inspect "${IMAGE}" \
    --format "{{ index .Config.Labels \"${label_key}\" }}")"
  if [[ "${label_have}" != "${label_want}" ]]; then
    echo "ERROR: label ${label_key}=${label_have}, expected ${label_want}" >&2
    exit 1
  fi
done

# GPU stack smoke: torch 2.12/cu132 on the 48-SM GB10, FlashInfer at the
# 801d57a pin (version.txt already reads 0.6.15 there), sparkinfer importable
# under its new name, upstream Int64 indexer fix present, dspark envs wired.
if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
podman run --rm --device nvidia.com/gpu=all -i \
  --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
import inspect

import flashinfer
import torch

import sparkinfer  # noqa: F401  (renamed from b12x; no import shim upstream)
from sparkinfer.attention.nsa_indexer import fused_indexer
from vllm import envs
from vllm.config.speculative import SpeculativeConfig  # noqa: F401

assert torch.__version__.startswith("2.12."), torch.__version__
assert flashinfer.__version__.startswith("0.6.15"), flashinfer.__version__
assert hasattr(envs, "VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH")
assert hasattr(envs, "VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE")

# Upstream 50ae819 (Int64 page offsets) must be present — it supersedes our
# i64 patch; a build without it reintroduces the direct-K IMA at ~49k ctx.
src = inspect.getsource(fused_indexer._score_tokens_direct_k)
assert "k_byte_off: Int64" in src, "direct-K Int64 offset fix missing"

# PR#3932 quantfix must be present (fi7ad08da = 801d57a + 4 commits):
# input_global_scale is its API marker in the cute_dsl b12x MoE.
import flashinfer.fused_moe.cute_dsl.b12x_moe as _fi_b12x_moe
assert "input_global_scale" in inspect.getsource(_fi_b12x_moe), \
    "PR#3932 quantfix content missing from FlashInfer"

# Baked vllm fixes: the broadcast mHC pre must be a registered custom op
# (cutlass dspark dynamo fix) — import the module to trigger registration.
import vllm.model_executor.kernels.mhc.tilelang  # noqa: F401
assert hasattr(torch.ops.vllm, "mhc_pre_broadcast_tilelang"), \
    "mhc_pre_broadcast_tilelang custom op not registered"

props = torch.cuda.get_device_properties(torch.cuda.current_device())
assert (int(props.major), int(props.minor)) == (12, 1)
assert int(props.multi_processor_count) == 48
print("GPU stack smoke: PASS",
      f"torch={torch.__version__} flashinfer={flashinfer.__version__}")
PY
fi

ds4_dry_run() {
  podman run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
    -e DRY_RUN=1 -e MODEL=/model "$@" "${IMAGE}" 2>&1
}

# Production dspark config wiring (our K=6 profile on the b12x-a8 backend).
prod_cfg="$(ds4_dry_run \
  -e MODE=dspark -e BACKEND=b12x-a8 -e TP_SIZE=2 -e DSPARK_TOKENS=6)"
grep -q 'mode=dspark backend=b12x-a8' <<<"${prod_cfg}"
grep -Eq 'method\W+dspark' <<<"${prod_cfg}"
grep -Eq 'num_speculative_tokens\W+6' <<<"${prod_cfg}"
grep -q -- '--moe-backend b12x' <<<"${prod_cfg}"
grep -q -- '--decode-context-parallel-size 1' <<<"${prod_cfg}"

# mtp2 on the b12x-a8 backend still wires up (boot success is a separate
# post-build test: the v19 mHC/3D warmup assert may or may not be fixed here).
mtp2_cfg="$(ds4_dry_run -e MODE=mtp2 -e BACKEND=b12x-a8 -e TP_SIZE=2)"
grep -q 'mode=mtp2 backend=b12x-a8' <<<"${mtp2_cfg}"
grep -Eq 'method\W+mtp' <<<"${mtp2_cfg}"

# lucifer-cutlass wires up — at fi801d57a this is a REAL fallback again
# (topk=256 DSV4 dispatches present, unlike stock 0.6.15).
cutlass_cfg="$(ds4_dry_run -e MODE=dspark -e BACKEND=lucifer-cutlass -e TP_SIZE=2)"
grep -q 'mode=dspark backend=lucifer-cutlass' <<<"${cutlass_cfg}"
grep -q 'kernel-config.moe_backend flashinfer_cutlass' <<<"${cutlass_cfg}"

printf 'BUILD-OK-v20p1 Image: %s\n' "${IMAGE}"
