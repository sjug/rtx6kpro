#!/usr/bin/env bash
set -euo pipefail

# DS4 v10 (Fathomless Firmament) image for DGX Spark (GB10, SM121, aarch64).
#
# Same source pins as the x86_64 v10 image from models/ds4dspark-v10.md
# (voipmonitor/vllm:fathomless-firmament-v15-vllmf5f4af3-b12x90172a5-cu132-20260709),
# retargeted at SM121 via the arch-parameterized blackwell-llm-docker branch
# spark/sm121-arm64 (see ../v9 for the original port and ../README.md for the
# full write-up):
#
# | Component     | Ref |
# |---------------|-----|
# | vLLM          | codex/ff-v15-mxfp4-online-mxfp8-20260709 @ f5f4af3 |
# | B12X          | voipmonitor/b12x codex/ff-v15-cute-compile-fallback-20260709 @ 90172a5 |
# | FlashInfer    | 5a73a36 |
# | DeepGEMM      | a6b593d (nv_dev tip; v9's SM121 MQA-logits fix still applies) |
# | InstantTensor | 85e7c5f |
# | CUTLASS       | d80a4e5 (same as v9) |
#
# Differences from the v9 Spark helper:
# - No VLLM_PATCH_FILE: the v9 indexer-warmup-fallback patch is obsolete —
#   B12X 90172a5 ships fused_indexer_decode_warmup_rows.
# - DeepGEMM SM121 MQA-logits patch kept (Spark-only; verified to apply at
#   a6b593d). Disable with DEEPGEMM_PATCH_FILE="" for strict doc parity.
# - humming-kernels 0.1.10 and pinned InstantTensor per the x86 v15 wrapper
#   (scripts/build-glm52-v15-final-image.sh in rtx6kpro).
#
# Run on a DGX Spark node (rusty/toby); executes inside ../blackwell-llm-docker.

cd "$(dirname "$0")/../blackwell-llm-docker"

export CONTAINER_ENGINE="${CONTAINER_ENGINE:-podman}"

export DATE_TAG="${DATE_TAG:-20260709}"
export IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-dspark-spark-sm121-vf5f4af3-b12x90172a5-cu132-${DATE_TAG}}"
export SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:spark-sm121-cu132-system-base-20260708}"
export BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:spark-sm121-cu132-build-base-20260708}"
# The v9 Spark base images are arch-correct and pin-identical (CUDA 13.2.1,
# torch 2.12.0+cu132, NCCL 2.30.4); reuse them instead of rebuilding.
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

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-5a73a36a7169ec5533ba474bb9204bed765dd297}"
export FLASHINFER_REF="${FLASHINFER_REF:-${FLASHINFER_COMMIT}}"
export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-${DEEPGEMM_COMMIT}}"
export DEEPGEMM_PATCH_FILE="${DEEPGEMM_PATCH_FILE-deepgemm-sm121-mqa-logits-arch-number-20260708.patch}"

export B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
export B12X_COMMIT="${B12X_COMMIT:-90172a504e96d246e07cb1ebad3b291532445560}"
export B12X_REF="${B12X_REF:-${B12X_COMMIT}}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_COMMIT="${VLLM_COMMIT:-f5f4af357e26643b355eb1190de7df1163bbcd98}"
export VLLM_REF="${VLLM_REF:-${VLLM_COMMIT}}"
export VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
export VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
export VLLM_PATCH_FILE="${VLLM_PATCH_FILE:-}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+fathomless.firmament.f5f4af3.b12x90172a5.sm121.cu132.${DATE_TAG}}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"

export INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO:-https://github.com/scitix/InstantTensor.git}"
export INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
export INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-${INSTANTTENSOR_COMMIT}}"

export CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
export CUTLASS_REF="${CUTLASS_REF:-d80a4e53b52b42550659a8696dab32705265e324}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-d80a4e53b52b42550659a8696dab32705265e324}"
export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.10}"
# fathomless-firmament vLLM declares these as runtime requirements; both have
# aarch64 wheels (matches the x86 v15 wrapper's VLLM_RUNTIME_EXTRA_PACKAGES).
export VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES:-nvtx==0.2.15 PyNvVideoCodec==2.1.0}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: this is an aarch64-native (DGX Spark) build; run on rusty/toby." >&2
  exit 1
fi

FLASHINFER_WHEEL_DIR=".tmp-flashinfer-wheels"
FLASHINFER_WHEEL_STASH=".tmp-flashinfer-wheels.disabled-ds4dspark-v10-spark-${DATE_TAG}"
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
