#!/usr/bin/env bash
set -euo pipefail

# DS4/DSpark + GLM unified v16.1 image for DGX Spark (GB10, SM121, aarch64):
# v16 with B12X advanced to pick up lukealonso's SM121/DGX-Spark
# optimization series (2026-07-14/15), notably:
#   f8dd59d Optimize DSV4F WO decode chain for DGX Spark
#   0b17124 Optimize Sparse MLA decode for DGX Spark
#   c368a83 perf(spark): optimize decode kernels for SM121
#   550527c/3c57a4c Optimize + per-platform tune FP8 dense GEMM
#   34f3b26/906d63d/1717469 RTX Blackwell + CuTe compile fixes
#
# B12X source is sjug/b12x branch
# codex/fathomless-firmament-v16-integration-20260716 @ 1bcc652: the two v16 fork
# commits (90172a5 CuTe compile
# callable, fe06f49 W4A8 resident-grid — voipmonitor v16-integration tip)
# rebased onto lukealonso/master tip 3c57a4c. Tree verified byte-identical
# to `git merge-tree 3c57a4c fe06f49` (d32926b7), which merges clean.
# The historical v16 pin (voipmonitor @ fe06f49) remains untouched in
# ../v16/ for exact reproducibility of the shipped v16 image.
#
# Everything else is pin-identical to v16 (see ../v16/ for the full pin
# table): vLLM 8f86f42, FlashInfer 801d57a, DeepGEMM a6b593d, NCCL dfab7c1,
# InstantTensor 85e7c5f, CUTLASS d80a4e5. The v17 CUDA13 supported-archs
# vLLM patch is deliberately NOT applied so the only variable vs the v16
# image is B12X (vLLM kernels stay family-generic 12.0, same as v16; DS4
# needs no 12.1a-only vLLM instructions).
#
# Note the b12x-build stage feeds vllm-build, so this bump recompiles vLLM
# (~3 h on GB10); flashinfer/deepgemm stages cache-hit.
#
# Run on a DGX Spark node (rusty/toby); executes inside ../blackwell-llm-docker.

cd "$(dirname "$0")/../blackwell-llm-docker"

export CONTAINER_ENGINE="${CONTAINER_ENGINE:-podman}"

export DATE_TAG="${DATE_TAG:-20260716}"
export IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v16p1-spark-sm121-vllm8f86f42-b12x1bcc652-fi801d57a-cu132-${DATE_TAG}}"
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

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/voipmonitor/flashinfer.git}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-801d57a08958c13d375ddbb6be3be4808f48a708}"
export FLASHINFER_REF="${FLASHINFER_REF:-codex/sm120-dspark-stack-20260711}"
export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-${DEEPGEMM_COMMIT}}"
export DEEPGEMM_PATCH_FILE="${DEEPGEMM_PATCH_FILE-deepgemm-sm121-mqa-logits-arch-number-20260708.patch}"

export B12X_REPO="${B12X_REPO:-https://github.com/sjug/b12x.git}"
export B12X_COMMIT="${B12X_COMMIT:-1bcc652708b26adb5c42761af05d840f2833843f}"
export B12X_REF="${B12X_REF:-codex/fathomless-firmament-v16-integration-20260716}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_COMMIT="${VLLM_COMMIT:-8f86f425102cee08745462615d54115eee275f9f}"
export VLLM_REF="${VLLM_REF:-codex/fathomless-firmament-v16-unified-20260712}"
export VLLM_PATCH_URL=
export VLLM_PATCH_SHA256=
export VLLM_PATCH_FILE=
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+fathomless.firmament.v16p1.vllm8f86f42.b12x1bcc652.fi801d57a.sm121.cu132.${DATE_TAG}}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"
export VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS:-serve-fathomless-firmament.sh serve-ds4-flash.sh serve-glm52-v16.sh}"

export CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
export CUTLASS_REF="${CUTLASS_REF:-d80a4e53b52b42550659a8696dab32705265e324}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-d80a4e53b52b42550659a8696dab32705265e324}"
export TRITON_KERNELS_REF=
export TRITON_KERNELS_COMMIT=

export INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO:-https://github.com/scitix/InstantTensor.git}"
export INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
export INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.10}"
export VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES:-nvtx==0.2.15 PyNvVideoCodec==2.0.4 nccl4py==0.3.1}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: this is an aarch64-native (DGX Spark) build; run on rusty/toby." >&2
  exit 1
fi

FLASHINFER_WHEEL_DIR=".tmp-flashinfer-wheels"
FLASHINFER_WHEEL_STASH=".tmp-flashinfer-wheels.disabled-ds4dspark-v16p1-spark-${DATE_TAG}"
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
# Post-build validation, identical to v16 (docker -> podman).
# DRY_RUN helper checks need no GPU and no model files.
# ---------------------------------------------------------------------------

podman run --rm --entrypoint /usr/local/bin/serve-glm52-v16.sh \
  -e DRY_RUN=1 \
  -e MODEL=lukealonso/GLM-5.2-NVFP4 \
  -e MTP=0 \
  -e DCP=2 \
  -e MOE_MODE=a16 \
  -e ONLINE_QUANT=mxfp8 \
  "${IMAGE}"

# The helper prints its DRY_RUN summary to stderr; capture both streams
# (the canonical recipe's stdout-only capture comes back empty under podman).
tp2_dspark_command="$(podman run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
  -e DRY_RUN=1 \
  -e MODE=dspark \
  -e BACKEND=lucifer-cutlass \
  -e TP_SIZE=2 \
  "${IMAGE}" 2>&1)"
grep -q -- '--gpu-memory-utilization 0.9465' <<<"${tp2_dspark_command}"

tp4_dspark_command="$(podman run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
  -e DRY_RUN=1 \
  -e MODE=dspark \
  -e BACKEND=lucifer-cutlass \
  -e TP_SIZE=4 \
  "${IMAGE}" 2>&1)"
grep -q -- '--gpu-memory-utilization 0.94' <<<"${tp4_dspark_command}"

podman run --rm --entrypoint /bin/bash "${IMAGE}" -lc \
  'grep -q "cooperative=True" /opt/venv/lib/python3.12/site-packages/b12x/moe/fused/dynamic.py'

podman run --rm --entrypoint /usr/local/bin/serve-fathomless-firmament.sh \
  -e MODEL_FAMILY=glm52 \
  -e DRY_RUN=1 \
  -e MODEL=lukealonso/GLM-5.2-NVFP4 \
  "${IMAGE}"

printf 'Image: %s\n' "${IMAGE}"
