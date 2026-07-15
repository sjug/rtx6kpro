#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2 v17 hybrid + DS4 unified (Fathomless Firmament) image for DGX Spark
# (GB10, SM121, aarch64).
#
# v17 delta on the v16 Spark recipe (build-ds4dspark-v16-spark-sm121-cu132.sh):
# only the vLLM and B12X pins change, plus the serve-glm52-hybrid-v17.sh
# launcher. Everything else (FlashInfer, DeepGEMM + SM121 patch, NCCL,
# InstantTensor, CUTLASS, arm64 base images) is pin-identical to v16.
# Canonical x86 recipe: blackwell-llm-docker
# build-fathomless-firmament-v17-cu132.sh @ 6d3d0aa (documented in
# models/glm5.2_v17.md); it wraps the v16 script the same way this wraps
# the v16 Spark env. Runs on the existing spark/sm121-arm64-v16 branch —
# the only upstream d104659..6d3d0aa build-infra changes are the 2>&1
# capture and overridable-launcher fixes this port already carries.
#
# | Component     | Ref |
# |---------------|-----|
# | vLLM          | codex/fathomless-firmament-v17-dcp-prefill-opt-20260714 @ 6ccc3eb |
# | B12X          | voipmonitor/b12x codex/fathomless-firmament-v17-nf3-nvfp4kv-20260714 @ 1377d5f |
# | FlashInfer    | voipmonitor/flashinfer codex/sm120-dspark-stack-20260711 @ 801d57a |
# | DeepGEMM      | a6b593d (SM121 MQA-logits patch still applies) |
# | NCCL          | nccl-canonical canonical/cu132-nccl2304-amd-noxml @ dfab7c1 |
# | InstantTensor | 85e7c5f (runtime loader, BUFFERED) |
# | CUTLASS       | d80a4e5 |
#
# Run on a DGX Spark node; executes inside ../blackwell-llm-docker.

cd "$(dirname "$0")/../blackwell-llm-docker"

export CONTAINER_ENGINE="${CONTAINER_ENGINE:-podman}"

export DATE_TAG="${DATE_TAG:-20260714}"
export IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v17-spark-sm121-vllm6ccc3eb-b12x1377d5f-fi801d57a-cu132-${DATE_TAG}}"
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

export B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
export B12X_COMMIT="${B12X_COMMIT:-1377d5f22c98de0c17d9b3f35a5b56d7587992fa}"
export B12X_REF="${B12X_REF:-codex/fathomless-firmament-v17-nf3-nvfp4kv-20260714}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_COMMIT="${VLLM_COMMIT:-6ccc3ebbd17edb05ce11b095a5b14f25839774dd}"
export VLLM_REF="${VLLM_REF:-codex/fathomless-firmament-v17-dcp-prefill-opt-20260714}"
export VLLM_PATCH_URL=
export VLLM_PATCH_SHA256=
# Spark-only: vLLM's CUDA>=13 CUDA_SUPPORTED_ARCHS lacks 12.1, collapsing
# TORCH_CUDA_ARCH_LIST=12.1a to plain "12.0"; the v17 NVFP4 KV kernels in
# cache_kernels.cu then fail at ptxas (cvt.e2m1x2.f32 needs an
# arch-specific target). Disable with VLLM_PATCH_FILE="".
export VLLM_PATCH_FILE="${VLLM_PATCH_FILE-vllm-sm121-cuda13-supported-archs-20260714.patch}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+fathomless.firmament.v17.vllm6ccc3eb.b12x1377d5f.fi801d57a.sm121.cu132.${DATE_TAG}}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"
export VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS:-serve-fathomless-firmament.sh serve-ds4-flash.sh serve-glm52-v16.sh serve-glm52-hybrid-v17.sh}"

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
  echo "ERROR: this is an aarch64-native (DGX Spark) build; run on a Spark node." >&2
  exit 1
fi

FLASHINFER_WHEEL_DIR=".tmp-flashinfer-wheels"
FLASHINFER_WHEEL_STASH=".tmp-flashinfer-wheels.disabled-ds4dspark-v17-spark-${DATE_TAG}"
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
# Post-build validation. DRY_RUN helper checks need no GPU and no model files.
# Ported from the canonical v17 recipe (docker -> podman); all captures use
# 2>&1 (the helpers print to stderr; stdout capture is empty under podman).
# The canonical TP6/TP8 topology loop is dropped: those need 6-8 GPUs and are
# not deployable on 1-GPU-per-node Sparks. TP4/DCP4 stays as a wiring check
# even though cross-node DCP is not usable (B12X DCP pool is CUDA-IPC,
# single-node); the deployment profile on Sparks is TP4/DCP1.
# ---------------------------------------------------------------------------

hybrid_dcp4="$(podman run --rm --entrypoint /usr/local/bin/serve-glm52-hybrid-v17.sh \
  -e DRY_RUN=1 \
  -e MODEL=/model \
  -e DCP=4 \
  -e MAX_BATCHED_TOKENS=3072 \
  "${IMAGE}" 2>&1)"
grep -q -- '--tensor-parallel-size 4' <<<"${hybrid_dcp4}"
grep -q -- '--decode-context-parallel-size 4' <<<"${hybrid_dcp4}"
grep -q -- '--kv-cache-dtype nvfp4_ds_mla' <<<"${hybrid_dcp4}"
grep -q -- '--quantization nvfp4_nf3_hybrid' <<<"${hybrid_dcp4}"
grep -q -- '--load-format instanttensor' <<<"${hybrid_dcp4}"
grep -q -- 'VLLM_DCP_PROJECT_BEFORE_MERGE=1' <<<"${hybrid_dcp4}"
grep -q -- 'VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=1' <<<"${hybrid_dcp4}"

# Spark deployment profile: TP4/DCP1 must resolve the workspace gate to off.
hybrid_dcp1="$(podman run --rm --entrypoint /usr/local/bin/serve-glm52-hybrid-v17.sh \
  -e DRY_RUN=1 \
  -e MODEL=/model \
  -e DCP=1 \
  "${IMAGE}" 2>&1)"
grep -q -- '--tensor-parallel-size 4' <<<"${hybrid_dcp1}"
grep -q -- '--decode-context-parallel-size 1' <<<"${hybrid_dcp1}"
grep -q -- 'VLLM_DCP_PROJECT_BEFORE_MERGE=0' <<<"${hybrid_dcp1}"
grep -q -- 'VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=0' <<<"${hybrid_dcp1}"

# Explicit workspace-off override must gate out on an eligible topology.
workspace_off="$(podman run --rm --entrypoint /usr/local/bin/serve-glm52-v16.sh \
  -e DRY_RUN=1 \
  -e MODEL=/model \
  -e TP=8 \
  -e DCP=4 \
  -e DCP_PREFILL_WORKSPACE=0 \
  "${IMAGE}" 2>&1)"
grep -q -- 'VLLM_DCP_PROJECT_BEFORE_MERGE=0' <<<"${workspace_off}"
grep -q -- 'VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=0' <<<"${workspace_off}"

# v16-carried checks: the image stays the unified GLM/DS4 base.
tp2_dspark_command="$(podman run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
  -e DRY_RUN=1 \
  -e MODE=dspark \
  -e BACKEND=lucifer-cutlass \
  -e TP_SIZE=2 \
  "${IMAGE}" 2>&1)"
grep -q -- '--gpu-memory-utilization 0.9465' <<<"${tp2_dspark_command}"

podman run --rm --entrypoint /bin/bash "${IMAGE}" -lc \
  'grep -q "cooperative=True" /opt/venv/lib/python3.12/site-packages/b12x/moe/fused/dynamic.py'

podman run --rm --entrypoint /usr/local/bin/serve-fathomless-firmament.sh \
  -e MODEL_FAMILY=glm52 \
  -e DRY_RUN=1 \
  -e MODEL=lukealonso/GLM-5.2-NVFP4 \
  "${IMAGE}" >/dev/null 2>&1

printf 'Image: %s\n' "${IMAGE}"
