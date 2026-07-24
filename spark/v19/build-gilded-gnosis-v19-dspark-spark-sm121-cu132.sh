#!/usr/bin/env bash
set -euo pipefail

# GG v19 "dspark-dev" image for DGX Spark (GB10, SM121, aarch64).
#
# New-recipe lineage (RTX Pro 6000 crew, 2026-07-19): vLLM dev/gilded-gnosis
# + B12X master + FlashInfer 0.6.15. Clean public endpoints only — no fork
# branches. One build patch remains:
#   - VLLM_PATCH_FILE = vllm-sm121-cuda13-supported-archs: the CUDA>=13
#     CUDA_SUPPORTED_ARCHS branch (CMakeLists.txt:121) STILL lacks 12.1 on
#     dev/gilded-gnosis (the 12.1 entry at line 124 is the CUDA<13 branch),
#     so 12.1a collapses to 12.0 and cache_kernels.cu dies at ptxas on
#     cvt.e2m1x2 — confirmed by the r1 build failure 2026-07-19. Still an
#     upstream candidate.
#   - B12X_PATCH_FILE empty: this vLLM ships the mHC kernel package (4-tuple
#     consumer), so the v18p2 mHC revert is unnecessary; the W4A16 path was
#     restructured upstream (NF3 + "preserve preplanned tile geometry across
#     launch op"), superseding the v18p3 ultra-tile re-pin patch.
#
# | Component     | Ref |
# |---------------|-----|
# | vLLM          | local-inference-lab/vllm dev/gilded-gnosis @ 371085e9e |
# | B12X          | lukealonso/b12x master @ c7dc733 (torch>=2.12: satisfied |
# |               | by the image's torch 2.12.0+cu132)                      |
# | FlashInfer    | flashinfer-ai/flashinfer @ ce40d25a (= 0.6.15, the pin  |
# |               | from the branch's tools/spark/versions.env; includes the |
# |               | PR #3932 input_global_scale content)                    |
# | DeepGEMM      | a6b593d (SM121 MQA-logits patch still applies) |
# | NCCL          | nccl-canonical canonical/cu132-nccl2304-amd-noxml @ dfab7c1 |
# | InstantTensor | 85e7c5f (runtime loader, BUFFERED) |
# | CUTLASS       | d80a4e5 |
#
# Run on a DGX Spark node; executes inside ../blackwell-llm-docker.

cd "$(dirname "$0")/../blackwell-llm-docker"

export CONTAINER_ENGINE="${CONTAINER_ENGINE:-podman}"

export DATE_TAG="${DATE_TAG:-20260719}"
export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v19-dspark-spark-sm121-vllm371085e-b12xc7dc733-fice40d25-cu132-${DATE_TAG}}"
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

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-ce40d25a4cd05b2f28c935b99c9ea5091b4e64d2}"
export FLASHINFER_REF="${FLASHINFER_REF:-${FLASHINFER_COMMIT}}"
export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-${DEEPGEMM_COMMIT}}"
export DEEPGEMM_PATCH_FILE="${DEEPGEMM_PATCH_FILE-deepgemm-sm121-mqa-logits-arch-number-20260708.patch}"

export B12X_REPO="${B12X_REPO:-https://github.com/lukealonso/b12x.git}"
export B12X_COMMIT="${B12X_COMMIT:-c7dc73322cc50609f843fa2bbcc53283a90003b3}"
export B12X_REF="${B12X_REF:-master}"
export B12X_PATCH_FILE=

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_COMMIT="${VLLM_COMMIT:-371085e9e4ee3471125d69cfbfcfc66864634ee4}"
export VLLM_REF="${VLLM_REF:-dev/gilded-gnosis}"
export VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
export VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
# Spark-only: vLLM's CUDA>=13 CUDA_SUPPORTED_ARCHS lacks 12.1 (see header).
# Verified to apply at 371085e9e (hunk offset only). Disable with
# VLLM_PATCH_FILE="".
export VLLM_PATCH_FILE="${VLLM_PATCH_FILE-vllm-sm121-cuda13-supported-archs-20260714.patch}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v19.dspark.vllm371085e.b12xc7dc733.fice40d25.sm121.cu132.${DATE_TAG}}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"
# dev/gilded-gnosis renamed the GLM helpers (serve-glm52.sh, not
# serve-glm52-v18.sh) and adds serve-ds4-flash-spark.sh; require only what
# exists there and what we deploy.
export VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS:-serve-ds4-flash.sh serve-ds4-flash-spark.sh serve-glm51.sh serve-glm52.sh}"

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

# Cached FlashInfer wheels on the build node are from older pins (801d57a
# era); force a source build at 0.6.15.
FLASHINFER_WHEEL_DIR=".tmp-flashinfer-wheels"
FLASHINFER_WHEEL_STASH=".tmp-flashinfer-wheels.disabled-gg-v19-spark-${DATE_TAG}"
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
# Post-build validation. The v18-line Grid48 / fast-DCP / hybrid-preset
# checks do not apply on this lineage (b12x master has no Grid48 mapping
# API — the one-grid decode became the unified b12x hybrid op — and the GLM
# helper family was renamed). Validation here: source-provenance labels, a
# GPU stack smoke, and DS4 helper DRY_RUN wiring for the recipe's dspark
# knobs. All DRY_RUN captures use 2>&1 (helper prints to stderr).
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

# GPU stack smoke: torch 2.12/cu132 on the 48-SM GB10, FlashInfer 0.6.15,
# b12x importable, and the two new dspark engine envs present.
if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
podman run --rm --device nvidia.com/gpu=all -i \
  --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
import flashinfer
import torch

import b12x  # noqa: F401
from vllm import envs
from vllm.config.speculative import SpeculativeConfig  # noqa: F401

assert torch.__version__.startswith("2.12."), torch.__version__
assert flashinfer.__version__.startswith("0.6.15"), flashinfer.__version__
assert hasattr(envs, "VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH")
assert hasattr(envs, "VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE")

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

# Luke's experimental dspark config through our env names (root-helper
# spellings of VLLM_ENABLE_DSPARK=1 NUM_SPECULATIVE_TOKENS=7
# DSPARK_SPS_CURVE=auto VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH=1
# VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=1; DCP=1 is the helper default
# and mandatory for dspark).
luke_cfg="$(ds4_dry_run \
  -e MODE=dspark -e BACKEND=b12x-a8 -e TP_SIZE=2 \
  -e DSPARK_TOKENS=7 -e DSPARK_CAPACITY=1 -e DSPARK_SPS_CURVE=auto \
  -e DSPARK_DYNAMIC_DRAFT_DEPTH=1 -e DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=1)"
grep -q 'mode=dspark backend=b12x-a8' <<<"${luke_cfg}"
grep -Eq 'method\W+dspark' <<<"${luke_cfg}"
grep -Eq 'num_speculative_tokens\W+7' <<<"${luke_cfg}"
grep -q 'sps_curve' <<<"${luke_cfg}"
grep -q -- '--moe-backend b12x' <<<"${luke_cfg}"
grep -q -- '--decode-context-parallel-size 1' <<<"${luke_cfg}"
grep '^DS4 launch:' <<<"${luke_cfg}" || true

# mtp2 on the b12x-a8 backend still wires up.
mtp2_cfg="$(ds4_dry_run -e MODE=mtp2 -e BACKEND=b12x-a8 -e TP_SIZE=2)"
grep -q 'mode=mtp2 backend=b12x-a8' <<<"${mtp2_cfg}"
grep -Eq 'method\W+mtp' <<<"${mtp2_cfg}"

# The v18-production fallback profile still wires up on this lineage.
cutlass_cfg="$(ds4_dry_run -e MODE=dspark -e BACKEND=lucifer-cutlass -e TP_SIZE=2)"
grep -q 'mode=dspark backend=lucifer-cutlass' <<<"${cutlass_cfg}"
grep -q 'kernel-config.moe_backend flashinfer_cutlass' <<<"${cutlass_cfg}"

printf 'Image: %s\n' "${IMAGE}"
