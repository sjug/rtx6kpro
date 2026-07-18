#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2 + DS4 unified Gilded Gnosis v18.4 Grid48 image for DGX Spark
# (GB10, SM121, aarch64). Grid48 is built directly from the reviewed B12X
# and vLLM branches; no Grid48 source patch or post-build hotfix layer is
# involved.
#
# v18.4 delta on the v18.3 Spark image: B12X adds the exact SM121/48-SM
# mapped grid and carries the v18.2 mHC compatibility revert plus the v18.3
# ultra-tile re-pin fix as source commits. vLLM selects Grid188 only on exact
# SM120/188-SM devices and Grid48 only on exact SM121/48-SM devices. All
# other component pins and the SM121 build patches remain unchanged.
# Canonical x86 recipe: blackwell-llm-docker build-gilded-gnosis-v18-cu132.sh
# @ 7f3cbc6 (documented in models/glm5.2_v18.md). Runs on the
# spark/sm121-arm64-v16 branch with upstream main @ 7f3cbc6 merged in.
#
# | Component     | Ref |
# |---------------|-----|
# | vLLM          | sjug/vllm codex/gilded-gnosis-grid48-sm121-20260718 @ df7a0b7 |
# | B12X          | sjug/b12x codex/nf3-grid48-sm121-20260718 @ 6b10833 |
# | FlashInfer    | sjug/flashinfer codex/sm120-dspark-stack-20260711 @ 801d57a |
# | DeepGEMM      | a6b593d (SM121 MQA-logits patch still applies) |
# | NCCL          | nccl-canonical canonical/cu132-nccl2304-amd-noxml @ dfab7c1 |
# | InstantTensor | 85e7c5f (runtime loader, BUFFERED) |
# | CUTLASS       | d80a4e5 |
#
# Run on a DGX Spark node; executes inside ../blackwell-llm-docker.

cd "$(dirname "$0")/../blackwell-llm-docker"

export CONTAINER_ENGINE="${CONTAINER_ENGINE:-podman}"

export DATE_TAG="${DATE_TAG:-20260718}"
export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v18p4-grid48-spark-sm121-vllmdf7a0b7-b12x6b10833-fi801d57a-cu132-${DATE_TAG}}"
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
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-801d57a08958c13d375ddbb6be3be4808f48a708}"
export FLASHINFER_REF="${FLASHINFER_REF:-codex/sm120-dspark-stack-20260711}"
export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-${DEEPGEMM_COMMIT}}"
export DEEPGEMM_PATCH_FILE="${DEEPGEMM_PATCH_FILE-deepgemm-sm121-mqa-logits-arch-number-20260708.patch}"

# Grid48 plus the two fixes carried by the active v18.3 image. Because both
# fixes are now source commits on this branch, B12X_PATCH_FILE must stay empty.
export B12X_REPO="${B12X_REPO:-https://github.com/sjug/b12x.git}"
export B12X_COMMIT="${B12X_COMMIT:-6b10833ce4880f988ddea9d089ec12058174037c}"
export B12X_REF="${B12X_REF:-codex/nf3-grid48-sm121-20260718}"
export B12X_PATCH_FILE=

export VLLM_REPO="${VLLM_REPO:-https://github.com/sjug/vllm.git}"
export VLLM_COMMIT="${VLLM_COMMIT:-df7a0b707f26a2dde80d4285e7b45a5e72f50e6f}"
export VLLM_REF="${VLLM_REF:-codex/gilded-gnosis-grid48-sm121-20260718}"
export VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
export VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
# Spark-only: vLLM's CUDA>=13 CUDA_SUPPORTED_ARCHS lacks 12.1, collapsing
# TORCH_CUDA_ARCH_LIST=12.1a to plain "12.0"; the NVFP4 MLA KV kernels in
# cache_kernels.cu then fail at ptxas (cvt.e2m1x2.f32 needs an arch-specific
# target). Verified against the v18 base of df7a0b7. Disable with
# VLLM_PATCH_FILE="".
export VLLM_PATCH_FILE="${VLLM_PATCH_FILE-vllm-sm121-cuda13-supported-archs-20260714.patch}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v18p4.grid48.vllmdf7a0b7.b12x6b10833.fi801d57a.sm121.cu132.${DATE_TAG}}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"
export VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS:-serve-gilded-gnosis.sh serve-fathomless-firmament.sh serve-ds4-flash.sh serve-glm52-v16.sh serve-glm52-v18.sh serve-glm52-hybrid-v17.sh serve-glm52-hybrid-v18.sh}"

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

# The cached FlashInfer wheel on the build node may be from a different FI
# commit (v16.2/v16.3 experiments); force a source build at the pinned commit.
FLASHINFER_WHEEL_DIR=".tmp-flashinfer-wheels"
FLASHINFER_WHEEL_STASH=".tmp-flashinfer-wheels.disabled-gg-v18-spark-${DATE_TAG}"
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
# Post-build validation, ported from the canonical v18 recipe (docker ->
# podman; --gpus -> CDI). All DRY_RUN captures use 2>&1 (the helpers print to
# stderr; stdout capture is empty under podman). TP8/TP6 dry-runs are wiring
# checks only — not deployable on 1-GPU-per-node Sparks; the deployment
# profiles here are hybrid TP4/DCP1 (sparky cluster) and DS4 TP2 (rusty/toby).
# ---------------------------------------------------------------------------

# Source-provenance labels must match the pins that were just built.
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

# GPU validation: consolidated GG runtime invariants, exact Grid48 mapping,
# real CuTe compilation/resource admission on the local 48-SM GB10, and the
# SM100+ nvfp4_ds_mla CUDA writer. Needs the GB10 GPU via CDI; skip with
# SKIP_GPU_CHECK=1 only when the build node GPU is busy.
if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
podman run --rm --device nvidia.com/gpu=all -i \
  -e B12X_PRINT_COMPILE_PROGRESS=1 \
  --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
from typing import get_args

import torch
from vllm import _custom_ops  # noqa: F401
from vllm import envs
from vllm.config.cache import CacheDType
from vllm.config.quantization import resolve_quantization_config
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.attention.backends.mla import b12x_mla_sparse
from vllm.v1.attention.ops import dcp_alltoall
from b12x.moe.fused.w4a16.kernel import (
    compile_w4a16_hybrid_mapped_grid48,
    w4a16_hybrid_mapped_grid48_mapping_proof,
)

assert hasattr(envs, "VLLM_DCP_QUERY_SPLIT")
assert hasattr(envs, "VLLM_B12X_MLA_CKV_GATHER")
assert hasattr(envs, "VLLM_NF3_MAPPED_GRID_DECODE")
assert hasattr(b12x_mla_sparse, "_global_causal_lens_for_ckv_gather")
assert hasattr(SpeculativeConfig, "_inherit_target_revision_for_mtp")
assert hasattr(dcp_alltoall, "_DCP_A2A_GRAPH_BUFFERS")
assert "nvfp4_ds_mla" in get_args(CacheDType)
assert hasattr(torch.ops._C_cache_ops, "concat_and_cache_nvfp4_mla")
assert resolve_quantization_config("nvfp4_nf3_hybrid", {"linear": {"weight": "mxfp8"}})
proof = w4a16_hybrid_mapped_grid48_mapping_proof()
assert proof["grid_x"] == 48
assert len(proof["fc1_tasks"]) == 128
assert len(proof["fc2_tasks"]) == 768
assert proof["fc1_per_cta_counts"] == (3,) * 32 + (2,) * 16
assert proof["fc2_per_cta_counts"] == (16,) * 48

props = torch.cuda.get_device_properties(torch.cuda.current_device())
assert (int(props.major), int(props.minor)) == (12, 1)
assert int(props.multi_processor_count) == 48
launch = compile_w4a16_hybrid_mapped_grid48(
    size_m=4,
    hidden_size=6144,
    intermediate_size=512,
    nv_num_experts=64,
    nf_num_experts=192,
    top_k=8,
    activation="silu",
    element_dtype="bf16",
    fast_math=True,
    sms=int(props.multi_processor_count),
    max_shared_mem=int(props.shared_memory_per_block_optin),
    force_tile_config=(64, 256, 64, 256),
)
assert launch.grid_x == 48
assert launch.blocks_per_sm == 1
assert launch.workspace_words == 194
assert launch.shared_memory_bytes == 45_184
assert launch.fc1_waves == 3
assert launch.fc2_waves == 16
assert launch.local_memory_bytes == 0
resources = dict(launch.codegen_resource_metadata)
assert resources["candidate"] == "w4a16_hybrid_mapped_grid48"
assert resources["architecture"] == "sm121"
assert resources["resident_capacity_ctas"] == 48
assert resources["whole_grid_resident"] is True
assert resources["one_cta_per_sm"] is True
print(
    "Grid48 CuTe compile: PASS",
    f"regs/thread={launch.registers_per_thread}",
    f"cubin={launch.generated_code_bytes} bytes",
)

# Exercise the SM100+ writer, not just its Python/C++ registration.  The
# stable-libtorch extension is loaded lazily and requires a CUDA driver.
torch.manual_seed(7)
num_blocks, block_size, num_tokens = 2, 16, 5
kv_c = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device="cuda")
k_pe = torch.randn(num_tokens, 64, dtype=torch.bfloat16, device="cuda")
slots = torch.tensor([0, 3, 7, 18, 29], dtype=torch.long, device="cuda")
scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
cache = torch.zeros(
    num_blocks, block_size, 432, dtype=torch.uint8, device="cuda"
)
_custom_ops.concat_and_cache_mla(
    kv_c, k_pe, cache, slots, "nvfp4_ds_mla", scale
)
torch.cuda.synchronize()
flat = cache.reshape(-1, 432)
selected = flat[slots]
assert (selected[:, :288] != 0).any(dim=1).all()
assert torch.equal(selected[:, 288:304], torch.zeros_like(selected[:, 288:304]))
assert torch.equal(
    selected[:, 304:432],
    k_pe.contiguous().view(torch.uint8).reshape(num_tokens, 128),
)
untouched = torch.ones(flat.shape[0], dtype=torch.bool, device="cuda")
untouched[slots] = False
assert torch.count_nonzero(flat[untouched]) == 0
print("nvfp4_ds_mla CUDA writer: PASS")
PY
fi

# Fast-DCP gate wiring (canonical checks; TP8 topologies are not deployable
# on Sparks but the gate logic must still resolve exactly as on x86).
tp8_dcp4="$(podman run --rm --entrypoint /usr/local/bin/serve-glm52-v18.sh \
  -e DRY_RUN=1 -e MODEL=/model -e TP=8 -e DCP=4 "${IMAGE}" 2>&1)"
grep -q '^VLLM_DCP_QUERY_SPLIT=1$' <<<"${tp8_dcp4}"
grep -q '^VLLM_B12X_MLA_CKV_GATHER=1$' <<<"${tp8_dcp4}"
grep -q -- '--load-format instanttensor' <<<"${tp8_dcp4}"

tp8_dcp8="$(podman run --rm --entrypoint /usr/local/bin/serve-glm52-v18.sh \
  -e DRY_RUN=1 -e MODEL=/model -e TP=8 -e DCP=8 "${IMAGE}" 2>&1)"
grep -q '^VLLM_DCP_QUERY_SPLIT=1$' <<<"${tp8_dcp8}"
grep -q '^VLLM_B12X_MLA_CKV_GATHER=1$' <<<"${tp8_dcp8}"

tp6_dcp3_mtp3="$(podman run --rm --entrypoint /usr/local/bin/serve-glm52-v18.sh \
  -e DRY_RUN=1 -e MODEL=/model -e TP=6 -e DCP=3 -e MTP=3 "${IMAGE}" 2>&1)"
grep -q -- '--tensor-parallel-size 6' <<<"${tp6_dcp3_mtp3}"
grep -q -- '--decode-context-parallel-size 3' <<<"${tp6_dcp3_mtp3}"
grep -q 'num_speculative_tokens.*3' <<<"${tp6_dcp3_mtp3}"

tp6_dcp6_mtp3="$(podman run --rm --entrypoint /usr/local/bin/serve-glm52-v18.sh \
  -e DRY_RUN=1 -e MODEL=/model -e TP=6 -e DCP=6 -e MTP=3 "${IMAGE}" 2>&1)"
grep -q -- '--tensor-parallel-size 6' <<<"${tp6_dcp6_mtp3}"
grep -q -- '--decode-context-parallel-size 6' <<<"${tp6_dcp6_mtp3}"

forced_off="$(podman run --rm --entrypoint /usr/local/bin/serve-glm52-v18.sh \
  -e DRY_RUN=1 -e MODEL=/model -e TP=8 -e DCP=4 \
  -e DCP_QUERY_SPLIT=0 -e DCP_CKV_GATHER=0 "${IMAGE}" 2>&1)"
grep -q '^VLLM_DCP_QUERY_SPLIT=0$' <<<"${forced_off}"
grep -q '^VLLM_B12X_MLA_CKV_GATHER=0$' <<<"${forced_off}"

# NF3 hybrid at the launcher defaults (TP4/DCP1) — exactly the sparky cluster
# deployment profile. The mapped-grid gate must be armed; exact SM121/48-SM
# admission selects Grid48. The TP8-only fast-DCP path and DCP workspace path
# must both gate off.
hybrid_default="$(podman run --rm --entrypoint /usr/local/bin/serve-glm52-hybrid-v18.sh \
  -e DRY_RUN=1 -e MODEL=/model "${IMAGE}" 2>&1)"
grep -q '^ONLINE_QUANT=nf3-mxfp8$' <<<"${hybrid_default}"
grep -q '^VLLM_NF3_GRID188_DECODE=1$' <<<"${hybrid_default}"
grep -q 'shared_experts' <<<"${hybrid_default}"
grep -q -- '--quantization nvfp4_nf3_hybrid' <<<"${hybrid_default}"
grep -q -- '--kv-cache-dtype nvfp4_ds_mla' <<<"${hybrid_default}"
grep -q -- '--load-format instanttensor' <<<"${hybrid_default}"
grep -q -- '--tensor-parallel-size 4' <<<"${hybrid_default}"
grep -q -- '--decode-context-parallel-size 1' <<<"${hybrid_default}"
grep -q '^VLLM_DCP_QUERY_SPLIT=0$' <<<"${hybrid_default}"
grep -q '^VLLM_B12X_MLA_CKV_GATHER=0$' <<<"${hybrid_default}"

# Unified dispatcher routes the hybrid family to the v18 preset.
podman run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e MODEL_FAMILY=glm52-hybrid -e DRY_RUN=1 -e MODEL=/model "${IMAGE}" 2>&1 \
  | grep -q '^ONLINE_QUANT=nf3-mxfp8$'

# DS4 helper checks (canonical set plus the Spark TP2 lucifer profile).
ds4_dry_run() {
  local mode="$1"
  local backend="$2"
  local tp="$3"
  podman run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
    -e DRY_RUN=1 \
    -e MODE="${mode}" \
    -e BACKEND="${backend}" \
    -e TP_SIZE="${tp}" \
    -e MODEL=/model \
    "${IMAGE}" 2>&1
}

ds4_mtp0="$(ds4_dry_run mtp0 b12x-a16 2)"
ds4_mtp2="$(ds4_dry_run mtp2 b12x-a8 2)"
ds4_dry_run mtp3 b12x-a8-dglin 2 >/dev/null
ds4_dspark_tp4="$(ds4_dry_run dspark lucifer-cutlass 4)"
ds4_dspark_tp2="$(ds4_dry_run dspark lucifer-cutlass 2)"
grep -q 'graph=512 load_format=instanttensor instanttensor_backend=BUFFERED' <<<"${ds4_mtp2}"
grep -q 'graph=384 load_format=instanttensor instanttensor_backend=BUFFERED' <<<"${ds4_dspark_tp4}"
grep -q -- '--gpu-memory-utilization 0.9465' <<<"${ds4_dspark_tp2}"
if grep -q -- '--revision' <<<"${ds4_mtp0}"; then
  echo 'ERROR: DS4 helper injected an HF revision for a local model path' >&2
  exit 1
fi

# v16-carried check: cooperative W4A8 resident-grid launch still present.
podman run --rm --entrypoint /bin/bash "${IMAGE}" -lc \
  'grep -q "cooperative=True" /opt/venv/lib/python3.12/site-packages/b12x/moe/fused/dynamic.py'

printf 'Image: %s\n' "${IMAGE}"
