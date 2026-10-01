FROM ghcr.io/astral-sh/uv@sha256:2f0a1c36ce6323e61bc81d39e692319669c03e385b592d670b6ba36dc884e3aa AS uv
FROM nvcr.io/nvidia/pytorch@sha256:237ecf9ac7373daf91b31bb4f86651ce1ce57b676366ed435aa1aba61dad81d5 AS flashinfer
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
COPY --from=uv /uv /usr/local/bin/uv
ENV TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_LOCAL_VERSION=lil.cu134.sm121.gdbd6238c6655 \
    FLASHINFER_SOURCE_COMMIT=dbd6238c6655b98195fdf77f04bba6facf5a38a4 \
    FLASHINFER_DISABLE_VERSION_CHECK=1 FLASHINFER_ENABLE_SM90=0 FLASHINFER_BUILD_NO_PIP=1 \
    BUILD_NVEP=0 BUILD_NCCL_EP=0 BUILD_NIXL_EP=0 MAX_JOBS=20 FLASHINFER_NVCC_THREADS=1
COPY build_flashinfer.py verify_flashinfer_component.py unpack_sources.py inputs.lock.json compiler-arm64.lock /opt/karmic-beta-refresh/
COPY inputs/ /opt/karmic-beta-refresh/inputs/
RUN /usr/bin/python3 /opt/karmic-beta-refresh/build_flashinfer.py

FROM 1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc AS candidate
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG BUILD_LOCK_INPUT
ARG SOURCE_LOCK_INPUT
ARG CACHE_INPUT
ARG VLLM_TREE_INPUT
ARG B12X_TREE_INPUT
ARG RECIPE_INPUT
ENV KARMIC_BETA_BUILD_LOCK_SHA256=${BUILD_LOCK_INPUT} SOURCE_LOCK_SHA256=${SOURCE_LOCK_INPUT} \
    PATH=/opt/venv/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a CUTE_DSL_ARCH=sm_121a \
    FLASHINFER_LOCAL_VERSION=lil.cu134.sm121.gdbd6238c6655 \
    FLASHINFER_SOURCE_COMMIT=dbd6238c6655b98195fdf77f04bba6facf5a38a4 \
    FLASHINFER_DISABLE_VERSION_CHECK=1 FLASHINFER_ENABLE_SM90=0 \
    BUILD_NVEP=0 BUILD_NCCL_EP=0 BUILD_NIXL_EP=0 MAX_JOBS=20 FLASHINFER_NVCC_THREADS=1 \
    PYTHONOPTIMIZE=0 LMCACHE_ENABLED=0 \
    XDG_CACHE_HOME=/cache/jit/${CACHE_INPUT} \
    VLLM_CACHE_ROOT=/cache/jit/${CACHE_INPUT}/vllm \
    TORCHINDUCTOR_CACHE_DIR=/cache/jit/${CACHE_INPUT}/inductor \
    TRITON_CACHE_DIR=/cache/jit/${CACHE_INPUT}/triton \
    B12X_COMPILE_CACHE_DIR=/cache/jit/${CACHE_INPUT}/b12x \
    B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/${CACHE_INPUT}/b12x/cute \
    SPARKINFER_COMPILE_CACHE_DIR=/cache/jit/${CACHE_INPUT}/b12x/compile \
    CUTE_DSL_CACHE_DIR=/cache/jit/${CACHE_INPUT}/cute-dsl \
    TORCH_EXTENSIONS_DIR=/cache/jit/${CACHE_INPUT}/torch-extensions \
    TVM_FFI_CACHE_DIR=/cache/jit/${CACHE_INPUT}/tvm-ffi \
    TVM_CACHE_DIR=/cache/jit/${CACHE_INPUT}/tvm \
    DG_JIT_CACHE_DIR=/cache/jit/${CACHE_INPUT}/deep-gemm \
    NUMBA_CACHE_DIR=/cache/jit/${CACHE_INPUT}/numba \
    CUDA_CACHE_PATH=/cache/jit/${CACHE_INPUT}/cuda
COPY install.py build.lock.json runtime.lock.json /opt/karmic-beta-refresh/
COPY upgrade_dependencies.py verify_compiler.py inputs.lock.json compiler-arm64.lock /opt/karmic-beta-refresh/
RUN --mount=type=bind,source=inputs,target=/opt/karmic-beta-refresh/inputs,ro \
    --mount=type=bind,source=refresh.tar,target=/opt/karmic-beta-refresh/refresh.tar,ro \
    --mount=type=bind,from=flashinfer,source=/artifacts/flashinfer,target=/opt/karmic-beta-refresh/flashinfer-wheels,ro \
    /opt/venv/bin/python /opt/karmic-beta-refresh/install.py \
    && /opt/venv/bin/python /opt/karmic-beta-refresh/upgrade_dependencies.py
LABEL org.opencontainers.image.title="Karmic beta 20261001 Spark SM121 candidate" \
    local-inference.base-image="1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc" \
    local-inference.publication="karmic-kraken-beta-20261001-020df706a373de8d" \
    local-inference.recipe.commit="6c0e9843bb962f483409b0296b225cca03fe8567" \
    local-inference.source-lock.sha256="${SOURCE_LOCK_INPUT}" \
    local-inference.serving.recipe.sha256="${RECIPE_INPUT}" \
    local-inference.cache.fingerprint="${CACHE_INPUT}" \
    local-inference.karmic-beta-build-lock.sha256="${BUILD_LOCK_INPUT}" \
    vllm.source-commit="4a379ed42881ee022aaf5d9ada554f9096e5acf3" \
    vllm.source-tree="${VLLM_TREE_INPUT}" \
    b12x.source-commit="b557d87850cc836268fd327ccd46eaa6a033cdbf" \
    b12x.source-tree="${B12X_TREE_INPUT}" \
    local-inference.cutlass-dsl.version="4.7.1" \
    local-inference.flashinfer.commit="dbd6238c6655b98195fdf77f04bba6facf5a38a4" \
    local-inference.flashinfer.arch="12.1a" \
    local-inference.quack.version="0.6.5" \
    local-inference.lmcache.commit="413ac987336a91b5cb3899ce20720e9ad9f1fa47" \
    local-inference.deviations="rust-dsml-parser-at-base;lmcache-at-base;vllm-native-wheel-at-base;compiler-requirement-overlay;ngc-system-cutlass-4.6.2-unused" \
    local-inference.status="build-candidate-not-model-qualified"
