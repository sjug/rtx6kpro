FROM 1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG BUILD_LOCK_INPUT
ARG SOURCE_LOCK_INPUT
ARG CACHE_INPUT
ARG VLLM_TREE_INPUT
ARG B12X_TREE_INPUT
ARG RECIPE_INPUT
ENV KARMIC_BETA_BUILD_LOCK_SHA256=${BUILD_LOCK_INPUT} SOURCE_LOCK_SHA256=${SOURCE_LOCK_INPUT} \
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
COPY install.py build.lock.json runtime.lock.json refresh.tar /opt/karmic-beta-refresh/
RUN python /opt/karmic-beta-refresh/install.py
LABEL org.opencontainers.image.title="Karmic beta 20260929 Spark SM121 candidate" \
    local-inference.base-image="1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc" \
    local-inference.publication="karmic-kraken-beta-20260929-7476365edc3d8ee5" \
    local-inference.recipe.commit="f28aea04babb0e6b0a5830354f1ef6fd2a1b3d3f" \
    local-inference.source-lock.sha256="${SOURCE_LOCK_INPUT}" \
    local-inference.serving.recipe.sha256="${RECIPE_INPUT}" \
    local-inference.cache.fingerprint="${CACHE_INPUT}" \
    local-inference.karmic-beta-build-lock.sha256="${BUILD_LOCK_INPUT}" \
    vllm.source-commit="99cbe782f3b67a7758a85bcc0e7938a1ec27012a" \
    vllm.source-tree="${VLLM_TREE_INPUT}" \
    b12x.source-commit="1b6cd278626aa6b44c519b5d9f4ad576c9af783b" \
    b12x.source-tree="${B12X_TREE_INPUT}" \
    local-inference.lmcache.commit="413ac987336a91b5cb3899ce20720e9ad9f1fa47" \
    local-inference.deviations="rust-dsml-parser-at-base;lmcache-at-base" \
    local-inference.status="build-candidate-not-model-qualified"
