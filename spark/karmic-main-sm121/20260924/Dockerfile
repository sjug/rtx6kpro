ARG NCCL_COMPONENT_IMAGE
FROM ${NCCL_COMPONENT_IMAGE} AS nccl2307
FROM 1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG BUILD_LOCK_INPUT
ARG SOURCE_LOCK_INPUT
ARG CACHE_INPUT
ARG VLLM_TREE_INPUT
ARG B12X_TREE_INPUT
ARG RECIPE_INPUT
ARG NCCL_LIBRARY_INPUT
ENV DS41_BUILD_LOCK_SHA256=${BUILD_LOCK_INPUT} SOURCE_LOCK_SHA256=${SOURCE_LOCK_INPUT} \
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
COPY install.py build.lock.json runtime.lock.json refresh.tar /opt/ds41-main-refresh/
COPY --from=nccl2307 /artifacts/nccl/ /opt/nccl-2.30.7/
RUN cd /opt/nccl-2.30.7 && sha256sum -c SHA256SUMS \
    && test "$(sha256sum lib/libnccl.so.2.30.7 | cut -d' ' -f1)" = "$NCCL_LIBRARY_INPUT" \
    && ldd lib/libnccl.so.2.30.7 > runtime-ldd.txt \
    && ! grep -q 'not found' runtime-ldd.txt
RUN python /opt/ds41-main-refresh/install.py
LABEL org.opencontainers.image.title="Karmic main 20260924 Spark DS4.1 candidate" \
    local-inference.base-image="1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc" \
    local-inference.source-lock.sha256="${SOURCE_LOCK_INPUT}" \
    local-inference.serving.recipe.sha256="${RECIPE_INPUT}" \
    local-inference.cache.fingerprint="${CACHE_INPUT}" \
    vllm.source-commit="1794dcf18454900263e0c66711af8ea4a1283ac1" \
    vllm.source-tree="${VLLM_TREE_INPUT}" \
    b12x.source-commit="a7d7d29b2ef8869086e0ceaa787321f17544e3c9" \
    b12x.source-tree="${B12X_TREE_INPUT}" \
    local-inference.lmcache.commit="361a85e88698174e9fd3b52fe372ceb40bd054f4" \
    local-inference.ds41-build-lock.sha256="${BUILD_LOCK_INPUT}" \
    local-inference.ds41.nccl.commit="73cf112295c33aee2b895f329f592f2a9b4b0f97" \
    local-inference.ds41.nccl.tree="3e7de6f92f0190d1afe9f05642e634cbf43ae4c9" \
    local-inference.ds41.nccl.library.sha256="${NCCL_LIBRARY_INPUT}" \
    local-inference.status="build-candidate-not-model-qualified"
