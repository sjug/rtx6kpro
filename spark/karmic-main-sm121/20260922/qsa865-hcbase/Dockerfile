FROM 88867036403cb3c9bc3026eb8c907ad64bc1af9c28d316d22b670dde4f561152
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG BACKPORT_LOCK_INPUT
ARG RUNTIME_LOCK_INPUT
ARG CACHE_INPUT
ARG TREE_INPUT
ARG B12X_TREE_INPUT
ARG RECIPE_INPUT
ENV QSA865_LOCK_SHA256=${BACKPORT_LOCK_INPUT} SOURCE_LOCK_SHA256=${RUNTIME_LOCK_INPUT} \
    PYTHONOPTIMIZE=0 XDG_CACHE_HOME=/cache/jit/${CACHE_INPUT} \
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
COPY install.py upstream.patch backport.lock.json runtime.lock.json refresh.tar /opt/qsa865/
RUN python /opt/qsa865/install.py
LABEL org.opencontainers.image.title="Karmic main HC-off baseline plus QSA865 only" \
    local-inference.base-image="88867036403cb3c9bc3026eb8c907ad64bc1af9c28d316d22b670dde4f561152" \
    local-inference.source-lock.sha256="${RUNTIME_LOCK_INPUT}" \
    local-inference.serving.recipe.sha256="${RECIPE_INPUT}" \
    local-inference.qsa865.lock.sha256="${BACKPORT_LOCK_INPUT}" \
    local-inference.qsa865.upstream="26b42cf9eb715b118e55f91f53fa0ab412a79c09" \
    local-inference.cache.fingerprint="${CACHE_INPUT}" \
    vllm.source-tree="${TREE_INPUT}" \
    vllm.source-commit="6afb99982576a7a2eb53d667189e859629e22739" \
    b12x.source-commit="4f3028b19c1d8290dc72b6f483aba40de23eae5a" \
    b12x.source-tree="${B12X_TREE_INPUT}" \
    local-inference.status="build-candidate-not-model-qualified"
