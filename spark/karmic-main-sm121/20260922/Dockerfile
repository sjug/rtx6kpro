FROM f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG LOCK_INPUT_SHA256
ARG RECIPE_INPUT_SHA256
ARG CACHE_INPUT
ARG QWEN_INPUT_SHA256
ARG GLM_INPUT_SHA256
ARG VLLM_TREE_INPUT
ARG B12X_TREE_INPUT
ENV SOURCE_LOCK_SHA256=${LOCK_INPUT_SHA256} PYTHONOPTIMIZE=0 \
    PYTHONPATH=/opt/jovian-judgement/vllm:/opt/jovian-judgement/b12x \
    B12X_ROCE_CACHE_DIR=/opt/b12x-roce-cache \
    QWEN_LAUNCHER_SHA256=${QWEN_INPUT_SHA256} GLM_LAUNCHER_SHA256=${GLM_INPUT_SHA256} \
    LMCACHE_ENABLED=0 VLLM_PLUGINS="" MAX_JOBS=20 CMAKE_BUILD_PARALLEL_LEVEL=20 \
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
COPY contracts.py install.py source.lock.json /opt/karmic-main-build/
COPY payload/ /opt/karmic-main-build/payload/
RUN python /opt/karmic-main-build/install.py
# No native vLLM or dependency rebuild. Package metadata remains that of the
# pinned base; the manifest and source labels identify the refreshed runtime.
RUN uv pip install --python /opt/venv/bin/python --no-index --no-deps --no-build-isolation \
      /opt/jovian-judgement/b12x
RUN python -c 'from b12x.comm.roce import _proxy; import sys; version=_proxy.load().roce_abi_version(); print("RoCE ABI",version); sys.exit(0 if version == 4 else 1)' \
    && CC=/bin/false python -c 'from b12x.comm.roce import _proxy; _proxy.load(); print("PREBUILT-PROXY-LOAD-PASS")'
COPY inherited/launchers/ /usr/local/bin/
RUN echo "$QWEN_LAUNCHER_SHA256  /usr/local/bin/serve-qwen38-flash-next-karmic-spark.sh" | sha256sum -c - \
    && echo "$GLM_LAUNCHER_SHA256  /usr/local/bin/serve-glm53-flash-karmic-spark.sh" | sha256sum -c - \
    && chmod 755 /usr/local/bin/serve-*-karmic-spark.sh
LABEL org.opencontainers.image.title="Karmic main DGX Spark SM121" \
      org.opencontainers.image.description="Source-locked main refresh; reused verified Karmic natives; model qualification pending" \
      local-inference.source-lock.sha256="${LOCK_INPUT_SHA256}" \
      local-inference.serving.recipe.sha256="${RECIPE_INPUT_SHA256}" \
      local-inference.cache.fingerprint="${CACHE_INPUT}" \
      local-inference.base-image="f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233" \
      local-inference.qwen.launcher.sha256="${QWEN_INPUT_SHA256}" \
      local-inference.glm.launcher.sha256="${GLM_INPUT_SHA256}" \
      vllm.source-commit="6afb99982576a7a2eb53d667189e859629e22739" \
      vllm.source-tree="${VLLM_TREE_INPUT}" \
      b12x.source-commit="4f3028b19c1d8290dc72b6f483aba40de23eae5a" \
      b12x.source-tree="${B12X_TREE_INPUT}" \
      local-inference.roce.proxy-abi="4" \
      local-inference.status="build-candidate-not-model-qualified"
