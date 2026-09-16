ARG BASE_IMAGE=74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c
ARG FLASHINFER_IMAGE
FROM ${FLASHINFER_IMAGE} AS flashinfer-wheels
FROM ${BASE_IMAGE} AS refreshed
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV PYTHONUNBUFFERED=1 PYTHONOPTIMIZE=0
COPY *.py source.lock.json upstream-r37.source.lock /build/r37/
COPY patches/ /build/r37/patches/
COPY inputs/ /build/r37/inputs/
RUN cp -a /opt/lmcache/source-r32 /opt/lmcache/source-r37 \
    && uv run --no-project --python /opt/venv/bin/python python /build/r37/refresh_sources.py \
    && uv run --no-project --python /opt/venv/bin/python python /build/r37/inputs/upstream/install_dependency_python_patches.py
# Native Engram/PLe I/O needs liburing headers even while offload is disabled.
RUN apt-get update \
    && apt-get install -y --no-install-recommends liburing2=2.5-1build1 liburing-dev=2.5-1build1 \
    && test "$(pkg-config --modversion liburing)" = 2.5

FROM refreshed AS lmcache-builder
# Match the pinned upstream R37 cpu-rebuild-cuda-reuse stage, not the
# FlashInfer stage: this rebuilds CPU modules and preserves the CUDA payload.
ENV NO_GPU_EXT=1 MAX_JOBS=8 NVCC_THREADS=2 TORCH_CUDA_ARCH_LIST=12.1a
RUN mkdir -p /build/lmcache /artifacts/lmcache \
    && git -C /opt/lmcache/source-r37 archive "$(git -C /opt/lmcache/source-r37 write-tree)" | tar -xf - -C /build/lmcache
RUN cp -a /opt/lmcache/source-r37/.git /build/lmcache/.git \
    && git -C /build/lmcache diff --quiet
WORKDIR /build/lmcache
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.5.5.dev0+glm53checkpoints.29bc5a2e \
    uv build --offline --python /opt/venv/bin/python --wheel --no-build-isolation --out-dir /artifacts/lmcache . \
    && uv run --no-project --python /opt/venv/bin/python python /build/r37/repack_lmcache.py \
       --source /build/lmcache --reference-source /opt/lmcache/source-r32 \
       --reference-package /opt/venv/lib/python3.12/site-packages/lmcache \
       --wheel-dir /artifacts/lmcache --mode cpu-rebuild-cuda-reuse \
    && cd /artifacts/lmcache && sha256sum *.whl > SHA256SUMS

FROM refreshed AS runtime
ARG CACHE_FINGERPRINT
ARG SOURCE_LOCK_SHA256
ARG RECIPE_SHA256
ARG RECIPE_COMMIT
ARG RECIPE_DIRTY
ARG BUILD_GLM_LAUNCHER_SHA256
ARG BUILD_QWEN_LAUNCHER_SHA256
ENV LMCACHE_ENABLED=0 VLLM_PLUGINS="" \
    GLM_LAUNCHER_SHA256=${BUILD_GLM_LAUNCHER_SHA256} QWEN_LAUNCHER_SHA256=${BUILD_QWEN_LAUNCHER_SHA256} \
    LOCAL_INFERENCE_CACHE_FINGERPRINT=${CACHE_FINGERPRINT} \
    XDG_CACHE_HOME=/cache/jit/${CACHE_FINGERPRINT} \
    VLLM_CACHE_ROOT=/cache/jit/${CACHE_FINGERPRINT}/vllm \
    VLLM_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/vllm \
    TRITON_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/torchinductor \
    CUTE_DSL_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/cute-dsl \
    B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/b12x/cute \
    B12X_COMPILE_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/b12x/compile \
    SPARKINFER_COMPILE_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/b12x/compile \
    TORCH_EXTENSIONS_DIR=/cache/jit/${CACHE_FINGERPRINT}/torch-extensions \
    VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/flashinfer-autotune \
    FLASHINFER_WORKSPACE_BASE=/cache/jit/${CACHE_FINGERPRINT}/flashinfer \
    TVM_FFI_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/tvm-ffi \
    TVM_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/tvm \
    TILELANG_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/tilelang \
    DG_JIT_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/deep-gemm \
    NUMBA_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/numba \
    CUDA_CACHE_PATH=/cache/jit/${CACHE_FINGERPRINT}/cuda \
    CUPY_CACHE_DIR=/cache/jit/${CACHE_FINGERPRINT}/cupy \
    B12X_ROCE_CACHE_DIR=/opt/jovian-judgement/b12x-roce
COPY --from=lmcache-builder /artifacts/lmcache/ /opt/local-inference/r37-native/lmcache/
COPY --from=flashinfer-wheels /artifacts/flashinfer/ /opt/local-inference/r37-native/flashinfer/
COPY source.lock.json /opt/glm53-flash/source.lock
COPY launchers/ /usr/local/bin/
COPY tests/ /opt/local-inference/r37-tests/
RUN printf '%s  %s\n' \
       "$BUILD_GLM_LAUNCHER_SHA256" /usr/local/bin/serve-glm53-flash-jj-r37-spark.sh \
       "$BUILD_QWEN_LAUNCHER_SHA256" /usr/local/bin/serve-qwen38-flash-next-jj-r37-spark.sh | sha256sum -c - \
    && cp -a /opt/local-inference/r32-native/flashkda-tests /opt/local-inference/r37-native/ \
    && chmod +x /usr/local/bin/serve-*-jj-r37-spark.sh \
    && uv run --no-project --python /opt/venv/bin/python python /build/r37/install_runtime.py \
    && uv run --no-project --python /opt/venv/bin/python python /build/r37/verify_native.py proxy \
    && CC=/bin/false uv run --no-project --python /opt/venv/bin/python python /build/r37/verify_native.py proxy \
    && XDG_CACHE_HOME=/opt/local-inference/r37-host-cache \
       uv run --no-project --python /opt/venv/bin/python python /build/r37/verify_native.py loader
LABEL local-inference.release.name="jj-r37-spark-sm121" \
      local-inference.status="build-candidate-not-serving-qualified" \
      local-inference.runtime.source-lock.sha256="${SOURCE_LOCK_SHA256}" \
      local-inference.runtime.native-artifact-base-id="74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c" \
      local-inference.runtime.native-artifact-reuse="R32 vLLM, FlashKDA, LMCache CUDA/interposer, RoCEnante; rebuild LMCache CPU and FlashInfer" \
      local-inference.flashinfer.commit="803c4664f4771ddc418f20a57f752469a237a825" \
      local-inference.recipe.sha256="${RECIPE_SHA256}" \
      local-inference.recipe.dirty="${RECIPE_DIRTY}" \
      local-inference.launcher.glm.sha256="${GLM_LAUNCHER_SHA256}" \
      local-inference.launcher.qwen.sha256="${QWEN_LAUNCHER_SHA256}" \
      local-inference.cache.fingerprint="${CACHE_FINGERPRINT}" \
      local-inference.lmcache.default="disabled" \
      local-inference.lmcache.rebuilt="CPU extensions; CUDA and interposer reused" \
      local-inference.b12x.spark-source-patches="none; R37 B12X integration tree applied unmodified" \
      local-inference.vllm.spark-overlay="unchanged R27 SM121 architecture and draft-head overlay" \
      local-inference.scope="R37 SM121 image build, no model serving qualification" \
      org.opencontainers.image.title="Jovian Judgement R37 Spark SM121" \
      org.opencontainers.image.description="R37 Spark candidate derived from R32, not yet serving qualified" \
      org.opencontainers.image.revision="${RECIPE_COMMIT}" \
      org.opencontainers.image.version="r37"
WORKDIR /
ENTRYPOINT ["/usr/local/bin/serve-glm53-flash-jj-r37-spark.sh"]
