# syntax=docker/dockerfile:1.7
# Derived from our exact qualified R29 SM121 image, not the RTX R32 runtime.
ARG BASE_IMAGE=0b15723cb87646bb4628c5cd67aa2a7c879ac5a27105109ea5db06c30113dc0b
FROM ${BASE_IMAGE} AS refreshed
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY source.lock.json refresh_sources.py contracts.py verify_native.py native_reuse.py /build/r32/
COPY patches /build/r32/patches
RUN cp -a /opt/lmcache/source-r29 /opt/lmcache/source-r32 \
    && python /build/r32/refresh_sources.py
RUN command -v uv

FROM refreshed AS lmcache-builder
# Export tracked source only. The copied R29 source retains old build/ outputs
# and generated version files that must not leak into a new Python wheel.
RUN mkdir /build/lmcache-python \
    && git -C /opt/lmcache/source-r32 archive "$(git -C /opt/lmcache/source-r32 write-tree)" \
       | tar -xf - -C /build/lmcache-python
WORKDIR /build/lmcache-python
# No native build, dependency resolution or repository download. Repack the
# Python wheel with exact compiled payloads from the declared ARM base.
RUN NO_NATIVE_EXT=1 SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.5.5.dev0+glm53checkpoints.35ad809f \
    uv build --offline --python /opt/venv/bin/python --wheel --no-build-isolation --out-dir /artifacts . \
    && python /build/r32/native_reuse.py pack \
    && cd /artifacts && sha256sum *.whl > lmcache-wheel.sha256

FROM refreshed AS runtime
ARG CACHE_FINGERPRINT
ARG SOURCE_LOCK_SHA256
ARG RECIPE_SHA256
ARG RECIPE_COMMIT
ARG VLLM_VERSION
ARG FLASHKDA_SHA256
ARG STABLE_NATIVE_SHA256
ARG R32_GLM_LAUNCHER_SHA256
ARG RECIPE_DIRTY
ENV LMCACHE_ENABLED=0 VLLM_PLUGINS="" \
    GLM_LAUNCHER_SHA256=${R32_GLM_LAUNCHER_SHA256} \
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
COPY --from=lmcache-builder /artifacts/ /opt/local-inference/r32-native/
COPY source.lock.json /opt/glm53-flash/source.lock
COPY install_runtime.py /build/r32/install_runtime.py
COPY launchers/ /usr/local/bin/
COPY tests/verify_jj_r32_runtime.py /opt/local-inference/verify_r32_runtime.py
COPY tests/verify_glm53_nvfp4_draft_head_sm121.py /opt/local-inference/verify_glm53_nvfp4_draft_head_sm121.py
COPY tests/run_flashkda_gate.py /opt/local-inference/run_flashkda_gate.py
COPY tests/run_r32_regressions.py /opt/local-inference/run_r32_regressions.py
COPY tests/flashkda_counts.py contracts.py verify_native.py native_reuse.py /opt/local-inference/
RUN cp -a /opt/local-inference/r29-native/flashkda-tests /opt/local-inference/r32-native/ \
    && python /build/r32/install_runtime.py \
    && python /build/r32/verify_native.py proxy \
    && CC=/bin/false python /build/r32/verify_native.py proxy
LABEL local-inference.release.name="jj-r32-spark-sm121" \
      local-inference.status="build-candidate" \
      local-inference.scope="glm53-qwen38-r32-spark-sm121-qualification" \
      local-inference.runtime.native-artifact-base-id="0b15723cb87646bb4628c5cd67aa2a7c879ac5a27105109ea5db06c30113dc0b" \
      local-inference.runtime.native-artifact-reuse="all native objects byte-identical to corrected R29" \
      local-inference.vllm.stable-native.sha256="${STABLE_NATIVE_SHA256}" \
      local-inference.runtime.source-lock.sha256="${SOURCE_LOCK_SHA256}" \
      local-inference.recipe.sha256="${RECIPE_SHA256}" \
      local-inference.recipe.dirty="${RECIPE_DIRTY}" \
      local-inference.launcher.glm.sha256="${R32_GLM_LAUNCHER_SHA256}" \
      org.opencontainers.image.revision="${RECIPE_COMMIT}" \
      local-inference.cache.fingerprint="${CACHE_FINGERPRINT}" \
      local-inference.flashkda.commit="3b225bf26bb8e218928a1fe14751cb48cf31d11b" \
      local-inference.flashkda.sha256="${FLASHKDA_SHA256}" \
      local-inference.flashkda.binary-manifest="/opt/local-inference/r28-native/flashkda.sha256" \
      local-inference.flashkda.patch.sha256="a9537532cf231babc942fa592666893268ca0617b24daeb8e78d3249e8c1d58a" \
      local-inference.flashkda.arch="12.1a" \
      local-inference.vllm.package-version="${VLLM_VERSION}" \
      local-inference.vllm.spark-overlay="cuda-arch-12.1-plus-glm53-nvfp4-draft-head-sm121" \
      local-inference.vllm.pr674.commit="superseded-in-r32" \
      local-inference.b12x.spark-source-patches="none; the r32 B12X integration tree is applied unmodified" \
      local-inference.lmcache.default="disabled" \
      org.opencontainers.image.title="Jovian Judgement R32 Spark SM121 runtime" \
      org.opencontainers.image.description="R32 candidate derived from qualified R29, matched Qwen and GLM profiles" \
      org.opencontainers.image.version="20260910"
WORKDIR /
ENTRYPOINT ["/usr/local/bin/serve-glm53-flash-jj-r32-spark.sh"]
