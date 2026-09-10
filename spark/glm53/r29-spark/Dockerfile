# syntax=docker/dockerfile:1.7
# Derived from our exact qualified R28 SM121 image, not the RTX R29 runtime.
ARG BASE_IMAGE=cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1
FROM ${BASE_IMAGE} AS refreshed
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY source.lock.json refresh_sources.py contracts.py verify_native.py /build/r29/
COPY patches /build/r29/patches
RUN python /build/r29/refresh_sources.py
RUN command -v uv

FROM refreshed AS native-builder
# R29 changes the output-buffer schema of a stable DS4 operator. Rebuild the
# containing extension; FlashKDA and the other native objects remain R28 bytes.
ENV TORCH_CUDA_ARCH_LIST=12.1a CMAKE_CUDA_ARCHITECTURES=121a
WORKDIR /opt/jovian-judgement/vllm
RUN cmake -S . -B /build/stable-native -G Ninja \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a \
       -DVLLM_TARGET_DEVICE=cuda -DVLLM_PYTHON_EXECUTABLE=/opt/venv/bin/python \
       -DNVCC_THREADS=2 \
    && cmake --build /build/stable-native --target _C_stable_libtorch --parallel 2 \
    && install -Dm755 /build/stable-native/_C_stable_libtorch.abi3.so /artifacts/_C_stable_libtorch.abi3.so
RUN python /build/r29/verify_native.py stable /artifacts/_C_stable_libtorch.abi3.so \
    && cd /artifacts && sha256sum _C_stable_libtorch.abi3.so > stable.sha256

FROM refreshed AS lmcache-builder
ENV TORCH_CUDA_ARCH_LIST=12.1a CMAKE_CUDA_ARCHITECTURES=121a \
    BUILD_WITH_CUDA=1 ENABLE_CXX11_ABI=1 LMCACHE_CUDA_MAJOR=13 MAX_JOBS=2 NVCC_THREADS=2
COPY inputs/lmcache.bundle /build/r29/inputs/lmcache.bundle
RUN python /build/r29/verify_native.py bundle /build/r29/inputs/lmcache.bundle \
    && git init /build/lmcache \
    && git -C /build/lmcache fetch /build/r29/inputs/lmcache.bundle refs/spark/jj-r29/lmcache-source \
    && git -C /build/lmcache checkout --detach FETCH_HEAD \
    && test "$(git -C /build/lmcache rev-parse HEAD^{tree})" = cd8a219da05eec6c7d6c24712f9ee1a9186c3567 \
    && test "$(git -C /build/lmcache rev-parse HEAD:lmcache)" = 7b1043f59605ae5035f557dbf19b7382fc7eaf3f
WORKDIR /build/lmcache
# uv uses the base interpreter/build dependencies; dependency resolution may
# not replace the frozen Torch/CUDA runtime. CUDAExtension consumes the Torch
# architecture list; no claim that CMake reads a same-named environment variable.
RUN command -v uv && command -v cmake \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.5.5.dev0+glm53checkpoints.dcd6ec92 \
       uv build --python /opt/venv/bin/python --wheel --no-build-isolation --out-dir /artifacts . \
    && make -C csrc/cumem_ipc_interposer CUDA_HOME=/usr/local/cuda \
    && install -Dm755 csrc/cumem_ipc_interposer/liblmcache_cumem_shareable.so /artifacts/liblmcache_cumem_shareable.so \
    && uv pip install --python /opt/venv/bin/python --no-deps --reinstall /artifacts/lmcache-*.whl \
    && cd / \
    && python /build/r29/verify_native.py lmcache /artifacts/liblmcache_cumem_shareable.so \
    && cd /artifacts && sha256sum *.whl *.so > lmcache-native.sha256

FROM refreshed AS runtime
ARG CACHE_FINGERPRINT
ARG SOURCE_LOCK_SHA256
ARG RECIPE_SHA256
ARG RECIPE_COMMIT
ARG VLLM_VERSION
ARG FLASHKDA_SHA256
ARG STABLE_NATIVE_SHA256
ARG R29_GLM_LAUNCHER_SHA256
ARG RECIPE_DIRTY
ENV LMCACHE_ENABLED=0 VLLM_PLUGINS="" \
    GLM_LAUNCHER_SHA256=${R29_GLM_LAUNCHER_SHA256} \
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
COPY --from=native-builder /artifacts/ /opt/local-inference/r29-native/
COPY --from=lmcache-builder /artifacts/ /opt/local-inference/r29-native/
COPY --from=lmcache-builder /build/lmcache/ /opt/lmcache/source-r29/
COPY source.lock.json /opt/glm53-flash/source.lock
COPY install_runtime.py /build/r29/install_runtime.py
COPY launchers/ /usr/local/bin/
COPY tests/verify_jj_r29_runtime.py /opt/local-inference/verify_r29_runtime.py
COPY tests/verify_glm53_nvfp4_draft_head_sm121.py /opt/local-inference/verify_glm53_nvfp4_draft_head_sm121.py
COPY tests/run_flashkda_gate.py /opt/local-inference/run_flashkda_gate.py
COPY tests/flashkda_counts.py contracts.py verify_native.py /opt/local-inference/
RUN cp -a /opt/local-inference/r28-native/flashkda-tests /opt/local-inference/r29-native/ \
    && python /build/r29/install_runtime.py \
    && python /build/r29/verify_native.py proxy \
    && CC=/bin/false python /build/r29/verify_native.py proxy
LABEL local-inference.release.name="jj-r29-spark-sm121" \
      local-inference.status="build-candidate" \
      local-inference.scope="glm53-qwen38-r29-spark-sm121-qualification" \
      local-inference.runtime.native-artifact-base-id="cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1" \
      local-inference.runtime.native-artifact-reuse="R28 except rebuilt _C_stable_libtorch and LMCache wheel" \
      local-inference.vllm.stable-native.sha256="${STABLE_NATIVE_SHA256}" \
      local-inference.runtime.source-lock.sha256="${SOURCE_LOCK_SHA256}" \
      local-inference.recipe.sha256="${RECIPE_SHA256}" \
      local-inference.recipe.dirty="${RECIPE_DIRTY}" \
      local-inference.launcher.glm.sha256="${R29_GLM_LAUNCHER_SHA256}" \
      org.opencontainers.image.revision="${RECIPE_COMMIT}" \
      local-inference.cache.fingerprint="${CACHE_FINGERPRINT}" \
      local-inference.flashkda.commit="3b225bf26bb8e218928a1fe14751cb48cf31d11b" \
      local-inference.flashkda.sha256="${FLASHKDA_SHA256}" \
      local-inference.flashkda.binary-manifest="/opt/local-inference/r28-native/flashkda.sha256" \
      local-inference.flashkda.patch.sha256="a9537532cf231babc942fa592666893268ca0617b24daeb8e78d3249e8c1d58a" \
      local-inference.flashkda.arch="12.1a" \
      local-inference.vllm.package-version="${VLLM_VERSION}" \
      local-inference.vllm.spark-overlay="cuda-arch-12.1-plus-glm53-nvfp4-draft-head-sm121" \
      local-inference.vllm.pr674.commit="superseded-in-r29" \
      local-inference.b12x.spark-source-patches="none; the r29 B12X integration tree is applied unmodified" \
      local-inference.lmcache.default="disabled" \
      org.opencontainers.image.title="Jovian Judgement R29 Spark SM121 runtime" \
      org.opencontainers.image.description="R29 candidate derived from qualified R28, matched Qwen and GLM profiles" \
      org.opencontainers.image.version="20260909"
WORKDIR /
ENTRYPOINT ["/usr/local/bin/serve-glm53-flash-jj-r29-spark.sh"]
