#!/usr/bin/env bash
set -euo pipefail
source_launcher=/opt/jovian-judgement/vllm/serve-ds4-flash.sh
expected=3542a3f6663503dc8697c85e27e5a0a67b80c68c9ffc76143a39403f2bf0d18c
[[ $(sha256sum "$source_launcher" | cut -d' ' -f1) == "$expected" ]] || exit 78
if [[ ${DRY_RUN:-0} != 1 ]]; then
  /opt/venv/bin/python /opt/ds4-vision/runtime-preflight.py
fi
export PYTHON_BIN=/opt/venv/bin/python
export VLLM_PLUGINS=
export VLLM_ENABLE_ROCE_ALLREDUCE=0
export LMCACHE_MODE=off LMCACHE_ENABLED=0
export DS4_MODEL_VARIANT=vision MODE=dspark DSPARK_DEPTH_MODE=fixed
export DRAFT_SAMPLE_METHOD=probabilistic REJECTION_SAMPLE_METHOD=standard
export ALLREDUCE_MODE=nccl
export CUTE_DSL_ARCH=sm_121a
export OMP_NUM_THREADS=2
export INSTANTTENSOR_BACKEND=BUFFERED
export INSTANTTENSOR_BUFFER_SIZE=1342177280
export INSTANTTENSOR_CONCURRENCY=1 INSTANTTENSOR_IO_DEPTH=3
export ENABLE_FLASHINFER_AUTOTUNE=0
# Keep image-owned NCCL and fingerprinted JIT paths. The pinned source launcher
# owns DS4 backend setup; positional max overrides its older high default.
exec bash "$source_launcher" "$@" \
  --default-chat-template-kwargs.reasoning_effort=max \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --mm-processor-cache-gb 0
