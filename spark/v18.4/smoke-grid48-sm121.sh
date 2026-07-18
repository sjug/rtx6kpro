#!/usr/bin/env bash
set -euo pipefail

# Standalone Grid48 compile/admission smoke. This intentionally allocates no
# model weights and does not start vLLM. Against v18p3, pass B12X_KERNEL from
# the reviewed Grid48 branch; against the source-built v18p4 image, omit it.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:-localhost/voipmonitor/vllm:gilded-gnosis-v18p3-spark-sm121-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718}
VALIDATOR=/tmp/validate-grid48-sm121.py
INSTALLED_KERNEL=/opt/venv/lib/python3.12/site-packages/b12x/moe/fused/w4a16/kernel.py

mount_args=(-v "$SCRIPT_DIR/validate-grid48-sm121.py:$VALIDATOR:ro")
if [[ -n "${B12X_KERNEL:-}" ]]; then
  if [[ ! -f "$B12X_KERNEL" ]]; then
    echo "B12X_KERNEL is not a file: $B12X_KERNEL" >&2
    exit 2
  fi
  B12X_KERNEL=$(realpath "$B12X_KERNEL")
  mount_args+=(-v "$B12X_KERNEL:$INSTALLED_KERNEL:ro")
fi

podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt label=disable \
  -e B12X_PRINT_COMPILE_PROGRESS=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "${mount_args[@]}" \
  --entrypoint /opt/venv/bin/python \
  "$IMAGE" \
  "$VALIDATOR"
