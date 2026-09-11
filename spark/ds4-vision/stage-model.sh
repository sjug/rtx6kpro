#!/usr/bin/env bash
set -euo pipefail
[[ $(hostname -s) == rusty ]] || exit 78
root=/home/jugs/git/ds4-vision
image=276f00868134
mkdir -p "$root/receipts" /home/jugs/.cache/huggingface
# CPU-only download using the already installed runtime. No GPU access, source
# changes, cache deletion, or inference-container mutation. Bound download RAM
# while the old model continues serving.
exec podman run --rm --pull never --name ds4-vision-model-download \
  --network host --memory 2g --memory-swap 2g --cpus 4 \
  -v /home/jugs/.cache/huggingface:/root/.cache/huggingface:rw \
  -v "$root/stage-model.py:/stage-model.py:ro" \
  -v "$root/receipts:/receipts:rw" \
  -e PYTHONUNBUFFERED=1 -e HF_HUB_DISABLE_XET=1 \
  --entrypoint python3 "$image" /stage-model.py
