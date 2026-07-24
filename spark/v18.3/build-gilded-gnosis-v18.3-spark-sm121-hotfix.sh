#!/usr/bin/env bash
# GG v18.3 for DGX Spark: derived hotfix layer on the local v18p2 image.
# See Containerfile.v18p3-hotfix header for the bug this fixes. Build this on
# every node that already has v18p2 loaded (rusty, toby) — the layer is
# python-only, so independently built images are functionally identical;
# verify by comparing sha256 of the patched kernel.py across nodes.
set -euo pipefail
cd "$(dirname "$0")"

PATCH=b12x-w4a16-ultra-tile-forced-repin-20260718.patch
# Repo layout keeps patches under blackwell-llm-docker; deployment dirs carry
# a copy next to this script.
[[ -f $PATCH ]] || cp "../blackwell-llm-docker/patches/$PATCH" .

TAG=localhost/voipmonitor/vllm:gilded-gnosis-v18p3-spark-sm121-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718

podman build -f Containerfile.v18p3-hotfix -t "$TAG" .
podman run --rm "$TAG" sha256sum /opt/venv/lib/python3.12/site-packages/b12x/moe/fused/w4a16/kernel.py
echo "Image: $TAG"
