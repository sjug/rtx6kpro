#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
image=${1:?image ID}
gpu=(podman run --rm --pull never --network none --device nvidia.com/gpu=all --ipc host
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1
  -e B12X_PRINT_COMPILE_PROGRESS=1 -v "$PWD:/kit:rw"
  -v /home/jugs/.cache/huggingface:/root/.cache/huggingface:ro)
"${gpu[@]}" -v /home/jugs/git/ds4-vision-r38/runtime-preflight.py:/preflight.py:ro \
  --entrypoint /opt/venv/bin/python "$image" /preflight.py | tee receipts/build/native.log
"${gpu[@]}" --entrypoint /bin/bash "$image" -c '
  set -euo pipefail
  cd /opt/jovian-judgement/vllm
  run() { /opt/venv/bin/python /opt/local-inference/r38-tests/run_required.py "$@"; }
  run --count 32 --confcutdir=tests/v1/spec_decode tests/v1/spec_decode/test_mtp_structured_output.py::test_gpu_sampler_rejects_drafts_after_grammar_termination
  run --count 2 --confcutdir=tests/v1/worker tests/v1/worker/test_gpu_batch_shard.py::test_shard_grammar_output
  run --count 2 --confcutdir=tests/v1/worker tests/v1/worker/test_async_verified_draft_counts.py
  run --count 1 --confcutdir=tests/v1/core tests/v1/core/test_scheduler.py::test_per_request_spec_decode_subtracts_invalid_drafts
  echo GRAMMAR-REPAIR-GATES-PASS
' | tee receipts/build/regressions.log
