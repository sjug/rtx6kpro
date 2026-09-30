#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p receipts
[[ ! -e receipts/baseline.log && ! -e receipts/baseline.xml ]] || {
  echo 'Existing baseline receipts; preserve them before another attempt' >&2; exit 78;
}
[[ -z $(podman ps -q) ]] || { echo 'GPU host is not idle'; exit 78; }
image=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
status=0
podman run --rm --pull never --device nvidia.com/gpu=all --ipc host \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e B12X_PRINT_COMPILE_PROGRESS=1 \
  -v "$PWD:/kit:rw" -v /home/jugs/.cache/huggingface:/root/.cache/huggingface:ro \
  --entrypoint /opt/venv/bin/python "$image" -m pytest -s -vv --confcutdir=/kit \
  /kit/test_r38_baseline_grammar.py::test_gpu_sampler_rejects_drafts_after_grammar_termination \
  --junitxml=/kit/receipts/baseline.xml > receipts/baseline.log 2>&1 || status=$?
python3 - "$status" <<'PY' | tee receipts/baseline-verdict.txt
import sys
import xml.etree.ElementTree as ET
root = ET.parse('receipts/baseline.xml').getroot()
cases = root.findall('.//testcase')
failed = [case for case in cases if case.find('failure') is not None]
if (int(sys.argv[1]) != 1 or len(cases) != 32 or len(failed) != 12
        or root.findall('.//error') or root.findall('.//skipped')):
    raise SystemExit('BASELINE INVALID: expected 32 cases, 12 semantic failures, no errors/skips')
for case in failed:
    text = case.find('failure').text or ''
    if 'expected_inputs' not in text or 'AssertionError' not in text:
        raise SystemExit('BASELINE INVALID: unexpected failure location')
print('R38-GRAMMAR-SEMANTIC-RED-CONFIRMED: 12 failures, 20 passes')
PY
