#!/usr/bin/env python3
"""Bind the approved 600K adapter amendment to its independent CLI gate."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
receipt = Path(sys.argv[1])
text = receipt.read_text()
qualification = root / 'receipts/qualification-500k-resumed.log'
if 'DS41-QUALIFICATION-PASS' not in qualification.read_text():
    raise RuntimeError('500K admission has not passed')
for marker in ('DS41-RESOURCE-RECEIPT', 'DS41-STOCK-NCCL-NATIVE-PASS',
               'DS41-FOUR-RANK-CLI-PASS', 'DS41-LOADER-LINKAGE-PASS'):
    if marker not in text:
        raise RuntimeError(f'Missing context gate: {marker}')
old_bytes = (root / 'runtime-files.json').read_bytes()
old = json.loads(old_bytes)
updated = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in old}
if {name for name in old if old[name] != updated[name]} != {'launch_contract.py', 'gate_ds41.py'}:
    raise RuntimeError('Unexpected context amendment scope')
pin = json.loads((root / 'candidate.json').read_text())
if pin['image_id'] not in text:
    raise RuntimeError('Gate receipt lacks candidate image identity')
amendment = {
    'reason': 'User-approved context-only 600K historical replay after successful 500K admission',
    'image_id': pin['image_id'], 'max_model_len': 600000,
    'prior_manifest_sha256': hashlib.sha256(old_bytes).hexdigest(),
    'prior_manifest': old, 'updated_manifest': updated,
    'qualification_sha256': hashlib.sha256(qualification.read_bytes()).hexdigest(),
    'gate_receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(),
}
with (root / 'context-amendment.json').open('x') as output:
    output.write(json.dumps(amendment, indent=2, sort_keys=True) + '\n')
(root / 'runtime-files.json').write_text(json.dumps(updated, indent=2, sort_keys=True) + '\n')
print('APPROVED-CONTEXT-AMENDMENT-FROZEN')
