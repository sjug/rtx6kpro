#!/usr/bin/env python3
"""Record the approved host-adapter IPC amendment without rewriting build evidence."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
receipt = Path(sys.argv[1])
text = receipt.read_text()
for marker in ('DS41-RESOURCE-RECEIPT', 'DS41-STOCK-NCCL-NATIVE-PASS',
               'DS41-FOUR-RANK-CLI-PASS', 'DS41-LOADER-LINKAGE-PASS'):
    if marker not in text:
        raise RuntimeError(f'Missing amendment gate: {marker}')
old = json.loads((root / 'runtime-files.json').read_text())
changed = {name for name, digest in old.items()
           if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest}
if changed != {'launch_contract.py', 'gate_ds41.py'}:
    raise RuntimeError(f'Unexpected amendment scope: {changed}')
amendment = {'reason': 'User approved upstream private shm 64g; unlimited tmpfs fails vLLM space check',
             'prior_manifest_sha256': hashlib.sha256((root / 'runtime-files.json').read_bytes()).hexdigest(),
             'prior_manifest': old, 'gate_receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest()}
(root / 'shm-amendment.json').write_text(json.dumps(amendment, indent=2, sort_keys=True) + '\n')
updated = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in old}
(root / 'runtime-files.json').write_text(json.dumps(updated, indent=2, sort_keys=True) + '\n')
print('APPROVED-SHM-AMENDMENT-FROZEN')
