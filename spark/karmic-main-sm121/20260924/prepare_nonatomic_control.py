"""Declare a diagnostic using the existing non-atomic FP32 split-K path.

Not an autotuning receipt: configurations are explicit diagnostic overrides.
The existing non-atomic implementation supports at most two slices. Four-way
atomic winners therefore become two-way controls; all other geometry stays.
"""
import hashlib
import json
from pathlib import Path
from claude_tuning_key_recon import CAPACITIES, KS, NS, record_key

root = Path(__file__).resolve().parent
source = root / 'receipts/combined-tuning.json'
raw = source.read_bytes()
prior = json.loads(raw)
records, changes = {}, []
for cap in (value for value in CAPACITIES if value <= 8):
    for k in sorted({*KS, 576, 15360}):
        for n in NS:
            old_key = record_key(cap, k, n, turbo=True)
            if old_key not in prior['records']:
                continue
            old = prior['records'][old_key]['assignment']
            assignment = dict(old)
            if assignment['split_k_slices'] == 4:
                assignment['split_k_slices'] = 2
            key = record_key(cap, k, n, turbo=False)
            records[key] = {'assignment': assignment, 'config': assignment,
                            'diagnostic_source_key': old_key}
            changes.append({'k': k, 'n': n, 'capacity': cap,
                            'original': old, 'control': assignment})
if len(records) != 64:
    raise RuntimeError('Unexpected decode control coverage')
result = {'identity': prior['identity'], 'records': records,
          'diagnostic_only': True, 'source_sha256': hashlib.sha256(raw).hexdigest(),
          'method': 'turbo=0; original 4-way becomes 2-way; all other geometry retained',
          'changes': changes}
with (root / 'receipts/non-atomic-complete-control-tuning.json').open('x') as stream:
    stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
print('NONATOMIC-CONTROL-PREPARED records=64')
