"""Record the one-file optional loading-policy diagnostic contract amendment."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
raw = (root / 'receipts/pre-loading-runtime-files.json').read_bytes()
old = json.loads(raw)
if (root / 'runtime-files.json').read_bytes() != raw:
    raise RuntimeError('Runtime manifest changed since loading-control snapshot')
prior = (root / 'receipts/pre-loading-launch_contract.py').read_bytes()
if hashlib.sha256(prior).hexdigest() != old['launch_contract.py']:
    raise RuntimeError('Prior contract snapshot does not match manifest')
new = dict(old)
for name, expected in old.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if name == 'launch_contract.py':
        new[name] = actual
    elif actual != expected:
        raise RuntimeError('Unrelated runtime change: ' + name)
with (root / 'loading-control-amendment.json').open('x') as stream:
    stream.write(json.dumps({'prior_manifest': old, 'manifest': new,
        'scope': 'optional CUDA_MODULE_LOADING LAZY/EAGER only; unset unchanged'},
        indent=2, sort_keys=True) + '\n')
(root / 'runtime-files.json').write_text(json.dumps(new, indent=2, sort_keys=True) + '\n')
print('LOADING-CONTROL-FROZEN')
