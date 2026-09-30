"""Freeze optional diagnostic controls, retaining the exact previous live kit."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
manifest_path = root / 'runtime-files.json'
manifest = json.loads(manifest_path.read_text())
remote = '/home/jugs/git/ds41-r38/karmic-main-20260924/launch_contract.py'
prior = None
for node in ('dusty', 'toby', 'rusty', 'kirby'):
    raw = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, f'cat {remote}'],
                         capture_output=True, check=True, timeout=30).stdout
    if hashlib.sha256(raw).hexdigest() != manifest['launch_contract.py']:
        raise RuntimeError(f'{node}: preceding contract mismatch')
    prior = raw
for name, expected in manifest.items():
    if name != 'launch_contract.py' and hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Unrelated runtime drift: {name}')
old = """    deterministic_moe = os.environ.get('B12X_DYNAMIC_DETERMINISTIC_OUTPUT')
    if deterministic_moe is not None:
        if deterministic_moe not in ('0', '1'):
            raise ValueError('B12X_DYNAMIC_DETERMINISTIC_OUTPUT must be 0 or 1')
        env['B12X_DYNAMIC_DETERMINISTIC_OUTPUT'] = deterministic_moe
"""
new = """    for name in ('B12X_DYNAMIC_DETERMINISTIC_OUTPUT', 'B12X_DENSE_SPLITK_TURBO',
                 'VLLM_DS41_L2_PREFETCH'):
        value = os.environ.get(name)
        if value is not None:
            if value not in ('0', '1'):
                raise ValueError(f'{name} must be 0 or 1')
            env[name] = value
"""
current = (root / 'launch_contract.py').read_bytes()
if prior.decode().count(old) != 1 or prior.decode().replace(old, new).encode() != current:
    raise RuntimeError('Contract differs beyond the reviewed optional controls')
updated = dict(manifest, **{'launch_contract.py': hashlib.sha256(current).hexdigest()})
with (root / 'receipts/short-controls-amendment.json').open('x') as receipt:
    json.dump({'prior_manifest': manifest, 'prior_launch_contract': prior.decode(),
               'manifest': updated, 'purpose': 'one-variable prefetch and dense reduction diagnostics'}, receipt, indent=2)
    receipt.write('\n')
manifest_path.write_text(json.dumps(updated, indent=2, sort_keys=True) + '\n')
print('SHORT-CONTROLS-FROZEN', updated['launch_contract.py'])
