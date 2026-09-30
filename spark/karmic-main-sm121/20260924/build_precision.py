"""Build and import-gate the approved worker-level BF16 reduced-precision-off image on idle nodes."""
from build_decision_row import main

SMOKE = '''
import contextlib, hashlib, importlib, io, json
from pathlib import Path
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
    raise RuntimeError('A visible GB10 is required')
root = Path('/opt/jovian-judgement')
lock_raw = Path('/opt/ds41-precision/ds41-precision.lock.json').read_bytes()
lock = json.loads(lock_raw)
loaded = {}
for target, entry in lock['targets'].items():
    if not target.startswith('vllm/vllm/') or not target.endswith('.py'):
        raise RuntimeError('Unexpected target')
    name = target.removeprefix('vllm/').removesuffix('.py').replace('/', '.')
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if path != (root / target).resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['output_sha256']:
        raise RuntimeError('Wrong imported module: ' + name)
    loaded[name] = module
worker = loaded['vllm.v1.worker.gpu_worker']
helper = loaded['vllm.v1.worker.ds41_precision']
if worker._ds41_precision is not helper or helper.SCHEMA != lock['precision_schema']:
    raise RuntimeError('Precision hook not linked')
for target, expected in lock['preserved'].items():
    if hashlib.sha256((root / target).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Inherited attention or capture helper changed')
provenance = lock['inherited_helper_provenance']
digest = hashlib.sha256(lock_raw).hexdigest()
for key in ('window_read_path', 'decision_read_path', 'indexer_read_path'):
    stub = json.loads(Path(provenance[key]).read_text())
    if (stub['kind'], stub['trees'], stub['precision_lock_sha256']) != ('precision-provenance-stub', lock['trees'], digest):
        raise RuntimeError('Inherited capture provenance differs: ' + key)
if hashlib.sha256(Path('/opt/ds41-indexer', provenance['indexer_era_lock']).read_bytes()).hexdigest() != lock['base_lock_sha256']:
    raise RuntimeError('Preserved indexer lock differs')
if hashlib.sha256(Path('/opt/ds41-window', provenance['window_era_original_lock']).read_bytes()).hexdigest() != lock['window_lock_sha256']:
    raise RuntimeError('Preserved window lock differs')
before = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
out = io.StringIO()
with contextlib.redirect_stdout(out):
    record = helper.apply(rank='smoke')
line = out.getvalue().strip()
if (not line.startswith(lock['apply_marker'] + ' rank=smoke allow_bf16_reduced_precision_reduction=False')
        or torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is not False
        or helper.status()['allow_bf16_reduced_precision_reduction'] is not False or record['before'] != before):
    raise RuntimeError('Precision flag did not take: ' + line)
print(line)
print('PRECISION-HELPER-IMPORT-PASS')
'''

if __name__ == '__main__':
    main(kind='precision', smoke=SMOKE, install_marker='PRECISION-INSTALL-PASS',
         smoke_marker='PRECISION-HELPER-IMPORT-PASS', wrapper='build_precision.py')
