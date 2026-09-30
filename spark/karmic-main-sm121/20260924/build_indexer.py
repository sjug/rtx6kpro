"""Build and import-gate the approved observation-only layer-2 image on idle nodes."""
from build_decision_row import main

SMOKE = '''
import hashlib, importlib, json
from pathlib import Path
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
    raise RuntimeError('A visible GB10 is required')
root = Path('/opt/jovian-judgement')
lock = json.loads(Path('/opt/ds41-indexer/ds41-indexer.lock.json').read_text())
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
attention = loaded['vllm.models.deepseek_v4_1.attention']
helper = loaded['vllm.models.deepseek_v4_1.ds41_indexer_capture']
if attention._indexer_capture is not helper or helper.SCHEMA != 'ds41-indexer-capture-v1':
    raise RuntimeError('Capture hook not linked')
for path in ('/opt/ds41-window/claude-window.lock.json',
             '/opt/ds41-decision-row/claude-decision-row.lock.json'):
    if json.loads(Path(path).read_text())['trees'] != lock['trees']:
        raise RuntimeError('Inherited capture provenance differs')
for target, expected in lock['preserved'].items():
    if hashlib.sha256((root / target).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Inherited helper changed')
print('INDEXER-HELPER-IMPORT-PASS')
'''

if __name__ == '__main__':
    main(kind='indexer', smoke=SMOKE, install_marker='INDEXER-INSTALL-PASS',
         smoke_marker='INDEXER-HELPER-IMPORT-PASS', wrapper='build_indexer.py')
