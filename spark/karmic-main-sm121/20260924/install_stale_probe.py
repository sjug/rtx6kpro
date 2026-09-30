"""Install exactly the reviewed host-side diagnostic hook; retain all natives."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path('/opt/ds41-stale-probe')
lock = json.loads((root / 'stale-probe.lock.json').read_text())
tree = Path('/opt/jovian-judgement/vllm')
target = tree / lock['source_path']
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for name, expected in lock['inputs'].items():
    if sha(root / name) != expected:
        raise RuntimeError('Diagnostic input drift: ' + name)
if sha(target) != lock['input_sha256']:
    raise RuntimeError('Base model_state is not the pinned input')
before = {str(p.relative_to(tree)): sha(p) for p in tree.rglob('*.so')}
if not before:
    raise RuntimeError('Missing inherited native inventory')
helper = target.with_name('claude_stale_state_probe.py')
if helper.exists():
    raise RuntimeError('Unexpected preexisting diagnostic helper')
shutil.copyfile(root / 'stale-probe-model_state.py', target)
shutil.copyfile(root / 'claude_stale_state_probe.py', helper)
for path in (target, helper):
    compile(path.read_bytes(), str(path), 'exec')
if sha(target) != lock['output_sha256'] or before != {
        str(p.relative_to(tree)): sha(p) for p in tree.rglob('*.so')}:
    raise RuntimeError('Diagnostic install changed unexpected content')
print('STALE-PROBE-INSTALL-PASS', flush=True)
