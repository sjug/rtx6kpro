"""Verify the complete base B12X tracked tree, then install two diagnostic files."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = Path('/opt/jovian-judgement/b12x')
lock = json.loads((HERE / 'diagnostic-topk.lock.json').read_text())
base = json.loads((HERE / 'runtime.lock.json').read_text())['sources']['b12x']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name, entry in base['files'].items():
    path = ROOT / name
    # Preserve the source manifest's symlink identity rather than dereferencing.
    content = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
    if hashlib.sha256(content).hexdigest() != entry['sha256']:
        raise RuntimeError(f'Base B12X tracked content mismatch: {name}')
target = ROOT / lock['target_path']
module = ROOT / lock['module_path']
if module.exists() or digest(target) != lock['input_sha256']:
    raise RuntimeError('Diagnostic base input mismatch')
for filename, key in [('diagnostic-mxfp4.py', 'output_sha256'),
                      ('deterministic_topk.py', 'module_sha256'),
                      ('diagnostic-topk.patch', 'patch_sha256')]:
    if digest(HERE / filename) != lock[key]:
        raise RuntimeError(f'Diagnostic artifact digest mismatch: {filename}')
native_before = {str(p): digest(p) for p in ROOT.rglob('*.so')}
shutil.copyfile(HERE / 'diagnostic-mxfp4.py', target)
shutil.copyfile(HERE / 'deterministic_topk.py', module)
native_after = {str(p): digest(p) for p in ROOT.rglob('*.so')}
if native_before != native_after:
    raise RuntimeError('Native files changed during Python-only overlay')
print('DS41-DETERMINISTIC-DIAGNOSTIC-INSTALL-PASS')
