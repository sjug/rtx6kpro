"""Fail-closed Python declaration repair; preserve all native files."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = Path('/opt/jovian-judgement/b12x')
lock = json.loads((HERE / 'moe-dependency.lock.json').read_text())
base = json.loads((HERE / 'runtime.lock.json').read_text())['sources']['b12x']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name, entry in base['files'].items():
    path = ROOT / name
    data = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise RuntimeError(f'Base source mismatch: {name}')
target = ROOT / lock['target_path']
if digest(target) != lock['input_sha256']:
    raise RuntimeError('MoE preparation input drift')
if digest(HERE / 'moe-dependency-preparation.py') != lock['output_sha256']:
    raise RuntimeError('MoE preparation output drift')
if digest(HERE / 'moe-dependency.patch') != lock['patch_sha256']:
    raise RuntimeError('MoE patch drift')
native = {str(p): digest(p) for p in ROOT.rglob('*.so')}
shutil.copyfile(HERE / 'moe-dependency-preparation.py', target)
if native != {str(p): digest(p) for p in ROOT.rglob('*.so')}:
    raise RuntimeError('Native artifact drift')
print('MOE-DEPENDENCY-INSTALL-PASS')
