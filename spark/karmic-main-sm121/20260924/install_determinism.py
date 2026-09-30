"""Install the reviewed selector over the verified dependency-only control."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = Path('/opt/jovian-judgement/b12x')
lock = json.loads((HERE / 'determinism.lock.json').read_text())
baseline = json.loads((HERE / 'runtime.lock.json').read_text())['sources']['b12x']['files']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name, entry in baseline.items():
    expected = entry['sha256']
    if name == lock['moe_dependency']['target_path']:
        expected = lock['moe_dependency']['output_sha256']
    path = ROOT / name
    data = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(f'Base source mismatch: {name}')
target = ROOT / lock['selector_path']
if digest(target) != lock['selector_input_sha256']:
    raise RuntimeError('Selector input mismatch')
if digest(HERE / 'determinism-tiled-topk.py') != lock['selector_output_sha256']:
    raise RuntimeError('Selector output mismatch')
if digest(HERE / 'claude-dsa-topk-tiebreak.patch') != lock['selector_patch_sha256']:
    raise RuntimeError('Selector patch mismatch')
native = {str(p): digest(p) for p in ROOT.rglob('*.so')}
shutil.copyfile(HERE / 'determinism-tiled-topk.py', target)
if digest(target) != lock['selector_output_sha256']:
    raise RuntimeError('Installed selector mismatch')
if native != {str(p): digest(p) for p in ROOT.rglob('*.so')}:
    raise RuntimeError('Native artifact drift')
print('DETERMINISM-INSTALL-PASS')
