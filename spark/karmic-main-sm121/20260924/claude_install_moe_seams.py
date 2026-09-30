"""Install the diagnostic MoE-seam helper inside an image build; prove nothing else changed.

Only RUN step of Dockerfile.claude-moe-seams over the activation-digest image
0b5c65a81b25. Refuses drifted inputs, a base whose installed helper is not
the v1 helper, runner sources that differ from the pinned ones the seams were
checked against, and any change beyond the one helper module.
"""
import hashlib
import json
from pathlib import Path
import shutil

if not __debug__:
    raise RuntimeError('Optimized Python is not admitted')
here = Path('/opt/ds41-moe-seams')
opt = Path('/opt/jovian-judgement')
lock = json.loads((here / 'claude-moe-seams.lock.json').read_text())


def digest(path):
    data = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def inventory():
    return {str(p): digest(p) for component in ('vllm', 'b12x') for p in (opt / component).rglob('*')
            if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts}


for name, expected in lock['inputs'].items():
    if digest(here / name) != expected:
        raise RuntimeError('Diagnostic input drift: ' + name)
for target, expected in lock['verify_unchanged'].items():
    if digest(opt / target) != expected:
        raise RuntimeError('Runner source differs from the pinned seam contract: ' + target)
for target, entry in lock['targets'].items():
    if digest(opt / target) != entry['input_sha256']:
        raise RuntimeError('Base is not the v1 activation-digest image: ' + target)
before = inventory()
if not any(name.endswith('.so') for name in before):
    raise RuntimeError('Missing inherited native inventory')
expected = dict(before)
for target, entry in lock['targets'].items():
    path = opt / target
    shutil.copyfile(here / entry['source'], path)
    compile(path.read_bytes(), str(path), 'exec')
    expected[str(path)] = entry['output_sha256']
after = inventory()
if after != expected:
    changed = sorted(set(after.items()) ^ set(expected.items()))
    raise RuntimeError('Change was not confined to the diagnostic layer: %r' % changed[:6])
(here / 'before-files.json').write_text(json.dumps(before, indent=2, sort_keys=True) + '\n')
(here / 'after-files.json').write_text(json.dumps(after, indent=2, sort_keys=True) + '\n')
print('MOE-SEAMS-INSTALL-PASS', flush=True)
