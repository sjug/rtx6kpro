"""Install the diagnostic activation-digest layer inside an image build; prove nothing else changed.

Only RUN step of Dockerfile.claude-act-trace over the Engram repair
candidate. Refuses drifted inputs, a base whose model.py is not the
candidate's, a preexisting helper, and any change beyond model.py plus the
one new helper module.
"""
import hashlib
import json
from pathlib import Path
import shutil

if not __debug__:
    raise RuntimeError('Optimized Python is not admitted')
here = Path('/opt/ds41-act-trace')
opt = Path('/opt/jovian-judgement')
lock = json.loads((here / 'claude-act-trace.lock.json').read_text())


def digest(path):
    data = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def inventory():
    return {str(p): digest(p) for component in ('vllm', 'b12x') for p in (opt / component).rglob('*')
            if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts}


for name, expected in lock['inputs'].items():
    if digest(here / name) != expected:
        raise RuntimeError('Diagnostic input drift: ' + name)
expected = None
for target, entry in lock['targets'].items():
    path = opt / target
    if entry['input_sha256'] is None:
        if path.exists() or path.is_symlink():
            raise RuntimeError('Unexpected preexisting diagnostic file: ' + target)
    elif digest(path) != entry['input_sha256']:
        raise RuntimeError('Base is not the Engram repair candidate: ' + target)
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
print('ACT-TRACE-INSTALL-PASS', flush=True)
