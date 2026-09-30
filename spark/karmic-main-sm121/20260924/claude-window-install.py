"""Install the diagnostic window-capture layer inside an image build; prove nothing else changed.

Only RUN step of claude-window.Dockerfile over the router fence candidate.
Refuses drifted inputs; a base whose router lock, fenced router source or full
file inventory is not the recorded candidate's (the router installer's own
preserved-files.json); a base whose attention.py is not the candidate's;
preexisting helpers; and any change beyond attention.py plus the two helper
modules. Also writes /opt/ds41-decision-row/claude-decision-row.lock.json as a
provenance stub so the UNCHANGED decision-row helper records this image's
source trees (it reads only the 'trees' key of that path); that stub is
outside the audited vLLM/B12X inventory and names the window lock it derives from.
"""
import hashlib
import json
from pathlib import Path
import shutil

if not __debug__:
    raise RuntimeError('Optimized Python is not admitted')
here = Path('/opt/ds41-window')
opt = Path('/opt/jovian-judgement')
router = Path('/opt/ds41-router-release')
decision_stub = Path('/opt/ds41-decision-row/claude-decision-row.lock.json')
lock_raw = (here / 'claude-window.lock.json').read_bytes()
lock = json.loads(lock_raw)


def digest(path):
    data = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def inventory():
    return {str(p): digest(p) for component in ('vllm', 'b12x') for p in (opt / component).rglob('*')
            if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts}


for name, expected in lock['inputs'].items():
    if digest(here / name) != expected:
        raise RuntimeError('Diagnostic input drift: ' + name)
for target, entry in lock['targets'].items():
    path = opt / target
    if entry['input_sha256'] is None:
        if path.exists() or path.is_symlink():
            raise RuntimeError('Unexpected preexisting diagnostic file: ' + target)
    elif digest(path) != entry['input_sha256']:
        raise RuntimeError('Base model preimage differs from the router candidate: ' + target)
if digest(router / lock['base_lock']) != lock['base_lock_sha256']:
    raise RuntimeError('Base is not the router fence candidate: router lock differs')
if digest(opt / 'b12x' / lock['base_router_target']['path']) != lock['base_router_target']['sha256']:
    raise RuntimeError('Base is not the router fence candidate: router source differs')
if decision_stub.exists() or decision_stub.parent.exists():
    raise RuntimeError('Base already carries a decision-row installation; the window kit derives from the router parent')
before = inventory()
if before != json.loads((router / 'preserved-files.json').read_text()):
    raise RuntimeError('Base inventory differs from the recorded router candidate')
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
decision_stub.parent.mkdir(parents=True)
decision_stub.write_text(json.dumps({
    'kind': 'window-capture-provenance-stub', 'trees': lock['trees'],
    'window_lock_sha256': hashlib.sha256(lock_raw).hexdigest(),
    'decision_row_lock_sha256': lock['decision_row_lock_sha256'],
    'note': 'read by the unchanged decision-row helper for source_trees only; not a decision-row installation'},
    indent=2, sort_keys=True) + '\n')
(here / 'before-files.json').write_text(json.dumps(before, indent=2, sort_keys=True) + '\n')
(here / 'after-files.json').write_text(json.dumps(after, indent=2, sort_keys=True) + '\n')
print('WINDOW-INSTALL-PASS', flush=True)
