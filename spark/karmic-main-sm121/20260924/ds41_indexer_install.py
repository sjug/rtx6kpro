"""Install the diagnostic indexer-capture layer inside an image build; prove nothing else changed.

Only RUN step of Dockerfile.ds41-indexer over the reviewed window-capture image.
Refuses drifted inputs; a base whose window lock is not the recorded one; a
base whose attention.py, decision-row helper or window helper differs from
that lock; a base whose full vLLM/B12X inventory is not the window
installer's own after-files.json; a preexisting indexer helper; and any
change beyond attention.py plus the one new helper module.

Inherited helper provenance (metadata only, no helper source mutation): the
unchanged decision-row and window helpers read the 'trees' key of
/opt/ds41-decision-row/claude-decision-row.lock.json and
/opt/ds41-window/claude-window.lock.json when they save. This installer
validates the original window lock (digest == base_lock_sha256), preserves it
byte for byte as claude-window.lock.original.json, preserves the window-era
decision-row stub as claude-decision-row.lock.original.json, and then writes
provenance stubs at both read paths naming this image's trees and the indexer
lock, so every sibling capture of one boot records the same source trees.
Both stub paths are outside the audited vLLM/B12X inventory.
"""
import hashlib
import json
from pathlib import Path
import shutil

if not __debug__:
    raise RuntimeError('Optimized Python is not admitted')
here = Path('/opt/ds41-indexer')
opt = Path('/opt/jovian-judgement')
window = Path('/opt/ds41-window')
decision = Path('/opt/ds41-decision-row')
lock_raw = (here / 'ds41-indexer.lock.json').read_bytes()
lock = json.loads(lock_raw)
provenance = lock['inherited_helper_provenance']


def digest(path):
    data = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def inventory():
    return {str(p): digest(p) for component in ('vllm', 'b12x') for p in (opt / component).rglob('*')
            if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts}


for name, expected in lock['inputs'].items():
    if digest(here / name) != expected:
        raise RuntimeError('Diagnostic input drift: ' + name)
if digest(window / lock['base_lock']) != lock['base_lock_sha256']:
    raise RuntimeError('Base is not the window-capture diagnostic: window lock differs')
for target, expected in lock['preserved'].items():
    if digest(opt / target) != expected:
        raise RuntimeError('Base helper differs from the window lock: ' + target)
for target, entry in lock['targets'].items():
    path = opt / target
    if entry['input_sha256'] is None:
        if path.exists() or path.is_symlink():
            raise RuntimeError('Unexpected preexisting diagnostic file: ' + target)
    elif digest(path) != entry['input_sha256']:
        raise RuntimeError('Base attention differs from the window-capture image: ' + target)
if digest(opt / 'b12x' / lock['base_router_target']['path']) != lock['base_router_target']['sha256']:
    raise RuntimeError('Base is not the router lineage: router source differs')
before = inventory()
if before != json.loads((window / 'after-files.json').read_text()):
    raise RuntimeError('Base inventory differs from the recorded window-capture installation')
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
# Inherited helper provenance: preserve originals, then point both read paths at this image's trees.
window_lock = window / lock['base_lock']
window_original = window / provenance['original_window_lock']
decision_stub = decision / 'claude-decision-row.lock.json'
decision_original = decision / provenance['original_decision_stub']
if window_original.exists() or decision_original.exists():
    raise RuntimeError('Preserved original provenance files already exist; refusing to overwrite')
if digest(window_lock) != lock['base_lock_sha256']:
    raise RuntimeError('Original window lock digest changed during installation')
inherited = json.loads(decision_stub.read_text())
if inherited.get('kind') != 'window-capture-provenance-stub' or inherited.get('trees') != lock['base_trees']:
    raise RuntimeError('Decision-row read path does not hold the window-capture provenance stub')
shutil.copyfile(window_lock, window_original)
shutil.copyfile(decision_stub, decision_original)
if digest(window_original) != lock['base_lock_sha256']:
    raise RuntimeError('Preserved window lock differs from the original')
indexer_lock_sha256 = hashlib.sha256(lock_raw).hexdigest()
for path, original, helper in ((window_lock, window_original.name, 'claude_window'),
                               (decision_stub, decision_original.name, 'claude_decision_row')):
    path.write_text(json.dumps({
        'kind': 'indexer-capture-provenance-stub', 'trees': lock['trees'], 'indexer_lock_sha256': indexer_lock_sha256,
        'window_lock_sha256': lock['base_lock_sha256'], 'decision_row_lock_sha256': lock['decision_row_lock_sha256'],
        'original': original, 'disposition': provenance['disposition'],
        'note': f'read by the unchanged {helper} helper for source_trees only; the original is preserved beside it'},
        indent=2, sort_keys=True) + '\n')
(here / 'before-files.json').write_text(json.dumps(before, indent=2, sort_keys=True) + '\n')
(here / 'after-files.json').write_text(json.dumps(after, indent=2, sort_keys=True) + '\n')
print('INDEXER-INSTALL-PASS', flush=True)
