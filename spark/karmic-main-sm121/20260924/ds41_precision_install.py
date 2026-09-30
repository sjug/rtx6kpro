"""Install the diagnostic worker-level precision layer inside an image build; prove nothing else changed.

Only RUN step of Dockerfile.ds41-precision over the reviewed indexer-capture image.
Refuses drifted inputs; a base whose indexer lock is not the recorded one; a base
whose attention.py or any of the three capture helpers differs from the indexer
lock; a base whose full vLLM/B12X inventory is not the indexer installer's own
after-files.json; a base whose worker source is not the router candidate's; a
preexisting precision helper; and any change beyond gpu_worker.py plus the one new
helper module.

Inherited capture provenance (metadata only, no helper source mutation): the
unchanged decision-row, window and indexer helpers read the 'trees' key of
/opt/ds41-decision-row/claude-decision-row.lock.json,
/opt/ds41-window/claude-window.lock.json and
/opt/ds41-indexer/ds41-indexer.lock.json when they save. This installer validates
the indexer-era stubs and lock at those paths, preserves them byte for byte under
explicit indexer-era names, leaves the window-era originals in place, and writes
provenance stubs at all three read paths naming this image's trees and the
precision lock, so every sibling capture of one boot records the same source
trees. All stub paths are outside the audited vLLM/B12X inventory.
"""
import hashlib
import json
from pathlib import Path
import shutil

if not __debug__:
    raise RuntimeError('Optimized Python is not admitted')
here = Path('/opt/ds41-precision')
opt = Path('/opt/jovian-judgement')
window = Path('/opt/ds41-window')
decision = Path('/opt/ds41-decision-row')
indexer = Path('/opt/ds41-indexer')
lock_raw = (here / 'ds41-precision.lock.json').read_bytes()
lock = json.loads(lock_raw)
provenance = lock['inherited_helper_provenance']


def digest(path):
    data = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def inventory():
    return {str(p): digest(p) for component in ('vllm', 'b12x') for p in (opt / component).rglob('*')
            if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts}


def read_stub(path, kind, trees):
    stub = json.loads(path.read_text())
    if stub.get('kind') != kind or stub.get('trees') != trees:
        raise RuntimeError('Provenance read path does not hold the expected ' + kind + ': ' + str(path))
    return stub


for name, expected in lock['inputs'].items():
    if digest(here / name) != expected:
        raise RuntimeError('Diagnostic input drift: ' + name)
indexer_lock = indexer / lock['base_lock']
if digest(indexer_lock) != lock['base_lock_sha256']:
    raise RuntimeError('Base is not the indexer-capture diagnostic: indexer lock differs')
for target, expected in lock['preserved'].items():
    if digest(opt / target) != expected:
        raise RuntimeError('Base helper differs from the indexer lock: ' + target)
for target, entry in lock['targets'].items():
    path = opt / target
    if entry['input_sha256'] is None:
        if path.exists() or path.is_symlink():
            raise RuntimeError('Unexpected preexisting diagnostic file: ' + target)
    elif digest(path) != entry['input_sha256']:
        raise RuntimeError('Base worker differs from the router candidate: ' + target)
if digest(opt / 'b12x' / lock['base_router_target']['path']) != lock['base_router_target']['sha256']:
    raise RuntimeError('Base is not the router lineage: router source differs')
# Indexer-era provenance at the three read paths, and the window-era originals beside two of them.
window_stub = window / 'claude-window.lock.json'
decision_stub = decision / 'claude-decision-row.lock.json'
for path in (window_stub, decision_stub):
    stub = read_stub(path, 'indexer-capture-provenance-stub', lock['base_trees'])
    if stub.get('indexer_lock_sha256') != lock['base_lock_sha256']:
        raise RuntimeError('Indexer-era stub names a different indexer lock: ' + str(path))
window_original = window / provenance['window_era_original_lock']
if not window_original.is_file() or digest(window_original) != lock['window_lock_sha256']:
    raise RuntimeError('Window-era original lock is missing or differs')
decision_original = decision / provenance['window_era_original_stub']
if not decision_original.is_file() or json.loads(decision_original.read_text()).get('kind') != 'window-capture-provenance-stub':
    raise RuntimeError('Window-era original decision-row stub is missing or differs')
before = inventory()
if before != json.loads((indexer / 'after-files.json').read_text()):
    raise RuntimeError('Base inventory differs from the recorded indexer-capture installation')
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
# Inherited helper provenance: preserve the indexer-era files, then point all three read paths at this image's trees.
preserved = ((window_stub, window / provenance['indexer_era_window_stub'], 'claude_window'),
             (decision_stub, decision / provenance['indexer_era_decision_stub'], 'claude_decision_row'),
             (indexer_lock, indexer / provenance['indexer_era_lock'], 'ds41_indexer_capture'))
if any(original.exists() for _, original, _ in preserved):
    raise RuntimeError('Preserved indexer-era provenance files already exist; refusing to overwrite')
for path, original, _ in preserved:
    shutil.copyfile(path, original)
    if digest(original) != digest(path):
        raise RuntimeError('Preserved provenance file differs from the original: ' + str(original))
if digest(indexer / provenance['indexer_era_lock']) != lock['base_lock_sha256']:
    raise RuntimeError('Preserved indexer lock differs from the recorded base lock')
precision_lock_sha256 = hashlib.sha256(lock_raw).hexdigest()
for path, original, helper in preserved:
    path.write_text(json.dumps({
        'kind': 'precision-provenance-stub', 'trees': lock['trees'], 'precision_lock_sha256': precision_lock_sha256,
        'indexer_lock_sha256': lock['base_lock_sha256'], 'window_lock_sha256': lock['window_lock_sha256'],
        'decision_row_lock_sha256': lock['decision_row_lock_sha256'],
        'original': original.name, 'disposition': provenance['disposition'],
        'note': f'read by the unchanged {helper} helper for source_trees only; the indexer-era file is preserved beside it'},
        indent=2, sort_keys=True) + '\n')
(here / 'before-files.json').write_text(json.dumps(before, indent=2, sort_keys=True) + '\n')
(here / 'after-files.json').write_text(json.dumps(after, indent=2, sort_keys=True) + '\n')
print('PRECISION-INSTALL-PASS', flush=True)
