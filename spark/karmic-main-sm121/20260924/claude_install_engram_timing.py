"""Install the Engram host timing layer inside an image build; prove nothing else changed.

Runs as the only RUN step of Dockerfile.claude-engram-timing over the ring-fix
candidate. It refuses drifted inputs, a base whose target files are not the
pinned blobs, a base without the ring fix, or any change beyond the patched
vLLM model.py and the one new helper module. B12X targets are "runtime-bound":
their patched copies stay in /opt/ds41-engram-timing and the helper binds the
edited functions at model import, so every B12X file must remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import shutil

if not __debug__:
    raise RuntimeError('Optimized Python is not admitted')
here = Path('/opt/ds41-engram-timing')
opt = Path('/opt/jovian-judgement')
lock = json.loads((here / 'claude-engram-timing.lock.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    return {str(p): sha(p) for base in (opt / 'vllm', opt / 'b12x')
            for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


for name, expected in lock['inputs'].items():
    if sha(here / name) != expected:
        raise RuntimeError('Diagnostic input drift: ' + name)
ring = opt / 'vllm/vllm/models/deepseek_v4_1/sparse_mla.py'
if sha(ring) != lock['base_ring_fix_sparse_mla_sha256']:
    raise RuntimeError('Base is not the ring-fix candidate')
for image_path, entry in lock['targets'].items():
    if sha(opt / image_path) != entry['input_sha256']:
        raise RuntimeError('Base file is not the pinned input: ' + image_path)
helper = opt / lock['helper']['target']
if helper.exists():
    raise RuntimeError('Unexpected preexisting timing helper')
before = inventory()
if not any(name.endswith('.so') for name in before):
    raise RuntimeError('Missing inherited native inventory')
files = {p: e for p, e in lock['targets'].items() if e['delivery'] == 'file'}
if set(files) != {'vllm/vllm/models/deepseek_v4_1/nvidia/model.py'} or any(
        e['delivery'] != 'runtime-bound' for p, e in lock['targets'].items() if p not in files):
    raise RuntimeError('Unexpected delivery plan')
for image_path, entry in files.items():
    shutil.copyfile(here / entry['source'], opt / image_path)
shutil.copyfile(here / lock['helper']['source'], helper)
expected = dict(before, **{str(opt / p): e['output_sha256'] for p, e in files.items()})
expected[str(helper)] = lock['helper']['sha256']
for path in [helper, *(opt / p for p in files)]:
    compile(path.read_bytes(), str(path), 'exec')
for image_path, entry in lock['targets'].items():
    compile((here / entry['source']).read_bytes(), entry['source'], 'exec')
after = inventory()
if after != expected:
    changed = sorted(set(after.items()) ^ set(expected.items()))
    raise RuntimeError('Change was not confined to the timing layer: %r' % changed[:6])
(here / 'before-files.json').write_text(json.dumps(before, indent=2, sort_keys=True) + '\n')
(here / 'after-files.json').write_text(json.dumps(after, indent=2, sort_keys=True) + '\n')
print('ENGRAM-TIMING-INSTALL-PASS', flush=True)
