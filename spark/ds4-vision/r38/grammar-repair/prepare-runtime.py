#!/usr/bin/env python3
"""Derive the serving kit from qualified R38 with identity-only runner changes."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
target = ROOT / 'runtime'
target.mkdir(exist_ok=True)
image = json.loads((ROOT / 'receipts/build/image-inspect.json').read_text())[0]['Id']
if not (ROOT / 'receipts/build/BUILD-OK').is_file():
    raise RuntimeError('Build gates have not passed')
text = (BASE / 'run-node.sh').read_text()
replacements = {
    'root=/home/jugs/git/ds4-vision-r38': 'root=/home/jugs/git/ds4-vision-r38/grammar-repair/runtime',
    'image=localhost/voipmonitor/vllm:jj-r38-spark-sm121': 'image=localhost/voipmonitor/vllm:jj-r38p-spark-sm121',
    'expected=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5': f'expected={image}',
    'name=ds4-vision-jj-r38-tp2': 'name=ds4-vision-jj-r38p-tp2',
}
for before, after in replacements.items():
    if text.count(before) != 1:
        raise RuntimeError(f'Runner drift: {before}')
    text = text.replace(before, after)
(target / 'run-node.sh').write_text(text)
for name in ('launch-in-container.sh', 'runtime-preflight.py', 'verify-model.py', 'parse-cli.py'):
    shutil.copyfile(BASE / name, target / name)
(target / 'test-render.py').write_text((BASE / 'test-render.py').read_text().replace(
    'jj-r38-spark-sm121', 'jj-r38p-spark-sm121'))
(target / 'receipts').mkdir(exist_ok=True)
shutil.copyfile(BASE / 'receipts/model-manifest.json', target / 'receipts/model-manifest.json')
files = sorted(p for p in target.rglob('*') if p.is_file() and p.name != 'runtime-files.sha256')
(target / 'runtime-files.sha256').write_text(''.join(
    f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(target)}\n' for p in files))
print(f'R38p runtime prepared: {image}; four identity-only runner substitutions')
