#!/usr/bin/env python3
"""Check the derivative inputs before probing hosts or building."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from contracts import file_sha, require, sha
sys.path.insert(0, str(ROOT))
from prepare import BASE, HEAD, PATCH_SHA, PATHS

pin = json.loads((ROOT / 'backport.lock.json').read_text())
require(pin['base_image_id'] == BASE and pin['upstream_head'] == HEAD, 'Pin drift')
require(file_sha(ROOT / 'upstream.patch') == PATCH_SHA == pin['patch_sha256'], 'Patch drift')
require(set(pin['before']) == set(pin['after']) == set(PATHS), 'Unexpected changed paths')
require(file_sha(ROOT.parent / 'source.lock.json') == pin['base_lock_sha256'], 'Base lock drift')
require(file_sha(ROOT / 'runtime.lock.json') == pin['runtime_lock_sha256'], 'Result lock drift')
require(pin['result_tree'] == '196f6e87a37480bca3b609a960e8969bcb9aa367', 'Result tree drift')
require(file_sha(ROOT / 'refresh.tar') == pin['refresh_sha256'], 'Source archive drift')
runtime = json.loads((ROOT / 'runtime.lock.json').read_text())
base = json.loads((ROOT.parent / 'source.lock.json').read_text())
require(runtime['sources']['b12x'] == base['sources']['b12x'], 'B12X drift')
require('lmcache_refresh' not in runtime, 'Unexpected LMCache refresh')
before = base['sources']['vllm']['files']
after = runtime['sources']['vllm']['files']
require({p for p in before.keys() | after.keys() if before.get(p) != after.get(p)} == set(PATHS), 'Not a narrow PR865 backport')
for name, digest in runtime['assets'].items():
    require(file_sha(ROOT.parent / name) == digest, f'Inherited gate asset drift: {name}')
files = ['Dockerfile', '.containerignore', 'install.py', 'prepare.py', 'preflight.py',
         'build.sh', 'gate.py', 'upstream.patch', 'backport.lock.json', 'runtime.lock.json', 'run-qwen.sh',
         'refresh.tar']
manifest = {name: file_sha(ROOT / name) for name in files}
# Parent gate and all its imported assets are build inputs too.
for path in ROOT.parent.rglob('*'):
    if path.is_file() and path.suffix in ('.py', '.sh') and not any(
        part in ('qsa865', 'qsa865-hcbase', 'qualification', 'receipts', 'build-receipts', '__pycache__', 'payload')
        for part in path.relative_to(ROOT.parent).parts
    ):
        manifest['../' + str(path.relative_to(ROOT.parent))] = file_sha(path)
if '--values' in sys.argv:
    for value in (file_sha(ROOT / 'backport.lock.json'), pin['runtime_lock_sha256'],
                  pin['cache_fingerprint'], pin['result_tree'], sha(json.dumps(manifest, sort_keys=True).encode()), pin['b12x_tree']):
        print(value)
elif '--manifest' in sys.argv:
    print(json.dumps(manifest, sort_keys=True, indent=2))
else:
    print('QSA865-LOCAL-INPUTS-PASS')
