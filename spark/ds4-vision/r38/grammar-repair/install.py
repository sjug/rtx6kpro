#!/usr/bin/env python3
"""Apply the exact repair and prove all other source/build products unchanged."""
import hashlib
import json
from pathlib import Path
import subprocess

KIT = Path(__file__).resolve().parent
ROOT = Path('/opt/jovian-judgement/vllm')


def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def inventory():
    result = {}
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if '.git' in relative.parts or '__pycache__' in relative.parts:
            continue
        if path.is_symlink():
            result[str(relative)] = 'symlink:' + str(path.readlink())
        elif path.is_file():
            result[str(relative)] = digest(path)
    return result


lock = json.loads((KIT / 'source.lock.json').read_text())
if digest(KIT / 'upstream.patch') != lock['patch_sha256']:
    raise RuntimeError('Patch digest mismatch')
before = inventory()
for name, values in lock['files'].items():
    if before.get(name) != values['before']:
        raise RuntimeError(f'Unexpected input bytes: {name}')
subprocess.run(['git', '-C', str(ROOT), 'apply', '--check', str(KIT / 'upstream.patch')], check=True)
subprocess.run(['git', '-C', str(ROOT), 'apply', str(KIT / 'upstream.patch')], check=True)
after = inventory()
changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
if changed != set(lock['files']):
    raise RuntimeError(f'Unexpected changed paths: {changed}')
for name, values in lock['files'].items():
    if after[name] != values['after']:
        raise RuntimeError(f'Unexpected output bytes: {name}')
(KIT / 'unchanged-products.json').write_text(json.dumps({
    'before_manifest_sha256': hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
    'after_manifest_sha256': hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
    'changed': sorted(changed), 'files_checked': len(before),
}, indent=2) + '\n')
print('EXACT-SEVEN-FILE-REFRESH-PASS', flush=True)
