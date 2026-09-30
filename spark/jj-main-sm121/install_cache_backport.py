#!/usr/bin/env python3
"""Apply the two-file backport and preserve the complete native payload manifest."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/opt/jovian-judgement/vllm')
BUILD = Path('/opt/jj-main-build')
KIT = BUILD / 'cache-agreement'

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    lock = json.loads((KIT / 'cache-agreement.lock.json').read_text())
    patch = KIT / 'cache-agreement.patch'
    if digest(patch) != lock['patch_sha256']:
        raise RuntimeError('Backport patch digest mismatch')
    manifest_path = BUILD / 'vllm-payload.json'
    manifest = json.loads(manifest_path.read_text())
    for category in ('tracked', 'build_products'):
        for name, expected in manifest[category].items():
            if digest(ROOT / name) != expected:
                raise RuntimeError(f'Base payload changed: {name}')
    expected_paths = {'vllm/v1/worker/b12x_startup.py', 'tests/v1/executor/test_b12x_startup.py'}
    if set(lock['files']) != expected_paths:
        raise RuntimeError('Unexpected backport surface')
    for name, record in lock['files'].items():
        if digest(ROOT / name) != record['before_sha256']:
            raise RuntimeError(f'Backport input differs: {name}')
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=ROOT, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=ROOT, check=True)
    (KIT / 'base-vllm-payload.json').write_bytes(manifest_path.read_bytes())
    for name, record in lock['files'].items():
        if digest(ROOT / name) != record['after_sha256']:
            raise RuntimeError(f'Backport result differs: {name}')
        if name not in manifest['tracked']:
            raise RuntimeError(f'Expected tracked source missing: {name}')
        manifest['tracked'][name] = record['after_sha256']
    for category in ('tracked', 'build_products'):
        for name, expected in manifest[category].items():
            if digest(ROOT / name) != expected:
                raise RuntimeError(f'Unexpected post-patch change: {name}')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('CACHE-AGREEMENT-PATCH-PASS native artifacts unchanged', flush=True)

if __name__ == '__main__':
    main()
