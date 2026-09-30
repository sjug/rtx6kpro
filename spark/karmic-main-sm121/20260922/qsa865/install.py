#!/usr/bin/env python3
"""Refresh tracked sources while preserving the parent's compiled artifacts."""
import json
import os
import tarfile
import importlib.util
import sys
from pathlib import Path

KIT = Path('/opt/qsa865')
sys.path.insert(0, '/opt/karmic-main-build')
from contracts import file_sha, require, safe_path, sha
from install import VLLM, B12X, verify_files, natives


def main():
    pin = json.loads((KIT / 'backport.lock.json').read_text())
    require(file_sha(KIT / 'backport.lock.json') == os.environ['QSA865_LOCK_SHA256'], 'Backport lock changed')
    parent = Path('/opt/karmic-main-build/source.lock.json')
    require(file_sha(parent) == pin['base_lock_sha256'], 'Wrong base lock')
    old = json.loads(parent.read_text())
    verify_files(VLLM, old['sources']['vllm']['files'])
    verify_files(B12X, old['sources']['b12x']['files'])
    before = natives()
    require(file_sha(KIT / 'runtime.lock.json') == pin['runtime_lock_sha256'], 'Result lock changed')
    new = json.loads((KIT / 'runtime.lock.json').read_text())
    require(file_sha(KIT / 'refresh.tar') == pin['refresh_sha256'], 'Refresh archive changed')
    with tarfile.open(KIT / 'refresh.tar') as bundle:
        expected = {component + '/' + p for component in ('vllm', 'b12x')
                    for p, entry in new['sources'][component]['files'].items()
                    if old['sources'][component]['files'].get(p) != entry}
        require(set(bundle.getnames()) == expected, 'Unexpected refresh inventory')
        for component, root in (('vllm', VLLM), ('b12x', B12X)):
            old_files = old['sources'][component]['files']
            new_files = new['sources'][component]['files']
            for name in old_files.keys() - new_files.keys():
                safe_path(root, name).unlink()
            for name, entry in new_files.items():
                if old_files.get(name) == entry:
                    continue
                member = bundle.getmember(component + '/' + name)
                require(member.isfile(), f'Not a regular source: {name}')
                data = bundle.extractfile(member).read()
                require(sha(data) == entry['sha256'], f'Refresh bytes changed: {name}')
                dest = safe_path(root, name)
                require(not dest.is_symlink(), f'Source symlink: {name}')
                require(name in old_files or not dest.exists(), f'Untracked build product collision: {name}')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                dest.chmod(0o755 if entry['mode'] == '100755' else 0o644)
            for name in old_files.keys() | new_files.keys():
                if old_files.get(name) == new_files.get(name):
                    continue
                path = safe_path(root, name)
                for pyc in (path.parent / '__pycache__').glob(path.stem + '.*.pyc'):
                    pyc.unlink()
    lm_root = Path(importlib.util.find_spec('lmcache').origin).parent
    lm_path = lm_root / 'v1/multiprocess/mq.py'
    lm = new['lmcache_refresh']
    require(file_sha(lm_path) == lm['before_sha256'], 'Wrong installed LMCache source')
    require(file_sha(KIT / 'lmcache-mq.py') == lm['after_sha256'], 'LMCache refresh changed')
    lm_path.write_bytes((KIT / 'lmcache-mq.py').read_bytes())
    for pyc in (lm_path.parent / '__pycache__').glob('mq.*.pyc'):
        pyc.unlink()
    verify_files(VLLM, new['sources']['vllm']['files'])
    verify_files(B12X, new['sources']['b12x']['files'])
    require(before == natives(), 'Native artifacts changed')
    # The inherited native gate also checks tracked Python through this manifest.
    payload_path = Path('/opt/karmic-build/vllm-payload.json')
    original = json.loads(payload_path.read_text())
    original['tracked'] = {p: entry['sha256'] for p, entry in new['sources']['vllm']['files'].items()}
    payload_path.write_text(json.dumps(original, sort_keys=True, indent=2) + '\n')
    parent.write_bytes((KIT / 'runtime.lock.json').read_bytes())
    print('QSA865-TRACKED-REFRESH-PASS')


if __name__ == '__main__':
    main()
