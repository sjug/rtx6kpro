#!/usr/bin/env python3
"""Install only declared tracked deltas; preserve all native/build products."""
import importlib.util
import json
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, '/opt/karmic-main-build')
from contracts import file_sha, require, safe_path, sha
from install import VLLM, B12X, verify_files, natives as inherited_natives

KIT = Path('/opt/ds41-main-refresh')


def natives():
    result = inherited_natives()
    library = Path('/opt/nccl-2.30.7/lib/libnccl.so.2.30.7')
    result[str(library)] = file_sha(library)
    return result


def main():
    require(__debug__, 'Optimized Python is not admitted')
    pin = json.loads((KIT / 'build.lock.json').read_text())
    require(file_sha(KIT / 'build.lock.json') == os.environ['DS41_BUILD_LOCK_SHA256'], 'Build lock mismatch')
    parent = Path('/opt/karmic-main-build/source.lock.json')
    require(file_sha(parent) == pin['base_lock_sha256'], 'Wrong base source lock')
    require(file_sha(KIT / 'runtime.lock.json') == pin['runtime_lock_sha256'], 'Wrong result source lock')
    require(file_sha(KIT / 'refresh.tar') == pin['refresh_sha256'], 'Wrong refresh payload')
    require(file_sha('/opt/nccl-2.30.7/lib/libnccl.so.2.30.7') == pin['nccl_component']['library_sha256'],
            'Stock NCCL object differs from locked builder output')
    require(file_sha('/opt/nccl-2.30.7/nccl.lock.json') == pin['nccl_lock_sha256'],
            'Stock NCCL component source lock differs')
    old, new = json.loads(parent.read_text()), json.loads((KIT / 'runtime.lock.json').read_text())
    lm_root = Path(importlib.util.find_spec('lmcache').origin).parent
    rows = [(name, root, old['sources'][name]['files'], new['sources'][name]['files'])
            for name, root in (('vllm', VLLM), ('b12x', B12X))]
    rows.append(('lmcache', lm_root, new['lmcache_refresh']['base_files'], new['lmcache_refresh']['files']))
    before = natives()
    for _, root, base, _ in rows:
        verify_files(root, base)
    with tarfile.open(KIT / 'refresh.tar') as bundle:
        expected = {component + '/' + p for component, _, base, target in rows
                    for p in target if base.get(p) != target[p]}
        require(len(bundle.getnames()) == len(expected) and set(bundle.getnames()) == expected,
                'Refresh member inventory mismatch')
        for component, root, base, target in rows:
            for name in base.keys() - target.keys():
                safe_path(root, name).unlink()
            for name, entry in target.items():
                if base.get(name) == entry:
                    continue
                member = bundle.getmember(component + '/' + name)
                require(member.isfile(), f'Non-regular replacement: {name}')
                data = bundle.extractfile(member).read()
                require(sha(data) == entry['sha256'], f'Wrong replacement bytes: {name}')
                dest = safe_path(root, name)
                require(not dest.is_symlink(), f'Symlink replacement: {name}')
                require(name in base or not dest.exists(), f'Untracked product collision: {name}')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                dest.chmod(0o755 if entry['mode'] == '100755' else 0o644)
            for name in base.keys() | target.keys():
                if base.get(name) != target.get(name):
                    path = safe_path(root, name)
                    for pyc in (path.parent / '__pycache__').glob(path.stem + '.*.pyc'):
                        pyc.unlink()
            verify_files(root, target)
    require(natives() == before, 'Native artifacts changed')
    payload_path = Path('/opt/karmic-build/vllm-payload.json')
    payload = json.loads(payload_path.read_text())
    payload['tracked'] = {p: e['sha256'] for p, e in new['sources']['vllm']['files'].items()}
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    parent.write_bytes((KIT / 'runtime.lock.json').read_bytes())
    (KIT / 'native-reuse.json').write_text(json.dumps(before, sort_keys=True, indent=2) + '\n')
    print('DS41-MAIN-TRACKED-REFRESH-PASS', flush=True)


if __name__ == '__main__':
    main()
