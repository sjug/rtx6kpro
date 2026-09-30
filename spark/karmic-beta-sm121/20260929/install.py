#!/usr/bin/env python3
"""Install the declared Karmic beta tracked deltas; preserve every native/build product."""
import json
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, '/opt/karmic-main-build')
from contracts import file_sha, require, safe_path, sha
from install import VLLM, B12X, verify_files, natives

KIT = Path('/opt/karmic-beta-refresh')


def main():
    require(__debug__, 'Optimized Python is not admitted')
    pin = json.loads((KIT / 'build.lock.json').read_text())
    require(file_sha(KIT / 'build.lock.json') == os.environ['KARMIC_BETA_BUILD_LOCK_SHA256'], 'Build lock mismatch')
    parent = Path('/opt/karmic-main-build/source.lock.json')
    require(file_sha(parent) == pin['base_lock_sha256'], 'Wrong base source lock')
    require(file_sha(KIT / 'runtime.lock.json') == pin['runtime_lock_sha256'], 'Wrong result source lock')
    require(file_sha(KIT / 'refresh.tar') == pin['refresh_sha256'], 'Wrong refresh payload')
    old, new = json.loads(parent.read_text()), json.loads((KIT / 'runtime.lock.json').read_text())
    rows = [(name, root, old['sources'][name]['files'], new['sources'][name]['files'])
            for name, root in (('vllm', VLLM), ('b12x', B12X))]
    for name, _, base, _ in rows:
        require(new['sources'][name]['base_files'] == base, f'Refresh was prepared from another base: {name}')
    before = natives()
    for _, root, base, _ in rows:
        verify_files(root, base)
    with tarfile.open(KIT / 'refresh.tar') as bundle:
        expected = {component + '/' + p for component, _, base, target in rows
                    for p in target if base.get(p) != target[p]}
        require(set(bundle.getnames()) == expected and len(bundle.getnames()) == len(expected),
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
    print('KARMIC-BETA-TRACKED-REFRESH-PASS', flush=True)


if __name__ == '__main__':
    main()
