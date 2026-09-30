#!/usr/bin/env python3
"""Refresh tracked sources only, preserving the declared native build products."""
import json
import os
import shutil
import tarfile
from pathlib import Path

from contracts import file_sha, load_lock, require, safe_path, sha

KIT = Path('/opt/karmic-main-build')
VLLM = Path('/opt/jovian-judgement/vllm')
B12X = Path('/opt/jovian-judgement/b12x')


def entry_sha(path, mode):
    if mode == '120000':
        require(path.is_symlink(), f'Missing symlink {path}')
        return sha(os.readlink(path).encode())
    require(path.is_file() and not path.is_symlink(), f'Missing regular source {path}')
    require(bool(path.stat().st_mode & 0o111) == (mode == '100755'), f'Source mode mismatch {path}')
    return file_sha(path)


def verify_files(root, files):
    for name, entry in files.items():
        path = safe_path(root, name)
        require(entry_sha(path, entry['mode']) == entry['sha256'], f'Source mismatch {path}')


def natives():
    result = {}
    for root in (VLLM, Path('/opt/venv/lib'), Path('/opt/nccl/lib')):
        for path in root.rglob('*'):
            if path.is_file() and ('.so' in path.name or path.suffix == '.a'):
                result[str(path)] = file_sha(path)
    require(any('_flashkda_C' in name for name in result), 'Missing FlashKDA native artifact')
    return result


def main():
    lock = load_lock(KIT)
    require(file_sha(KIT / 'source.lock.json') == os.environ['SOURCE_LOCK_SHA256'], 'Lock digest mismatch')
    original = json.loads(Path('/opt/karmic-build/vllm-payload.json').read_text())
    before = natives()
    products = original['build_products']
    for name, expected in products.items():
        require(file_sha(VLLM / name) == expected, f'Base build product changed {name}')
    for component, root in (('vllm', VLLM), ('b12x', B12X)):
        row = lock['sources'][component]
        verify_files(root, row['base_files'])
        archive = KIT / 'payload' / f'{component}.tar'
        require(file_sha(archive) == row['archive_sha256'], f'Archive changed: {component}')
        for name in row['base_files'].keys() - row['files'].keys():
            safe_path(root, name).unlink()
        with tarfile.open(archive) as bundle:
            members = {member.name: member for member in bundle.getmembers()}
            require(set(members) == set(row['files']), f'Archive path inventory mismatch {component}')
            for name, entry in row['files'].items():
                path = safe_path(root, name)
                member = members[name]
                if component == 'vllm' and name in products:
                    raise RuntimeError(f'Tracked update collides with preserved build product {name}')
                path.parent.mkdir(parents=True, exist_ok=True)
                if entry['mode'] == '120000':
                    require(member.issym(), f'Expected symlink {name}')
                    require(sha(member.linkname.encode()) == entry['sha256'], f'Symlink mismatch {name}')
                    if path.exists() or path.is_symlink():
                        path.unlink()
                    path.symlink_to(member.linkname)
                else:
                    require(member.isfile(), f'Expected regular member {name}')
                    data = bundle.extractfile(member).read()
                    require(sha(data) == entry['sha256'], f'Archive blob mismatch {name}')
                    require(not path.is_symlink(), f'Refusing source write through symlink {name}')
                    path.write_bytes(data)
                    path.chmod(0o755 if entry['mode'] == '100755' else 0o644)
        verify_files(root, row['files'])
        for path in root.rglob('__pycache__'):
            shutil.rmtree(path)
    # Do not leave a Git checkout claiming the old B12X revision after refresh.
    if (B12X / '.git').exists():
        shutil.rmtree(B12X / '.git')
    require(before == natives(), 'Native artifact bytes changed during Python refresh')
    for name, expected in products.items():
        require(file_sha(VLLM / name) == expected, f'Build product changed {name}')
    link = Path('/opt/venv/lib/python3.12/site-packages/triton_kernels')
    require(link.is_symlink() and link.resolve() == VLLM / 'vllm/third_party/triton_kernels', 'Triton symlink changed')
    require((link / 'matmul_ogs.py').is_file(), 'Triton product missing')
    original['tracked'] = {name: entry['sha256'] for name, entry in lock['sources']['vllm']['files'].items()}
    Path('/opt/karmic-build/vllm-payload.json').write_text(json.dumps(original, sort_keys=True, indent=2) + '\n')
    (KIT / 'native-reuse.json').write_text(json.dumps(before, sort_keys=True, indent=2) + '\n')
    print('TRACKED-REFRESH-NATIVE-IDENTITY-PASS', flush=True)


if __name__ == '__main__':
    main()
