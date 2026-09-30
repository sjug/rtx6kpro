#!/usr/bin/env python3
"""Freeze stock NCCL without adding a checkout or trusting archive packaging."""
import io
import json
import posixpath
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / '20260922'))
from contracts import file_sha, git_tree, require

COMMIT = '73cf112295c33aee2b895f329f592f2a9b4b0f97'


def main():
    require(__debug__, 'Python optimization is forbidden')
    api = 'repos/NVIDIA/nccl'
    meta = json.loads(subprocess.check_output(['gh', 'api', f'{api}/git/commits/{COMMIT}']))
    data = subprocess.check_output(['gh', 'api', f'{api}/tarball/{COMMIT}'])
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for item in archive:
            if item.isdir():
                continue
            require(item.isfile() or item.issym(), f'Unsupported archive member {item.name}')
            name = item.name.split('/', 1)[1]
            require(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts,
                    f'Unsafe source path {name}')
            require(name not in files, f'Duplicate source path {name}')
            mode = '120000' if item.issym() else ('100755' if item.mode & 0o111 else '100644')
            files[name] = (mode, item.linkname.encode() if item.issym() else archive.extractfile(item).read())
    require(git_tree(files) == meta['tree']['sha'], 'Stock NCCL source tree mismatch')
    # A normalized payload is stable across GitHub archive regeneration.
    with tarfile.open(ROOT / 'nccl-source.tar', 'w') as archive:
        for name, (mode, contents) in sorted(files.items()):
            item = tarfile.TarInfo(name)
            if mode == '120000':
                target = contents.decode()
                resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
                require(not target.startswith('/') and resolved in files
                        and files[resolved][0] != '120000', f'Unsafe symlink: {name} -> {target}')
                require(not any(other.startswith(name + '/') for other in files),
                        f'Symlink used as source directory: {name}')
                item.type, item.linkname, item.mode = tarfile.SYMTYPE, target, 0o777
                archive.addfile(item)
            else:
                item.mode, item.size = int(mode[-3:], 8), len(contents)
                archive.addfile(item, io.BytesIO(contents))
    lock = {'repository': 'NVIDIA/nccl', 'tag': 'v2.30.7-1', 'commit': COMMIT,
            'tree': meta['tree']['sha'], 'payload_sha256': file_sha(ROOT / 'nccl-source.tar'),
            'dockerfile_sha256': file_sha(ROOT / 'Dockerfile.nccl'),
            'max_jobs': 20, 'target': 'sm_121', 'version_code': 23007,
            'provenance': 'User-approved stock substitution, not Luke patched binary'}
    (ROOT / 'nccl.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    print(json.dumps(lock, indent=2))


if __name__ == '__main__':
    main()
