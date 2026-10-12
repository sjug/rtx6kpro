#!/usr/bin/env python3
"""Verify pinned Git blob inventories, then unpack source archives without Git metadata."""
import hashlib
import gzip
import json
import tarfile
from pathlib import Path


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifact_sha(path, row):
    if row.get('digest_kind') == 'uncompressed-tar-sha256':
        with gzip.open(path, 'rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    return sha(path)


def inspect_archive(path, row, verify_digest=True):
    if verify_digest:
        require(artifact_sha(path, row) == row['sha256'], f'Archive digest mismatch: {path}')
    tree_file = path.parent / row['tree_file']
    require(sha(tree_file) == row['tree_sha256'], 'Source tree inventory drift')
    tree = json.loads(tree_file.read_text())
    expected = {r['path']: r for r in tree['tree'] if r['type'] == 'blob'}
    result = {}
    with tarfile.open(path) as bundle:
        for member in bundle.getmembers():
            if member.isdir():
                continue
            if row.get('format') == 'canonical-git-blobs-v1':
                name = member.name
            else:
                parts = member.name.split('/', 1)
                if len(parts) != 2 or not parts[1]:
                    continue
                name = parts[1]
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, f'Unsafe member: {name}')
            require(name not in result and name in expected, f'Unexpected member: {name}')
            require(member.isfile() or member.issym(), f'Unsupported member: {name}')
            data = member.linkname.encode() if member.issym() else bundle.extractfile(member).read()
            mode = '120000' if member.issym() else ('100755' if member.mode & 0o111 else '100644')
            oid = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            require((mode, oid) == (expected[name]['mode'], expected[name]['sha']), f'Git blob mismatch: {name}')
            if member.issym():
                require(not Path(member.linkname).is_absolute(), f'Absolute link: {name}')
            result[name] = (mode, data)
    require(set(result) == set(expected), f'Archive inventory mismatch: {path}')
    return result


def canonicalize_archive(source, destination, row):
    """Git blobs define identity; tar transport timestamps/prefixes do not."""
    import io
    transport = dict(row)
    transport.pop('format', None)
    files = inspect_archive(source, transport, verify_digest=False)
    with destination.open('wb') as output, gzip.GzipFile(fileobj=output, filename='', mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w', format=tarfile.GNU_FORMAT) as bundle:
            for name, (mode, data) in sorted(files.items()):
                member = tarfile.TarInfo(name)
                member.mode = 0o755 if mode == '100755' else 0o644
                if mode == '120000':
                    member.type, member.linkname = tarfile.SYMTYPE, data.decode()
                    bundle.addfile(member)
                else:
                    member.size = len(data)
                    bundle.addfile(member, io.BytesIO(data))


def main():
    root = Path(__file__).resolve().parent
    lock = json.loads((root / 'inputs.lock.json').read_text())
    destination = root / 'build/flashinfer'
    require(not destination.exists(), 'Refusing to overwrite source build tree')
    for name, row in lock['sources'].items():
        files = inspect_archive(root / 'inputs' / row['file'], row)
        base = destination / row['destination']
        for relative, (mode, data) in files.items():
            path = base / relative
            require(path.parent.resolve().is_relative_to(destination.resolve()), f'Escaping source: {relative}')
            path.parent.mkdir(parents=True, exist_ok=True)
            require(not path.exists() and not path.is_symlink(), f'Source collision: {relative}')
            if mode == '120000':
                require((path.parent / data.decode()).resolve().is_relative_to(destination.resolve()),
                        f'Escaping symlink: {relative}')
                path.symlink_to(data.decode())
            else:
                path.write_bytes(data)
                path.chmod(0o755 if mode == '100755' else 0o644)
        print(f'SOURCE-ARCHIVE-PASS {name}', flush=True)


if __name__ == '__main__':
    main()
