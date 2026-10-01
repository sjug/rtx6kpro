#!/usr/bin/env python3
"""Download pinned source archives and ARM64 wheels, without creating Git checkouts."""
import hashlib
import json
import subprocess
import urllib.request
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(url, path):
    # Existing input files are reused only after their remote identity is verified below.
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.download')
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)


def api(endpoint):
    return json.loads(subprocess.check_output(['gh', 'api', endpoint], text=True))


def normalized_tree(tree):
    if tree.get('truncated'):
        raise RuntimeError('Truncated source inventory')
    return {'tree': sorted(({k: row[k] for k in ('path', 'mode', 'type', 'sha')}
                            for row in tree['tree'] if row['type'] in ('blob', 'commit')),
                           key=lambda row: row['path'])}


def restore_inputs(root=ROOT):
    """Restore exactly the frozen inputs; never resolve versions or rewrite the lock."""
    from unpack_sources import canonicalize_archive, artifact_sha
    lock = json.loads((root / 'inputs.lock.json').read_text())
    inputs = root / 'inputs'
    inputs.mkdir(exist_ok=True)
    for row in lock['wheels'].values():
        path = inputs / row['file']
        download(row['url'], path)
        if digest(path) != row['sha256']:
            raise RuntimeError(f'Locked wheel drift: {path}')
    for row in lock['sources'].values():
        tree_file = inputs / row['tree_file']
        if not tree_file.exists():
            tree = normalized_tree(api(f"repos/{row['repository']}/git/trees/{row['commit']}?recursive=1"))
            data = (json.dumps(tree, sort_keys=True, indent=2) + '\n').encode()
            if hashlib.sha256(data).hexdigest() != row['tree_sha256']:
                raise RuntimeError('Locked Git tree drift')
            tree_file.write_bytes(data)
        if digest(tree_file) != row['tree_sha256']:
            raise RuntimeError('Locked Git tree inventory drift')
        path = inputs / row['file']
        if not path.exists():
            transport = path.with_suffix('.transport')
            download(row['url'], transport)
            temporary = path.with_suffix('.canonical')
            canonicalize_archive(transport, temporary, row)
            if artifact_sha(temporary, row) != row['sha256']:
                raise RuntimeError('Locked canonical archive drift')
            temporary.replace(path)
            transport.unlink()
        if artifact_sha(path, row) != row['sha256']:
            raise RuntimeError('Locked canonical archive digest mismatch')
    print('LOCKED-INPUTS-RESTORED; locks unchanged', flush=True)


if __name__ == '__main__':
    if sys.argv[1:]:
        raise SystemExit('No arguments: this command restores the existing frozen lock only')
    restore_inputs()
