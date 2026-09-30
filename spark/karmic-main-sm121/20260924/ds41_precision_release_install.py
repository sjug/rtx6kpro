"""Install only the worker BF16 reduced-precision disable over the router parent; preserve natives.

Only RUN step of Dockerfile.ds41-precision-release. Refuses: recipe drift; a parent whose
router lock is not the recorded one; any router source that differs from the lock's
before-manifest; an image inventory that is not the router installer's own recorded
preserved-files.json; missing natives; capture instrumentation or a preexisting helper; a
worker that is not the router preimage; and any inventory change beyond gpu_worker.py plus
the one new helper module. Parent lock files and install directories are left untouched,
so the router's own GPU gates still read their original locks.
"""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()).hexdigest()


def inventory(root):
    return {str(p): sha(p) for c in ('vllm', 'b12x') for p in (root / c).rglob('*')
            if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts}


def install(here=Path('/opt/ds41-precision-release'), root=Path('/opt/jovian-judgement'),
            parent=Path('/opt/ds41-router-release')):
    if not __debug__:
        raise RuntimeError('Optimized Python not admitted')
    lock = json.loads((here / 'ds41-precision-release.lock.json').read_text())
    for name, expected in lock['inputs'].items():
        if sha(here / name) != expected:
            raise RuntimeError('Recipe drift: ' + name)
    if sha(parent / lock['base_lock']) != lock['base_lock_sha256']:
        raise RuntimeError('Parent router lock differs')
    for component, files in lock['before'].items():
        for name, entry in files.items():
            if sha(root / component / name) != entry['sha256']:
                raise RuntimeError('Base source drift: ' + component + '/' + name)
    before = inventory(root)
    if before != json.loads((parent / 'preserved-files.json').read_text()):
        raise RuntimeError('Native/source inventory differs from the recorded router parent')
    if not any(p.endswith('.so') for p in before):
        raise RuntimeError('Missing inherited native inventory')
    for path in lock['capture_paths_absent']:
        if (root / 'vllm' / path).exists():
            raise RuntimeError('Capture instrumentation present in the parent: ' + path)
    expected = dict(before)
    for path, entry in lock['targets'].items():
        target = root / 'vllm' / path
        if entry['input_sha256'] is None:
            if target.exists() or target.is_symlink():
                raise RuntimeError('Unexpected preexisting file: ' + path)
        elif sha(target) != entry['input_sha256']:
            raise RuntimeError('Worker preimage differs from the router parent: ' + path)
    for path, entry in lock['targets'].items():
        target = root / 'vllm' / path
        shutil.copyfile(here / entry['source'], target)
        compile(target.read_bytes(), str(target), 'exec')
        expected[str(target)] = entry['output_sha256']
    after = inventory(root)
    if after != expected:
        changed = sorted(set(after.items()) ^ set(expected.items()))
        raise RuntimeError('Undeclared source/native delta: %r' % changed[:6])
    for component, files in lock['after'].items():
        for name, entry in files.items():
            if sha(root / component / name) != entry['sha256']:
                raise RuntimeError('Installed source drift: ' + component + '/' + name)
    (here / 'before-files.json').write_text(json.dumps(before, indent=2, sort_keys=True) + '\n')
    (here / 'preserved-files.json').write_text(json.dumps(after, indent=2, sort_keys=True) + '\n')
    print(lock['install_pass'], flush=True)


if __name__ == '__main__':
    install()
