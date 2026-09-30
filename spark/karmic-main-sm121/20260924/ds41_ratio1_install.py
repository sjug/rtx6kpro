"""Install only the ratio-1 extend BF16 pin over the exact precision release; preserve everything else.

Only RUN step of Dockerfile.ds41-ratio1. Refuses: recipe drift; a parent whose release lock is
not the recorded one; any release source differing from the lock's before-manifest; an image
inventory that is not the release installer's own preserved-files.json; missing natives; an
attention preimage that is not the release's; and any inventory change beyond attention.py.
The release's lock and install directory are left untouched.
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


def install(here=Path('/opt/ds41-ratio1'), root=Path('/opt/jovian-judgement'),
            parent=Path('/opt/ds41-precision-release')):
    if not __debug__:
        raise RuntimeError('Optimized Python not admitted')
    lock = json.loads((here / 'ds41-ratio1.lock.json').read_text())
    for name, expected in lock['inputs'].items():
        if sha(here / name) != expected:
            raise RuntimeError('Recipe drift: ' + name)
    if sha(parent / lock['base_lock']) != lock['base_lock_sha256']:
        raise RuntimeError('Parent release lock differs')
    for component, files in lock['before'].items():
        for name, entry in files.items():
            if sha(root / component / name) != entry['sha256']:
                raise RuntimeError('Base source drift: ' + component + '/' + name)
    before = inventory(root)
    if before != json.loads((parent / 'preserved-files.json').read_text()):
        raise RuntimeError('Native/source inventory differs from the recorded release')
    if not any(p.endswith('.so') for p in before):
        raise RuntimeError('Missing inherited native inventory')
    expected = dict(before)
    for path, entry in lock['targets'].items():
        target = root / 'vllm' / path
        if sha(target) != entry['input_sha256']:
            raise RuntimeError('Attention preimage differs from the release: ' + path)
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
