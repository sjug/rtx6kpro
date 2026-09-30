"""Verify every tracked base file, install exactly eleven targets, preserve natives."""
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    data = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError('Optimized Python is not admitted')
    here = Path('/opt/ds41-engram-repair')
    root = Path('/opt/jovian-judgement')
    lock = json.loads((here / 'engram-repair.lock.json').read_text())
    for name, expected in lock['inputs'].items():
        if digest(here / name) != expected:
            raise RuntimeError('Recipe input drift: ' + name)
    for component, files in lock['before'].items():
        for name, entry in files.items():
            if digest(root / component / name) != entry['sha256']:
                raise RuntimeError('Base source drift: ' + component + '/' + name)

    def inventory():
        return {str(p): digest(p) for component in ('vllm', 'b12x')
                for p in (root / component).rglob('*')
                if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts}

    before = inventory()
    if not any(p.endswith('.so') for p in before):
        raise RuntimeError('Native layer absent')
    expected = dict(before)
    for name, target in lock['targets'].items():
        path = root / name
        if digest(path) != target['input_sha256']:
            raise RuntimeError('Target preimage differs: ' + name)
        shutil.copyfile(here / target['source'], path)
        if path.suffix == '.py':
            compile(path.read_bytes(), str(path), 'exec')
        expected[str(path)] = target['output_sha256']
    if inventory() != expected:
        raise RuntimeError('Undeclared source/native change')
    for component, files in lock['after'].items():
        for name, entry in files.items():
            if digest(root / component / name) != entry['sha256']:
                raise RuntimeError('Installed source drift: ' + component + '/' + name)
    (here / 'before-files.json').write_text(json.dumps(before, sort_keys=True) + '\n')
    (here / 'after-files.json').write_text(json.dumps(expected, sort_keys=True) + '\n')
    print('ENGRAM-REPAIR-INSTALL-PASS', flush=True)


if __name__ == '__main__':
    main()
