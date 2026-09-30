"""Install only the frozen router fence; reject diagnostics and preserve natives."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path('/opt/ds41-router-release')
ROOT = Path('/opt/jovian-judgement')


def sha(path):
    return hashlib.sha256(str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError('Optimized Python not admitted')
    lock = json.loads((HERE / 'router-release.lock.json').read_text())
    if sha(Path('/opt/ds41-engram-repair/engram-repair.lock.json')) != lock['base_lock_sha256']:
        raise RuntimeError('Parent lock differs')
    for name, expected in lock['inputs'].items():
        if sha(HERE / name) != expected:
            raise RuntimeError('Recipe drift: ' + name)
    for component, files in lock['before'].items():
        for name, entry in files.items():
            if sha(ROOT / component / name) != entry['sha256']:
                raise RuntimeError('Base source drift: ' + component + '/' + name)
    def inventory():
        return {str(p): sha(p) for c in ('vllm', 'b12x') for p in (ROOT / c).rglob('*')
                if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts}
    before = inventory()
    parent_inventory = json.loads(Path('/opt/ds41-engram-repair/after-files.json').read_text())
    if before != parent_inventory:
        raise RuntimeError('Native/source inventory differs from the recorded clean parent')
    if not any(p.endswith('.so') for p in before) or any('claude_act_trace' in p for p in before):
        raise RuntimeError('Missing natives or diagnostic parent')
    target = ROOT / 'b12x' / lock['target']
    if sha(target) != lock['input_sha256']:
        raise RuntimeError('Router preimage mismatch')
    shutil.copyfile(HERE / 'router-prefill-release-before.py', target)
    compile(target.read_bytes(), str(target), 'exec')
    expected = dict(before)
    expected[str(target)] = lock['output_sha256']
    if inventory() != expected:
        raise RuntimeError('Undeclared source/native delta')
    for component, files in lock['after'].items():
        for name, entry in files.items():
            if sha(ROOT / component / name) != entry['sha256']:
                raise RuntimeError('Installed source drift: ' + name)
    (HERE / 'preserved-files.json').write_text(json.dumps(expected, sort_keys=True) + '\n')
    print('ROUTER-RELEASE-INSTALL-PASS', flush=True)


if __name__ == '__main__':
    main()
