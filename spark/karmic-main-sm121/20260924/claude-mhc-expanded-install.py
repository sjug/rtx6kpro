"""Install the expanded-mHC capture layer inside an image build; prove nothing else changed.

Only RUN step of Dockerfile.claude-mhc-expanded over the precision release image. Refuses drifted inputs,
a base whose release lock or full vLLM/B12X inventory is not the release installer's own record
(/opt/ds41-precision-release/preserved-files.json), a model preimage that is not the release model, a
preexisting helper, and any change beyond the model file plus the one helper module.

    install(here, opt, release)   paths are parameters so the local tests run it on a temporary tree
"""
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    data = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def inventory(opt):
    return {str(p): digest(p) for component in ('vllm', 'b12x') for p in (opt / component).rglob('*')
            if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts}


def install(here=Path('/opt/ds41-mhc-expanded'), opt=Path('/opt/jovian-judgement'),
            release=Path('/opt/ds41-precision-release'), require_native=True):
    if not __debug__:
        raise RuntimeError('Optimized Python is not admitted')
    lock = json.loads((here / 'claude-mhc-expanded.lock.json').read_text())
    for name, expected in lock['inputs'].items():
        if digest(here / name) != expected:
            raise RuntimeError('Diagnostic input drift: ' + name)
    if digest(release / lock['base_lock']) != lock['base_lock_sha256']:
        raise RuntimeError('Base is not the precision release: release lock differs')
    for target, entry in lock['targets'].items():
        path = opt / target
        if entry['input_sha256'] is None:
            if path.exists() or path.is_symlink():
                raise RuntimeError('Unexpected preexisting diagnostic file: ' + target)
        elif digest(path) != entry['input_sha256']:
            raise RuntimeError('Base model preimage differs from the release: ' + target)
    before = inventory(opt)
    if before != json.loads((release / 'preserved-files.json').read_text()):
        raise RuntimeError('Base inventory differs from the recorded release installation')
    if require_native and not any(name.endswith('.so') for name in before):
        raise RuntimeError('Missing inherited native inventory')
    expected = dict(before)
    for target, entry in lock['targets'].items():
        path = opt / target
        shutil.copyfile(here / entry['source'], path)
        compile(path.read_bytes(), str(path), 'exec')
        expected[str(path)] = entry['output_sha256']
    after = inventory(opt)
    if after != expected:
        changed = sorted(set(after.items()) ^ set(expected.items()))
        raise RuntimeError('Change was not confined to the diagnostic layer: %r' % changed[:6])
    (here / 'before-files.json').write_text(json.dumps(before, indent=2, sort_keys=True) + '\n')
    (here / 'after-files.json').write_text(json.dumps(after, indent=2, sort_keys=True) + '\n')
    print(lock['install_pass_marker'], flush=True)


if __name__ == '__main__':
    install()
