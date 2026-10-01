"""Check the imported compiler, not just the first distribution metadata entry."""
from pathlib import Path
import importlib.metadata as metadata


def compiler_metadata_path(dist):
    candidates = [p for p in dist.files or ()
                  if str(p).endswith(('.dist-info/METADATA', '.egg-info/PKG-INFO'))]
    if len(candidates) != 1:
        raise RuntimeError('Cannot locate compiler consumer metadata')
    return Path(dist.locate_file(candidates[0])).resolve()


def validate_compiler_metadata(updated=None):
    """Check every visible consumer copy, including the one serving resolves first."""
    audit = {}
    for name in ('vllm', 'b12x'):
        paths = {compiler_metadata_path(dist) for dist in metadata.distributions(name=name)}
        selected = compiler_metadata_path(metadata.distribution(name))
        if not paths or selected not in paths or (updated is not None and selected not in updated):
            raise RuntimeError(f'Uncovered selected compiler consumer: {name} {selected}')
        for path in paths:
            targets = [(path, 'Requires-Dist: ')]
            if path.parent.name.endswith('.egg-info'):
                targets.append((path.parent / 'requires.txt', ''))
            for target, prefix in targets:
                lines = [line for line in target.read_text().splitlines()
                         if line.startswith(prefix + 'nvidia-cutlass-dsl')]
                if (not lines and target.name != 'PKG-INFO') or any(
                        '4.7.1' not in line or '4.6.2' in line for line in lines):
                    raise RuntimeError(f'Wrong compiler requirement: {name} {target}: {lines}')
        audit[name] = {'selected': str(selected), 'copies': sorted(map(str, paths))}
    return audit


def validate_cutlass(module, root=Path('/opt/venv')):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(root) or module.__version__ != '4.7.1':
        raise RuntimeError(f'Wrong imported CUTLASS: {path}, version={module.__version__}')
    return {'module': str(path), 'version': module.__version__}


if __name__ == '__main__':
    import cutlass
    print(validate_cutlass(cutlass))
