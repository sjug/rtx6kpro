"""Check the imported compiler, not just the first distribution metadata entry."""
from pathlib import Path


def validate_cutlass(module, root=Path('/opt/venv')):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(root) or module.__version__ != '4.7.1':
        raise RuntimeError(f'Wrong imported CUTLASS: {path}, version={module.__version__}')
    return {'module': str(path), 'version': module.__version__}


if __name__ == '__main__':
    import cutlass
    print(validate_cutlass(cutlass))
