"""Require every FlashInfer nvcc kernel requested by this gate to use its wheel."""
from pathlib import Path


def enforce_aot(core, cache):
    cache = Path(cache).resolve()
    used = []
    original = core.JitSpecNvcc.build_and_load

    def denied(*args, **kwargs):
        raise RuntimeError('FlashInfer compilation is forbidden in the runtime gate')

    def checked(spec, *args, **kwargs):
        path = Path(spec.aot_path).resolve()
        if not spec.is_aot or not path.is_relative_to(cache):
            raise RuntimeError(f'Missing rebuilt AOT kernel: {spec.name}: {path}')
        result = original(spec, *args, **kwargs)
        used.append({'name': spec.name, 'path': str(path)})
        return result

    core.JitSpecNvcc.build = denied
    core.JitSpecNvcc.build_and_load = checked
    return used
