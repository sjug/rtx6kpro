#!/usr/bin/env python3
"""Replace only declared compiler/FlashInfer distributions; audit all other native bytes."""
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, '/opt/karmic-main-build')
from contracts import file_sha, require
from install import natives

ROOT = Path('/opt/karmic-beta-refresh')
COMPILER = ('nvidia-cutlass-dsl', 'nvidia-cutlass-dsl-libs-base', 'nvidia-cutlass-dsl-libs-core',
            'nvidia-cutlass-dsl-libs-cu12', 'nvidia-cutlass-dsl-libs-cu13')
REPLACED = (*COMPILER, 'quack-kernels', 'torch-c-dlpack-ext', 'flashinfer-python', 'flashinfer-jit-cache')


def owned_paths():
    paths = set()
    for name in REPLACED:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        for path in dist.files or ():
            paths.add(os.path.abspath(dist.locate_file(path)))
    return paths


def update_compiler_metadata():
    """Keep the source-first vLLM/B12X install; update only their compiler requirements.

    vLLM's native wheel and version metadata are retained, as in the prior refresh.
    A compiler-only requirement overlay prevents that inherited metadata requiring
    4.6.2 after installing 4.7.1. No wheel version or source identity is invented.
    """
    import base64
    import csv
    import hashlib
    import io
    import re
    changes = {}
    updated = set()
    for name in ('vllm', 'b12x'):
        distributions = list(metadata.distributions(name=name))
        require(distributions, f'Cannot locate {name} distribution')
        for dist in distributions:
            files = list(dist.files or ())
            candidates = [p for p in files if str(p).endswith(('.dist-info/METADATA', '.egg-info/PKG-INFO'))]
            require(len(candidates) == 1, f'Cannot locate {name} metadata')
            relative = candidates[0]
            path = Path(dist.locate_file(relative))
            updated.add(path.resolve())
            targets = [(relative, path, r'Requires-Dist: ')]
            if path.parent.name.endswith('.egg-info'):
                requires = path.parent / 'requires.txt'
                require(requires.is_file(), f'Missing source requirements: {name}')
                targets.append((None, requires, ''))
            for relative, path, prefix in targets:
                before = path.read_text()
                pattern = r'(?m)^(' + prefix + r'nvidia-cutlass-dsl[^\n]*)4\.6\.2'
                after = re.sub(pattern, lambda m: m.group(1) + '4.7.1', before)
                # Preserve versions, extra sections and all unrelated requirements.
                require('4.6.2' not in '\n'.join(line for line in after.splitlines()
                                                if line.startswith(prefix + 'nvidia-cutlass-dsl')),
                        f'Unrecognized compiler requirement: {name} {path}')
                if after == before:
                    continue
                data = after.encode()
                if relative is not None and path.parent.name.endswith('.dist-info'):
                    record = path.parent / 'RECORD'
                    require(record.is_file(), f'Missing distribution RECORD: {name}')
                    rows = list(csv.reader(io.StringIO(record.read_text())))
                    matched = 0
                    for row in rows:
                        if row[0] == str(relative):
                            row[1] = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
                            row[2] = str(len(data))
                            matched += 1
                    require(matched == 1, f'Missing metadata RECORD row: {name}')
                    stream = io.StringIO()
                    csv.writer(stream, lineterminator='\n').writerows(rows)
                    record.write_text(stream.getvalue())
                path.write_bytes(data)
                changes.setdefault(name, []).append({
                    'path': str(path),
                    'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
                    'after_sha256': hashlib.sha256(data).hexdigest()})
    from verify_compiler import validate_compiler_metadata
    validate_compiler_metadata(updated)
    return changes


def main():
    lock = json.loads((ROOT / 'build.lock.json').read_text())
    require(file_sha(ROOT / 'inputs.lock.json') == lock['inputs_lock_sha256'], 'Dependency lock drift')
    require(file_sha(ROOT / 'compiler-arm64.lock') == lock['compiler_lock_sha256'], 'Wheel lock drift')
    before, allowed_before = natives(), owned_paths()
    subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '--offline', '--no-deps',
                    '--require-hashes', '--find-links', str(ROOT / 'inputs'),
                    '-r', str(ROOT / 'compiler-arm64.lock')], check=True)
    # The separately checked NGC component supplies wheels through a temporary mount.
    wheels = ROOT / 'flashinfer-wheels'
    outputs = sorted(wheels.glob('*.whl'))
    require(len(outputs) == 2 and sum('aarch64.whl' in p.name for p in outputs) == 1,
            'Expected Python and ARM64 FlashInfer JIT wheels')
    for wheel in outputs:
        with zipfile.ZipFile(wheel) as bundle:
            require(not any(name.startswith(('b12x/', 'flashinfer/b12x/')) for name in bundle.namelist()),
                    'FlashInfer wheel must not shadow our pinned B12X source')
    subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '--offline', '--no-deps',
                    '--reinstall', *map(str, outputs)], check=True)
    after, allowed_after = natives(), owned_paths()
    changes = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    require(changes <= allowed_before | allowed_after, f'Unexpected native replacement: {changes - allowed_before - allowed_after}')
    for name in COMPILER:
        require(metadata.version(name) == '4.7.1', f'Compiler mismatch: {name}')
    require(metadata.version('quack-kernels') == '0.6.5', 'QuACK mismatch')
    site = Path('/opt/venv/lib/python3.12/site-packages')
    compiler_path = site / 'nvidia_cutlass_dsl/dsl_packages'
    require((compiler_path / 'cutlass/__init__.py').is_file(), 'Compiler import root missing')
    (site / '00-karmic-cutlass.pth').write_text(f'import sys; sys.path.insert(0, {str(compiler_path)!r})\n')
    subprocess.run([sys.executable, str(ROOT / 'verify_compiler.py')], check=True)
    metadata_changes = update_compiler_metadata()
    from verify_compiler import validate_compiler_metadata
    metadata_copies = validate_compiler_metadata()
    import flashinfer
    import flashinfer_jit_cache
    require(flashinfer.__git_commit__ == 'dbd6238c6655b98195fdf77f04bba6facf5a38a4', 'FlashInfer source mismatch')
    require(flashinfer_jit_cache.__git_version__ == flashinfer.__git_commit__, 'JIT source mismatch')
    (ROOT / 'dependency-upgrade.json').write_text(json.dumps({
        'before': before, 'after': after, 'changed_native_paths': sorted(changes),
        'compiler_requirement_overlays': metadata_changes,
        'compiler_consumer_metadata': metadata_copies,
        'system_compiler_deviation': 'NGC system Python retains 4.6.2; serving and all runtime gates use /opt/venv and check imported CUTLASS 4.7.1',
        'flashinfer_build_environment': json.loads((wheels / 'build-environment.json').read_text()),
        'wheels': {p.name: file_sha(p) for p in outputs}}, sort_keys=True, indent=2) + '\n')
    # The foundation gate audits this full post-upgrade native inventory.
    Path('/opt/karmic-main-build/native-reuse.json').write_text(json.dumps(after, sort_keys=True, indent=2) + '\n')
    print('CUTLASS-471-FLASHINFER-UPGRADE-PASS', flush=True)


if __name__ == '__main__':
    main()
