#!/usr/bin/env python3
"""Replace only declared compiler/FlashInfer/runtime distributions; audit all other native bytes.

FlashInfer 0.7.1 carries B12X: its wheel installs flashinfer/experimental/b12x and a top-level
`b12x` alias package. The kit's B12X root (/opt/jovian-judgement/b12x, first on PYTHONPATH) is
refreshed from that same subtree by install.py, so the base's b12x 1.3.0 wheel copy is removed,
the FlashInfer wheels are installed, and the root package is proven byte-identical to the wheel's
embedded implementation. The root keeps its egg-info, so `b12x` distribution metadata resolves.
"""
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
RUNTIME = ('nccl4py', 'nccl-extensions', 'nvidia-cudnn-frontend')   # FlashInfer 0.7.1 runtime dependencies
FLASHINFER = ('flashinfer-python', 'flashinfer-jit-cache', 'flashinfer-jit-cache-sm121a')
REPLACED = (*COMPILER, 'quack-kernels', 'torch-c-dlpack-ext', *FLASHINFER, *RUNTIME, 'b12x')
FLASHINFER_COMMIT = 'b4c35e6ec1d712f6b832679c08ce5719fae57cd6'
B12X_ROOT = Path('/opt/jovian-judgement/b12x')


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


DEPENDENCY_OVERLAY = (
    # (prefix-independent pattern, replacement): FlashInfer 0.7.1 replaces 0.6.18 and the
    # b12x extra resolves to FlashInfer's own extra, as vLLM 19f2c20e declares upstream.
    (r'(?m)^(Requires-Dist: |)flashinfer-python==0\.6\.18(\.post1)?(\s*$)', r'\1flashinfer-python==0.7.1\3'),
    (r'(?m)^(Requires-Dist: )b12x==1\.3\.0; extra == "b12x"(\s*$)', r'\1flashinfer-python[b12x]==0.7.1; extra == "b12x"\2'),
    (r'(?m)^b12x==1\.3\.0(\s*$)', r'flashinfer-python[b12x]==0.7.1\1'),
)


def update_compiler_metadata():
    """Keep the source-first vLLM/B12X install; update only their declared dependencies.

    vLLM's native wheel and version metadata are retained, as in the prior refresh.
    A requirement overlay prevents that inherited metadata requiring CUTLASS 4.6.2 and
    FlashInfer 0.6.18 after installing 4.7.1 and 0.7.1; the b12x extra follows upstream.
    No wheel version or source identity is invented.
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
                if name == 'vllm':
                    for find, replacement in DEPENDENCY_OVERLAY:
                        after = re.sub(find, replacement, after)
                    require('flashinfer-python==0.6.18' not in after and 'b12x==1.3.0' not in after,
                            f'Unrecognized FlashInfer requirement: {name} {path}')
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
    require(file_sha(ROOT / 'runtime-arm64.lock') == lock['runtime_lock_sha256_deps'], 'Runtime wheel lock drift')
    before, allowed_before = natives(), owned_paths()
    site = Path('/opt/venv/lib/python3.12/site-packages')
    require(B12X_ROOT.joinpath('b12x/__init__.py').is_file() and B12X_ROOT.joinpath('b12x.egg-info/PKG-INFO').is_file(),
            'B12X root or its source metadata is missing')
    require(metadata.version('b12x') == '1.3.0' and (site / 'b12x-1.3.0.dist-info').is_dir(), 'Unexpected base B12X wheel state')
    for name in ('nccl4py', 'nvidia-cudnn-frontend'):
        require(metadata.version(name) in ('0.3.1', '1.27.0'), f'Unexpected base {name} version: {metadata.version(name)}')
    subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '--offline', '--no-deps',
                    '--require-hashes', '--find-links', str(ROOT / 'inputs'),
                    '-r', str(ROOT / 'compiler-arm64.lock')], check=True)
    # FlashInfer 0.7.1 runtime dependencies (recipe 05eaafa linux-arm64/qwen38-runtime.lock pins):
    # nccl4py 0.3.1 -> 0.4.1 and nccl-extensions 0.1.0 share the `nccl` namespace (0.3.1 shipped
    # its own nccl/ep, which 0.4.1 drops, so the pair must be replaced together); cudnn-frontend 1.30.
    subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '--offline', '--no-deps',
                    '--require-hashes', '--reinstall', '--find-links', str(ROOT / 'inputs'),
                    '-r', str(ROOT / 'runtime-arm64.lock')], check=True)
    # The base installed the archived b12x 1.3.0 wheel beside the source root; the FlashInfer
    # wheel now owns the site-packages `b12x` package, so the stale wheel copy goes first.
    subprocess.run(['uv', 'pip', 'uninstall', '--python', sys.executable, 'b12x'], check=True)
    require(not (site / 'b12x').exists() and not (site / 'b12x-1.3.0.dist-info').exists(), 'Base B12X wheel copy remains')
    require(metadata.version('b12x') == '1.3.0', 'B12X source metadata (egg-info) must remain resolvable')
    # The separately checked NGC component supplies wheels through a temporary mount.
    wheels = ROOT / 'flashinfer-wheels'
    outputs = sorted(wheels.glob('*.whl'))
    require(len(outputs) == 3, 'Expected the FlashInfer Python, JIT-cache shim and sm121a provider wheels')
    (python_wheel,) = [p for p in outputs if p.name.startswith('flashinfer_python-')]
    (shim_wheel,) = [p for p in outputs if p.name.startswith('flashinfer_jit_cache-')]
    (provider_wheel,) = [p for p in outputs if p.name.startswith('flashinfer_jit_cache_sm121a-')]
    require(python_wheel.name.endswith('-py3-none-any.whl') and 'aarch64' in shim_wheel.name
            and 'aarch64' in provider_wheel.name, f'Unexpected FlashInfer wheel tags: {[p.name for p in outputs]}')
    with zipfile.ZipFile(python_wheel) as bundle:
        names = set(bundle.namelist())
    require('b12x/__init__.py' in names and 'flashinfer/experimental/b12x/_api.py' in names,
            'FlashInfer wheel must carry B12X and its b12x alias package')
    subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '--offline', '--no-deps',
                    '--reinstall', *map(str, outputs)], check=True)
    after, allowed_after = natives(), owned_paths()
    changes = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    require(changes <= allowed_before | allowed_after, f'Unexpected native replacement: {changes - allowed_before - allowed_after}')
    for name in COMPILER:
        require(metadata.version(name) == '4.7.1', f'Compiler mismatch: {name}')
    require(metadata.version('quack-kernels') == '0.6.5', 'QuACK mismatch')
    for name, version in (('nccl4py', '0.4.1'), ('nccl-extensions', '0.1.0'), ('nvidia-cudnn-frontend', '1.30.0'),
                          ('apache-tvm-ffi', '0.1.11')):
        require(metadata.version(name) == version, f'Runtime dependency mismatch: {name} {metadata.version(name)}')
    compiler_path = site / 'nvidia_cutlass_dsl/dsl_packages'
    require((compiler_path / 'cutlass/__init__.py').is_file(), 'Compiler import root missing')
    (site / '00-karmic-cutlass.pth').write_text(f'import sys; sys.path.insert(0, {str(compiler_path)!r})\n')
    subprocess.run([sys.executable, str(ROOT / 'verify_compiler.py')], check=True)
    metadata_changes = update_compiler_metadata()
    from verify_compiler import validate_compiler_metadata
    metadata_copies = validate_compiler_metadata()
    import flashinfer
    import flashinfer_jit_cache
    require(flashinfer.__git_commit__ == FLASHINFER_COMMIT, 'FlashInfer source mismatch')
    require(flashinfer_jit_cache.__git_version__ == flashinfer.__git_commit__, 'JIT source mismatch')
    require(metadata.version('flashinfer-python') == flashinfer.__version__ == metadata.version('flashinfer-jit-cache-sm121a'),
            'FlashInfer distribution versions disagree')
    providers = flashinfer_jit_cache.get_jit_cache_providers()
    require(len(providers) == 1 and providers[0].distribution == 'flashinfer-jit-cache-sm121a'
            and providers[0].cuda_architectures == frozenset({'sm121a'}), f'Unexpected JIT cache providers: {providers}')
    # The B12X root (first on PYTHONPATH, asserted by the foundation gates and both launchers)
    # must be the package the FlashInfer wheel embeds, byte for byte, and must win the import.
    import b12x
    # The root package executes _api.py as `b12x` (kit initializer), so __file__ is the root's _api.py.
    require(Path(b12x.__file__).resolve() == (B12X_ROOT / 'b12x/_api.py').resolve(), f'Wrong B12X import: {b12x.__file__}')
    require(all(callable(getattr(b12x, name, None)) or isinstance(getattr(b12x, name, None), type)
                for name in ('Plan', 'PreparationSession', 'list_ops', 'find_op')), 'B12X root lacks its public API')
    embedded = Path(flashinfer.__file__).resolve().parent / 'experimental/b12x'
    root_files = {p.relative_to(B12X_ROOT / 'b12x'): file_sha(p) for p in (B12X_ROOT / 'b12x').rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}
    embedded_files = {p.relative_to(embedded): file_sha(p) for p in embedded.rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts and p.relative_to(embedded).parts[0] != '_compat'}
    # Identity is the wheel payload: every embedded file is in the root with the same bytes, apart
    # from the initializer (the kit's, pinned in runtime.lock.json); the root's only extras are the
    # subtree's documentation files, which the wheel does not package.
    runtime_lock = json.loads((ROOT / 'runtime.lock.json').read_text())
    init = runtime_lock['sources']['b12x']['root_initializer']
    differing = {str(p) for p in embedded_files if root_files.get(p) != embedded_files[p]}
    require(differing == {'__init__.py'}, f'B12X root differs from the FlashInfer-embedded package: {sorted(differing)[:10]}')
    require(root_files[Path('__init__.py')] == init['sha256'], 'B12X root initializer is not the pinned kit initializer')
    extras = sorted(str(p) for p in root_files.keys() - embedded_files.keys())
    require(all(p.endswith('.md') for p in extras), f'Unexpected B12X root extras: {extras[:10]}')
    # One live package: the wheel's alias stub resolves to the root, and submodules reached
    # through flashinfer.experimental.b12x.* are the root's module objects.
    import flashinfer.experimental.b12x as aliased
    require(aliased is b12x, 'flashinfer.experimental.b12x is not the root package')
    from b12x._lib import runtime_control as canonical
    from flashinfer.experimental.b12x._lib import runtime_control as mirrored
    require(canonical is mirrored and Path(canonical.__file__).resolve().is_relative_to(B12X_ROOT),
            'Duplicate B12X module state across import paths')
    b12x_identity = {'root': str(B12X_ROOT / 'b12x'), 'embedded': str(embedded), 'files': len(embedded_files),
                     'root_extras': extras, 'initializer_sha256': init['sha256']}
    (ROOT / 'dependency-upgrade.json').write_text(json.dumps({
        'before': before, 'after': after, 'changed_native_paths': sorted(changes),
        'compiler_requirement_overlays': metadata_changes,
        'compiler_consumer_metadata': metadata_copies,
        'system_compiler_deviation': 'NGC system Python retains 4.6.2; serving and all runtime gates use /opt/venv and check imported CUTLASS 4.7.1',
        'flashinfer_build_environment': json.loads((wheels / 'build-environment.json').read_text()),
        'runtime_dependencies': {name: metadata.version(name) for name in RUNTIME},
        'b12x': b12x_identity, 'removed_distributions': ['b12x 1.3.0 wheel copy (site-packages); the source egg-info remains'],
        'wheels': {p.name: file_sha(p) for p in outputs}}, sort_keys=True, indent=2) + '\n')
    # The foundation gate audits this full post-upgrade native inventory.
    Path('/opt/karmic-main-build/native-reuse.json').write_text(json.dumps(after, sort_keys=True, indent=2) + '\n')
    print('CUTLASS-471-FLASHINFER-071-UPGRADE-PASS', flush=True)


if __name__ == '__main__':
    main()
