#!/usr/bin/env python3
"""Freeze the October 8 Karmic beta (20261008.2) as a tracked-source refresh of the SM121 base.

Base image 1a7a8acf carries vLLM e77be225 (SM121 overlay plus PR865) and B12X 10a553ef.
vLLM 19f2c20e descends from the base. B12X no longer has a repository of its own: since
FlashInfer 0.7.1 it lives at flashinfer/experimental/b12x and the FlashInfer wheel installs
the `b12x` import package. This kit keeps the base's B12X root (/opt/jovian-judgement/b12x,
first on PYTHONPATH, asserted by the foundation gates and both launchers) and refreshes its
package from the frozen FlashInfer archive's experimental subtree, so the root and the wheel's
copy are byte-identical. vLLM's native inputs are identical apart from the deferred Rust DSML
parser and two dependency-metadata files (requirements/cuda.txt, setup.py: FlashInfer 0.7.1
replaces 0.6.18 and the b12x extra), which the compiled wheel never reads; the installed
metadata is overlaid at assembly. B12X's C sources changed (RoCE proxy ABI 5, loader
reformatting): the proxy is rebuilt and gated in-image, the loader compiles at first use as
before. Compiler libraries, FlashInfer and its three runtime dependencies are replaced in a
separate, audited offline step. LMCache stays at the base 413ac987 and remains disabled.
"""
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from unpack_sources import artifact_sha, inspect_archive

ROOT = Path(__file__).resolve().parent
# SM121 foundation tree: base image 1a7a8acf, its source lock, shared contracts and gates.
FOUNDATION = ROOT.parents[1] / 'karmic-main-sm121/20260922'
sys.path.insert(0, str(FOUNDATION))
from contracts import file_sha, git_tree, manifest, overlay, require, sha

spec = importlib.util.spec_from_file_location('previous_prepare', FOUNDATION / 'qsa865/prepare.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

BASE = '1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc'
VLLM = '19f2c20ed4d64ae22bd009ceacadd85e2a81e354'
# B12X source: the FlashInfer commit whose experimental/b12x subtree is the package.
FLASHINFER = 'b4c35e6ec1d712f6b832679c08ce5719fae57cd6'
B12X_SUBTREE = 'flashinfer/experimental/b12x/'
B12X_EXCLUDED = ('_compat/',)   # the wheel-only `b12x` alias package; the root is the real package
PUBLICATION = 'karmic-kraken-beta-20261008.2'
PUBLISHED_IMAGE = 'ghcr.io/local-inference-lab/vllm@sha256:cf8e87217901a88a258beeb545a52e893323671735b926cd25bd3879e48e965e'
# Foundation provenance copied from the base lock; kept apart from the beta publication.
INHERITED = ('base_build_lock_sha256', 'native_inputs_sha256', 'native_metadata', 'published_image',
             'recipe_commit', 'lmcache_refresh')
RECIPE = '05eaafa9473068844c0342f93286ed9759443251'
NATIVE = ('csrc', 'cmake', 'rust', 'requirements', 'CMakeLists.txt', 'setup.py',
          'pyproject.toml', '.gitmodules', 'rust-toolchain.toml', 'build_rust.sh', 'MANIFEST.in')
# Opt-in Rust frontend (VLLM_USE_RUST_FRONTEND=1) only; the compiled extension is not rebuilt.
RUST_DEFERRED = (
    'rust/src/parser/src/tool/deepseek_dsml/deepseek_v32.rs',
    'rust/src/parser/src/tool/deepseek_dsml/deepseek_v4.rs',
    'rust/src/parser/src/tool/deepseek_dsml/deepseek_v41/tests.rs',
    'rust/src/parser/src/tool/deepseek_dsml/mod.rs',
)
# Upstream wheel-release tooling (5ef18472), never read at runtime; kept at the base bytes.
TOOLING_DEFERRED = ('.dockerignore', 'tools/jovian_wheel_release/Dockerfile')
# Dependency metadata inputs of the (not rebuilt) native wheel: shipped as text, never compiled.
METADATA_INPUTS = ('requirements/cuda.txt', 'setup.py')
# B12X C sources that changed with the FlashInfer move: proxy ABI 5, loader reformatting.
B12X_NATIVE_CHANGED = ('b12x/comm/roce/_roce_proxy.c', 'b12x/loader/_batch.c', 'b12x/loader/_bounce.c',
                       'b12x/loader/_cuda_range.h', 'b12x/loader/_direct.c', 'b12x/loader/_gds_checkpoint.c',
                       'b12x/loader/_gds_owner.c', 'b12x/loader/_gds_reader.c', 'b12x/loader/_ple_reader.c',
                       'b12x/loader/_row_plan.h', 'b12x/loader/_storage.c')
PROXY_ABI = 5
SHIPPED_SUFFIXES = ('.py', '.json', '.md', '.rst', '.sh', '.txt', '.yaml', '.yml', '.jinja', '.toml', '.lock')


def full(component, ref):
    return previous.git(component, 'rev-parse', ref).decode().strip()


def beta_overlay(files):
    """SM121 overlay for this beta: the CMake architecture list only.

    The draft-head capability gate that the foundation overlay patched is upstream since
    vLLM #995 (7e847a8f72): the NVFP4 MTP draft head accepts the SM120 family, 12.1 included.
    """
    result = dict(files)
    name = 'vllm/models/glm5next/nvidia/mtp_draft_head.py'
    mode, data = result[name]
    require(b'if (major, minor) != (12, 0):' not in data and b'SM120-family' in data,
            'Draft-head capability gate is not the upstream family check (vLLM #995)')
    name, before, after, count = 'CMakeLists.txt', ';11.0;12.0")', ';11.0;12.0;12.1")', 2
    mode, data = result[name]
    require(data.count(before.encode()) == count, f'Overlay context changed: {name}')
    result[name] = (mode, data.replace(before.encode(), after.encode()))
    return result


def b12x_from_flashinfer(dependencies):
    """The B12X package as frozen inside the FlashInfer archive, as a `b12x/...` tree."""
    row = dependencies['sources']['flashinfer']
    require(row['commit'] == FLASHINFER, 'Frozen FlashInfer archive is not the pinned B12X carrier')
    files = inspect_archive(ROOT / 'inputs' / row['file'], row)
    package = {}
    for name, entry in files.items():
        if not name.startswith(B12X_SUBTREE):
            continue
        relative = name[len(B12X_SUBTREE):]
        if relative.startswith(B12X_EXCLUDED):
            continue
        package['b12x/' + relative] = entry
    require('b12x/__init__.py' in package and 'b12x/comm/roce/_roce_proxy.c' in package, 'B12X subtree incomplete')
    return package


ROOT_INIT = ROOT / 'b12x_root_init.py'


def root_package(dependencies):
    """The B12X root: the embedded package with the kit's initializer in place of the alias stub.

    The embedded __init__.py only aliases flashinfer.experimental.b12x to `b12x`; the real
    bootstrap (execute _api.py as the package, install the alias finder) is the wheel's _compat
    initializer, rooted at the embedded directory. b12x_root_init.py does that bootstrap rooted
    here, so the root is the single live package for both import paths.
    """
    package = b12x_from_flashinfer(dependencies)
    stub = package['b12x/__init__.py'][1]
    require(b'sys.modules[__name__] = importlib.import_module("b12x")' in stub, 'Embedded b12x/__init__.py is not the alias stub')
    package['b12x/__init__.py'] = ('100644', ROOT_INIT.read_bytes())
    return package


def base_vllm(commit):
    """Replay the base tree: upstream commit, SM121 overlay, exact PR865 patch."""
    files = overlay(previous.source('vllm', commit))
    patch = FOUNDATION / 'qsa865/upstream.patch'
    require(file_sha(patch) == previous.PATCH_SHA, 'PR865 patch drift')
    with tempfile.TemporaryDirectory(prefix='karmic-beta-base-') as scratch:
        work = Path(scratch)
        for name in previous.PATHS:
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(files[name][1])
        subprocess.run(['git', 'apply', '--whitespace=error', str(patch)], cwd=work, check=True)
        for name in previous.PATHS:
            files[name] = (files[name][0], (work / name).read_bytes())
    return files


def require_pr865_merged(files):
    """PR865 must already be present in beta: its patch must reverse-apply exactly."""
    with tempfile.TemporaryDirectory(prefix='karmic-beta-pr865-') as scratch:
        work = Path(scratch)
        for name in previous.PATHS:
            require(name in files, f'PR865 path missing in beta: {name}')
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(files[name][1])
        result = subprocess.run(['git', 'apply', '--reverse', '--check', '--whitespace=nowarn',
                                 str(FOUNDATION / 'qsa865/upstream.patch')], cwd=work,
                                capture_output=True, text=True)
        require(result.returncode == 0, 'PR865 is not present in beta: ' + result.stderr.strip())


def main():
    require(__debug__, 'Optimized Python is not admitted')
    dependencies = json.loads((ROOT / 'inputs.lock.json').read_text())
    for row in [*dependencies['wheels'].values(), *dependencies['sources'].values()]:
        require(artifact_sha(ROOT / 'inputs' / row['file'], row) == row['sha256'], 'Dependency input drift')
        if 'tree_file' in row:
            require(file_sha(ROOT / 'inputs' / row['tree_file']) == row['tree_sha256'], 'Source inventory drift')
    vllm = full('vllm', VLLM)
    require(vllm == VLLM, 'Pinned vLLM commit does not resolve to itself')
    b12x = FLASHINFER
    base_path = FOUNDATION / 'qsa865/runtime.lock.json'
    lock = json.loads(base_path.read_text())
    old_commit_v = lock['sources']['vllm']['commit']
    old_commit_b = lock['sources']['b12x']['commit']
    require(subprocess.run(['git', '-C', '/home/jugs/git/vllm', 'merge-base', '--is-ancestor',
                            old_commit_v, vllm]).returncode == 0, 'vllm base is not an ancestor of beta')
    # The archived B12X repository ends at 236ddff0; the base 10a553ef is its ancestor, and the
    # FlashInfer subtree continues that line (checked by package identity below, not ancestry).
    require(subprocess.run(['git', '-C', '/home/jugs/git/b12x', 'merge-base', '--is-ancestor',
                            old_commit_b, '236ddff04a5f'], capture_output=True).returncode == 0,
            'b12x base is not an ancestor of the archived B12X tip')
    old_v = base_vllm(old_commit_v)
    old_b = previous.source('b12x', old_commit_b)
    for name, files in (('vllm', old_v), ('b12x', old_b)):
        require(manifest(files) == lock['sources'][name]['files'], f'Independent base replay failed: {name}')

    old_native = previous.git('vllm', 'ls-tree', '-r', old_commit_v, '--', *NATIVE).decode().splitlines()
    new_native = previous.git('vllm', 'ls-tree', '-r', vllm, '--', *NATIVE).decode().splitlines()
    moved = sorted({line.split('\t', 1)[1] for line in set(old_native) ^ set(new_native)})
    require(moved == sorted(RUST_DEFERRED + METADATA_INPUTS), f'Unexpected vLLM native input change: {moved}')

    upstream_v = beta_overlay(previous.source('vllm', vllm))
    require_pr865_merged(upstream_v)
    new_v = dict(upstream_v)
    for name in RUST_DEFERRED + TOOLING_DEFERRED:
        if name in old_v:
            new_v[name] = old_v[name]
        else:
            new_v.pop(name, None)
    new_b = root_package(dependencies)
    # Only the package is carried; the archived repository's tests, validation corpus and
    # packaging files leave the root. The compiler gate runs FlashInfer's tests/experimental/b12x.
    native_b = sorted(p for p in old_b.keys() | new_b.keys() if p.startswith('b12x/')
                      and Path(p).suffix in ('.c', '.cc', '.cpp', '.cu', '.h', '.hpp') and old_b.get(p) != new_b.get(p))
    require(native_b == sorted(B12X_NATIVE_CHANGED + ('b12x/preparation/_memory.cpp',)),
            f'Undeclared B12X native source change: {native_b}')
    require('b12x/preparation/_memory.cpp' not in new_b, 'Removed B12X native source reappeared')
    require(b'#define ROCE_ABI_VERSION %d' % PROXY_ABI in new_b['b12x/comm/roce/_roce_proxy.c'][1],
            'RoCE proxy ABI is not the declared one')

    with tarfile.open(ROOT / 'refresh.tar', 'w') as bundle:
        for component, old, new, commit in (('vllm', old_v, new_v, vllm), ('b12x', old_b, new_b, b12x)):
            changed = sorted(p for p in old.keys() | new.keys() if old.get(p) != new.get(p))
            bad = [p for p in changed if p in new and Path(p).suffix not in SHIPPED_SUFFIXES
                   and p not in B12X_NATIVE_CHANGED]
            require(not bad, f'Non-admitted refresh input for {component}: {bad[:10]}')
            if component == 'b12x':
                # Shipped native sources are the declared ones; repository-level files (tests,
                # benchmarks, packaging) leave the root with the archived repository.
                natives = [p for p in changed if p in new and Path(p).suffix in ('.c', '.cc', '.cpp', '.cu', '.h', '.hpp')]
                require(sorted(natives) == sorted(B12X_NATIVE_CHANGED), f'B12X native source changed: {natives}')
                removed = [p for p in changed if p not in new]
                require(all(not p.startswith('b12x/') or p == 'b12x/preparation/_memory.cpp'
                            or p in old and Path(p).suffix in SHIPPED_SUFFIXES for p in removed),
                        f'Unexpected B12X package removal: {[p for p in removed if p.startswith("b12x/")][:10]}')
            row = lock['sources'][component]
            row.update(base_commit=row['commit'], commit=commit, base_files=manifest(old),
                       files=manifest(new), changed_paths=changed, refreshed_tree=git_tree(new),
                       upstream_tree=(previous.git('flashinfer', 'rev-parse', commit + ':' + B12X_SUBTREE.rstrip('/'))
                                      if component == 'b12x' else
                                      previous.git(component, 'rev-parse', commit + '^{tree}')).decode().strip(),
                       package_tree=git_tree({p[len(component) + 1:]: v for p, v in new.items()
                                              if p.startswith(component + '/')}))
            row.pop('local_backport', None)
            for name in changed:
                if name not in new:
                    continue
                mode, data = new[name]
                require(mode != '120000', f'New symlink not qualified: {component}/{name}')
                item = tarfile.TarInfo(component + '/' + name)
                item.size, item.mode = len(data), int(mode[-3:], 8)
                bundle.addfile(item, io.BytesIO(data))
    missing = [key for key in INHERITED if key not in lock]
    require(not missing, f'Base lock lacks foundation provenance: {missing}')
    foundation = {key: lock.pop(key) for key in INHERITED}
    foundation.update(image_id=BASE, source_lock_sha256=file_sha(base_path),
                      source_lock_path='spark/karmic-main-sm121/20260922/qsa865/runtime.lock.json')
    lock['foundation'] = foundation
    lock['publication'] = {'tag': PUBLICATION, 'image': PUBLISHED_IMAGE, 'recipe_commit': RECIPE,
                           'channel': 'karmic-kraken-beta', 'vllm_commit': vllm, 'flashinfer_commit': FLASHINFER,
                           'b12x_commit': b12x, 'b12x_source': 'flashinfer/experimental/b12x at the FlashInfer commit',
                           'use': 'source identity only; the published image is linux/amd64 SM120 and is not run'}
    lock['sources']['vllm']['deferred_metadata'] = {
        'paths': list(METADATA_INPUTS),
        'reason': 'Dependency metadata of the native wheel (FlashInfer 0.7.1 and the b12x extra); shipped as text, '
                  'the compiled extension is not rebuilt, the installed vLLM metadata is overlaid at assembly.'}
    lock['sources']['b12x']['native_changed'] = {
        'paths': list(B12X_NATIVE_CHANGED), 'removed': ['b12x/preparation/_memory.cpp'],
        'reason': 'RoCE proxy ABI 5 (rails, four HCAs) rebuilt in-image into B12X_ROCE_CACHE_DIR and gated; '
                  'loader C sources reformatted upstream, compiled at first use through the host compiler as before.'}
    lock['sources']['b12x']['root_initializer'] = {
        'path': 'b12x/__init__.py', 'sha256': file_sha(ROOT_INIT), 'source': 'b12x_root_init.py',
        'reason': 'the embedded __init__.py is an alias stub; the kit initializer bootstraps _api.py rooted here '
                  'and installs the flashinfer.experimental.b12x alias finder, as the wheel _compat package does'}
    lock['sources']['b12x']['carrier'] = {'repository': 'local-inference-lab/flashinfer', 'commit': FLASHINFER,
                                          'subtree': B12X_SUBTREE, 'excluded': list(B12X_EXCLUDED)}
    lock['proxy_abi'] = PROXY_ABI
    lock['sources']['vllm']['sm121_overlay'] = ('CMakeLists.txt architecture list only; the draft-head '
                                                 'capability gate is upstream since vLLM #995')
    lock['sources']['vllm']['deferred_native'] = {
        'paths': list(RUST_DEFERRED),
        'reason': 'Opt-in Rust frontend (VLLM_USE_RUST_FRONTEND=1, default off); compiled extension and '
                  'these sources stay at the base bytes. The Python DSML parser carries the fix.'}
    lock['sources']['vllm']['deferred_tooling'] = {
        'paths': list(TOOLING_DEFERRED),
        'reason': 'Upstream wheel-release build files (5ef18472), not shipped or read at runtime; kept at the base bytes.'}
    lock['sources']['vllm']['pr865'] = 'merged upstream in beta (01f1b874c7); reverse-apply check passed'
    lock['base_image_id'] = BASE
    lock['composition'] = ('October 8 Karmic beta 20261008.2 (vLLM integration/karmic-kraken-beta; B12X from '
                           'FlashInfer integration/karmic-kraken-beta experimental/b12x) over SM121 base 1a7a8acf '
                           'with the unchanged SM121 overlay')
    lock['lmcache_note'] = 'LMCache kept at the base 413ac987 (beta publishes 820af25f); disabled in all deployments'
    lock['compiler_upgrade'] = {'version': '4.7.1', 'inputs_lock_sha256': file_sha(ROOT / 'inputs.lock.json'),
                                'flashinfer_commit': dependencies['sources']['flashinfer']['commit'],
                                'flashinfer_version': '0.7.1', 'jit_cache_provider': 'flashinfer-jit-cache-sm121a',
                                'runtime_dependencies': {k: dependencies['wheels'][k]['version']
                                                         for k in ('nccl4py', 'nccl-extensions', 'nvidia-cudnn-frontend')},
                                'quack_version': '0.6.5', 'native_reuse': 'vLLM/FlashKDA/Torch/NCCL preserved; compiler '
                                'libraries, FlashInfer (with B12X inside) and its three runtime dependencies rebuilt or replaced'}
    lock['cache_fingerprint'] = 'karmic-beta-20261009-' + sha(
        (BASE + git_tree(new_v) + git_tree(new_b) + file_sha(ROOT / 'inputs.lock.json')).encode())[:20]
    (ROOT / 'runtime.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    pin = {'base_image_id': BASE, 'base_lock_sha256': file_sha(base_path),
           'runtime_lock_sha256': file_sha(ROOT / 'runtime.lock.json'),
           'refresh_sha256': file_sha(ROOT / 'refresh.tar'),
           'publication': PUBLICATION, 'recipe_commit': RECIPE,
           'vllm_commit': vllm, 'b12x_commit': b12x,
           'flashinfer_commit': FLASHINFER,
           'inputs_lock_sha256': file_sha(ROOT / 'inputs.lock.json'),
           'compiler_lock_sha256': file_sha(ROOT / 'compiler-arm64.lock'),
           'runtime_lock_sha256_deps': file_sha(ROOT / 'runtime-arm64.lock'),
           'cache_fingerprint': lock['cache_fingerprint']}
    (ROOT / 'build.lock.json').write_text(json.dumps(pin, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: {'commit': r['commit'], 'tree': r['refreshed_tree'],
                          'changed_paths': len(r['changed_paths'])} for k, r in lock['sources'].items()},
                     indent=2))


if __name__ == '__main__':
    main()
