#!/usr/bin/env python3
"""Freeze the full October 5 Karmic beta as a tracked-source refresh of the SM121 base.

Base image 1a7a8acf carries vLLM e77be225 (SM121 overlay plus PR865) and B12X 10a553ef.
Beta is a descendant of both. The tracked refresh changes Python/text files; vLLM's native
inputs are identical apart from the opt-in Rust DSML parser, which stays at the base bytes
(declared deviation). Compiler libraries and FlashInfer are replaced in a separate,
audited offline step. LMCache stays at the base 413ac987 and remains disabled.
"""
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from unpack_sources import artifact_sha

ROOT = Path(__file__).resolve().parent
# SM121 foundation tree: base image 1a7a8acf, its source lock, shared contracts and gates.
FOUNDATION = ROOT.parents[1] / 'karmic-main-sm121/20260922'
sys.path.insert(0, str(FOUNDATION))
from contracts import file_sha, git_tree, manifest, overlay, require, sha

spec = importlib.util.spec_from_file_location('previous_prepare', FOUNDATION / 'qsa865/prepare.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

BASE = '1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc'
VLLM = 'f1c2508f1018f47ee8a78478a819d202f359887b'
B12X = '52640cb15d4ad1c7230f747e72c45d45dc2da681'
PUBLICATION = 'karmic-kraken-beta-20261005-4102660abd8a98b1'
PUBLISHED_IMAGE = 'ghcr.io/local-inference-lab/vllm@sha256:fd74019fc217a541060093d18427c36330df6ac5700651b55593dafdbb936920'
# Foundation provenance copied from the base lock; kept apart from the beta publication.
INHERITED = ('base_build_lock_sha256', 'native_inputs_sha256', 'native_metadata', 'published_image',
             'recipe_commit', 'lmcache_refresh')
RECIPE = 'e881183c06a6f64dffb87afb24c7e7703240dca3'
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
SHIPPED_SUFFIXES = ('.py', '.json', '.md', '.rst', '.sh', '.txt', '.yaml', '.yml', '.jinja', '.toml', '.lock')


def full(component, ref):
    return previous.git(component, 'rev-parse', ref).decode().strip()


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
    vllm, b12x = full('vllm', VLLM), full('b12x', B12X)
    require((vllm, b12x) == (VLLM, B12X), 'Pinned commits do not resolve to themselves')
    base_path = FOUNDATION / 'qsa865/runtime.lock.json'
    lock = json.loads(base_path.read_text())
    old_commit_v = lock['sources']['vllm']['commit']
    old_commit_b = lock['sources']['b12x']['commit']
    for component, old, new in (('vllm', old_commit_v, vllm), ('b12x', old_commit_b, b12x)):
        require(subprocess.run(['git', '-C', '/home/jugs/git/' + component, 'merge-base', '--is-ancestor',
                                old, new]).returncode == 0, f'{component} base is not an ancestor of beta')
    old_v = base_vllm(old_commit_v)
    old_b = previous.source('b12x', old_commit_b)
    for name, files in (('vllm', old_v), ('b12x', old_b)):
        require(manifest(files) == lock['sources'][name]['files'], f'Independent base replay failed: {name}')

    old_native = previous.git('vllm', 'ls-tree', '-r', old_commit_v, '--', *NATIVE).decode().splitlines()
    new_native = previous.git('vllm', 'ls-tree', '-r', vllm, '--', *NATIVE).decode().splitlines()
    moved = sorted({line.split('\t', 1)[1] for line in set(old_native) ^ set(new_native)})
    require(moved == sorted(RUST_DEFERRED), f'Unexpected vLLM native input change: {moved}')

    upstream_v = overlay(previous.source('vllm', vllm))
    require_pr865_merged(upstream_v)
    new_v = dict(upstream_v)
    for name in RUST_DEFERRED + TOOLING_DEFERRED:
        if name in old_v:
            new_v[name] = old_v[name]
        else:
            new_v.pop(name, None)
    new_b = previous.source('b12x', b12x)
    require(old_b['b12x/comm/roce/_roce_proxy.c'] == new_b['b12x/comm/roce/_roce_proxy.c'],
            'RoCE proxy native source changed')

    with tarfile.open(ROOT / 'refresh.tar', 'w') as bundle:
        for component, old, new, commit in (('vllm', old_v, new_v, vllm), ('b12x', old_b, new_b, b12x)):
            changed = sorted(p for p in old.keys() | new.keys() if old.get(p) != new.get(p))
            bad = [p for p in changed if p in new and Path(p).suffix not in SHIPPED_SUFFIXES]
            require(not bad, f'Non-admitted refresh input for {component}: {bad[:10]}')
            if component == 'b12x':
                natives = [p for p in changed if Path(p).suffix in ('.c', '.cc', '.cpp', '.cu', '.h', '.hpp')]
                require(not natives, f'B12X native source changed: {natives}')
            row = lock['sources'][component]
            row.update(base_commit=row['commit'], commit=commit, base_files=manifest(old),
                       files=manifest(new), changed_paths=changed, refreshed_tree=git_tree(new),
                       upstream_tree=previous.git(component, 'rev-parse', commit + '^{tree}').decode().strip(),
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
                           'channel': 'karmic-kraken-beta', 'vllm_commit': vllm, 'b12x_commit': b12x,
                           'use': 'source identity only; the published image is linux/amd64 SM120 and is not run'}
    lock['sources']['vllm']['deferred_native'] = {
        'paths': list(RUST_DEFERRED),
        'reason': 'Opt-in Rust frontend (VLLM_USE_RUST_FRONTEND=1, default off); compiled extension and '
                  'these sources stay at the base bytes. The Python DSML parser carries the fix.'}
    lock['sources']['vllm']['deferred_tooling'] = {
        'paths': list(TOOLING_DEFERRED),
        'reason': 'Upstream wheel-release build files (5ef18472), not shipped or read at runtime; kept at the base bytes.'}
    lock['sources']['vllm']['pr865'] = 'merged upstream in beta (01f1b874c7); reverse-apply check passed'
    lock['base_image_id'] = BASE
    lock['composition'] = ('October 5 Karmic beta (vLLM integration/karmic-kraken-beta, B12X '
                           'integration/karmic-kraken-beta) over SM121 base 1a7a8acf with the unchanged SM121 overlay')
    lock['lmcache_note'] = 'LMCache kept at the base 413ac987 (beta publishes 820af25f); disabled in all deployments'
    lock['compiler_upgrade'] = {'version': '4.7.1', 'inputs_lock_sha256': file_sha(ROOT / 'inputs.lock.json'),
                                'flashinfer_commit': dependencies['sources']['flashinfer']['commit'],
                                'quack_version': '0.6.5', 'native_reuse': 'vLLM/FlashKDA/Torch/NCCL preserved; compiler libraries and FlashInfer rebuilt'}
    lock['cache_fingerprint'] = 'karmic-beta-20261005-' + sha(
        (BASE + git_tree(new_v) + git_tree(new_b) + file_sha(ROOT / 'inputs.lock.json')).encode())[:20]
    (ROOT / 'runtime.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    pin = {'base_image_id': BASE, 'base_lock_sha256': file_sha(base_path),
           'runtime_lock_sha256': file_sha(ROOT / 'runtime.lock.json'),
           'refresh_sha256': file_sha(ROOT / 'refresh.tar'),
           'publication': PUBLICATION, 'recipe_commit': RECIPE,
           'vllm_commit': vllm, 'b12x_commit': b12x,
           'inputs_lock_sha256': file_sha(ROOT / 'inputs.lock.json'),
           'compiler_lock_sha256': file_sha(ROOT / 'compiler-arm64.lock'),
           'cache_fingerprint': lock['cache_fingerprint']}
    (ROOT / 'build.lock.json').write_text(json.dumps(pin, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: {'commit': r['commit'], 'tree': r['refreshed_tree'],
                          'changed_paths': len(r['changed_paths'])} for k, r in lock['sources'].items()},
                     indent=2))


if __name__ == '__main__':
    main()
