#!/usr/bin/env python3
"""Freeze the September 24 main refresh, retaining the reviewed SM121/PR865 port."""
import importlib.util
import io
import json
import posixpath
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / '20260922'
sys.path.insert(0, str(PARENT))
from contracts import file_sha, git_tree, manifest, overlay, require, sha

spec = importlib.util.spec_from_file_location('previous_prepare', PARENT / 'qsa865/prepare.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
BASE = '1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc'
VLLM = '1794dcf18454900263e0c66711af8ea4a1283ac1'
B12X = 'a7d7d29b2ef8869086e0ceaa787321f17544e3c9'
LMCACHE = '361a85e88698174e9fd3b52fe372ceb40bd054f4'


def wheel_package(files):
    """Derive the two verified wheel-layout differences from upstream bytes."""
    result = {}
    for path, (mode, data) in files.items():
        if not path.startswith('lmcache/'):
            continue
        if mode == '120000':
            require(path == 'lmcache/v1/distributed/bitmap_ops/README.md',
                    f'Unqualified wheel symlink: {path}')
            target = posixpath.normpath(posixpath.join(posixpath.dirname(path), data.decode()))
            require(target in files and files[target][0] == '100644', f'Invalid wheel link: {path}')
            mode, data = files[target]
        if path == 'lmcache/lmcache_frontend/run_mp_server_with_frontend.sh':
            require(mode == '100755', 'LMCache script source mode changed')
            mode = '100644'
        result[path[8:]] = (mode, data)
    return result


def lm_source(commit):
    """Read the publisher without adding a remote or checkout; prove the full tree."""
    repo = 'repos/local-inference-lab/LMCache'
    meta = json.loads(subprocess.check_output(['gh', 'api', f'{repo}/git/commits/{commit}']))
    archive = subprocess.check_output(['gh', 'api', f'{repo}/tarball/{commit}'])
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle:
            if not (member.isfile() or member.issym()):
                continue
            name = member.name.split('/', 1)[1]
            mode = '120000' if member.issym() else ('100755' if member.mode & 0o111 else '100644')
            files[name] = (mode, member.linkname.encode() if member.issym()
                           else bundle.extractfile(member).read())
    require(git_tree(files) == meta['tree']['sha'], 'LMCache archive does not reproduce commit tree')
    return files


def patched_vllm(commit):
    files = overlay(previous.source('vllm', commit))
    patch = PARENT / 'qsa865/upstream.patch'
    require(file_sha(patch) == previous.PATCH_SHA, 'PR865 patch drift')
    with tempfile.TemporaryDirectory(prefix='ds41-compose-') as scratch:
        work = Path(scratch)
        for name in previous.PATHS:
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(files[name][1])
        subprocess.run(['git', 'apply', '--check', '--whitespace=error', str(patch)], cwd=work, check=True)
        subprocess.run(['git', 'apply', '--whitespace=error', str(patch)], cwd=work, check=True)
        for name in previous.PATHS:
            files[name] = (files[name][0], (work / name).read_bytes())
    return files


def main():
    require(__debug__, 'Optimized Python is not admitted')
    from check_nccl import check
    nccl = check()
    base_path = PARENT / 'qsa865/runtime.lock.json'
    lock = json.loads(base_path.read_text())
    old_v = patched_vllm(lock['sources']['vllm']['commit'])
    old_b = previous.source('b12x', lock['sources']['b12x']['commit'])
    for name, files in (('vllm', old_v), ('b12x', old_b)):
        require(manifest(files) == lock['sources'][name]['files'], f'Independent base replay failed: {name}')
    native = ('csrc', 'cmake', 'rust', 'requirements', 'CMakeLists.txt', 'setup.py',
              'pyproject.toml', '.gitmodules', 'rust-toolchain.toml', 'build_rust.sh', 'MANIFEST.in')
    require(previous.git('vllm', 'ls-tree', '-r', lock['sources']['vllm']['commit'], '--', *native) ==
            previous.git('vllm', 'ls-tree', '-r', VLLM, '--', *native), 'vLLM native inputs changed')
    new_v, new_b = patched_vllm(VLLM), previous.source('b12x', B12X)
    old_lm = lm_source(lock['lmcache_refresh']['commit'])
    new_lm = lm_source(LMCACHE)
    changed_lm = sorted(p for p in old_lm.keys() | new_lm.keys() if old_lm.get(p) != new_lm.get(p))
    require(all(Path(p).suffix in ('.py', '.json', '.md', '.rst') for p in changed_lm),
            'LMCache native or packaging inputs changed')
    lm_old = wheel_package(old_lm)
    lm_new = wheel_package(new_lm)
    lock['lmcache_refresh'] = {'commit': LMCACHE, 'base_commit': lock['lmcache_refresh']['commit'],
                              'upstream_tree': git_tree(new_lm), 'base_tree': git_tree(old_lm),
                              'base_files': manifest(lm_old), 'files': manifest(lm_new),
                              'changed_paths': changed_lm}
    lock['lmcache_refresh']['wheel_layout'] = {
        'lmcache_frontend/run_mp_server_with_frontend.sh': 'installed regular 0644; source 0755; bytes unchanged',
        'v1/distributed/bitmap_ops/README.md': 'installed regular 0644; bytes from verified source symlink target'}
    sets = [('vllm', old_v, new_v), ('b12x', old_b, new_b), ('lmcache', lm_old, lm_new)]
    with tarfile.open(ROOT / 'refresh.tar', 'w') as bundle:
        for component, old, new in sets:
            changed = sorted(p for p in old.keys() | new.keys() if old.get(p) != new.get(p))
            require(all(Path(p).suffix in ('.py', '.json', '.md', '.rst', '.sh') for p in changed),
                    f'Non-Python refresh input outside allowlist: {component}')
            if component != 'lmcache':
                row = lock['sources'][component]
                row.update(base_commit=row['commit'], commit=VLLM if component == 'vllm' else B12X,
                           base_files=manifest(old), files=manifest(new), changed_paths=changed,
                           refreshed_tree=git_tree(new),
                           package_tree=git_tree({p[len(component)+1:]: v for p, v in new.items()
                                                  if p.startswith(component + '/')}))
                row['upstream_tree'] = previous.git(component, 'rev-parse', row['commit'] + '^{tree}').decode().strip()
            for name in changed:
                if name not in new:
                    continue
                mode, data = new[name]
                require(mode != '120000', f'New symlink not qualified: {component}/{name}')
                item = tarfile.TarInfo(component + '/' + name)
                item.size, item.mode = len(data), int(mode[-3:], 8)
                bundle.addfile(item, io.BytesIO(data))
    lock['base_image_id'] = BASE
    lock['composition'] = 'September 24 Karmic main plus unchanged SM121 and PR865 overlays'
    lock['ds41_nccl'] = nccl
    lock['cache_fingerprint'] = 'karmic-main-20260924-' + sha((BASE + git_tree(new_v) + git_tree(new_b) + LMCACHE + file_sha(ROOT / 'nccl.lock.json')).encode())[:20]
    (ROOT / 'runtime.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    pin = {'base_image_id': BASE, 'base_lock_sha256': file_sha(base_path),
           'nccl_lock_sha256': file_sha(ROOT / 'nccl.lock.json'),
           'runtime_lock_sha256': file_sha(ROOT / 'runtime.lock.json'),
           'refresh_sha256': file_sha(ROOT / 'refresh.tar'),
           'publication': 'karmic-kraken-20260924-291c78003da67f8d',
           'recipe_commit': '626f7af3205cee128c65de44891fa04825ec3df6',
           'qsa865_patch_sha256': previous.PATCH_SHA,
           'cache_fingerprint': lock['cache_fingerprint']}
    (ROOT / 'build.lock.json').write_text(json.dumps(pin, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: {'commit': r['commit'], 'tree': r['refreshed_tree'],
                          'changed_paths': len(r['changed_paths'])} for k, r in lock['sources'].items()}, indent=2))


if __name__ == '__main__':
    main()
