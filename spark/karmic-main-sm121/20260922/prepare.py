#!/usr/bin/env python3
"""Freeze local Git blobs into a source-verified, offline Spark refresh context."""
import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

from contracts import file_sha, git_tree, manifest, overlay, require, sha

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parents[1] / 'karmic-beta-sm121'
PINS = {
    'vllm': ('57a80980bbf4b40398de7ed851b23e55a3a4c50e', '6afb99982576a7a2eb53d667189e859629e22739'),
    'b12x': ('e9ce547767ff9ee6509faf294fa1b4e2380dfbf5', '4f3028b19c1d8290dc72b6f483aba40de23eae5a'),
}
NATIVE_ROOTS = ('csrc/', 'cmake/', 'rust/', 'requirements/', 'CMakeLists.txt',
                'setup.py', 'pyproject.toml', '.gitmodules', 'rust-toolchain.toml',
                'build_rust.sh', 'MANIFEST.in')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def source(repo, ref):
    files = {}
    # archive deliberately does not follow submodules; their reused installed
    # products are protected separately. Compare every Git native-input entry.
    listing = git(repo, 'ls-tree', '-r', '-z', ref).split(b'\0')
    for entry in filter(None, listing):
        info, name = entry.split(b'\t', 1)
        mode, kind, oid = info.decode().split()
        if kind == 'commit':
            continue
        require(kind == 'blob', f'Unexpected Git type: {kind}')
        files[name.decode()] = (mode, git(repo, 'cat-file', 'blob', oid))
    return files


def archive(path, files):
    with tarfile.open(path, 'w') as out:
        for name, (mode, data) in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.mtime = 0
            info.mode = int(mode[-3:], 8)
            if mode == '120000':
                info.type = tarfile.SYMTYPE
                info.linkname = data.decode()
                out.addfile(info)
            else:
                info.size = len(data)
                out.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--git-root', type=Path, default=Path('/home/jugs/git'))
    args = parser.parse_args()
    payload = ROOT / 'payload'
    payload.mkdir(exist_ok=True)
    lock = {
        'schema': 1,
        'base_image_id': 'f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233',
        'published_image': 'ghcr.io/local-inference-lab/vllm@sha256:e637cd058b47adbcfa8bf97561c3392d7855108c6ea149f6a0c9b78e3dc0ba78',
        'recipe_commit': '0ab249fe88f004c286f7100a2b33f66639901fb1',
        'base_build_lock_sha256': file_sha(OLD / 'build.lock.json'),
        'proxy_abi': 4,
        'composition': 'non-linear beta-to-main source replacement; native reuse is direct input-tree equality, not ancestry',
        'sources': {}, 'assets': {},
        'native_metadata': 'Base vLLM distribution metadata retained; runtime identity is the tracked source manifest and image labels.',
    }
    for component, (base, target) in PINS.items():
        repo = args.git_root / component
        if component == 'vllm':
            native_before = git(repo, 'ls-tree', '-r', base, '--', *NATIVE_ROOTS)
            native_after = git(repo, 'ls-tree', '-r', target, '--', *NATIVE_ROOTS)
            require(native_before == native_after, 'vLLM native/dependency inputs changed; native reuse refused')
            lock['native_inputs_sha256'] = sha(native_before)
        old, new = source(repo, base), source(repo, target)
        changed = sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))
        for name in changed:
            if component == 'vllm':
                require(Path(name).suffix in ('.py', '.md', '.json', '.sh'), f'Unqualified vLLM refresh path {name}')
            else:
                require(Path(name).suffix in ('.py', '.md', '.json') or name in
                        ('.github/workflows/lil-cu134-wheel-release.yml', 'b12x/comm/roce/_roce_proxy.c'),
                        f'Unqualified B12X refresh path {name}')
        if component == 'vllm':
            old, new = overlay(old), overlay(new)
            require(git_tree(old) == '02a457e2e933d0fb20d5786835110546acf7d8f2', 'Base overlay tree mismatch')
        # The package subtree has no submodules, so its independently computed
        # tree can be anchored even when the repository has external gitlinks.
        package = {name[len(component)+1:]: value for name, value in new.items() if name.startswith(component + '/')}
        dest = payload / f'{component}.tar'
        archive(dest, new)
        entry = {
            'base_commit': base, 'commit': target,
            'upstream_tree': git(repo, 'rev-parse', target + '^{tree}').decode().strip(),
            'package_tree': git_tree(package),
            'refreshed_tree': git_tree(new),
            'archive_sha256': file_sha(dest), 'changed_paths': changed,
            'base_files': manifest(old), 'files': manifest(new),
        }
        lock['sources'][component] = entry
    # Preserve the corrected launcher bytes and the inherited native checks.
    assets = (
        'launchers/serve-qwen38-flash-next-karmic-spark.sh',
        'launchers/serve-glm53-flash-karmic-spark.sh',
        'verify_launch_arguments.py', 'gate_flashkda.py',
        'tests/verify_glm53_nvfp4_draft_head_sm121.py',
        'tests/workspace_fixture.py',
        'tests/run_r38_regressions.py', 'tests/flashkda_counts.py',
    )
    for name in assets:
        data = (OLD / name).read_bytes()
        if name == 'launchers/serve-glm53-flash-karmic-spark.sh':
            require(data.count(b'roce_abi_version() == 3') == 1, 'Unexpected GLM proxy check')
            data = data.replace(b'roce_abi_version() == 3', b'roce_abi_version() == 4')
        dest = ROOT / 'inherited' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        lock['assets']['inherited/' + name] = sha(data)
    native_gate = (OLD / 'verify_runtime.py').read_text()
    require(native_gate.count('roce_abi_version() == 3') == 1, 'Unexpected base proxy gate')
    native_gate = native_gate.replace('roce_abi_version() == 3', 'roce_abi_version() == 4')
    check = '            require(path.is_file(), f\'Missing {path}\')'
    require(native_gate.count(check) == 1, 'Native file check context changed')
    native_gate = native_gate.replace(check,
        '            if path.is_symlink():\n'
        '                require(hashlib.sha256(os.readlink(path).encode()).hexdigest() == expected, f\'Changed symlink {path}\')\n'
        '                continue\n' + check)
    dest = ROOT / 'inherited/verify_runtime.py'
    dest.write_text(native_gate)
    lock['assets']['inherited/verify_runtime.py'] = file_sha(dest)
    base_lock = ROOT / 'inherited/base-build.lock.json'
    base_lock.write_bytes((OLD / 'build.lock.json').read_bytes())
    lock['assets']['inherited/base-build.lock.json'] = file_sha(base_lock)
    origin = ROOT / 'recipe-origin.json'
    origin.write_text(json.dumps({
        'repository': '/home/jugs/git/rtx6kpro',
        'commit': git(ROOT, 'rev-parse', 'HEAD').decode().strip(),
        'status': git(ROOT, 'status', '--porcelain', '--', '.').decode(),
        'identity': 'The complete recipe-inputs digest is authoritative; commit may precede these uncommitted files.',
    }, sort_keys=True, indent=2) + '\n')
    lock['assets']['recipe-origin.json'] = file_sha(origin)
    runner = (OLD / 'run-qwen-tp2-node.sh').read_text()
    for before, after in (
        (':karmic-spark-sm121', ':karmic-main-spark-sm121'),
        ('qwen38-flash-next-nvfp4-karmic-tp2', 'qwen38-flash-next-nvfp4-karmic-main-tp2'),
        ('02a457e2e933d0fb20d5786835110546acf7d8f2', lock['sources']['vllm']['refreshed_tree']),
        ('9f20c0e6b9d49a42bc19f45429ac44758cf5fe76', lock['sources']['b12x']['refreshed_tree']),
    ):
        require(runner.count(before) == 1, f'Runner replacement context changed: {before}')
        runner = runner.replace(before, after)
    (ROOT / 'run-qwen-tp2-node.sh').write_text(runner)
    lock['assets']['run-qwen-tp2-node.sh'] = sha(runner.encode())
    data = (OLD / 'qualify_qwen.py').read_bytes()
    (ROOT / 'qualify_qwen.py').write_bytes(data)
    lock['assets']['qualify_qwen.py'] = sha(data)
    flash_tests = ROOT / 'inherited/flashkda-tests'
    require((flash_tests / 'spark-test-adaptation.json').is_file(), 'Stage retained FlashKDA test receipt first')
    for path in sorted(flash_tests.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            lock['assets'][str(path.relative_to(ROOT))] = file_sha(path)
    lock['cache_fingerprint'] = 'karmic-main-' + sha(json.dumps({
        'base': lock['base_image_id'], 'native': lock['native_inputs_sha256'],
        'sources': {name: row['archive_sha256'] for name, row in lock['sources'].items()},
    }, sort_keys=True).encode())[:24]
    (ROOT / 'source.lock.json').write_text(json.dumps(lock, sort_keys=True, indent=2) + '\n')
    print('PREPARED', lock['cache_fingerprint'])
    for name, row in lock['sources'].items():
        print(name, row['commit'], 'package tree', row['package_tree'], 'changed paths', len(row['changed_paths']))


if __name__ == '__main__':
    main()
