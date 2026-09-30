#!/usr/bin/env python3
"""Compose PR865 over the frozen Spark tree without changing source checkouts."""
import json
import base64
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from contracts import file_sha, git_tree, manifest, require, sha
from contracts import overlay

BASE = '88867036403cb3c9bc3026eb8c907ad64bc1af9c28d316d22b670dde4f561152'
HEAD = '26b42cf9eb715b118e55f91f53fa0ab412a79c09'
PATCH_SHA = '4a70945def2dda5540d04c4a18cf138f3908ae5329745b62753a5c106061f113'
TARGETS = {'vllm': 'e77be22511ab91ecf217760524b7579c366cca2a',
           'b12x': '10a553ef980571f23a073f930cb386fbc8a77e07'}
LMCACHE = '413ac987336a91b5cb3899ce20720e9ad9f1fa47'
PATHS = (
    'tests/models/qwen4_exp/test_b12x_qsa.py',
    'tests/v1/cudagraph/test_cudagraph_manager.py',
    'vllm/models/qwen4_exp/nvidia/b12x_qsa.py',
    'vllm/models/qwen4_exp/nvidia/model_state.py',
    'vllm/v1/worker/gpu/cudagraph_utils.py',
    'vllm/v1/worker/gpu/input_batch.py',
)


def git(component, *args):
    return subprocess.check_output(['git', '-C', '/home/jugs/git/' + component, *args])


def source(component, ref):
    files = {}
    with tarfile.open(fileobj=io.BytesIO(git(component, 'archive', ref))) as bundle:
        for member in bundle.getmembers():
            if not (member.isfile() or member.issym()):
                continue
            mode = '120000' if member.issym() else ('100755' if member.mode & 0o111 else '100644')
            files[member.name] = (mode, member.linkname.encode() if member.issym()
                                  else bundle.extractfile(member).read())
    return files


def remote_lmcache(ref, name):
    response = json.loads(subprocess.check_output([
        'gh', 'api', f'repos/local-inference-lab/LMCache/contents/{name}?ref={ref}']))
    data = base64.b64decode(response['content'])
    import hashlib
    oid = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    require(oid == response['sha'], f'GitHub blob mismatch: {name}')
    return data


def compose():
    require(file_sha(ROOT / 'upstream.patch') == PATCH_SHA, 'Patch digest changed')
    lock = json.loads((ROOT.parent / 'source.lock.json').read_text())
    row = lock['sources']['vllm']
    archive = ROOT.parent / 'payload/vllm.tar'
    require(file_sha(archive) == row['archive_sha256'], 'Base archive changed')
    base_files = {}
    with tarfile.open(archive) as bundle:
        for name, entry in row['files'].items():
            member = bundle.getmember(name)
            data = member.linkname.encode() if member.issym() else bundle.extractfile(member).read()
            require(sha(data) == entry['sha256'], f'Base blob changed: {name}')
            base_files[name] = (entry['mode'], data)
    require(git_tree(base_files) == row['refreshed_tree'], 'Base tracked tree changed')
    native = ('csrc', 'cmake', 'rust', 'requirements', 'CMakeLists.txt', 'setup.py',
              'pyproject.toml', '.gitmodules', 'rust-toolchain.toml', 'build_rust.sh', 'MANIFEST.in')
    require(git('vllm', 'ls-tree', '-r', row['commit'], '--', *native) ==
            git('vllm', 'ls-tree', '-r', TARGETS['vllm'], '--', *native), 'Native inputs moved')
    files = overlay(source('vllm', TARGETS['vllm']))
    upstream_files = dict(files)
    with tempfile.TemporaryDirectory(prefix='qsa865-') as scratch:
        work = Path(scratch)
        for name in PATHS:
            dest = work / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(files[name][1])
        for options in (['--check'], []):
            subprocess.run(['git', 'apply', '--whitespace=error', *options,
                            str(ROOT / 'upstream.patch')], cwd=work, check=True)
        for name in PATHS:
            files[name] = (files[name][0], (work / name).read_bytes())
    before = {name: manifest(upstream_files)[name] for name in PATHS}
    old_tree = row['refreshed_tree']
    row['files'] = manifest(files)
    row['refreshed_tree'] = git_tree(files)
    row['package_tree'] = git_tree({name[5:]: value for name, value in files.items()
                                    if name.startswith('vllm/')})
    row['local_backport'] = {'pr': 865, 'head': HEAD, 'patch_sha256': PATCH_SHA}
    row['base_files'] = manifest(base_files)
    row['base_commit'] = row['commit']
    row['commit'] = TARGETS['vllm']
    row['upstream_tree'] = git('vllm', 'rev-parse', TARGETS['vllm'] + '^{tree}').decode().strip()
    row['changed_paths'] = sorted(p for p in base_files.keys() | files.keys() if base_files.get(p) != files.get(p))
    require(all(Path(p).suffix in ('.py', '.json', '.md') for p in row['changed_paths']), 'Unexpected vLLM path')
    b_row = lock['sources']['b12x']
    b_old = source('b12x', b_row['commit'])
    require(manifest(b_old) == b_row['files'], 'Base B12X source differs from lock')
    b_new = source('b12x', TARGETS['b12x'])
    b_changed = sorted(p for p in b_old.keys() | b_new.keys() if b_old.get(p) != b_new.get(p))
    native_loader_paths = {f'b12x/loader/{name}' for name in
                          ('_batch.c', '_bounce.c', '_cuda_range.h', '_direct.c', '_gds_checkpoint.c',
                           '_gds_owner.c', '_pool.c', '_pool_api.h', '_storage.c')}
    require(all(Path(p).suffix in ('.py', '.json', '.md', '.yml') or p in native_loader_paths
                for p in b_changed), 'Unexpected B12X native path')
    require(b_old['b12x/comm/roce/_roce_proxy.c'] == b_new['b12x/comm/roce/_roce_proxy.c'], 'Proxy ABI changed')
    b_row.update(base_commit=b_row['commit'], commit=TARGETS['b12x'], base_files=manifest(b_old),
                 files=manifest(b_new), changed_paths=b_changed, refreshed_tree=git_tree(b_new),
                 upstream_tree=git('b12x', 'rev-parse', TARGETS['b12x'] + '^{tree}').decode().strip(),
                 package_tree=git_tree({p[5:]: v for p, v in b_new.items() if p.startswith('b12x/')}))
    # Install changed tracked blobs only. No generated/native build products enter this tar.
    with tarfile.open(ROOT / 'refresh.tar', 'w') as bundle:
        for component, old, new in (('vllm', base_files, files), ('b12x', b_old, b_new)):
            for name in sorted(new):
                if old.get(name) == new[name]:
                    continue
                mode, data = new[name]
                require(mode != '120000', f'New symlink is not qualified: {name}')
                item = tarfile.TarInfo(component + '/' + name)
                item.size = len(data)
                item.mode = int(mode[-3:], 8)
                bundle.addfile(item, io.BytesIO(data))
    lm_before = remote_lmcache('688bee14e157b64623d93c07fc0d4db93470e12f', 'lmcache/v1/multiprocess/mq.py')
    lm_after = remote_lmcache(LMCACHE, 'lmcache/v1/multiprocess/mq.py')
    (ROOT / 'lmcache-mq.py').write_bytes(lm_after)
    tests = ROOT / 'lmcache-tests/tests/v1/multiprocess'
    tests.mkdir(parents=True, exist_ok=True)
    for path in (tests, tests.parent, tests.parent.parent):
        (path / '__init__.py').write_text('')
    for name in ('test_mq.py', 'test_mq_handler_helpers.py'):
        (tests / name).write_bytes(remote_lmcache(LMCACHE, 'tests/v1/multiprocess/' + name))
    lock['lmcache_refresh'] = {'commit': LMCACHE, 'before_sha256': sha(lm_before), 'after_sha256': sha(lm_after)}
    lock['base_image_id'] = BASE
    lock['composition'] = 'September 23 Karmic main sources plus PR865 on retained Spark natives'
    lock['cache_fingerprint'] = 'karmic-main-qsa865-' + sha((BASE + row['refreshed_tree'] +
                                                          b_row['refreshed_tree'] + LMCACHE).encode())[:20]
    (ROOT / 'runtime.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    pin = {'base_image_id': BASE, 'upstream_head': HEAD,
           'upstream_base': 'e77be22511ab91ecf217760524b7579c366cca2a',
           'patch_sha256': PATCH_SHA, 'before': before,
           'after': {name: row['files'][name] for name in PATHS},
           'base_tree': old_tree, 'result_tree': row['refreshed_tree'],
           'package_tree': row['package_tree'],
           'base_lock_sha256': file_sha(ROOT.parent / 'source.lock.json'),
           'runtime_lock_sha256': file_sha(ROOT / 'runtime.lock.json'),
           'cache_fingerprint': lock['cache_fingerprint']}
    pin.update(refresh_sha256=file_sha(ROOT / 'refresh.tar'),
               b12x_tree=b_row['refreshed_tree'], publication='karmic-kraken-20260923-f37d447a16799e72',
               recipe_commit='459593df3dab9cc5861c0a66dfed709f1a5f2f72',
               lmcache_tests={str(p.relative_to(ROOT)): file_sha(p) for p in (ROOT / 'lmcache-tests').rglob('*.py')})
    (ROOT / 'backport.lock.json').write_text(json.dumps(pin, indent=2, sort_keys=True) + '\n')
    runner = (ROOT.parent / 'run-qwen-hc-diagnostic.sh').read_text()
    replacements = {
        ':karmic-main-spark-sm121': ':karmic-main-qsa865-spark-sm121',
        'NAME=${NAME:-qwen38-flash-next-nvfp4-karmic-main-tp2}':
            'NAME=${NAME:-qwen38-flash-next-nvfp4-karmic-main-qsa865-tp2}',
        '31cb715f3511073f893900248ba8e737646ca50d': row['refreshed_tree'],
        '6a30d32d796df362b5b6a1568f7de86a8c1693d2': b_row['refreshed_tree'],
        '# Diagnostic only: unset preserves the image default; no general env passthrough.':
            '# Saved HC-off profile; HC=1 remains an explicit diagnostic control.\n'
            'VLLM_QWEN3_8_FLASH_NEXT_HC_TP=${VLLM_QWEN3_8_FLASH_NEXT_HC_TP:-0}\n'
            '# Preserve the failing 32-row capture envelope to qualify the source fix.\n'
            '[[ $MAX_NUM_SEQS == 4 && $NUM_SPECULATIVE_TOKENS == 3 ]] || { echo "QSA865 profile requires four sequences and MTP3" >&2; exit 78; }',
        '  --recurrent-checkpoint-policy "${RECURRENT_CHECKPOINT_POLICY}"\n)':
            '  --recurrent-checkpoint-policy "${RECURRENT_CHECKPOINT_POLICY}"\n'
            '  --max-cudagraph-capture-size 32\n)',
        'ROLE=${ROLE:?set ROLE=head|worker|stop}':
            'ROLE=${ROLE:?set ROLE=head|worker}\n'
            '[[ $ROLE != stop ]] || { echo "Stop the pair gracefully and retain containers; this runner never removes them" >&2; exit 78; }',
    }
    for before, after in replacements.items():
        require(runner.count(before) == 1, f'Runner context changed: {before}')
        runner = runner.replace(before, after)
    (ROOT / 'run-qwen.sh').write_text(runner)
    print(json.dumps({key: pin[key] for key in ('base_tree', 'result_tree', 'package_tree')}, indent=2))


if __name__ == '__main__':
    compose()
