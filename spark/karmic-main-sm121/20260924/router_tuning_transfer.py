"""Snapshot each rank's parent selections and seed the candidate without overwriting.

This freezes available choices, not query identities: boot-specific KV geometry
can introduce new bind-time queries. Compare post-boot receipts before attribution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_activation_trace import idle, ssh, REMOTE

ROOT = Path(__file__).resolve().parent
NODES = ('dusty', 'toby', 'rusty', 'kirby')
NAME = 'ds41-flash-karmic-main-tp4'
PARENT = 'ee03505df7a3b5b251d657ab201f62ac9d74eb2aee033cc5a094d7996caab4c5'
RELATIVE = 'preparation/6115b03c7a814701d5610b3e6aecf82e30fdbb46a122d70441228538b9bd1e5e.json'
CACHE = '/cache/jit/ds41-engram-b3e4f0ada602fab6f3ce/b12x/' + RELATIVE
HOST_CACHE = '/home/jugs/.cache/vllm-jj-ds41-tp4'


def validate(data):
    parsed = json.loads(data)
    if not isinstance(parsed.get('identity'), dict) or not parsed.get('records'):
        raise RuntimeError('Not a populated B12X selection cache')
    identity_hash = hashlib.sha256(json.dumps(parsed['identity'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if identity_hash != Path(RELATIVE).stem:
        raise RuntimeError('Selection identity does not match the cache filename')
    return parsed


def snapshot(out, candidate=False):
    image_id, cache = PARENT, CACHE
    if candidate:
        build = json.loads((ROOT / 'receipts/router-release-build-receipt.json').read_text())
        raw = (ROOT / 'router-release.lock.json').read_bytes()
        lock = json.loads(raw)
        if hashlib.sha256(raw).hexdigest() != build['lock_sha256']:
            raise RuntimeError('Candidate lock changed')
        image_id = build['image_id']
        cache = '/cache/jit/' + lock['cache_fingerprint'] + '/b12x/' + RELATIVE
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'image_id': image_id, 'cache_path': cache, 'nodes': {}}
    identity = None
    for node in NODES:
        info = json.loads(ssh(node, 'podman inspect ' + NAME))[0]
        if not info['State']['Running'] or info['Image'].removeprefix('sha256:') != image_id:
            raise RuntimeError('Wrong live image: ' + node)
        env = dict(item.split('=', 1) for item in info['Config']['Env'])
        base = str(Path(cache).parent.parent)
        if env.get('B12X_COMPILE_CACHE_DIR') != base:
            raise RuntimeError('Live cache namespace differs: ' + node)
        listing = ssh(node, shlex.join(['podman', 'exec', NAME, '/opt/venv/bin/python', '-c',
            'import json,pathlib,sys; print(json.dumps(sorted(p.name for p in pathlib.Path(sys.argv[1]).glob("*.json"))))',
            str(Path(cache).parent)]))
        if json.loads(listing) != [Path(RELATIVE).name]:
            raise RuntimeError('Unexpected preparation-cache inventory: ' + node)
        data = ssh(node, shlex.join(['podman', 'exec', NAME, 'cat', cache]))
        parsed = validate(data)
        if identity is not None and identity != parsed['identity']:
            raise RuntimeError('Selection identities differ across nodes')
        identity = parsed['identity']
        (out / (node + '.json')).write_text(data)
        manifest['nodes'][node] = {'sha256': hashlib.sha256(data.encode()).hexdigest(),
                                   'records': len(parsed['records']), 'identity': parsed['identity']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('ROUTER-SELECTIONS-SNAPSHOTTED', out, flush=True)


def seed(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['image_id'] != PARENT or manifest['cache_path'] != CACHE or set(manifest['nodes']) != set(NODES):
        raise RuntimeError('Wrong selection provenance')
    build = json.loads((ROOT / 'receipts/router-release-build-receipt.json').read_text())
    lock_bytes = (ROOT / 'router-release.lock.json').read_bytes()
    lock = json.loads(lock_bytes)
    if hashlib.sha256(lock_bytes).hexdigest() != build['lock_sha256']:
        raise RuntimeError('Candidate build lock mismatch')
    idle()
    receipt = {}
    for node in NODES:
        data = (out / (node + '.json')).read_bytes()
        expected = manifest['nodes'][node]['sha256']
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError('Selection snapshot changed: ' + node)
        validate(data)
        image = json.loads(ssh(node, 'podman image inspect ' + build['image_id']))[0]
        env = dict(item.split('=', 1) for item in image['Config']['Env'])
        cache_base = '/cache/jit/' + lock['cache_fingerprint'] + '/b12x'
        if image['Id'].removeprefix('sha256:') != build['image_id'] or env.get('B12X_COMPILE_CACHE_DIR') != cache_base:
            raise RuntimeError('Wrong candidate cache identity: ' + node)
        target = HOST_CACHE + cache_base.removeprefix('/cache') + '/' + RELATIVE
        staged = REMOTE + '/router-selection-' + node + '.json'
        subprocess.run(['scp', str(out / (node + '.json')), node + ':' + staged], check=True)
        code = '''import hashlib, pathlib, sys
source, destination, expected = sys.argv[1:]
data = pathlib.Path(source).read_bytes()
if hashlib.sha256(data).hexdigest() != expected:
    raise RuntimeError('Staged selection digest differs')
target = pathlib.Path(destination)
if target.exists():
    if target.read_bytes() != data:
        raise RuntimeError('Refuse to overwrite existing different candidate selections')
else:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)
print(hashlib.sha256(target.read_bytes()).hexdigest())
'''
        actual = ssh(node, shlex.join(['python3', '-c', code, staged, target, expected])).strip()
        if actual != expected:
            raise RuntimeError('Seed verification differs: ' + node)
        receipt[node] = {'target': target, 'sha256': actual, 'image_id': build['image_id']}
        print('ROUTER-SELECTIONS-SEEDED', node, actual, flush=True)
    (out / 'seeded.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('snapshot', 'seed'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidate', action='store_true', help='Snapshot the built router candidate after boot')
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / 'receipts'):
        parser.error('Selection receipts must stay under this task')
    if args.candidate and args.mode != 'snapshot':
        parser.error('--candidate is a snapshot option only')
    if args.mode == 'snapshot':
        snapshot(args.out, candidate=args.candidate)
    else:
        seed(args.out)
