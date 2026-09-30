"""Restore only the clean parent's image pin, preserving every runtime setting."""
import datetime
import argparse
import hashlib
import json
from pathlib import Path

from build_activation_trace import idle, ssh

ROOT = Path(__file__).resolve().parent


def restored_parent(amendment, current, manifest):
    if current != amendment['candidate'] or manifest != amendment['manifest']:
        raise RuntimeError('Current candidate does not match the recorded router amendment')
    parent, prior = amendment['prior_candidate'], amendment['prior_manifest']
    if parent.get('diagnostic', {}).get('kind') != 'engram-prequeued-native-io-fix':
        raise RuntimeError('Amendment does not restore the Engram parent')
    payload = json.dumps(parent, sort_keys=True, indent=2) + '\n'
    if hashlib.sha256(payload.encode()).hexdigest() != prior['candidate.json']:
        raise RuntimeError('Recorded prior candidate digest differs')
    if {k: v for k, v in prior.items() if k != 'candidate.json'} != {k: v for k, v in manifest.items() if k != 'candidate.json'}:
        raise RuntimeError('Restoration would change other runtime files')
    return parent, prior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amendment', type=Path, help='Restore the exact prior identity saved by the router candidate selector')
    args = parser.parse_args()
    current = json.loads((ROOT / 'candidate.json').read_text())
    manifest = json.loads((ROOT / 'runtime-files.json').read_text())
    prior = None
    if args.amendment:
        parent, prior = restored_parent(json.loads(args.amendment.read_text()), current, manifest)
    else:
        parent = json.loads((ROOT / 'engram-repair-ee03505df7a3-candidate-amendment.json').read_text())['candidate']
    expected_parent = json.loads((ROOT / 'engram-repair-ee03505df7a3-candidate-amendment.json').read_text())['candidate']
    if parent != expected_parent:
        raise RuntimeError('Restored identity differs from the frozen clean parent')
    skip = {'image_id', 'diagnostic'}
    if {k: v for k, v in parent.items() if k not in skip} != {k: v for k, v in current.items() if k not in skip}:
        raise RuntimeError('Restoring the parent would change more than image/source identity')
    for name, expected in manifest.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Runtime drift: ' + name)
    idle()
    for node in ('dusty', 'toby', 'rusty', 'kirby'):
        info = json.loads(ssh(node, 'podman image inspect ' + parent['image_id']))[0]
        if info['Id'].removeprefix('sha256:') != parent['image_id']:
            raise RuntimeError('Wrong parent on ' + node)
        labels = info['Config']['Labels']
        for key, expected in {
            'local-inference.ds41.diagnostic.kind': parent['diagnostic']['kind'],
            'local-inference.ds41.diagnostic.lock.sha256': parent['diagnostic']['lock_sha256'],
            'vllm.source-tree': parent['diagnostic']['vllm_tree'],
            'b12x.source-tree': parent['diagnostic']['b12x_tree'],
        }.items():
            if labels.get(key) != expected:
                raise RuntimeError('Parent label differs on ' + node + ': ' + key)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    payload = json.dumps(parent, sort_keys=True, indent=2) + '\n'
    updated = dict(manifest, **{'candidate.json': hashlib.sha256(payload.encode()).hexdigest()})
    if prior is not None and updated != prior:
        raise RuntimeError('Restored runtime manifest differs from the recorded parent')
    with (ROOT / ('router-parent-' + stamp + '-amendment.json')).open('x') as stream:
        stream.write(json.dumps({'prior_candidate': current, 'prior_manifest': manifest,
                                 'candidate': parent, 'manifest': updated,
                                 'scope': 'unqualified matched router control, profile unchanged'}, indent=2) + '\n')
    (ROOT / 'candidate.json').write_text(payload)
    (ROOT / 'runtime-files.json').write_text(json.dumps(updated, sort_keys=True, indent=2) + '\n')
    print('ROUTER-PARENT-PINNED', parent['image_id'], flush=True)


if __name__ == '__main__':
    main()
