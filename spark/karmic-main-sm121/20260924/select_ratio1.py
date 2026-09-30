"""Pin the ratio-1 diagnostic image over the exact release pin, or restore that exact release pin.

Only candidate.json and its manifest entry change; an amendment records both sides. Selection
requires the current pin to be the gated precision release; restoration requires the recorded
prior to be that release again. Launch profile, contract and every other runtime file untouched.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import precision_release

ROOT = Path(__file__).resolve().parent
KIND = 'ratio1-bf16-diagnostic'


def ratio1_build(root=ROOT):
    raw = (root / 'ds41-ratio1.lock.json').read_bytes()
    lock, digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
    build = json.loads((root / 'receipts/ratio1-build-receipt.json').read_text())
    directory = root / 'receipts' / Path(build['directory']).name
    if build['lock_sha256'] != digest or (directory / 'BUILD-OK').read_text().strip() != build['image_id']:
        raise RuntimeError('Ratio-1 build receipt differs from the current lock or its gate result')
    return lock, digest, build


def check_manifest(manifest, root=ROOT):
    for name, expected in manifest.items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Runtime file drift: ' + name)


def plan(current, manifest, amendment=None, root=ROOT):
    check_manifest(manifest, root)
    lock, digest, build = ratio1_build(root)
    if amendment is None:
        release_lock, _, _ = precision_release.validate_build(current, root)
        if lock['base_image_id'] != current['image_id']:
            raise RuntimeError('Ratio-1 diagnostic is not based on the pinned release image')
        pin = dict(current, image_id=build['image_id'])
        pin['diagnostic'] = {'kind': KIND, 'lock_sha256': digest,
                             'vllm_tree': lock['trees']['vllm'], 'b12x_tree': lock['trees']['b12x']}
        labels = {'local-inference.ds41.diagnostic.kind': KIND, 'local-inference.ds41.diagnostic.lock.sha256': digest,
                  'local-inference.ds41.diagnostic.base-image': lock['base_image_id'],
                  'local-inference.cache.fingerprint': lock['cache_fingerprint'],
                  'vllm.source-tree': lock['trees']['vllm'], 'b12x.source-tree': lock['trees']['b12x']}
        prior = None
    else:
        if current != amendment['candidate'] or manifest != amendment['manifest']:
            raise RuntimeError('Current runtime differs from the ratio-1 amendment')
        if current.get('diagnostic', {}).get('kind') != KIND:
            raise RuntimeError('Current pin is not the ratio-1 diagnostic')
        pin, prior = amendment['prior_candidate'], amendment['prior_manifest']
        release_lock, _, _ = precision_release.validate_build(pin, root)
        if hashlib.sha256((json.dumps(pin, sort_keys=True, indent=2) + '\n').encode()).hexdigest() != prior['candidate.json']:
            raise RuntimeError('Prior candidate digest differs')
        skip = {'image_id', 'diagnostic'}
        if {k: v for k, v in pin.items() if k not in skip} != {k: v for k, v in current.items() if k not in skip}:
            raise RuntimeError('Restoration would change more than image and source identity')
        labels = precision_release.expected_labels(pin, release_lock)
    payload = json.dumps(pin, sort_keys=True, indent=2) + '\n'
    updated = dict(manifest, **{'candidate.json': hashlib.sha256(payload.encode()).hexdigest()})
    if prior is not None and updated != prior:
        raise RuntimeError('Restored manifest differs from the recorded release pin')
    return pin, payload, updated, labels


def main():
    from build_activation_trace import idle, ssh
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', type=Path, help='ratio-1 amendment to reverse exactly')
    args = parser.parse_args()
    current = json.loads((ROOT / 'candidate.json').read_text())
    manifest = json.loads((ROOT / 'runtime-files.json').read_text())
    amendment = json.loads(args.restore.read_text()) if args.restore else None
    pin, payload, updated, labels = plan(current, manifest, amendment)
    idle()
    for node in ('dusty', 'toby', 'rusty', 'kirby'):
        info = json.loads(ssh(node, 'podman image inspect ' + pin['image_id']))[0]
        if info['Id'].removeprefix('sha256:') != pin['image_id']:
            raise RuntimeError('Wrong image on ' + node)
        for key, value in labels.items():
            if info['Config']['Labels'].get(key) != value:
                raise RuntimeError('Image label differs on ' + node + ': ' + key)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name = ('ratio1-restore-' if args.restore else 'ratio1-') + stamp + '-amendment.json'
    with (ROOT / name).open('x') as stream:
        stream.write(json.dumps({'prior_candidate': current, 'prior_manifest': manifest, 'candidate': pin,
                                 'manifest': updated,
                                 'scope': 'restore exact precision release' if args.restore else
                                          'ratio-1 extend BF16 pin diagnostic at pinned KV; not qualified'},
                                indent=2) + '\n')
    (ROOT / 'candidate.json').write_text(payload)
    (ROOT / 'runtime-files.json').write_text(json.dumps(updated, sort_keys=True, indent=2) + '\n')
    print('PRECISION-RELEASE-RESTORED' if args.restore else 'RATIO1-DIAGNOSTIC-PINNED', pin['image_id'], flush=True)


if __name__ == '__main__':
    main()
