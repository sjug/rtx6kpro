"""Pin the gated precision release over the exact router parent, or restore that exact parent.

Selection requires the current pin to be the router parent (kind, lock, trees, gated image),
every runtime file to match its manifest, and all four nodes idle with the release image
carrying the lock's labels. Only candidate.json and its manifest entry change; an amendment
records both sides. --restore takes that amendment and returns the exact router pin.
Launch profile, contract and every other runtime file are untouched.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import precision_release as release
from build_activation_trace import idle, ssh

ROOT = Path(__file__).resolve().parent


def check_manifest(manifest, root=ROOT):
    for name, expected in manifest.items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Runtime file drift: ' + name)


def plan(current, manifest, amendment=None, root=ROOT):
    """(new pin, updated manifest, expected image labels) without touching nodes or files."""
    check_manifest(manifest, root)
    if amendment is None:
        pin = release.candidate(current, root)
        lock, _, _ = release.validate_build(pin, root)
        labels = release.expected_labels(pin, lock)
        prior = None
    else:
        pin, prior = release.restored(amendment, current, manifest, root)
        labels = {'local-inference.ds41.diagnostic.kind': release.PARENT_KIND,
                  'local-inference.ds41.diagnostic.lock.sha256': pin['diagnostic']['lock_sha256'],
                  'vllm.source-tree': pin['diagnostic']['vllm_tree'],
                  'b12x.source-tree': pin['diagnostic']['b12x_tree']}
    payload = json.dumps(pin, sort_keys=True, indent=2) + '\n'
    updated = dict(manifest, **{'candidate.json': hashlib.sha256(payload.encode()).hexdigest()})
    if prior is not None and updated != prior:
        raise RuntimeError('Restored manifest differs from the recorded router parent')
    return pin, payload, updated, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', type=Path, help='release amendment to reverse exactly')
    args = parser.parse_args()
    current = json.loads((ROOT / 'candidate.json').read_text())
    manifest = json.loads((ROOT / 'runtime-files.json').read_text())
    amendment = json.loads(args.restore.read_text()) if args.restore else None
    pin, payload, updated, labels = plan(current, manifest, amendment)
    idle()
    for node in release.NODES:
        info = json.loads(ssh(node, 'podman image inspect ' + pin['image_id']))[0]
        if info['Id'].removeprefix('sha256:') != pin['image_id']:
            raise RuntimeError('Wrong image on ' + node)
        for key, value in labels.items():
            if info['Config']['Labels'].get(key) != value:
                raise RuntimeError('Image label differs on ' + node + ': ' + key)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name = ('precision-release-restore-' if args.restore else 'precision-release-') + stamp + '-amendment.json'
    with (ROOT / name).open('x') as stream:
        stream.write(json.dumps({'prior_candidate': current, 'prior_manifest': manifest,
                                 'candidate': pin, 'manifest': updated,
                                 'scope': ('restore exact router parent' if args.restore else
                                           'precision release candidate, normal profile; not qualified')},
                                indent=2) + '\n')
    (ROOT / 'candidate.json').write_text(payload)
    (ROOT / 'runtime-files.json').write_text(json.dumps(updated, sort_keys=True, indent=2) + '\n')
    print('ROUTER-PARENT-RESTORED' if args.restore else 'PRECISION-RELEASE-PINNED', pin['image_id'], flush=True)


if __name__ == '__main__':
    main()
