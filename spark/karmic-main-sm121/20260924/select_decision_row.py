"""Select only the installation-gated capture image; do not change launch arguments."""
import datetime
import argparse
import hashlib
import json
from pathlib import Path

from build_activation_trace import idle, ssh

ROOT = Path(__file__).resolve().parent


def candidate(parent, lock, receipt, lock_sha, kind='decision-row-capture'):
    if kind not in ('decision-row-capture', 'window-capture', 'indexer-capture', 'precision-capture', 'engram-fault-inject'):
        raise RuntimeError('Unsupported diagnostic kind')
    if parent['image_id'] != lock['base_image_id']:
        raise RuntimeError('Capture diagnostic must start from its declared parent')
    if receipt['lock_sha256'] != lock_sha:
        raise RuntimeError('Build receipt is for a different capture lock')
    result = dict(parent)
    result['image_id'] = receipt['image_id']
    result['diagnostic'] = {
        'kind': kind, 'lock_sha256': lock_sha,
        'vllm_tree': lock['trees']['vllm'], 'b12x_tree': lock['trees']['b12x'],
    }
    return result


def restored(amendment, current, manifest, lock):
    if current != amendment['candidate'] or manifest != amendment['manifest']:
        raise RuntimeError('Current runtime differs from the capture amendment')
    parent, prior = amendment['prior_candidate'], amendment['prior_manifest']
    if parent['image_id'] != lock['base_image_id']:
        raise RuntimeError('Restoration is not to the declared router parent')
    if parent.get('diagnostic', {}).get('kind') != lock['base_kind']:
        raise RuntimeError('Wrong parent diagnostic kind')
    payload = json.dumps(parent, sort_keys=True, indent=2) + '\n'
    if hashlib.sha256(payload.encode()).hexdigest() != prior['candidate.json']:
        raise RuntimeError('Prior candidate digest differs')
    if {k: v for k, v in prior.items() if k != 'candidate.json'} != {
            k: v for k, v in manifest.items() if k != 'candidate.json'}:
        raise RuntimeError('Restoration would change other runtime files')
    return parent, prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('decision-row', 'window', 'indexer', 'precision', 'engram-fault'), default='decision-row')
    parser.add_argument('--restore', type=Path, help='Restore the exact parent saved in a capture amendment')
    args = parser.parse_args()
    current = json.loads((ROOT / 'candidate.json').read_text())
    manifest = json.loads((ROOT / 'runtime-files.json').read_text())
    for name, expected in manifest.items():
        if Path(name).name != name or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Runtime file drift: ' + name)
    stem, kind = {'decision-row': ('claude-decision-row', 'decision-row-capture'),
                  'precision': ('ds41-precision', 'precision-capture'),
                  'indexer': ('ds41-indexer', 'indexer-capture'),
                  'window': ('claude-window', 'window-capture'),
                  'engram-fault': ('claude-engram-fault-inject', 'engram-fault-inject')}[args.kind]
    raw = (ROOT / (stem + '.lock.json')).read_bytes()
    lock = json.loads(raw)
    receipt = json.loads((ROOT / ('receipts/' + args.kind + '-build-receipt.json')).read_text())
    expected_prior = None
    if args.restore:
        pin, expected_prior = restored(json.loads(args.restore.read_text()), current, manifest, lock)
    else:
        pin = candidate(current, lock, receipt, hashlib.sha256(raw).hexdigest(), kind)
    build_dir = ROOT / 'receipts' / Path(receipt['directory']).name
    if (build_dir / 'BUILD-OK').read_text().strip() != receipt['image_id']:
        raise RuntimeError('Missing matching diagnostic installation gate result')
    idle()
    for node in ('dusty', 'toby', 'rusty', 'kirby'):
        info = json.loads(ssh(node, 'podman image inspect ' + pin['image_id']))[0]
        if info['Id'].removeprefix('sha256:') != pin['image_id']:
            raise RuntimeError('Wrong capture image on ' + node)
        labels = info['Config']['Labels']
        expected = {
            'local-inference.ds41.diagnostic.kind': pin['diagnostic']['kind'],
            'local-inference.ds41.diagnostic.lock.sha256': pin['diagnostic']['lock_sha256'],
            'vllm.source-tree': pin['diagnostic']['vllm_tree'],
            'b12x.source-tree': pin['diagnostic']['b12x_tree'],
        }
        if not args.restore:
            expected.update({
                'local-inference.ds41.diagnostic.base-image': lock['base_image_id'],
                'local-inference.ds41.diagnostic.base-lock.sha256': lock['base_lock_sha256'],
                'local-inference.status': 'diagnostic-only-not-qualified',
            })
        for key, value in expected.items():
            if labels.get(key) != value:
                raise RuntimeError('Capture label differs on ' + node + ': ' + key)
    payload = json.dumps(pin, sort_keys=True, indent=2) + '\n'
    updated = dict(manifest, **{'candidate.json': hashlib.sha256(payload.encode()).hexdigest()})
    if expected_prior is not None and updated != expected_prior:
        raise RuntimeError('Restored manifest differs from the saved parent')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    with (ROOT / (args.kind + '-' + stamp + '-amendment.json')).open('x') as stream:
        stream.write(json.dumps({'prior_candidate': current, 'prior_manifest': manifest,
                                 'candidate': pin, 'manifest': updated,
                                 'scope': ('restore exact router parent' if args.restore else
                                           args.kind + ' diagnostic image only; not a qualified serving profile')}, indent=2) + '\n')
    (ROOT / 'candidate.json').write_text(payload)
    (ROOT / 'runtime-files.json').write_text(json.dumps(updated, sort_keys=True, indent=2) + '\n')
    print('ROUTER-PARENT-RESTORED' if args.restore else 'DIAGNOSTIC-PINNED', args.kind, pin['image_id'], flush=True)


if __name__ == '__main__':
    main()
