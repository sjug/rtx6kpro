"""Select only the gated router image; preserve the matched parent's launch contract."""
import datetime
import hashlib
import json
from pathlib import Path

from build_activation_trace import idle, ssh

ROOT = Path(__file__).resolve().parent


def candidate(parent, lock, receipt, lock_sha):
    if parent['image_id'] != lock['base_image_id']:
        raise RuntimeError('Router comparison must start from its declared parent')
    if receipt['lock_sha256'] != lock_sha:
        raise RuntimeError('Build receipt is for a different router lock')
    result = dict(parent)
    result['image_id'] = receipt['image_id']
    result['diagnostic'] = {
        'kind': 'router-stage-release-candidate', 'lock_sha256': lock_sha,
        'vllm_tree': lock['trees']['vllm'], 'b12x_tree': lock['trees']['b12x'],
    }
    return result


def main():
    current = json.loads((ROOT / 'candidate.json').read_text())
    manifest = json.loads((ROOT / 'runtime-files.json').read_text())
    for name, expected in manifest.items():
        if Path(name).name != name or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Runtime file drift: ' + name)
    raw = (ROOT / 'router-release.lock.json').read_bytes()
    lock = json.loads(raw)
    receipt = json.loads((ROOT / 'receipts/router-release-build-receipt.json').read_text())
    pin = candidate(current, lock, receipt, hashlib.sha256(raw).hexdigest())
    build_dir = ROOT / 'receipts' / Path(receipt['directory']).name
    if (build_dir / 'BUILD-OK').read_text().strip() != pin['image_id']:
        raise RuntimeError('Missing matching build gate result')
    idle()
    for node in ('dusty', 'toby', 'rusty', 'kirby'):
        info = json.loads(ssh(node, 'podman image inspect ' + pin['image_id']))[0]
        if info['Id'].removeprefix('sha256:') != pin['image_id']:
            raise RuntimeError('Wrong router image on ' + node)
        labels = info['Config']['Labels']
        expected = {
            'local-inference.ds41.diagnostic.kind': pin['diagnostic']['kind'],
            'local-inference.ds41.diagnostic.lock.sha256': pin['diagnostic']['lock_sha256'],
            'vllm.source-tree': pin['diagnostic']['vllm_tree'],
            'b12x.source-tree': pin['diagnostic']['b12x_tree'],
            'local-inference.cache.fingerprint': lock['cache_fingerprint'],
            'local-inference.status': 'candidate-not-qualified',
        }
        for key, value in expected.items():
            if labels.get(key) != value:
                raise RuntimeError('Router label differs on ' + node + ': ' + key)
    payload = json.dumps(pin, sort_keys=True, indent=2) + '\n'
    updated = dict(manifest, **{'candidate.json': hashlib.sha256(payload.encode()).hexdigest()})
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    with (ROOT / ('router-fence-' + stamp + '-amendment.json')).open('x') as stream:
        stream.write(json.dumps({'prior_candidate': current, 'prior_manifest': manifest,
                                 'candidate': pin, 'manifest': updated,
                                 'scope': 'router fence causal arm, profile unchanged; not qualified'}, indent=2) + '\n')
    (ROOT / 'candidate.json').write_text(payload)
    (ROOT / 'runtime-files.json').write_text(json.dumps(updated, sort_keys=True, indent=2) + '\n')
    print('ROUTER-CANDIDATE-PINNED', pin['image_id'], flush=True)


if __name__ == '__main__':
    main()
