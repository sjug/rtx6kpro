#!/usr/bin/env python3
"""Bind the runtime recipe to a completed, reviewed stock NCCL component."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / '20260922'))
from contracts import file_sha, require
from check_nccl import check


def main(receipt):
    row = check()
    receipt = Path(receipt)
    require((receipt / 'BUILD-OK').is_file(), 'Component has no BUILD-OK')
    require((receipt / 'status.txt').read_text().strip() == 'exit_status=0', 'Component did not finish successfully')
    require(file_sha(receipt / 'nccl.lock.json') == file_sha(ROOT / 'nccl.lock.json'), 'Component source lock drift')
    require(file_sha(receipt / 'Dockerfile.nccl') == row['dockerfile_sha256'], 'Component recipe drift')
    image = (receipt / 'image.id').read_text().strip().removeprefix('sha256:')
    inspect = json.loads((receipt / 'image-inspect.json').read_text())[0]
    require(inspect['Id'].removeprefix('sha256:') == image, 'Component image identity mismatch')
    labels = inspect.get('Labels') or inspect['Config']['Labels']
    for key, value in [('commit', row['commit']), ('recipe.sha256', row['dockerfile_sha256']),
                       ('source.sha256', row['payload_sha256'])]:
        require(labels.get('local-inference.nccl.' + key) == value, 'Component label drift: ' + key)
    digest, path = (receipt / 'SHA256SUMS').read_text().split()
    require(path == 'lib/libnccl.so.2.30.7' and len(digest) == 64 and set(digest) <= set('0123456789abcdef'),
            'Malformed library checksum')
    pin = json.loads((ROOT / 'build.lock.json').read_text())
    require(pin['nccl_lock_sha256'] == file_sha(ROOT / 'nccl.lock.json'), 'Runtime source lock drift')
    pin['nccl_component'] = {'image_id': image, 'library_sha256': digest,
                             'source_lock_sha256': pin['nccl_lock_sha256']}
    (ROOT / 'build.lock.json').write_text(json.dumps(pin, indent=2, sort_keys=True) + '\n')
    print('NCCL-COMPONENT-PIN-PASS', image, digest)


if __name__ == '__main__':
    main(sys.argv[1])
