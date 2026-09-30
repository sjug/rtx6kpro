#!/usr/bin/env python3
"""Freeze deployment identity from a successful runtime build receipt."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main(receipt):
    receipt = Path(receipt)
    if not (receipt / 'BUILD-OK').is_file() or (receipt / 'status.txt').read_text().strip() != 'exit_status=0':
        raise RuntimeError('Runtime build did not pass all gates')
    image = json.loads((receipt / 'image-inspect.json').read_text())[0]
    labels = image.get('Labels') or image['Config']['Labels']
    recipe = json.loads((receipt / 'recipe-inputs.json').read_text())
    for name in ('gate_ds41.py', 'launch_contract.py', 'upstream-launch.json', 'seccomp-io-uring.json'):
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != recipe[name]:
            raise RuntimeError(f'Gated serving input changed after build: {name}')
    pin = {'image_id': image['Id'].removeprefix('sha256:'),
           'build_lock_sha256': hashlib.sha256((ROOT / 'build.lock.json').read_bytes()).hexdigest(),
           'source_lock_sha256': hashlib.sha256((ROOT / 'runtime.lock.json').read_bytes()).hexdigest(),
           'recipe_sha256': labels['local-inference.serving.recipe.sha256']}
    if labels['local-inference.ds41-build-lock.sha256'] != pin['build_lock_sha256']:
        raise RuntimeError('Runtime build lock changed after assembly')
    (ROOT / 'candidate.json').write_text(json.dumps(pin, sort_keys=True, indent=2) + '\n')
    names = ('candidate.json', 'run_node.py', 'runtime.py', 'launch_contract.py', 'gate_ds41.py',
             'upstream-launch.json', 'seccomp-io-uring.json', 'model-manifest.json')
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    (ROOT / 'runtime-files.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    print('DS41-DEPLOYMENT-KIT-FROZEN', pin['image_id'])


if __name__ == '__main__':
    main(sys.argv[1])
