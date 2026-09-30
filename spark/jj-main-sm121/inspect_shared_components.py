#!/usr/bin/env python3
"""Read exact local component identities before spending native build time."""
import json
from pathlib import Path
import subprocess

lock = json.loads(Path(__file__).with_name('build.lock.json').read_text())
records = {}
for name, image in lock['components'].items():
    if name == 'vllm':
        continue
    result = subprocess.run(['podman', 'image', 'inspect', image], check=True,
                            capture_output=True, text=True)
    items = json.loads(result.stdout)
    if len(items) != 1 or items[0]['Id'].removeprefix('sha256:') != image:
        raise RuntimeError(f'{name}: image identity mismatch')
    if items[0].get('Architecture') != 'arm64':
        raise RuntimeError(f'{name}: not ARM64')
    # Keep actual recipe labels, not hashes of potentially hardened newer recipes.
    records[name] = items[0]
print(json.dumps(records, indent=2))
