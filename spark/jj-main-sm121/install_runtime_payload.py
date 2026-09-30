#!/usr/bin/env python3
"""Install native wheel products beside byte-preserved tracked runtime sources."""
import hashlib
import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

ROOT = Path('/opt/jovian-judgement/vllm')
SITE = Path(sysconfig.get_paths()['purelib'])


def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    tracked = {}
    for path in ROOT.rglob('*'):
        if path.is_file():
            tracked[str(path.relative_to(ROOT))] = digest(path)
    wheel_package = SITE / 'vllm'
    if not wheel_package.is_dir():
        raise RuntimeError('vLLM wheel package missing')
    products = {}
    for source in wheel_package.rglob('*'):
        if not source.is_file() or '__pycache__' in source.parts:
            continue
        relative = Path('vllm') / source.relative_to(wheel_package)
        destination = ROOT / relative
        if destination.exists():
            if digest(source) != digest(destination):
                raise RuntimeError(f'Wheel changed tracked source: {relative}')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            products[str(relative)] = digest(destination)
    for relative, expected in tracked.items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Tracked source changed: {relative}')
    triton = ROOT / 'vllm/third_party/triton_kernels'
    if not (triton / 'matmul_ogs.py').is_file():
        raise RuntimeError('Built Triton kernel payload is absent')
    link = SITE / 'triton_kernels'
    # Only the fresh image's venv is modified. The NGC foundation stays intact.
    if link.exists() or link.is_symlink():
        raise RuntimeError(f'Unexpected pre-existing venv Triton kernels: {link}')
    link.symlink_to(triton, target_is_directory=True)
    if not link.resolve().is_dir():
        raise RuntimeError('Triton kernels symlink does not resolve')
    receipt = Path('/opt/jj-main-build/vllm-payload.json')
    receipt.write_text(json.dumps({'tracked': tracked, 'build_products': products}, indent=2) + '\n')
    subprocess.run(['uv', 'pip', 'freeze', '--python', '/opt/venv/bin/python'], check=True)


if __name__ == '__main__':
    main()
