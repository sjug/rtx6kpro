#!/usr/bin/env python3
"""Build FlashInfer in the original NGC component environment, before serving assembly."""
import json
import subprocess
import sys
from pathlib import Path

from unpack_sources import main as unpack

ROOT = Path(__file__).resolve().parent


def main():
    # Install only the hash-pinned compiler upgrade into this clean component stage.
    subprocess.run(['uv', 'pip', 'install', '--system', '--break-system-packages', '--python', sys.executable,
                    '--offline', '--no-deps', '--require-hashes', '--find-links', str(ROOT / 'inputs'),
                    '-r', str(ROOT / 'compiler-arm64.lock')], check=True)
    # Run in a fresh interpreter so .pth processing reflects the upgraded compiler.
    subprocess.run([sys.executable, str(ROOT / 'verify_flashinfer_component.py'), '--foundation-only'], check=True)
    unpack()
    source = ROOT / 'build/flashinfer'
    subprocess.run([sys.executable, str(source / 'ci/lil_wheels/write_build_metadata.py')], check=True)
    output = Path('/artifacts/flashinfer')
    for path in (source, source / 'flashinfer-jit-cache'):
        subprocess.run(['uv', 'build', '--python', sys.executable, '--offline', '--wheel',
                        '--no-build-isolation', '--out-dir', str(output), str(path)], check=True)
    wheels = sorted(output.glob('*.whl'))
    subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '--offline',
                    '--no-deps', '--target', '/opt/flashinfer-check', *map(str, wheels)], check=True)
    # Inspect in a new process: the installed wheel must supply both imports.
    import os
    subprocess.run([sys.executable, str(ROOT / 'verify_flashinfer_component.py')],
                   env=dict(os.environ, PYTHONPATH='/opt/flashinfer-check'), check=True)
    (output / 'build-environment.json').write_text(json.dumps({
        'interpreter': sys.executable, 'compiler': '4.7.1',
        'setuptools': '81.0.0', 'wheel': '0.48.0', 'apache-tvm-ffi': '0.1.13.post3',
        'source': 'dbd6238c6655b98195fdf77f04bba6facf5a38a4'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
