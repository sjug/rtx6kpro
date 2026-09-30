"""Fail-closed entrypoint for the approved upstream-derived DS4.1 profile."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys

from launch_contract import render, validate_chunking_image

ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def kit_digest():
    manifest = json.loads((ROOT / 'runtime-files.json').read_text())
    for name, expected in manifest.items():
        require('/' not in name and name not in ('.', '..'), 'Unsafe runtime manifest path')
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        require(actual == expected, f'Runtime file drift: {name}')
    return hashlib.sha256((ROOT / 'runtime-files.json').read_bytes()).hexdigest()


def disk_probe():
    from array import array
    from b12x.loader._native import load
    native = load()
    shard = Path('/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/'
                 'fb2764a5cf321eaa5070ca8f9e892818f477c16d/model-00001-of-00048.safetensors')
    rows = array('q', [0, 7, 15])
    weights, scales = bytearray(3 * 256), bytearray(3 * 8)
    reader = native.ple_reader(16, 16, 0, 16, 256, 8, 128, 128)
    try:
        native.ple_reader_add(reader, 0, str(shard), 37, False)
        native.ple_reader_add(reader, 0, str(shard), 8205, True)
        native.ple_reader_run(reader, rows, weights, scales, len(rows))
        with shard.open('rb', buffering=0) as source:
            for width, offset, output in ((256, 37, weights), (8, 8205, scales)):
                expected = bytearray()
                for row in rows:
                    source.seek(offset + row * width)
                    expected.extend(source.read(width))
                require(output == expected, 'io_uring checkpoint byte parity failed')
    finally:
        del reader
    print('DS41-IO-URING-BYTE-PARITY-PASS', flush=True)


def main():
    require(__debug__, 'Optimized Python is not admitted')
    require(kit_digest() == os.environ['DS41_KIT_SHA256'], 'Runner/entrypoint identity mismatch')
    from diagnostic_overlay import validate as validate_overlay
    pin = json.loads((ROOT / 'candidate.json').read_text())
    validate_overlay(pin, ROOT, installed=True)
    if 'DS41_DECISION_ROW_BLOCKS' in os.environ:
        require(pin.get('diagnostic', {}).get('kind') in ('decision-row-capture', 'window-capture', 'indexer-capture', 'precision-capture',
                                                              'precision-release-candidate', 'ratio1-bf16-diagnostic',
                                                              'mhc-expanded-capture'),
                'KV block pin is restricted to decision-row-capture or window-capture')
    row = render(os.environ['DS41_NODE'])
    validate_chunking_image(pin, row['env'])
    for key, value in row['env'].items():
        require(os.environ.get(key) == value, f'Effective environment differs: {key}')
    for key in row['unset']:
        require(key not in os.environ, f'Unexpected inherited tuning: {key}')
    require(resource.getrlimit(resource.RLIMIT_MEMLOCK)[0] == resource.RLIM_INFINITY,
            'Effective memlock is bounded; approval needed for override')
    require(os.environ['LD_LIBRARY_PATH'].split(':')[0] == row['ld_library_path_prefix'],
            'Wrong library search order after Bash activation')
    from gate_ds41 import check_nccl, check_cli, check_loader, check_limits
    check_limits()
    check_nccl()
    check_cli()
    check_loader()
    disk_probe()
    print(json.dumps(row, sort_keys=True), flush=True)
    os.execv(row['model'][0], row['model'])


if __name__ == '__main__':
    main()
