#!/usr/bin/env python3
"""Check startup selection before inference, preserving rank-specific log evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

NAME = 'qwen38-flash-next-nvfp4-karmic-main-tp2'
MARKER = 'Sharding HyperConnection projections across 2 ranks'


def check_log(log, enabled):
    # Absence alone is not evidence: require completed model construction first.
    if 'Model loading took' not in log:
        raise RuntimeError('Incomplete model-loading log; cannot check HC selection')
    if (MARKER in log) != enabled:
        raise RuntimeError('HC sharding log disagrees with the selected arm')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['baseline', 'hc-off', 'hc-on-return'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + 1200
    while True:
        try:
            with urllib.request.urlopen('http://dusty:8000/v1/models', timeout=10) as response:
                models = json.load(response)
            if [m['id'] for m in models['data']] != ['Qwen3.8-Flash-Next']:
                raise RuntimeError('Unexpected endpoint identity')
            break
        except OSError as error:
            if time.monotonic() >= deadline:
                raise RuntimeError('HC boot gate readiness deadline exceeded') from error
            print(f'Waiting for HC boot log gate: {error}', flush=True)
            time.sleep(10)
    enabled = args.arm != 'hc-off'
    for node in ('dusty', 'kirby'):
        result = subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', '-o',
            'ConnectTimeout=10', node, f'podman logs --timestamps {NAME}'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=60, check=True)
        (args.out / f'{node}-hc-boot.log').write_text(result.stdout)
        check_log(result.stdout, enabled)
        inspect = json.loads(subprocess.check_output(['ssh', '-n', '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=10', node, f'podman inspect {NAME}'], text=True, timeout=30))[0]
        original = json.loads((args.out / f'{node}-container.json').read_text())[0]
        if not inspect['State']['Running'] or inspect['Id'] != original['Id']:
            raise RuntimeError('Container changed during readiness')
        if args.arm != 'baseline':
            expected = f'VLLM_QWEN3_8_FLASH_NEXT_HC_TP={int(enabled)}'
            if expected not in inspect['Config']['Env']:
                raise RuntimeError('HC environment does not match arm')
    print(f'HC-BOOT-SELECTION-PASS: {args.arm}; correctness still required', flush=True)


if __name__ == '__main__':
    main()
