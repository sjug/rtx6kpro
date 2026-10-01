#!/usr/bin/env python3
"""Require a matched current-production baseline, then compare complete grids."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PRODUCTION_IMAGE = '500ae05b98da0658c1a5e1820387f96f2121c5659bd7954ad1c5861f20934f05'
PROFILES = {
    'qwen': ('Qwen3.8-Flash-Next', '7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd'),
    'glm': ('GLM-5.3-Flash', '175ae8ce3b5af842b0d0140dbeb43e9cfc557c49'),
}
HARNESS = '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3'


def validate_identity(data, model, image):
    metadata = data['run_metadata']
    name, revision = PROFILES[model]
    expected = {'image_id': image, 'checkpoint_revision': revision,
                'recurrent_checkpoint_policy': 'aligned'}
    if model == 'qwen':
        expected['hc_tp'] = '0'
    for key, value in expected.items():
        if str(metadata.get(key)) != value:
            raise RuntimeError(f'Production comparison identity mismatch: {key}')
    if data['metadata']['model'] != name:
        raise RuntimeError('Production comparison model mismatch')
    if metadata.get('llm_decode_bench_sha256', metadata.get('harness_sha256')) != HARNESS:
        raise RuntimeError('Production comparison harness mismatch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=PROFILES)
    parser.add_argument('--validate-baseline', action='store_true')
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path, nargs='?')
    args = parser.parse_args()
    validate_identity(json.loads(args.baseline.read_text()), args.model, PRODUCTION_IMAGE)
    if args.validate_baseline:
        # Complete-grid validation, including errors/capacity and all C1/C2/C4 cells.
        import importlib.util
        spec = importlib.util.spec_from_file_location('grid_reader', ROOT.parents[2] / 'spark/glm53/r38-spark/qualification/compare-qwen-grids.py')
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        data, _ = reader.load(args.baseline)
        if set(data['metadata']['concurrency_levels']) != {1, 2, 4}:
            raise RuntimeError('Production baseline requires C1/C2/C4')
        print('PRODUCTION-BASELINE-VALID')
        return
    if args.candidate is None:
        parser.error('candidate receipt required')
    candidate = json.loads(args.candidate.read_text())
    baseline = json.loads(args.baseline.read_text())
    validate_identity(candidate, args.model, os.environ['EXPECTED_IMAGE_ID'])
    if baseline['metadata'].get('max_total_tokens') != candidate['metadata'].get('max_total_tokens'):
        raise RuntimeError('Production comparison token budget mismatch')
    subprocess.run([sys.executable, str(ROOT.parent / 'compare-grids.py'),
                    str(args.baseline), str(args.candidate)], check=True)


if __name__ == '__main__':
    main()
