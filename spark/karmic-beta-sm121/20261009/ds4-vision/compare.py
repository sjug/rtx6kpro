#!/usr/bin/env python3
"""Compare a same-window fresh-boot production DS4 Vision grid (image 500ae05b) with this pinned candidate, both on the frozen harness cc9bb06a."""
import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
from statistics import geometric_mean

parser = argparse.ArgumentParser()
parser.add_argument('baseline', type=Path)
parser.add_argument('candidate', type=Path)
args = parser.parse_args()
candidate_image = os.environ['EXPECTED_IMAGE_ID']
if not re.fullmatch(r'[0-9a-f]{64}', candidate_image):
    raise RuntimeError('Expected a gated candidate image ID')
source = Path(__file__).resolve().parents[3] / 'glm53/r38-spark/qualification/compare-qwen-grids.py'
spec = importlib.util.spec_from_file_location('grid_reader', source)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
left, _ = reader.load(args.baseline)
right, _ = reader.load(args.candidate)
for key in ('version', 'model', 'decode_mode', 'duration_per_test',
            'decode_warmup_seconds', 'context_lengths', 'concurrency_levels',
            'chat_template_kwargs', 'temperature', 'prefill_mode', 'max_tokens',
            'ignore_eos', 'max_total_tokens'):
    if left['metadata'][key] != right['metadata'][key]:
        raise ValueError(f'Protocol drift: {key}')
baseline_image = os.environ['BASELINE_IMAGE_ID']  # the same-window production grid's image, never a pinned default
if not re.fullmatch(r'[0-9a-f]{64}', baseline_image):
    raise RuntimeError('Expected the production baseline image ID')
for data, image in ((left, baseline_image), (right, candidate_image)):
    for key, expected in {
        'image_id': image,
        'checkpoint_revision': '6821d6ad3681a4b137b066b76094fa82ebd0a380',
        'harness_sha256': 'cc9bb06a4f1142d2bf08afbcfc0e57d8a465a0bd180eee74e276dea70a82071c',
        'nodes': 'rusty,toby',
    }.items():
        if data['run_metadata'].get(key) != expected:
            raise ValueError(f'Identity mismatch: {key}')

def delta(a, b):
    return {'production': a, 'candidate': b, 'change_pct': 100 * (b / a - 1)}

summary = []
for c in (1, 2, 4):
    summary.append({'concurrency': c, **{
        metric: delta(*(geometric_mean(row[metric] for row in data['results']
                                        if row['concurrency'] == c)
                        for data in (left, right)))
        for metric in ('aggregate_tps', 'server_steps_per_s', 'server_accept_len_effective')}})
print(json.dumps({
    'scope': 'Same-window fresh-boot production comparison, one grid each; boot variance not separated',
    'inputs': {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
               for name, path in (('baseline', args.baseline), ('candidate', args.candidate))},
    'summary': summary,
    'prefill': [{'context_tokens': int(context),
                **delta(left['prefill'][context]['tok_per_sec'], right['prefill'][context]['tok_per_sec'])}
               for context in sorted(left['prefill'], key=int)],
}, indent=2, allow_nan=False))
