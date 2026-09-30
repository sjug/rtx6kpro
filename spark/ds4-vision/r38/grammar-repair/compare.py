#!/usr/bin/env python3
"""Compare the historical R38 DS4 Vision grid with this pinned candidate."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import geometric_mean

parser = argparse.ArgumentParser()
parser.add_argument('baseline', type=Path)
parser.add_argument('candidate', type=Path)
args = parser.parse_args()
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
for data, image in ((left, 'ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5'),
                    (right, 'ab3ed5285a81c9fd4df3c9001021c420368e7667177470aa2e055420f89ecdcf')):
    for key, expected in {
        'image_id': image,
        'checkpoint_revision': '6821d6ad3681a4b137b066b76094fa82ebd0a380',
        'harness_sha256': '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3',
        'nodes': 'rusty,toby',
    }.items():
        if data['run_metadata'].get(key) != expected:
            raise ValueError(f'Identity mismatch: {key}')

def delta(a, b):
    return {'r38': a, 'r38p': b, 'change_pct': 100 * (b / a - 1)}

summary = []
for c in (1, 2, 4):
    summary.append({'concurrency': c, **{
        metric: delta(*(geometric_mean(row[metric] for row in data['results']
                                        if row['concurrency'] == c)
                        for data in (left, right)))
        for metric in ('aggregate_tps', 'server_steps_per_s', 'server_accept_len_effective')}})
print(json.dumps({
    'scope': 'Historical R38 comparison, not a matched multi-boot study',
    'inputs': {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
               for name, path in (('baseline', args.baseline), ('candidate', args.candidate))},
    'summary': summary,
    'prefill': [{'context_tokens': int(context),
                **delta(left['prefill'][context]['tok_per_sec'], right['prefill'][context]['tok_per_sec'])}
               for context in sorted(left['prefill'], key=int)],
}, indent=2, allow_nan=False))
