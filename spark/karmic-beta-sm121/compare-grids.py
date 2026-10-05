#!/usr/bin/env python3
"""Read-only comparison of complete matched standard-harness grids."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import geometric_mean

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prior_compare',
    ROOT.parent / 'glm53/r38-spark/qualification/compare-qwen-grids.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
parser = argparse.ArgumentParser()
parser.add_argument('baseline', type=Path)
parser.add_argument('candidate', type=Path)
# A checkpoint qualification compares two revisions on an otherwise matched grid.
parser.add_argument('--checkpoint-change', action='store_true')
args = parser.parse_args()
baseline, left = prior.load(args.baseline)
candidate, right = prior.load(args.candidate)
for key in ('version', 'decode_mode', 'duration_per_test', 'decode_warmup_seconds',
            'context_lengths', 'ignore_eos', 'max_tokens', 'chat_template_kwargs',
            'temperature', 'prefill_mode', 'concurrency_levels'):
    if baseline['metadata'][key] != candidate['metadata'][key]:
        raise RuntimeError(f'Protocol drift: {key}')
for key in ('recurrent_checkpoint_policy',) if args.checkpoint_change else ('checkpoint_revision', 'recurrent_checkpoint_policy'):
    if not baseline['run_metadata'].get(key) or baseline['run_metadata'][key] != candidate['run_metadata'].get(key):
        raise RuntimeError(f'Identity drift: {key}')
def harness(data):
    meta = data['run_metadata']
    return meta.get('llm_decode_bench_sha256', meta.get('harness_sha256'))
if not harness(baseline) or harness(baseline) != harness(candidate):
    raise RuntimeError('Harness digest differs')
if baseline['metadata']['model'] != candidate['metadata']['model']:
    raise RuntimeError('Model names differ; establish alias equivalence before comparison')
def values(a, b):
    return {'baseline': a, 'candidate': b, 'change_pct': (b / a - 1) * 100}
metrics = ('aggregate_tps', 'server_steps_per_s', 'server_accept_len_effective')
report = {'repeatable_gain_proven': False, 'grids_valid': True,
    'checkpoint_change': args.checkpoint_change, 'inputs': {
    name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for name, path in (('baseline', args.baseline), ('candidate', args.candidate))},
    'summary': [], 'cells': [], 'prefill': []}
for concurrency in baseline['metadata']['concurrency_levels']:
    report['summary'].append({'concurrency': concurrency, **{
        metric: values(geometric_mean(row[metric] for (c, _), row in left.items() if c == concurrency),
                       geometric_mean(row[metric] for (c, _), row in right.items() if c == concurrency))
        for metric in metrics}})
for key in sorted(left):
    report['cells'].append({'concurrency': key[0], 'context_tokens': key[1],
        **{metric: values(left[key][metric], right[key][metric]) for metric in metrics}})
for context in sorted(baseline['prefill'], key=int):
    a, b = baseline['prefill'][context], candidate['prefill'][context]
    report['prefill'].append({'context_tokens': int(context),
        **values(a['tok_per_sec'], b['tok_per_sec']),
        'baseline_server_validation': a.get('server_validation'),
        'candidate_server_validation': b.get('server_validation')})
print(json.dumps(report, indent=2, allow_nan=False))
