"""Explicit 14 original cells plus one clean supplement, never a rewritten grid."""
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import geometric_mean

paths = list(map(Path, sys.argv[1:]))
if len(paths) != 3:
    raise ValueError('Supply baseline, original grid, and supplemental grid')
baseline, original, supplement = [json.loads(p.read_text()) for p in paths]
keys = {(c, m) for c in (1, 2, 4) for m in (0, 16384, 32768, 65536, 131072)}
def index(data, expected):
    rows = {(r['concurrency'], r['context_tokens']): r for r in data['results']}
    if set(rows) != expected or len(rows) != len(data['results']):
        raise ValueError('Missing or duplicate cells')
    return rows

a, b, s = index(baseline, keys), index(original, keys), index(supplement, {(4, 16384)})
if not b[4, 16384]['warmup_timed_out']:
    raise ValueError('Original receipt no longer records the known contaminated cell')
for field in ('model', 'decode_mode', 'duration_per_test', 'decode_warmup_seconds',
              'chat_template_kwargs', 'temperature', 'max_tokens', 'ignore_eos', 'max_total_tokens'):
    if not baseline['metadata'][field] == original['metadata'][field] == supplement['metadata'][field]:
        raise ValueError(f'Decode protocol changed: {field}')
for field in ('checkpoint_revision', 'harness_sha256', 'nodes'):
    if not baseline['run_metadata'][field] == original['run_metadata'][field] == supplement['run_metadata'][field]:
        raise ValueError(f'Identity changed: {field}')
if original['run_metadata']['image_id'] != supplement['run_metadata']['image_id']:
    raise ValueError('Candidate image changed')
# In-memory comparison only; both raw files remain immutable.
b = {**b, **s}
for rows in (a, b):
    for row in rows.values():
        for flag in ('num_errors', 'warmup_timed_out', 'capacity_limited', 'underfilled'):
            if row[flag]:
                raise ValueError(f'Invalid selected row: {flag}')
        for metric in ('aggregate_tps', 'server_steps_per_s', 'server_accept_len_effective'):
            if not math.isfinite(row[metric]) or row[metric] <= 0:
                raise ValueError(f'Invalid metric: {metric}')

summary = []
for concurrency in (1, 2, 4):
    item = {'concurrency': concurrency}
    for metric in ('aggregate_tps', 'server_steps_per_s', 'server_accept_len_effective'):
        av, bv = [geometric_mean(r[metric] for (c, _), r in rows.items() if c == concurrency)
                  for rows in (a, b)]
        item[metric] = {'r38': av, 'r38p': bv, 'change_pct': 100 * (bv / av - 1)}
    summary.append(item)
print(json.dumps({
    'scope': 'Historical R38 versus 14 valid original R38p cells plus separate C4/16K repeat; not a single clean sweep or matched multi-boot study',
    'inputs': {name: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
               for name, p in zip(('baseline', 'original_flagged_grid', 'supplement'), paths)},
    'replacement_cell': [4, 16384], 'summary': summary,
}, indent=2, allow_nan=False))
