#!/usr/bin/env python3
"""Compare qualified whole profiles, explicitly not a matched-policy engine A/B."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from statistics import geometric_mean

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('grid', ROOT / 'glm53/r38-spark/qualification/compare-qwen-grids.py')
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)
paths = [Path(p) for p in sys.argv[1:]]
if len(paths) != 2:
    raise SystemExit('baseline and candidate paths required')
base, left = grid.load(paths[0])
candidate, right = grid.load(paths[1])
for key in ('version', 'decode_mode', 'duration_per_test', 'decode_warmup_seconds',
            'context_lengths', 'ignore_eos', 'max_tokens', 'chat_template_kwargs',
            'temperature', 'prefill_mode', 'concurrency_levels', 'model'):
    if base['metadata'][key] != candidate['metadata'][key]:
        raise SystemExit('Benchmark protocol drift: ' + key)
if base['run_metadata'].get('checkpoint_revision') != candidate['run_metadata'].get('checkpoint_revision'):
    raise SystemExit('Checkpoint drift')
if (base['run_metadata'].get('recurrent_checkpoint_policy'), candidate['run_metadata'].get('recurrent_checkpoint_policy')) != ('aligned', 'request_boundaries'):
    raise SystemExit('Unexpected profile comparison')
def harness(data):
    m = data['run_metadata']
    return m.get('llm_decode_bench_sha256', m.get('harness_sha256'))
if not harness(base) or harness(base) != harness(candidate):
    raise SystemExit('Benchmark harness drift')
def comparison(a, b):
    return {'r38': a, 'candidate': b, 'change_pct': 100 * (b / a - 1)}
report = {'scope': 'whole-profile comparison; multiple defaults changed; no engine-only attribution',
          'repeatability_established': False,
          'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
          'summary': [], 'prefill': [], 'cells': []}
metrics = ('aggregate_tps', 'server_steps_per_s', 'server_accept_len_effective')
for c in (1, 2, 4):
    report['summary'].append({'concurrency': c, **{
        m: comparison(geometric_mean(r[m] for (cc, _), r in left.items() if cc == c),
                      geometric_mean(r[m] for (cc, _), r in right.items() if cc == c)) for m in metrics}})
for key in sorted(left):
    report['cells'].append({'concurrency': key[0], 'context_tokens': key[1],
                            **{m: comparison(left[key][m], right[key][m]) for m in metrics}})
for ctx in sorted(base['prefill'], key=int):
    report['prefill'].append({'context_tokens': int(ctx),
                             **comparison(base['prefill'][ctx]['tok_per_sec'], candidate['prefill'][ctx]['tok_per_sec'])})
print(json.dumps(report, indent=2, allow_nan=False))
