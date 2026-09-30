"""Compare retained synthetic snapshots on the workstation, never on a serving GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from claude_stale_state_probe import diff

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
report = json.loads((args.out / 'report.json').read_text())
if report['status'] != 'captured' or not report['reproduced']:
    raise RuntimeError('Capture must reproduce the divergence')
digests = json.loads((args.out / 'snapshots.sha256.json').read_text())
for name, expected in digests.items():
    if hashlib.sha256((args.out / 'snapshots' / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Snapshot digest mismatch: ' + name)
environment = {
    'python': sys.version, 'torch': torch.__version__, 'device': 'cpu'}
(args.out / 'analysis-environment.json').write_text(json.dumps(environment, indent=2) + '\n')
for node in ('dusty', 'toby', 'rusty', 'kirby'):
    paths = [args.out / 'snapshots' / f'{run}-{node}.pt' for run in report['runs']]
    snapshots = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    for snapshot in snapshots:
        if snapshot['seq_len'] != 393 or snapshot['prompt_len'] != 385:
            raise RuntimeError(f'{node}: captured wrong verification step')
        if len(snapshot['layers']) != 54 or snapshot['omitted']:
            raise RuntimeError(f'{node}: incomplete layer coverage')
    result = diff(*paths)
    result['step_input_values'] = [
        {key: value.tolist() for key, value in s['step_inputs'].items()} for s in snapshots]
    (args.out / f'diff-{node}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(node, json.dumps({key: result[key] for key in ('seq_len', 'step_inputs', 'step_input_values', 'extra')}), flush=True)
    print('changed_layers', len(result['layers']), flush=True)
print('STALE-CAPTURE-LOCAL-DIFF-PASS')
