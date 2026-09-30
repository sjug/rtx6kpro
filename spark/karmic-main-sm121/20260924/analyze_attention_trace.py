"""Compare complete diagnostic tensor bytes between cold forwards on one rank."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

# This is an offline CPU byte comparison, not a serving or compilation policy.
# Avoid sixteen-thread launch overhead for hundreds of small tensor checks.
torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument('--engram-only', action='store_true', help='Compare the localized projection without paging in unrelated layers')
parser.add_argument('traces', type=Path, nargs='+')
args = parser.parse_args()
if len(args.traces) < 2:
    raise RuntimeError('At least two traces are required')
baseline = torch.load(args.traces[0], map_location='cpu', weights_only=True, mmap=True)
expected_layers = 5 if 'operators' in baseline else 40
if len(baseline['layers']) != expected_layers:
    raise RuntimeError('Incomplete baseline trace')
report = []


def add_logical_scale_view(trace):
    for row in trace.get('operators', []):
        if row['name'] != 'engram_projection' or 'quantized_scale_mma' not in row['outputs']:
            continue
        m, k = row['inputs']['source'].shape
        rr = torch.arange(m)[:, None]
        gg = torch.arange(k // 32)[None, :]
        mma = row['outputs']['quantized_scale_mma']
        logical = mma[rr % 32, (rr // 32) % 4, rr // 128, gg % 4, gg // 4, 0]
        row['outputs']['quantized_scale_mma_logical'] = logical
        row['logical_scale_layout_agrees'] = torch.equal(logical, row['outputs']['quantized_scale_rows'][0])


add_logical_scale_view(baseline)


def engram_identity(trace):
    records = []
    for row in trace.get('operators', []):
        if not row['name'].startswith('engram_'):
            continue
        item = {'name': row['name'], 'logical_scale_layout_agrees': row.get('logical_scale_layout_agrees'),
                'addresses': row.get('addresses')}
        for key in ('dense_config', 'dense_policy', 'dense_rows', 'dense_programs'):
            if key in row:
                item[key] = row[key]
        if row['name'] == 'engram_projection':
            item['replay_comparisons'] = {
                name: compare_tensor(row['outputs']['projected_kv'], row['outputs'][name])
                for name in ('dense_replay', 'dense_replay_synchronized')
                if name in row['outputs']}
        for group in ('inputs', 'kwargs', 'outputs'):
            item[group] = {}
            for key, value in row[group].items():
                raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                entry = {'sha256': hashlib.sha256(raw).hexdigest(),
                         'shape': list(value.shape), 'dtype': str(value.dtype)}
                if row['name'] == 'engram_epochs' and group == 'outputs':
                    entry['values'] = value.tolist()
                item[group][key] = entry
        records.append(item)
    return records


def compare_tensor(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        raise RuntimeError('Trace geometry drift')
    exact = torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
    result = {'exact_bytes': exact, 'shape': list(a.shape)}
    if not exact:
        result.update(differing_elements=int(torch.count_nonzero(a != b)),
                      max_abs=float((a.float() - b.float()).abs().max()))
    return result


for path in args.traces[1:]:
    other = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    add_logical_scale_view(other)
    if other['node'] != baseline['node'] or len(other['layers']) != expected_layers:
        raise RuntimeError('Cross-rank or incomplete trace comparison')
    layers = []
    for left, right in zip(baseline['layers'], other['layers'], strict=True):
        if args.engram_only:
            break
        if (left['layer'], left['prefix']) != (right['layer'], right['prefix']):
            raise RuntimeError('Layer identity drift')
        row = {'layer': left['layer']}
        for field in ('positions', 'input', 'output'):
            a, b = left[field], right[field]
            if a.dtype != b.dtype or a.shape != b.shape:
                raise RuntimeError('Trace geometry drift')
            exact = torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
            item = {'exact_bytes': exact, 'shape': list(a.shape)}
            if not exact:
                item.update(differing_elements=int(torch.count_nonzero(a != b)),
                            max_abs=float((a.float() - b.float()).abs().max()))
            row[field] = item
        layers.append(row)
    operators = []
    for i, (left, right) in enumerate(zip(baseline.get('operators', []), other.get('operators', []), strict=True)):
        if args.engram_only and not left['name'].startswith('engram_'):
            continue
        for field in ('name', 'after_attention_layer', 'query'):
            if left.get(field) != right.get(field):
                raise RuntimeError('Operator identity drift: ' + field)
        row = {'index': i, 'name': left['name'], 'after_attention_layer': left['after_attention_layer']}
        for group in ('inputs', 'kwargs', 'outputs'):
            if left[group].keys() != right[group].keys():
                raise RuntimeError('Operator tensor inventory drift')
            row[group] = {key: compare_tensor(value, right[group][key]) for key, value in left[group].items()}
        operators.append(row)
    report.append({'baseline': str(args.traces[0]), 'other': str(path), 'operators': operators,
                   'comparison_scope': 'engram-only' if args.engram_only else 'full-trace',
                   'engram_baseline': engram_identity(baseline), 'engram_other': engram_identity(other),
                   'first_differing_input': next((r['layer'] for r in layers if not r['input']['exact_bytes']), None),
                   'first_differing_output': next((r['layer'] for r in layers if not r['output']['exact_bytes']), None),
                   'layers': layers})
print(json.dumps(report, indent=2))
