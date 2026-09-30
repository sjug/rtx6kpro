"""Summarize per-rank Engram traces and the replicated-output invariant."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
nodes = ('dusty', 'toby', 'rusty', 'kirby')
lengths = sorted({int(p.stem.split('-')[1]) for p in root.glob('*-analysis.json')})
for length in lengths:
    data = {node: json.loads((root / f'{node}-{length}-analysis.json').read_text())
            for node in nodes}
    print('LENGTH', length)
    for node, comparisons in data.items():
        for repeat, comparison in enumerate(comparisons, 1):
            changed = []
            for row in comparison['operators']:
                if not row['name'].startswith('engram_'):
                    continue
                for group in ('inputs', 'kwargs', 'outputs'):
                    for name, value in row[group].items():
                        if not value['exact_bytes']:
                            changed.append((row['name'], group, name, value))
            print(node, 'repeat', repeat, json.dumps(changed))
    count = len(data[nodes[0]]) + 1
    for repeat in range(count):
        for op, group, field in (
            ('engram_allreduce', 'outputs', 'rows'),
            ('engram_projection', 'inputs', 'source'),
            ('engram_projection', 'outputs', 'projected_kv'),
            ('engram_mix', 'inputs', 'arg0'),
            ('engram_mix', 'outputs', 'return'),
        ):
            hashes = {}
            for node in nodes:
                comparison = data[node][max(0, repeat - 1)]
                key = 'engram_baseline' if repeat == 0 else 'engram_other'
                row, = (r for r in comparison[key] if r['name'] == op)
                hashes[node] = row[group][field]['sha256']
            print('CROSS-RANK', repeat, op, group, field,
                  'exact=' + str(len(set(hashes.values())) == 1), json.dumps(hashes))
    for node in nodes:
        for repeat in range(count):
            comparison = data[node][max(0, repeat - 1)]
            key = 'engram_baseline' if repeat == 0 else 'engram_other'
            row, = (r for r in comparison[key] if r['name'] == 'engram_epochs')
            print('EPOCHS', node, repeat,
                  {k: v['values'] for k, v in row['outputs'].items()})
