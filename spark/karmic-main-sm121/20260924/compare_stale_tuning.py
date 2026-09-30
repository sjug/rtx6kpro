"""Record diagnostic boot selection changes; do not equate retuning with a fix."""
import json
from pathlib import Path
from claude_dense_regression import enumerate_winners

root = Path(__file__).resolve().parent
_, old = enumerate_winners(root / 'receipts/dense-release-live-tuning-initial.json', turbo=False)
baseline = {tuple(row[:3]): row[3] for row in old}
report = {}
for node in ('dusty', 'toby', 'rusty', 'kirby'):
    _, new = enumerate_winners(root / f'receipts/stale-probe-tuning-{node}.json', turbo=False)
    current = {tuple(row[:3]): row[3] for row in new}
    if current.keys() != baseline.keys():
        raise RuntimeError('Dense shape inventory changed on ' + node)
    changes = [{'K_N_capacity': key, 'prior': baseline[key], 'current': value}
               for key, value in current.items() if baseline[key] != value]
    report[node] = {'programs': len(current), 'changed': changes}
    print(node, len(current), 'programs;', len(changes), 'changed assignments')
(root / 'receipts/stale-probe-tuning-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
