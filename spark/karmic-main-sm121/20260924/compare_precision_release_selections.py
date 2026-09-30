"""Offline comparison of the passing diagnostic and failed clean-release selections."""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_decision_reference import exact_regime

ROOT = Path(__file__).resolve().parent
PASSING = ROOT / 'receipts/decision-row-matched8192-nccl-standard-upstream-capture-20260926T233036Z'
RELEASE = ROOT / 'receipts/precision-release-full-qualification-20260927-r2'


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), {'path': str(path.relative_to(ROOT)),
                             'sha256': hashlib.sha256(raw).hexdigest()}


def compare():
    result = {'scope': 'selection differences, not proof of execution or causality', 'nodes': {}}
    for node in ('dusty', 'toby', 'rusty', 'kirby'):
        old, old_source = load(PASSING / f'{node}-selection.json')
        new, new_source = load(RELEASE / 'initial-selections' / f'{node}.json')
        after, after_source = load(RELEASE / 'post-first-replay-selections' / f'{node}.json')
        a, b, c = old['records'], new['records'], after['records']
        common = set(a) & set(b)
        changed = {key: {'passing': a[key]['config'], 'release': b[key]['config']}
                   for key in sorted(common) if a[key]['config'] != b[key]['config']}
        prior_regime, current_regime = exact_regime(old, 81389), exact_regime(new, 80022)
        result['nodes'][node] = {
            'sources': [old_source, new_source, after_source],
            'passing_records': len(a), 'release_records': len(b), 'common_records': len(common),
            'changed_configurations': changed,
            'passing_attention_regime': prior_regime, 'release_attention_regime': current_regime,
            'changed_attention_plans': [key for key in sorted(prior_regime)
                                        if prior_regime[key] != current_regime[key]],
            'during_replays': {
                'added': sorted(set(c) - set(b)), 'removed': sorted(set(b) - set(c)),
                'changed_records': [key for key in sorted(set(b) & set(c)) if b[key] != c[key]],
            },
        }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = compare()
    with args.out.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    for node, row in report['nodes'].items():
        print(node, len(row['changed_configurations']), row['changed_attention_plans'], row['during_replays'])
