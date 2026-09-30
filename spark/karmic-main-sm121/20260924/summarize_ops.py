"""Print the first changed operator tensors from complete per-rank trace analyses."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
for path in sorted(a.directory.glob('*-analysis.json')):
    for comparison in json.loads(path.read_text()):
        print(path.name, comparison['other'])
        found = False
        for op in comparison.get('operators', []):
            changed = [(group, key, data) for group in ('inputs', 'kwargs', 'outputs')
                       for key, data in op[group].items() if not data['exact_bytes']]
            if changed:
                print('FIRST_CHANGED_OPERATOR', op['index'], op['name'],
                      'after_attention_layer', op['after_attention_layer'])
                for group, key, data in changed:
                    print(group, key, json.dumps(data, sort_keys=True))
                found = True
                break
        if not found:
            print('OPERATORS_EXACT', len(comparison.get('operators', [])))
        print('attention_first_input', comparison['first_differing_input'],
              'attention_first_output', comparison['first_differing_output'])
