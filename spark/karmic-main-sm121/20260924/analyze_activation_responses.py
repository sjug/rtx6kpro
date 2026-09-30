"""Join nonmodal responses to covered trace epochs. Clock-based joins are evidence, not causal proof."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from claude_act_analyze import load, step_key


def correlate(responses, traces, analysis):
    counts = defaultdict(Counter)
    for r in responses:
        counts[r['length']][r['signature']] += 1
    modes = {n: counter.most_common(1)[0][0] for n, counter in counts.items()
             if counter.most_common(1)[0][1] * 2 > sum(counter.values())}
    epochs = defaultdict(list)
    for r in traces:
        epochs[r['epoch']].append(r)
    differences = {d['epoch']: d for d in analysis['divergent_steps']}
    result = []
    for response in responses:
        if response['length'] not in modes or response['signature'] == modes[response['length']]:
            continue
        row = {'response': response, 'verdict': 'inconclusive-untraced-or-unmatched'}
        candidates = [r for r in traces if r['rank'] == 0 and r['rows'] == response['length']
                      and response['started_unix'] <= r['wall'] <= response['ended_unix']]
        if len(candidates) == 1:
            epoch = candidates[0]['epoch']
            records = epochs[epoch]
            covered = (len(records) == 4 and {r['rank'] for r in records} == set(range(4))
                       and len({step_key(r) for r in records}) == 1
                       and all(not r.get('overflow') and not r.get('dropped_before', 0)
                               and any(n.startswith('layers.') for n in r['names'])
                               and bool({'<output>', '<output>:0'} & set(r['names'])) for r in records))
            row['epoch'] = epoch
            if covered and epoch in differences:
                row.update(verdict='activation-divergence-observed', finding=differences[epoch])
            elif covered:
                # Only call a record modal if its whole vector is a strict majority in its input group.
                modal = True
                for record in records:
                    vectors = Counter((tuple(r['names']), tuple(map(tuple, r['hashes']))) for r in traces
                                      if r['rank'] == record['rank'] and step_key(r) == step_key(record))
                    vector = (tuple(record['names']), tuple(map(tuple, record['hashes'])))
                    if sum(vectors.values()) < 3 or vectors[vector] * 2 <= sum(vectors.values()):
                        modal = False
                if modal:
                    row['verdict'] = 'model-body-digests-modal-investigate-post-model'
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--trace-block', type=int, required=True)
    args = parser.parse_args()
    trace_dir = args.experiment / f'traces-{args.trace_block:02d}'
    analysis = json.loads((trace_dir / 'analysis.json').read_text())
    traces = [r for r in load(list(trace_dir.glob('*/*.jsonl'))) if r.get('arm') == analysis['arm']]
    responses = []
    for path in sorted(args.experiment.glob('block-*/records.json')):
        responses.extend(dict(r, block=path.parent.name) for r in json.loads(path.read_text()))
    report = {'note': 'Joins require one head trace within the client request interval; host clock offsets were not corrected. '
                      'Modal digests narrow the search but do not prove equality of all internal state.',
              'responses': len(responses), 'findings': correlate(responses, traces, analysis)}
    output = trace_dir / 'response-correlation.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
