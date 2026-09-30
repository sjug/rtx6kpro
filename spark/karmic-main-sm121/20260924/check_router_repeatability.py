"""Validate completed full-model router comparison receipts, never just exit 0."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(summary, records, cycles=1000):
    lengths = {254, 256, 258}
    if len(summary) != 3 or {r['length'] for r in summary} != lengths:
        raise RuntimeError('Incomplete shape coverage')
    if len(records) != cycles * 3:
        raise RuntimeError('Incomplete request count')
    keys = {(r['cycle'], r['length']) for r in records}
    if keys != {(cycle, length) for cycle in range(cycles) for length in lengths}:
        raise RuntimeError('Duplicate or missing request identities')
    from collections import Counter
    totals, modes = {}, {}
    for row in summary:
        counts = Counter(r['signature'] for r in records if r['length'] == row['length'])
        nonmodal = cycles - max(counts.values())
        if (row['trials'] != cycles or row['unique_signatures'] != len(counts)
                or row['signature_counts'] != dict(counts)
                or row['nonmodal_trials'] != nonmodal):
            raise RuntimeError('Summary does not match individual requests')
        totals[row['length']] = nonmodal
        modes[row['length']] = max(counts, key=counts.get)
    return {'nonmodal_by_length': totals, 'nonmodal_total': sum(totals.values()),
            'modal_signatures': modes,
            'requests': len(records),
            'long_unexplained_gaps': sum(bool(r['unexplained_ttft_ge_5s']) for r in records)}


def validate_raw(probe, records):
    for row in records:
        path = probe / f'cycle-{row["cycle"]:03d}' / f'{row["length"]}-0.json'
        receipt = json.loads(path.read_text())
        choice = receipt['response']['choices'][0]
        signature = hashlib.sha256(json.dumps({key: choice[key] for key in
            ('message', 'finish_reason', 'logprobs')}, sort_keys=True).encode()).hexdigest()
        if signature != row['signature']:
            raise RuntimeError('Raw response differs from record: ' + str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--require-clean', action='store_true')
    args = parser.parse_args()
    if (args.receipt / 'exit-code').read_text().strip() != '0':
        raise RuntimeError('Observed run did not finish successfully')
    if 'INTERLEAVED-DIAGNOSTIC-COMPLETE' not in (args.receipt / 'run.log').read_text().splitlines():
        raise RuntimeError('Missing completion marker')
    for node in ('dusty', 'toby', 'rusty', 'kirby'):
        for kind in ('container', 'kernel'):
            if (args.receipt / f'{node}-{kind}.exit-code').read_text().strip() != '0':
                raise RuntimeError('Incomplete log collection')
    probe = args.receipt / 'probe'
    records = json.loads((probe / 'records.json').read_text())
    result = validate(json.loads((probe / 'summary.json').read_text()), records)
    validate_raw(probe, records)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.require_clean and (result['nonmodal_total'] or result['long_unexplained_gaps']):
        raise RuntimeError('Router comparison is not clean')


if __name__ == '__main__':
    main()
