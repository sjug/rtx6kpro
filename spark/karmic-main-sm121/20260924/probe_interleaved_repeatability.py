"""Randomly interleave frozen cold prompts to separate shape from time effects.

Uses the existing probe unchanged for requests, exact answers and cache checks.
This is a diagnostic, not a promotion gate. Cross-cycle signatures, timings and
the full individual request/response/cache receipts are retained.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

from cache_metrics import total

ROOT = Path(__file__).resolve().parent
TIMINGS = {
    'ttft': 'vllm:time_to_first_token_seconds_sum',
    'prefill': 'vllm:request_prefill_time_seconds_sum',
    'queue': 'vllm:request_queue_time_seconds_sum',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lengths', default='127,128,143,247,248,249,250,251,252,253,254,255,256,257,258,271,383,384,511,512,767,1023')
    parser.add_argument('--cycles', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20260925)
    args = parser.parse_args()
    lengths = [int(n) for n in args.lengths.split(',')]
    if not lengths or len(set(lengths)) != len(lengths) or min(lengths) < 100 or args.cycles < 2:
        parser.error('Require unique lengths >=100 and at least two cycles')
    args.out.mkdir(parents=True, exist_ok=False)
    script = ROOT / 'probe_repeatability.py'
    rng = random.Random(args.seed)
    order = [rng.sample(lengths, len(lengths)) for _ in range(args.cycles)]
    (args.out / 'manifest.json').write_text(json.dumps({
        'seed': args.seed, 'orders': order, 'probe_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    signatures = {n: Counter() for n in lengths}
    records = []
    for cycle, permutation in enumerate(order):
        out = args.out / f'cycle-{cycle:03d}'
        command = [sys.executable, '-u', str(script), '--corpus',
                   str(ROOT / 'receipts/determinism-corpus.json'), '--out', str(out),
                   '--lengths', ','.join(map(str, permutation)), '--repeats', '1']
        with (args.out / f'cycle-{cycle:03d}.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Request/cold-cache/correctness gate failed: ' + str(out))
        for length in permutation:
            data = json.loads((out / f'{length}-0.json').read_text())
            choice = data['response']['choices'][0]
            signature = hashlib.sha256(json.dumps({key: choice[key] for key in
                ('message', 'finish_reason', 'logprobs')}, sort_keys=True).encode()).hexdigest()
            cache = json.loads((out / f'{length}-0-cache.json').read_text())
            before, after = cache['before'], cache['after_samples'][-1]
            timings = {label: total(after['raw'], metric) - total(before['raw'], metric)
                       for label, metric in TIMINGS.items()}
            row = {'cycle': cycle, 'length': length, 'signature': signature,
                   'started_unix': before['unix'], 'ended_unix': after['unix'],
                   'elapsed_s': data['elapsed_s'], 'server_seconds': timings,
                   'unexplained_ttft_ge_5s': timings['ttft'] - timings['prefill'] - timings['queue'] >= 5}
            signatures[length][signature] += 1
            records.append(row)
        summary = [{'length': n, 'trials': sum(groups.values()), 'unique_signatures': len(groups),
                    'nonmodal_trials': sum(groups.values()) - max(groups.values()),
                    'signature_counts': dict(groups)} for n, groups in signatures.items()]
        (args.out / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'cycle': cycle, 'completed_requests': len(records),
                          'divergent_lengths': [row for row in summary if row['unique_signatures'] > 1]}), flush=True)
    print('INTERLEAVED-DIAGNOSTIC-COMPLETE', flush=True)


if __name__ == '__main__':
    main()
