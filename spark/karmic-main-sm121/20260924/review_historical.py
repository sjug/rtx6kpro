#!/usr/bin/env python3
"""Review completed replay receipts against independently sampled cache counters."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent / 'receipts'
samples = [json.loads(line) for line in (root / 'historical-524k-metrics.jsonl').read_text().splitlines()]
# Preserve the exact completed-admission snapshot preceding the first replay;
# the independent sampler can otherwise straddle that preceding completion.
prior = json.loads((root / 'qualification-600k/needle-262000-cache-metrics.json').read_text())
samples += prior['after_samples']
samples.sort(key=lambda s: s['unix'])
rows = []
for path in sorted((root / 'historical-524k-600k').glob('*-cold-*.json')):
    record = json.loads(path.read_text())
    if 'response' not in record:
        continue
    start = record['started_unix']
    end = start + record['elapsed_s']
    before = [s for s in samples if s['unix'] <= start]
    after = [s for s in samples if s['unix'] >= end]
    if not before or not after:
        continue
    baseline = before[-1]
    # Metrics publish asynchronously. A subsequent request may already be admitted,
    # but successes delimit completion and hits must remain unchanged throughout.
    completed = next((s for s in after if s['successes'] >= baseline['successes'] + 1), None)
    if completed is None:
        continue
    window = [s for s in samples if baseline['unix'] <= s['unix'] <= completed['unix']]
    response = record['response']
    count = response['usage']['prompt_tokens']
    admissions = [s['queries'] - baseline['queries'] for s in window]
    clean = (count in admissions and completed['successes'] - baseline['successes'] == 1
             and all(s['hits'] == baseline['hits'] and s['preemptions'] == baseline['preemptions']
                     and s['running'] <= 1 and s['waiting'] == 0 for s in window))
    choice = response['choices'][0]
    first = choice['logprobs']['content'][0]['top_logprobs']
    probs = {item['token']: item['logprob'] for item in first}
    rows.append({'label': record['label'], 'elapsed_s': record['elapsed_s'],
                 'prompt_tokens': count, 'answer': choice['message']['content'],
                 'exact': choice['message']['content'].strip() == record['expected'],
                 'finish_reason': choice['finish_reason'], 'cold_counter_verified': clean,
                 'input_sha256': record['input_sha256'],
                 'code_minus_identity_logprob': probs['739'] - probs['510']
                     if '739' in probs and '510' in probs else None,
                 'counter_before': {k: v for k, v in baseline.items() if k != 'raw'},
                 'counter_completed': {k: v for k, v in completed.items() if k != 'raw'},
                 'receipt_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
originals = [row for row in rows if row['label'].startswith('orig-')]
series = None
if len(originals) == 3:
    baseline = originals[0]['counter_before']
    completed = originals[-1]['counter_completed']
    window = [s for s in samples if baseline['unix'] <= s['unix'] <= completed['unix']]
    series = {
        'queries': completed['queries'] - baseline['queries'],
        'successes': completed['successes'] - baseline['successes'],
        'cold_verified': (completed['queries'] - baseline['queries'] == 3 * 524288
                          and completed['successes'] - baseline['successes'] == 3
                          and all(s['hits'] == baseline['hits']
                                  and s['preemptions'] == baseline['preemptions']
                                  and s['running'] <= 1 and s['waiting'] == 0 for s in window)),
        'note': 'Aggregate interval covers all three originals without relying on inter-request sampling races',
    }
full_series = None
if len(rows) == 4:
    baseline = rows[0]['counter_before']
    completed = rows[-1]['counter_completed']
    window = [s for s in samples if baseline['unix'] <= s['unix'] <= completed['unix']]
    full_series = {
        'cold_verified': (completed['queries'] - baseline['queries'] == sum(r['prompt_tokens'] for r in rows)
                          and completed['successes'] - baseline['successes'] == 4
                          and all(s['hits'] == baseline['hits']
                                  and s['preemptions'] == baseline['preemptions']
                                  and s['running'] <= 1 and s['waiting'] == 0 for s in window)),
        'queries': completed['queries'] - baseline['queries'],
        'successes': completed['successes'] - baseline['successes'],
    }
print(json.dumps({'trials': rows, 'original_series': series, 'full_series': full_series}, indent=2))
