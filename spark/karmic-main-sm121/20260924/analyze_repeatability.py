"""Summarize complete logprob equality separately from top-token correctness."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('receipts', type=Path)
a = p.parse_args()
report = []
for group in json.loads((a.receipts / 'report.json').read_text()):
    n, count = group['length'], group['repeats']
    responses = [json.loads((a.receipts / f'{n}-{i}.json').read_text())['response']
                 for i in range(count)]
    choices = [response['choices'][0] for response in responses]
    tokens = [choice['logprobs']['content'] for choice in choices]
    positions = []
    for position in range(max(map(len, tokens))):
        rows = [row[position] if position < len(row) else None for row in tokens]
        common = None
        maps = []
        for row in rows:
            values = {} if row is None else {entry['token']: entry['logprob'] for entry in row['top_logprobs']}
            maps.append(values)
            common = set(values) if common is None else common.intersection(values)
        spreads = {token: max(row[token] for row in maps) - min(row[token] for row in maps)
                   for token in common}
        positions.append({'position': position,
                          'complete_record_equal': all(row == rows[0] for row in rows),
                          'top_tokens': [None if row is None else row['token'] for row in rows],
                          'common_top_logprob_tokens': len(common),
                          'max_common_logprob_spread': max(spreads.values(), default=None)})
    report.append({'length': n, 'repeats': count, 'full_signature_equal': group['identical'],
                   'answers': [choice['message'].get('content') for choice in choices],
                   'finish_reasons': [choice['finish_reason'] for choice in choices],
                   'positions': positions})
print(json.dumps(report, indent=2))
