"""Fail-closed replay of the original 524K failure, without changing its wording."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request
import uuid
from cache_metrics import idle_snapshot, finish


def first_token_margin(choice):
    """Report the observed greedy margin; this is not a numerical oracle."""
    first = choice['logprobs']['content'][0]
    alternatives = [item for item in first.get('top_logprobs', [])
                    if item['token'] != first['token']]
    if not alternatives:
        raise RuntimeError('Missing competing first-token logprobs')
    competitor = max(alternatives, key=lambda item: item['logprob'])
    selected, competing = first['logprob'], competitor['logprob']
    if not all(math.isfinite(value) for value in (selected, competing)):
        raise RuntimeError('Nonfinite first-token logprobs')
    return {'token': first['token'], 'logprob': selected,
            'competing_token': competitor['token'], 'competing_logprob': competing,
            'margin_nats': selected - competing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base-url', default='http://dusty:8000')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '1b89c7f4d1e1047f6cf7ebe6000d6aff242981cbdbe8d8ce2a742ff61a7d6db6':
        raise RuntimeError('Historical input file identity changed')
    spec = json.loads(raw)
    if hashlib.sha256(spec['messages'][0]['content'].encode()).hexdigest() != 'b410c6b19d492e83ecb775805266d91eed15ded5e3afd0cf264c25aae41bc7e4':
        raise RuntimeError('Historical prompt identity changed')
    if args.repeats < 3:
        raise RuntimeError('At least three cold trials required')
    args.out.mkdir(parents=True, exist_ok=False)
    signatures, report = [], []
    for repeat in range(args.repeats):
        body = {'model': 'DeepSeek-V4.1-Flash', 'messages': spec['messages'],
                'chat_template_kwargs': spec['chat_template_kwargs'], 'temperature': 0,
                'max_tokens': 64, 'logprobs': True, 'top_logprobs': 20,
                'cache_salt': uuid.uuid4().hex}
        before = idle_snapshot(args.base_url)
        start = time.monotonic()
        request = urllib.request.Request(args.base_url + '/v1/chat/completions',
            json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=3600) as response:
            result = json.load(response)
        elapsed = time.monotonic() - start
        (args.out / f'{repeat}.json').write_text(json.dumps({
            'request': body, 'response': result, 'elapsed_s': elapsed}, indent=1) + '\n')
        if result['usage']['prompt_tokens'] != 524288:
            raise RuntimeError('Historical token count changed')
        cached = finish(args.base_url, before, 524288, args.out / f'{repeat}-cache.json')
        if cached != 0:
            raise RuntimeError('Historical replay was not cold')
        choice = result['choices'][0]
        if not (choice.get('logprobs') or {}).get('content'):
            raise RuntimeError('Missing logprobs')
        signature = {key: choice[key] for key in ('message', 'finish_reason', 'logprobs')}
        signatures.append(signature)
        row = {'repeat': repeat, 'elapsed_s': elapsed,
               'correct': choice['finish_reason'] == 'stop'
                    and (choice['message'].get('content') or '').strip() == spec['expected'],
               'identical': signature == signatures[0], 'cached_tokens': cached,
               'first_token': first_token_margin(choice)}
        report.append(row)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    if not all(row['correct'] and row['identical'] for row in report):
        raise SystemExit('ORIGINAL-NEEDLE-FAIL: inspect correctness and determinism separately')
    print('ORIGINAL-NEEDLE-PASS', flush=True)


if __name__ == '__main__':
    main()
