#!/usr/bin/env python3
"""Adapt PR865's exact counting oracle to our TP2/four-sequence profile.

HTTP tokenization uses the serving tokenizer. This observes workload overlap,
not the engine's actual graph-row dispatch; never claim dispatch coverage from it.
"""
import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request
from pathlib import Path


def post(base, route, payload, timeout=900):
    req = urllib.request.Request(base + route, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def check(text, expected, choice, usage, count):
    return bool(text) and expected.startswith(text) and choice['finish_reason'] == 'length' and usage['completion_tokens'] == count


def cache_metrics(base):
    with urllib.request.urlopen(base + '/metrics', timeout=30) as response:
        lines = response.read().decode().splitlines()
    result = {}
    for key in ('prefix_cache_queries_total', 'prefix_cache_hits_total'):
        values = [float(line.rsplit(' ', 1)[1]) for line in lines
                  if line.startswith('vllm:' + key + '{')]
        if not values:
            raise RuntimeError('Missing prefix counter: ' + key)
        result[key] = sum(values)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://dusty:8000')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    model = 'Qwen3.8-Flash-Next'
    stop = threading.Event()
    records = []
    lock = threading.Lock()
    def save(record):
        with lock:
            records.append(record)
            with (args.out / 'requests.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
    def complete(prompt, count, label, expected=None):
        payload = {'model': model, 'prompt': prompt, 'max_tokens': count,
                   'temperature': 0, 'ignore_eos': True, 'return_token_ids': True}
        record = {'label': label, 'request': payload, 'start': time.time()}
        try:
            before = cache_metrics(args.base) if label.startswith('serial-') else None
            response = post(args.base, '/v1/completions', payload)
            choice = response['choices'][0]
            record.update(response=response, end=time.time(), ok=bool(choice['text']))
            if before is not None:
                after = cache_metrics(args.base)
                record['prefix_counter_delta'] = {key: after[key] - value for key, value in before.items()}
            if expected is not None:
                record['expected'] = expected
                record['ok'] = check(choice['text'], expected, choice, response['usage'], count)
        except Exception as error:
            record.update(end=time.time(), ok=False, error=repr(error))
        save(record)
        return record['ok']
    def short_stream(index):
        count = 0
        while not stop.is_set():
            count += 1
            complete(f'Note {index}-{count}: list three colors, separated by commas:', 4, f'short-{index}-{count}')
            stop.wait(0.25)
    for round_id in range(args.rounds):
        cases = []
        # Vary lengths, but mixed short/long scheduling supplies graph pressure.
        # Length alone does not establish the engine's actual forward sizes.
        for index, tail in enumerate((16, 17, 18, 24, 32, 33, 64)):
            text = f'Run qsa865-{time.time_ns()} session {index}. Continue the sequence exactly.\n' + ''.join(f'{n}, ' for n in range(1, 20000))
            ids = post(args.base, '/tokenize', {'model': model, 'prompt': text, 'add_special_tokens': False})['tokens']
            length = 2848 * 4 + tail
            expected = post(args.base, '/detokenize', {'model': model, 'tokens': ids[length:length + 1200]})['prompt']
            cases.append((ids[:length], expected, f'round-{round_id}-length-{length}'))
        stop.clear()
        threads = [threading.Thread(target=short_stream, args=(i,)) for i in range(2)]
        # Leave one of the four server slots for short requests.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(complete, prompt, 400, label, expected) for prompt, expected, label in cases]
            time.sleep(3)
            for thread in threads:
                thread.start()
            results = [future.result() for future in futures]
        stop.set()
        for thread in threads:
            thread.join()
        if not all(results):
            raise RuntimeError('Counting continuation failed; see full receipts')
        print(f'COUNTING-ROUND-PASS {round_id + 1}: 7/7', flush=True)
    longs = [r for r in records if r['label'].startswith('round-')]
    shorts = [r for r in records if r['label'].startswith('short-')]
    overlap = sum(any(s['start'] < r['end'] and s['end'] > r['start'] for s in shorts) for r in longs)
    if not shorts or not all(r['ok'] for r in records) or not overlap:
        raise RuntimeError('Short traffic failed or workload overlap absent')
    for tail in (16, 17, 18, 24, 32, 33):
        text = f'Serial qsa865-{time.time_ns()}. Continue the sequence exactly.\n' + ''.join(f'{n}, ' for n in range(1, 20000))
        ids = post(args.base, '/tokenize', {'model': model, 'prompt': text, 'add_special_tokens': False})['tokens']
        length = 56960 + tail
        expected = post(args.base, '/detokenize', {'model': model, 'tokens': ids[length:length + 1200]})['prompt']
        for repeat in range(2):
            if not complete(ids[:length], 400, f'serial-{length}-repeat-{repeat}', expected):
                raise RuntimeError('Serial long-context counting failed')
        print(f'SERIAL-COUNTING-PASS length={length}; cached token usage retained, graph dispatch not observed', flush=True)
    with urllib.request.urlopen(args.base + '/health', timeout=10) as response:
        healthy = response.status == 200
    if not healthy:
        raise RuntimeError('Unhealthy after counting')
    summary = {'long_passed': len(longs), 'short_passed': len(shorts),
               'serial_passed': 12, 'healthy_after': healthy,
               'overlapped_long_requests': overlap, 'graph_dispatch_coverage': 'predicted, not observed'}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.out / 'COUNTING-OK').write_text('Exact continuations passed with overlapping client workload.\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
