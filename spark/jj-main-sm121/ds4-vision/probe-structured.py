#!/usr/bin/env python3
"""Repeated speculative grammar exercise, not a replay of the lost crash input."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--base-url', default='http://rusty:8000')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(a.base_url + '/metrics', timeout=30) as r:
        (a.out / 'before-metrics.txt').write_bytes(r.read())

    def trial(i):
        payload = {
            'model': 'DeepSeek-V4-Flash-Vision-Exp',
            'messages': [{'role': 'user', 'content': f'Calculate {17+i} times 23 minus 58. Return JSON with answer as an integer and check as the string verified.'}],
            'chat_template_kwargs': {'thinking': False},
            'temperature': 0, 'max_tokens': 256,
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'calculation', 'strict': True, 'schema': {
                    'type': 'object', 'properties': {
                        'answer': {'type': 'integer'},
                        'check': {'type': 'string', 'enum': ['verified']}},
                    'required': ['answer', 'check'], 'additionalProperties': False}}},
        }
        receipt = {'request': payload, 'started': time.time()}
        try:
            req = urllib.request.Request(a.base_url + '/v1/chat/completions', json.dumps(payload).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=300) as r:
                receipt['response'] = json.load(r)
            c = receipt['response']['choices'][0]
            if c['finish_reason'] != 'stop' or json.loads(c['message']['content']) != {'answer': (17+i)*23-58, 'check': 'verified'}:
                raise RuntimeError('Structured output correctness failed')
            receipt['passed'] = True
        except Exception as exc:
            receipt['error'] = repr(exc)
            raise
        finally:
            receipt['elapsed_s'] = time.time() - receipt['started']
            (a.out / f'{i:03d}.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return i

    for start in range(0, 32, 4):
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i in pool.map(trial, range(start, start+4)):
                print(f'STRUCTURED-CASE-PASS {i}', flush=True)
    with urllib.request.urlopen(a.base_url + '/metrics', timeout=30) as r:
        (a.out / 'after-metrics.txt').write_bytes(r.read())
    print('STRUCTURED-STRESS-PASS 32', flush=True)


if __name__ == '__main__':
    main()
