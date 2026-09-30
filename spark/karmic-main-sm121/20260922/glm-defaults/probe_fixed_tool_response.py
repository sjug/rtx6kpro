#!/usr/bin/env python3
"""Replay one frozen final-tool input with identical seeds across sampler arms."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://sparky:8000')
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()
    source = args.corpus.read_bytes()
    body = json.loads(source)
    digest = hashlib.sha256(source).hexdigest()
    if args.output.exists() or args.trials < 1 or body.get('temperature') != 1.0:
        raise SystemExit('Require a new receipt, positive trials and frozen temperature one')
    if body['messages'][-1]['role'] != 'tool' or body['messages'][-1]['content'] != '391':
        raise SystemExit('Unexpected final tool response')
    passed = 0
    with args.output.open('x') as output:
        for seed in range(args.trials):
            payload = {**body, 'seed': seed, 'cache_salt': f'fixed-tool-{args.arm}-{seed}'}
            request = urllib.request.Request(args.base_url.rstrip('/') + '/v1/chat/completions',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
            choice = result['choices'][0]
            answer = choice['message'].get('content') or ''
            valid = choice['finish_reason'] == 'stop' and answer.strip() == 'FINAL: 391'
            passed += valid
            row = {'arm': args.arm, 'seed': seed, 'corpus_sha256': digest,
                   'strict_format_pass': valid, 'response': result}
            output.write(json.dumps(row) + '\n')
            output.flush()
            if not valid or (seed + 1) % 10 == 0:
                print(json.dumps({'seed': seed, 'passed_so_far': passed,
                                  'answer': answer if not valid else None}), flush=True)
        summary = {'kind': 'diagnostic_summary', 'arm': args.arm, 'trials': args.trials,
                   'strict_format_passes': passed, 'corpus_sha256': digest}
        output.write(json.dumps(summary) + '\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
