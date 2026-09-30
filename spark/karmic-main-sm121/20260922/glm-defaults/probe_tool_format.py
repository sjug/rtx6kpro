#!/usr/bin/env python3
"""Matched seeded tool-format diagnostic; never emits a qualification PASS."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE_SHA256 = 'ef1a02e1112ccc6d394f4870466632c4a9476cf5fd388743cd4f56f831ba216a'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://sparky:8000')
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or args.trials < 1:
        raise SystemExit('Require a new receipt and positive trial count')
    source = Path(__file__).resolve().parents[3] / 'glm53/verify-semantic-admission.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise SystemExit('Shared semantic probe changed')
    spec = importlib.util.spec_from_file_location('glm_semantic', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.post_json
    passed = 0
    with args.output.open('x') as output:
        for seed in range(args.trials):
            exchanges = []

            def captured(url, payload, timeout=300):
                body = {**copy.deepcopy(payload), 'temperature': 1.0, 'seed': seed,
                        'cache_salt': f'tool-format-{args.arm}-{seed}'}
                response = original(url, body, timeout)
                exchanges.append({'request': body, 'response': response})
                return response

            module.post_json = captured
            receipt = {'arm': args.arm, 'seed': seed, 'temperature': 1.0,
                       'source_sha256': SOURCE_SHA256, 'exchanges': exchanges}
            try:
                receipt['result'] = module.tool_gate(args.base_url.rstrip('/') + '/v1/chat/completions', 'GLM-5.3-Flash')
                receipt['strict_format_pass'] = True
                passed += 1
            except AssertionError as error:
                receipt['strict_format_pass'] = False
                receipt['error'] = str(error)
            output.write(json.dumps(receipt) + '\n')
            output.flush()
            print(json.dumps({'arm': args.arm, 'seed': seed,
                              'strict_format_pass': receipt['strict_format_pass'],
                              'error': receipt.get('error')}), flush=True)
        summary = {'kind': 'diagnostic_summary', 'arm': args.arm,
                   'trials': args.trials, 'strict_format_passes': passed,
                   'note': 'Same initial prompts/seeds; generated tool messages can differ. Not a qualification gate.'}
        output.write(json.dumps(summary) + '\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
