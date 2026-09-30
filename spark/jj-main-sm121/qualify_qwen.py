#!/usr/bin/env python3
"""Bounded first-completion admission followed by the existing Qwen battery."""
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base-url', default='http://dusty:8000')
    parser.add_argument('--mtp0', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    expected_image = '6633e678fee74f5e1060290a01df812d34d10edcf30e34dd1c9bf7db88260ef3'
    container = 'qwen38-flash-next-nvfp4-jj-main-tp2'
    for node in ('dusty', 'kirby'):
        raw = subprocess.check_output(['ssh', '-n', '-o', 'BatchMode=yes', '-o',
            'ConnectTimeout=10', node, 'podman', 'inspect', container], text=True)
        record = json.loads(raw)[0]
        if not record['State']['Running'] or record['Image'] != expected_image:
            raise RuntimeError(f'{node}: wrong candidate or stopped container')
        (args.out / f'{node}-container.json').write_text(raw)
    model = 'Qwen3.8-Flash-Next'
    deadline = time.monotonic() + 1200
    while True:
        try:
            with urllib.request.urlopen(args.base_url + '/v1/models', timeout=10) as stream:
                models = json.load(stream)
            if [entry['id'] for entry in models['data']] != [model]:
                raise RuntimeError(f'Wrong endpoint model identities: {models}')
            break
        except OSError as error:
            if time.monotonic() >= deadline:
                raise RuntimeError('Readiness deadline exceeded') from error
            print(f'Waiting for API: {error}', flush=True)
            time.sleep(10)
    payload = {'model': model, 'messages': [{'role': 'user', 'content':
               'What is 17 * 23 - 58? Reply with only the integer answer.'}],
               'temperature': 0, 'max_tokens': 128,
               'chat_template_kwargs': {'enable_thinking': False}}
    request = urllib.request.Request(args.base_url + '/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as stream:
        response = json.load(stream)
    (args.out / 'first-completion.json').write_text(json.dumps({'request': payload, 'response': response}, indent=2) + '\n')
    choice = response['choices'][0]
    if choice['finish_reason'] != 'stop' or choice['message']['content'].strip() != '333':
        raise RuntimeError(f'First completion failed: {response}')
    print('FIRST-COMPLETION-PASS', flush=True)
    qwen = ROOT / 'spark/qwen38-flash-next'
    r27 = ROOT / 'spark/glm53/r27-spark/tests'
    common = ['--base-url', args.base_url, '--model', model]
    commands = []
    if not args.mtp0:
        commands.append([qwen / 'verify-semantic-admission.py', *common, '--runs', '3',
                         '--receipt-file', args.out / 'semantic.jsonl'])
    lengths = ['2784', '2785', '131072'] if args.mtp0 else ['2848', '2849', '131072', '262000']
    commands.append([qwen / 'verify-native-context.py', *common, '--needle', '739184',
                     '--lengths', *lengths, '--max-tokens', '64',
                     '--receipt-file', args.out / 'native-context.jsonl'])
    if not args.mtp0:
        commands.extend([
            [qwen / 'benchmark-fixed-token-acceptance.py', *common,
             '--corpus-file', qwen / 'fixed-token-acceptance-corpus.json',
             '--waves', '3', '--max-tokens', '512', '--output', args.out / 'fixed-token-acceptance.json'],
            [r27 / 'probe-spec-padded-transition.py', *common, '--phases', '1,3,1,2,4,1',
             '--rounds', '3', '--max-tokens', '256', '--spec-tokens', '3',
             '--capture-sizes', '1,2,4,8,16,24,32', '--receipt-file', args.out / 'padded-transition.jsonl'],
            [r27 / 'probe-prefix-reuse.py', *common, '--label', 'jj-main-cache-agreement-aligned-mtp3',
             '--receipt-file', args.out / 'prefix-reuse.jsonl'],
        ])
    for command in commands:
        print('RUN', *map(str, command), flush=True)
        subprocess.run([sys.executable, '-u', *map(str, command)], check=True)
    (args.out / 'CORRECTNESS-OK').write_text('Existing Qwen battery passed; benchmark and promotion are separate.\n')
    print('QWEN-CORRECTNESS-BATTERY-PASS', flush=True)


if __name__ == '__main__':
    main()
