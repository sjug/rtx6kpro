#!/usr/bin/env python3
"""Replay the supplied proxy trace locally without exporting proxy credentials.

The trace lacks wire raw_request bytes. This is a documented reconstruction,
not a byte-exact replay. Responses stay in ignored private receipts.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


def reconstruct(trace):
    keys = ('role', 'content', 'tool_calls', 'tool_call_id', 'reasoning_content')
    messages = [{k: v for k, v in message.items() if k in keys and v is not None}
                for message in trace['input_history']]
    allowed = ('temperature', 'top_p', 'top_k', 'presence_penalty', 'frequency_penalty',
               'max_completion_tokens', 'max_tokens', 'tool_choice', 'stop', 'seed')
    params = {k: v for k, v in trace['params'].items() if k in allowed}
    tools = trace['params'].get('tools') or trace['tools']
    return {**params, 'model': 'Qwen3.8-Flash-Next', 'messages': messages,
            'tools': tools, 'stream': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=False)
    data = args.trace.read_bytes()
    trace = json.loads(data)
    payload = reconstruct(trace)
    wire = json.dumps(payload).encode()
    metadata = {'trace_sha256': hashlib.sha256(data).hexdigest(),
                'request_sha256': hashlib.sha256(wire).hexdigest(),
                'messages': len(payload['messages']), 'tools': len(payload['tools']),
                'transport': 'non-streaming; stream_options omitted',
                'equivalence': 'reconstructed from input_history and params; raw_request absent'}
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for iteration in range(2):
        start = time.time()
        request = urllib.request.Request('http://dusty:8000/v1/chat/completions',
            data=wire, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=1800) as response:
            result = json.load(response)
        (args.out / f'private-response-{iteration}.json').write_text(json.dumps(result))
        choice = result['choices'][0]
        message = choice['message']
        if choice['finish_reason'] not in ('stop', 'tool_calls') or not (message.get('content') or message.get('tool_calls')):
            raise RuntimeError('Replay did not complete meaningfully; inspect private receipt')
        print(json.dumps({'iteration': iteration, 'seconds': time.time() - start,
                          'finish_reason': choice['finish_reason'], 'usage': result.get('usage')}), flush=True)
    (args.out / 'REPLAY-COMPLETED').write_text('Two reconstructed requests completed; manual output and health review still required.\n')


if __name__ == '__main__':
    main()
