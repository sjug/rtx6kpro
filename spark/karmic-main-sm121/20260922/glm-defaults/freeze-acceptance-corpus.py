#!/usr/bin/env python3
"""Render the existing acceptance prompts once with the GLM serving convention.

Print generated corpus JSON to stdout. No inference, runtime changes or writes.
Use the existing fixed-token acceptance probe with this corpus across arms.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
probe_path = ROOT.parents[2] / 'qwen38-flash-next/benchmark-fixed-token-acceptance.py'
spec = importlib.util.spec_from_file_location('probe', probe_path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def main():
    entries = []
    kwargs = {'reasoning_effort': 'high', 'clear_thinking': False}
    for i, prompt in enumerate(probe.PROMPTS):
        result = probe.post_json('http://sparky:8000/tokenize', {
            'model': 'GLM-5.3-Flash',
            'messages': [{'role': 'user', 'content': prompt}],
            'chat_template_kwargs': kwargs,
            'add_generation_prompt': True,
        })
        tokens = result.get('tokens')
        if not tokens or not all(type(t) is int and t >= 0 for t in tokens):
            raise SystemExit('Invalid tokenization')
        entries.append({'id': f'prompt-{i}', 'input_ids': tokens,
                        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
    print(json.dumps({'format': 1, 'model': 'GLM-5.3-Flash',
                      'chat_template_kwargs': kwargs, 'prompts': entries}, indent=2))


if __name__ == '__main__':
    main()
