#!/usr/bin/env python3
"""Print immutable GDN screen results; never modify benchmark outputs."""
import json
from pathlib import Path
from statistics import geometric_mean

root = Path('/home/jugs/git/llm-inference-bench/results/runs/qwen3.8-flash-next/nvfp4-4p89/2026-09-r38-qwen-gdn-prefill-screen/prefill')
arms = {'b12x': [], 'flashinfer': []}
for path in sorted(root.glob('*.json')):
    run = json.loads(path.read_text())
    metadata = run['run_metadata']
    if metadata['measurement_role'] != 'measured':
        continue
    arm = metadata['gdn_screen_arm']
    arms[arm].append((path.name, run))
for arm, records in arms.items():
    for name, run in records:
        print(arm, name)
        for context, value in run['prefill'].items():
            server = value['server_validation']
            print(context, 'client', value['tok_per_sec'], 'server', server['tok_per_sec'],
                  'cached', server['cached_tokens'], 'samples', value['samples'],
                  'invalid', server['invalid_reason'])
if all(arms.values()):
    print('context,b12x_client,flashinfer_client,delta_pct,b12x_server,flashinfer_server,server_delta_pct')
    for context in ['8192', '16384', '32768', '65536', '131072']:
        client = {arm: geometric_mean(run['prefill'][context]['tok_per_sec'] for _, run in records)
                  for arm, records in arms.items()}
        server = {arm: geometric_mean(run['prefill'][context]['server_validation']['tok_per_sec'] for _, run in records)
                  for arm, records in arms.items()}
        print(f"{context},{client['b12x']:.2f},{client['flashinfer']:.2f},"
              f"{100*(client['flashinfer']/client['b12x']-1):.3f},"
              f"{server['b12x']:.2f},{server['flashinfer']:.2f},"
              f"{100*(server['flashinfer']/server['b12x']-1):.3f}")
