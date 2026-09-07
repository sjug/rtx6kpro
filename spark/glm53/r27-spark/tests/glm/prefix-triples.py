"""Cache diagnostics, not throughput benchmarks. Preserve actual termination reasons."""
import argparse
import datetime
import hashlib
import json
import time

import importlib.util
from pathlib import Path

import os
ROOT = Path(os.environ.get('OUT_DIR', Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location('matrix', Path(__file__).resolve().parent / 'prefix-matrix.py')
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


def run(name, base, extension, code, first_budget):
    corpus = {}
    for stage, length in [('base', base), ('extension', extension)]:
        messages, _ = matrix.needle.exact_needle_prompt(matrix.URL, matrix.MODEL, length, code, 120)
        tokens = matrix.needle.tokenize_chat(matrix.URL, matrix.MODEL, messages, 120)
        corpus[stage] = dict(messages=messages, tokens=tokens)
    fixture = ROOT / f'triple-{name}-inputs.json'
    with fixture.open('x') as file:
        json.dump(corpus, file)
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    with (ROOT / f'triple-{name}.jsonl').open('x') as receipt:
        for stage, source, budget in [('first', 'base', first_budget), ('repeat', 'base', 64), ('extension', 'extension', 128)]:
            prefix = f'triple-{name}-{stage}'
            raw, before = matrix.metrics()
            (ROOT / f'{prefix}-before.metrics').write_text(raw)
            started = datetime.datetime.now().astimezone().isoformat()
            print(started, prefix, 'START', flush=True)
            result, elapsed = matrix.needle.post_json(matrix.URL + '/v1/chat/completions', {
                'model': matrix.MODEL, 'messages': corpus[source]['messages'],
                'chat_template_kwargs': {'enable_thinking': False},
                'temperature': 0, 'max_tokens': budget, 'return_token_ids': True,
            }, 1800)
            time.sleep(1)
            raw, after = matrix.metrics()
            (ROOT / f'{prefix}-after.metrics').write_text(raw)
            delta = {k: v-before.get(k, 0) for k, v in after.items()}
            choice = result['choices'][0]
            usage = result.get('usage', {})
            hits = sum(v for k, v in delta.items() if k.startswith('vllm:prefix_cache_hits_total{'))
            queries = sum(v for k, v in delta.items() if k.startswith('vllm:prefix_cache_queries_total{'))
            total = sum(v for k, v in delta.items() if k.startswith('vllm:prompt_tokens_total{'))
            length = len(corpus[source]['tokens'])
            record = dict(name=name, stage=stage, started_at=started, elapsed_seconds=elapsed,
                max_tokens=budget, prompt_tokens=length, finish_reason=choice['finish_reason'],
                needle_present=code in (choice['message'].get('content') or ''),
                cache_hits=hits, cache_queries=queries, isolated=total==length,
                common_prefix_tokens=matrix.common(corpus['base']['tokens'], corpus['extension']['tokens']),
                corpus_sha256=digest, usage=usage, metric_deltas=delta, response=result)
            receipt.write(json.dumps(record)+'\n')
            receipt.flush()
            print(json.dumps({k:v for k,v in record.items() if k not in ('response','metric_deltas')}), flush=True)
            assert record['isolated'] and usage['prompt_tokens']==length
            if stage != 'first' or first_budget != 16:
                assert record['needle_present'] and choice['finish_reason']=='stop'
    print('TRIPLE COMPLETE', name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['short', 'long'])
    args = parser.parse_args()
    if args.mode == 'short':
        run('short-budget16', 4096, 8192, '58371692', 16)
        run('short-normal', 4096, 8192, '71629485', 128)
    else:
        run('long-budget16', 262000, 1048000, '739526', 16)
