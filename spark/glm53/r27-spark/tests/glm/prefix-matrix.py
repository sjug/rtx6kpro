"""GLM prefix-cache pair matrix (frozen r26/r22 corpus replayed through /v1/completions).
Cache diagnostics, not throughput benchmarks. Adapted from the 2026-09-06 R26/R22 control."""
"""Three fixed token-ID extension pairs for R26/R22 Mamba cache comparison."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.request

import os
ROOT = Path(os.environ.get('OUT_DIR', Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location('needle', REPO / 'spark/qwen38-flash-next/verify-native-context.py')
needle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(needle)
URL = os.environ.get('BASE_URL', 'http://sparky:8000')
MODEL = os.environ.get('MODEL', 'GLM-5.3-Flash')
# The frozen r26/r22 corpus (identical input IDs) is copied into OUT_DIR by the driver.
FIXTURE = ROOT / 'prefix-matrix-inputs.json'


def common(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def metrics():
    with urllib.request.urlopen(URL + '/metrics', timeout=30) as response:
        raw = response.read().decode()
    values = {}
    for line in raw.splitlines():
        if line.startswith(('vllm:prefix_cache_hits_total{', 'vllm:prefix_cache_queries_total{', 'vllm:prompt_tokens_by_source_total{', 'vllm:prompt_tokens_total{')):
            key, value = line.rsplit(' ', 1)
            values[key] = float(value)
    return raw, values


def prepare():
    assert not FIXTURE.exists(), 'Do not overwrite the shared experiment corpus'
    pairs = []
    for name, length, extension, code, alignment in (
        ('long-unaligned', 131072, 262144, '75193681', None),
        ('long-aligned', 131072, 262144, '62981473', 131072),
        ('short-unaligned', 4096, 8192, '94817362', None),
    ):
        def build(n):
            messages, _ = needle.exact_needle_prompt(URL, MODEL, n, code, 120)
            return needle.tokenize_chat(URL, MODEL, messages, 120)
        a, b = build(length), build(extension)
        if alignment is not None:
            a = build(length + alignment - common(a, b))
            assert common(a, b) == alignment and alignment % 2048 == 0
        pair = dict(name=name, code=code, base=a, extension=b, common_prefix_tokens=common(a, b))
        pairs.append(pair)
        print(name, len(a), len(b), pair['common_prefix_tokens'], flush=True)
    FIXTURE.write_text(json.dumps(pairs, separators=(',', ':')))
    print('CORPUS_SHA256', hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), flush=True)


def run(arm):
    pairs = json.loads(FIXTURE.read_text())
    receipt = ROOT / f'prefix-matrix-{arm}.jsonl'
    with receipt.open('x') as output:
        for pair in pairs:
            for stage in ('base', 'extension'):
                tokens = pair[stage]
                prefix = f'{arm}-{pair["name"]}-{stage}'
                before_raw, before = metrics()
                (ROOT / f'{prefix}-before.metrics').write_text(before_raw)
                start = datetime.datetime.now().astimezone().isoformat()
                print(start, prefix, len(tokens), flush=True)
                result, elapsed = needle.post_json(URL + '/v1/completions', {
                    'model': MODEL, 'prompt': tokens, 'max_tokens': 128,
                    'temperature': 0, 'return_token_ids': True,
                }, 1200)
                time.sleep(1)
                after_raw, after = metrics()
                (ROOT / f'{prefix}-after.metrics').write_text(after_raw)
                delta = {key: after[key]-before.get(key, 0) for key in after}
                hits = sum(v for k,v in delta.items() if k.startswith('vllm:prefix_cache_hits_total{'))
                fresh = sum(v for k,v in delta.items() if 'source="local_compute"' in k)
                total = sum(v for k,v in delta.items() if k.startswith('vllm:prompt_tokens_total{'))
                usage = result.get('usage', {})
                choice = result['choices'][0]
                valid = pair['code'] in choice.get('text', '') and choice['finish_reason']=='stop' and usage.get('prompt_tokens') == len(tokens)
                record = dict(arm=arm, pair=pair['name'], stage=stage, common_prefix_tokens=pair['common_prefix_tokens'], prompt_tokens=len(tokens),
                    corpus_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), started_at=start, finished_at=datetime.datetime.now().astimezone().isoformat(), elapsed_seconds=elapsed,
                    cached_tokens=hits, fresh_tokens=fresh, metric_deltas=delta, usage=usage, response=result, valid=valid, isolated=total==len(tokens))
                output.write(json.dumps(record)+'\n')
                output.flush()
                print(json.dumps({k:v for k,v in record.items() if k not in ('response', 'metric_deltas')}), flush=True)
                assert valid and total==len(tokens), record
    print('MATRIX COMPLETE', arm, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arm')  # 'prepare' or a release label such as r27
    args = parser.parse_args()
    prepare() if args.arm == 'prepare' else run(args.arm)
