"""Measured cache accounting for a quiet endpoint without usage-detail flags."""
import json
import re
import time
import urllib.request


def total(text, name):
    values = re.findall(r'^' + re.escape(name) + r'(?:\{[^\n]*\})?\s+([0-9.eE+-]+)$', text, re.M)
    if not values:
        raise RuntimeError(f'Missing metric {name}')
    return sum(float(value) for value in values)


def snapshot(base_url):
    with urllib.request.urlopen(base_url + '/metrics', timeout=15) as response:
        raw = response.read().decode()
    return {'unix': time.time(), 'raw': raw,
            'queries': total(raw, 'vllm:prefix_cache_queries_total'),
            'hits': total(raw, 'vllm:prefix_cache_hits_total'),
            'successes': total(raw, 'vllm:request_success_total'),
            'preemptions': total(raw, 'vllm:num_preemptions_total'),
            'running': total(raw, 'vllm:num_requests_running'),
            'waiting': total(raw, 'vllm:num_requests_waiting')}


def idle_snapshot(base_url):
    before = snapshot(base_url)
    if before['running'] or before['waiting']:
        raise RuntimeError('Cache accounting requires an idle endpoint')
    return before


def delta(before, after, length):
    queries = after['queries'] - before['queries']
    hits = after['hits'] - before['hits']
    completed = after['successes'] - before['successes']
    if queries != length or completed != 1:
        raise RuntimeError(f'Ambiguous request accounting: queries={queries}, successes={completed}')
    if after['preemptions'] != before['preemptions']:
        raise RuntimeError('Request preempted during cache accounting')
    if hits < 0 or hits > queries:
        raise RuntimeError('Invalid cache counter delta')
    return hits


def finish(base_url, before, length, receipt, *, expected_queries=None):
    # Prompt-logprob requests deliberately skip prefix reads and query stats
    # (pinned sampling_params.py / KVCacheManager.record_prefix_cache_stats).
    queries = length if expected_queries is None else expected_queries
    deadline = time.monotonic() + 30
    samples = []
    while True:
        after = snapshot(base_url)
        samples.append(after)
        receipt.write_text(json.dumps({'before': before, 'after_samples': samples}, indent=2) + '\n')
        if after['successes'] > before['successes'] and (queries == 0 or after['queries'] > before['queries']):
            return delta(before, after, queries)
        if time.monotonic() >= deadline:
            raise RuntimeError('Cache counters did not advance after completed request')
        time.sleep(1)
