#!/usr/bin/env python3
"""Frozen original 524K needle under mixed load: a correctness gate, with determinism reported separately.

The request is the original gate's request (qualify_original_needle.py): the frozen input's
messages sent exactly as stored, including the original archive identity
510c94b1bb4f42d2998093bd6b9c3d99, with that file's chat_template_kwargs, temperature 0,
max_tokens 64 and top-20 logprobs. Only cache_salt varies, one fresh value per trial, so the
prefix cache cannot serve it without changing any prompt byte.

Each trial reuses the conversation gate's machinery unchanged (claude_gate_conversations.py,
imported, never modified): an idle baseline, the long request observed running before the
four three-turn conversations start, engine sampling while it runs, the historical overlap
rule, and aggregate cache accounting that must prove every request of the window cold.

Correctness (gates): every trial answers exactly '739184, 482617' with finish_reason stop, the
prompt is exactly 524288 tokens, overlap is observed, and accounting proves the window cold
with no preemption and no foreign request. Integrity failures stop at once; a wrong answer is
recorded, the remaining trials still run, and the gate then fails.

Determinism (reported, never gated): per-trial response signatures, first-token margins and
the scheduling evidence of each trial (samples, running/waiting maxima, turn timing). Trials
overlap differently by construction, so chunk boundaries and batch composition can differ;
equal signatures are not required and unequal ones are not a defect by themselves.

Usage (serving head, after the full suite; no node actions):
  python3 claude_gate_frozen_mixed_needle.py --out receipts/<qualification>/frozen-mixed-needle [--trials 3]
Success marker: FROZEN-MIXED-NEEDLE-PASS.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import claude_gate_conversations as conv  # noqa: E402  (imported unchanged)
from qualify_original_needle import first_token_margin  # noqa: E402

INPUT = HERE.parents[1] / 'ds41/r38/receipts/20260916/admission-k7-1m-u80/needle-524288-input.json'
INPUT_SHA256 = '1b89c7f4d1e1047f6cf7ebe6000d6aff242981cbdbe8d8ce2a742ff61a7d6db6'
CONTENT_SHA256 = 'b410c6b19d492e83ecb775805266d91eed15ded5e3afd0cf264c25aae41bc7e4'
ARCHIVE = 'Archive identity 510c94b1bb4f42d2998093bd6b9c3d99'
PROMPT_TOKENS = 524288
MIN_TRIALS = 2
NEEDLE_LABEL = 'frozen-needle'
TURN_LABEL = re.compile(r'concurrent-\d+-turn\d')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_input(path):
    """The frozen input, refused unless file, prompt text and archive identity are the recorded ones."""
    raw = Path(path).read_bytes()
    if sha(raw) != INPUT_SHA256:
        raise RuntimeError('Frozen input file identity changed')
    spec = json.loads(raw)
    if len(spec['messages']) != 1 or sha(spec['messages'][0]['content'].encode()) != CONTENT_SHA256:
        raise RuntimeError('Frozen prompt identity changed')
    if conv.ARCHIVE_ID.findall(spec['messages'][0]['content']) != [ARCHIVE]:
        raise RuntimeError('Original archive identity is not present exactly once')
    return spec


def needle_body(spec, salt):
    """The original gate's request with this trial's salt; the messages object is passed through untouched."""
    if not re.fullmatch(r'[0-9a-f]{32}', salt):
        raise ValueError('cache_salt must be 32 lowercase hex characters')
    return {'model': conv.MODEL, 'messages': spec['messages'], 'chat_template_kwargs': spec['chat_template_kwargs'],
            'temperature': 0, 'max_tokens': 64, 'logprobs': True, 'top_logprobs': 20, 'cache_salt': salt}


def check_body(body, spec):
    """What the wire carries: every message byte and the archive identity exactly as frozen."""
    sent = json.loads(json.dumps(body))
    if sent['messages'] != spec['messages'] or sha(sent['messages'][0]['content'].encode()) != CONTENT_SHA256:
        raise RuntimeError('Needle messages differ from the frozen input')
    if conv.ARCHIVE_ID.findall(sent['messages'][0]['content']) != [ARCHIVE]:
        raise RuntimeError('Needle archive identity differs from the original')


def signature(choice):
    payload = {key: choice[key] for key in ('message', 'finish_reason', 'logprobs')}
    return sha(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def post(base_url, body, out, label, *, timeout=3600):
    request = urllib.request.Request(base_url + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        (out / f'{label}-http-error.json').write_text(json.dumps({
            'status': error.code, 'body': error.read().decode(errors='replace'),
            'request': body, 'started_unix': start}, indent=1) + '\n')
        raise
    receipt = {'label': label, 'started_unix': start, 'elapsed_s': time.time() - start,
               'request': body, 'response': result}
    (out / f'{label}.json').write_text(json.dumps(receipt, indent=1) + '\n')
    return receipt


def evaluate(receipt, spec):
    """Correctness of one needle response; raises only on an invalid response."""
    result = receipt['response']
    if result['usage']['prompt_tokens'] != PROMPT_TOKENS:
        raise RuntimeError('Frozen prompt token count changed')
    choice = result['choices'][0]
    if not (choice.get('logprobs') or {}).get('content'):
        raise RuntimeError('Missing needle logprobs')
    answer = choice['message'].get('content') or ''
    return {'answer': answer, 'finish_reason': choice['finish_reason'],
            'correct': choice['finish_reason'] == 'stop' and answer.strip() == spec['expected'],
            'first_token': first_token_margin(choice), 'signature': signature(choice)}


def trial(index, base_url, spec, root, metrics, used_salts):
    out = root / f'trial-{index}'
    out.mkdir()
    salt = uuid.uuid4().hex
    if salt in used_salts:
        raise RuntimeError('cache_salt reused')
    used_salts.add(salt)
    body = needle_body(spec, salt)
    check_body(body, spec)
    client = conv.Client(base_url, out)
    baseline = conv.wait_idle(base_url, metrics.snapshot, metrics.idle_snapshot)
    (out / 'baseline.prom').write_text(baseline['raw'])
    stop, samples, errors = threading.Event(), [], []
    sampler = None
    with ThreadPoolExecutor(max_workers=conv.CONVERSATIONS + 1) as pool:
        submitted = time.time()
        long_run = pool.submit(post, base_url, body, out, NEEDLE_LABEL)
        deadline = time.monotonic() + conv.IDLE_TIMEOUT_S
        while True:
            if long_run.done():
                long_run.result()
                raise RuntimeError('Needle finished before overlap was established')
            started = metrics.snapshot(base_url)
            if started['running'] >= 1:
                (out / 'overlap-start.prom').write_text(started['raw'])
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('No running needle observed')
            time.sleep(0.2)
        sampler = threading.Thread(target=conv.sample_engine, daemon=True,
                                   args=(stop, long_run, samples, errors, metrics.snapshot, base_url))
        sampler.start()
        try:
            list(pool.map(lambda i: conv.conversation(client, i), range(conv.CONVERSATIONS)))
        finally:
            stop.set()
            sampler.join()
            (out / 'samples.json').write_text(json.dumps({'samples': samples, 'sampler_errors': errors}, indent=1) + '\n')
        receipt = long_run.result()
    turns = [t for t in client.turns if TURN_LABEL.fullmatch(t['label'])]
    after = conv.settle(base_url, metrics.snapshot, baseline, len(turns) + 1)
    (out / 'after.prom').write_text(after['raw'])
    evidence = conv.mixed_evidence(baseline, submitted, started, samples, errors, turns, receipt)
    (out / 'evidence.json').write_text(json.dumps(evidence, indent=1) + '\n')
    # Accounting first: foreign requests, preemption or an unprovable cold needle fail closed.
    accounting = conv.concurrent_verdict(baseline, after, [t['prompt_tokens'] for t in turns],
                                         needle_prompt_tokens=receipt['response']['usage']['prompt_tokens'])
    (out / 'accounting.json').write_text(json.dumps(accounting, indent=1) + '\n')
    if not evidence['overlap_observed']:
        raise RuntimeError(f'Trial {index}: mixed-load overlap was not observed; see evidence.json')
    row = {'trial': index, 'cache_salt': salt, 'elapsed_s': receipt['elapsed_s'], **evaluate(receipt, spec),
           'accounting': accounting,
           'scheduling': {key: evidence[key] for key in (
               'samples', 'samples_while_long_active', 'samples_running_ge_2_while_long_active',
               'max_running_while_long_active', 'max_waiting_while_long_active', 'queue_observed',
               'long_submitted_unix', 'long_ended_unix')} | {
               'turns_completed_before_long_end': len(evidence['turns_completed_before_long_end']),
               'turn_windows': [[t['started_unix'], t['ended_unix']] for t in turns]}}
    (out / 'trial.json').write_text(json.dumps(row, indent=1) + '\n')
    print(json.dumps({key: row[key] for key in ('trial', 'answer', 'correct', 'elapsed_s', 'first_token')}), flush=True)
    return row


def determinism(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['signature'], []).append(row['trial'])
    return {'scope': 'reported, not gated: trials overlap differently, so chunk boundaries and batch '
                     'composition may differ; equal signatures are not required',
            'signature_groups': groups, 'all_trials_same_signature': len(groups) == 1,
            'first_token_margins_nats': [row['first_token']['margin_nats'] for row in rows],
            'scheduling': {row['trial']: row['scheduling'] for row in rows}}


def main(argv=None):
    p = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    p.add_argument('--base-url', default='http://dusty:8000')
    p.add_argument('--input', type=Path, default=INPUT)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--trials', type=int, default=3)
    a = p.parse_args(argv)
    if a.trials < MIN_TRIALS:
        p.error(f'At least {MIN_TRIALS} matched mixed trials are required')
    spec = load_input(a.input)
    a.out.mkdir(parents=True, exist_ok=False)
    import cache_metrics as metrics

    def digest(name):
        return sha((HERE / name).read_bytes())
    summary = {'base_url': a.base_url, 'input': str(a.input), 'input_sha256': INPUT_SHA256,
               'archive_identity': ARCHIVE, 'trials_requested': a.trials,
               'sources': {name: digest(name) for name in (Path(__file__).name, 'claude_gate_conversations.py',
                                                           'cache_metrics.py', 'qualify_original_needle.py')}}
    rows, used = [], set()
    try:
        for index in range(a.trials):
            rows.append(trial(index, a.base_url, spec, a.out, metrics, used))
    finally:
        summary['correctness'] = {'trials': [{k: r[k] for k in ('trial', 'answer', 'finish_reason', 'correct',
                                                                  'cache_salt', 'elapsed_s')} for r in rows],
                                  'trials_completed': len(rows)}
        if rows:
            summary['determinism'] = determinism(rows)
        (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    wrong = [r['trial'] for r in rows if not r['correct']]
    if wrong:
        raise SystemExit(f'FROZEN-MIXED-NEEDLE-FAIL: wrong answer in trial(s) {wrong}; see summary.json')
    print('FROZEN-MIXED-NEEDLE-PASS', flush=True)


if __name__ == '__main__':
    main()
