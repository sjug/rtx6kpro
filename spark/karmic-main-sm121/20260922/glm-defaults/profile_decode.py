#!/usr/bin/env python3
"""Capture bounded established decode through the unmodified benchmark harness.

The generated throughput results are instrumented diagnostics, not scores.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request

import experiment

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
HARNESS = Path('/home/jugs/git/llm-inference-bench')
BASE_URL = 'http://sparky:8000'


def request(path, post=False):
    req = urllib.request.Request(BASE_URL + path, data=b'' if post else None)
    with urllib.request.urlopen(req, timeout=15) as response:
        return response.read().decode()


def count(metrics, kind):
    values = re.findall(r'^vllm:num_requests_' + kind + r'\{[^\n]*\} ([\d.e+-]+)$', metrics, re.M)
    if not values:
        raise RuntimeError('Missing request counter: ' + kind)
    return sum(float(v) for v in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--concurrency', type=int, choices=(1, 4), required=True)
    args = parser.parse_args()
    if not __debug__:
        raise SystemExit('Assertions must remain enabled')
    out = args.receipt.resolve()
    if 'SCREEN-COMPLETE:' not in (out / 'driver.log').read_text():
        raise SystemExit('Completed correctness screen required')
    lock, record, digest = experiment.resolve((ROOT / 'profile.lock.json').read_bytes(), 'no-prefetch-profile')
    rendered = json.loads((out / 'effective-profile.json').read_text())
    if rendered != {'effective_profile_sha256': digest, **record}:
        raise SystemExit('Profile receipt mismatch')
    for node in ('sparky', 'buddy', 'rocky', 'lucky'):
        live = json.loads(subprocess.check_output(['ssh', '-n', node, 'podman', 'inspect', experiment.name('no-prefetch-profile')]))[0]
        prior = json.loads((out / f'{node}-container.json').read_text())[0]
        if (live['Id'], live['State']['StartedAt']) != (prior['Id'], prior['State']['StartedAt']):
            raise SystemExit('Container or boot changed on ' + node)
        if not live['State']['Running'] or live['State']['OOMKilled'] or live['Image'] != lock['image_id']:
            raise SystemExit('Unhealthy or wrong image on ' + node)
    for name, expected in {
        'run_bench.sh': '5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2',
        'llm_decode_bench.py': '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3',
    }.items():
        if hashlib.sha256((HARNESS / name).read_bytes()).hexdigest() != expected:
            raise SystemExit('Harness identity changed: ' + name)
    metrics = request('/metrics')
    if count(metrics, 'running') or count(metrics, 'waiting'):
        raise SystemExit('Endpoint must be idle before profiling')
    env = dict(os.environ, RESULTS_REPO=str(REPO), HOST=BASE_URL, MODEL='GLM-5.3-Flash',
               MODEL_FAMILY='glm-5.3-flash', MODEL_VARIANT='nvfp4',
               CAMPAIGN='2026-09-karmic-sm121-qualification', WORKLOAD='profiling',
               VARIANT=f'no-prefetch-k3-profile-c{args.concurrency}', CONCURRENCY=str(args.concurrency),
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    command = [str(HARNESS / 'run_bench.sh'), '--no-resume', '--contexts', '0',
               '--skip-prefill', '--duration', '20', '--display-mode', 'plain',
               '--calibration-cache', str(out / f'profile-c{args.concurrency}-calibration.json'),
               '--metadata', 'instrumented=true', '--metadata', 'performance_verdict=not_applicable',
               '--metadata', 'image_id=' + lock['image_id'],
               '--metadata', 'checkpoint_revision=' + lock['checkpoint_revision'],
               '--metadata', 'speculative_tokens=3',
               '--metadata', 'effective_profile_sha256=' + digest]
    events = out / f'profile-c{args.concurrency}-events.jsonl'
    with events.open('x') as receipt, (out / f'profile-c{args.concurrency}-harness.log').open('x') as log:
        def event(kind, **data):
            receipt.write(json.dumps({'time_unix': time.time(), 'event': kind, **data}) + '\n')
            receipt.flush()
            print(kind, data, flush=True)
        child = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        child.stdin.write(b'n\n')
        child.stdin.close()
        triggered = False
        stable_since = None
        deadline = time.monotonic() + 300
        try:
            event('HARNESS_STARTED', pid=child.pid, command=command)
            while child.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('Bounded profiling workload exceeded 300 seconds')
                metrics = request('/metrics')
                running, waiting = count(metrics, 'running'), count(metrics, 'waiting')
                if running > args.concurrency or waiting:
                    raise RuntimeError('Unexpected concurrent or queued traffic')
                if running == args.concurrency:
                    stable_since = stable_since or time.monotonic()
                    if not triggered and time.monotonic() - stable_since >= 3:
                        request('/start_profile', post=True)
                        triggered = True
                        event('PROFILE_STARTED', running=running, waiting=waiting)
                else:
                    stable_since = None
                time.sleep(0.5)
            if child.returncode != 0 or not triggered:
                raise RuntimeError(f'Incomplete profile workload: exit={child.returncode}, triggered={triggered}')
            event('HARNESS_COMPLETE', exit_code=child.returncode)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                child.wait(timeout=30)
            if triggered:
                request('/stop_profile', post=True)
                event('PROFILE_STOP_REQUESTED')
    print('PROFILE-WORKLOAD-COMPLETE: inspect all four rank traces; not a benchmark verdict')


if __name__ == '__main__':
    main()
