#!/usr/bin/env python3
"""Local driver for one stale-state observe capture on the running diagnostic image.

Subcommands
  preflight  local only: corpora, prompt digests, control validation, helper
             and lock identity. Touches no node and sends no request.
  run        the capture: identity checks, atomic control deployment to all
             four ranks, prime (fresh meadow/826493 385) twice, original 385
             twice, marker/snapshot/counter checks, disarm, collect.
  diff       per rank, diff the two snapshots with the helper baked into the
             pinned image (`/opt/ds41-stale-probe/claude_stale_state_probe.py`)
             under the image's own Python, CPU only, no network. Refuses to run
             beside a live serving container unless --alongside-serving.

Fail-closed rules in `run`
- Before any request: every rank's serving container runs the pinned
  diagnostic image with the kit label and the dense-release environment; no
  control file and no snapshot with the new run names exists on any rank;
  the control validates with the helper's own `parse_control`; it is written
  atomically (temp file in the same directory, then `mv -T`) on all four
  ranks and its sha256 is re-read on each.
- The helper loads the control lazily at the next real engine step. The
  first prime therefore doubles as the arming probe: its prompt digest is
  checked locally to differ from the control digest, so it cannot consume an
  action. After it, every rank must log `control loaded, 2 actions`, or the
  driver disarms and stops before any original request.
- Per request: cold cache (cache_metrics), exact answer, and spec-decode
  counter deltas of exactly 1 draft, 7 draft tokens, 3 generated tokens.
  One draft per request means one 8-token verify step, so the helper, which
  matches on prompt digest and step shape rather than request identity, can
  consume at most one action per original request. Primes must add no
  observe marker; each original must add exactly one observe marker per rank
  naming its run, and the snapshot file must exist on every rank.
- Any failure disarms (renames the control away on every rank) and exits
  nonzero; snapshots and receipts are kept.

Usage
  python3 claude_stale_capture.py preflight
  python3 claude_stale_capture.py run --out receipts/stale-capture-<tag>
  python3 claude_stale_capture.py diff --out receipts/stale-capture-<tag> [--alongside-serving]
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
NODES = ('dusty', 'toby', 'rusty', 'kirby')
NAME = 'ds41-flash-karmic-main-tp4'
BASE_URL = 'http://dusty:8000'
MODEL = 'DeepSeek-V4.1-Flash'
HOST_CACHE = '~/.cache/vllm-jj-ds41-tp4'          # mounted at /cache by run_node.py
CONTROL = f'{HOST_CACHE}/ds41-stale-control.json'
SNAPSHOTS = f'{HOST_CACHE}/ds41-stale-state'
HELPER_IN_IMAGE = '/opt/ds41-stale-probe/claude_stale_state_probe.py'
ORIGINAL = ('receipts/determinism-corpus.json', '385', ' filler', '739184')
PRIME = ('receipts/fresh-lookup-corpus-20260925.json', '385', ' meadow', '826493')
EXPECTED_SPEC = {'vllm:spec_decode_num_drafts_total': 1, 'vllm:spec_decode_num_draft_tokens_total': 7,
                 'vllm:generation_tokens_total': 3}
METRIC = re.compile(r'^(vllm:[a-z_0-9]+)(\{[^\n]*\})?\s+([0-9.eE+-]+)$', re.M)


# --------------------------------------------------------------------------
# Pure helpers (stdlib; unit tested)
# --------------------------------------------------------------------------

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus_entry(path, key, filler, code):
    corpus = json.loads((ROOT / path).read_text())
    entry = corpus[key]
    if entry.get('filler', ' filler') != filler or entry.get('retrieval_code', '739184') != code:
        raise RuntimeError(f'{path}[{key}] is not the expected {filler!r}/{code} corpus entry')
    tokens = entry['tokenize']['tokens']
    if len(tokens) != int(key) or entry['tokenize']['count'] != int(key):
        raise RuntimeError(f'{path}[{key}] token count differs from its key')
    return entry


def build_control(prompt_sha, stamp):
    runs = (f'{stamp}-orig-first', f'{stamp}-orig-repeat')
    return {'prompt_token_sha256': prompt_sha, 'scheduled_tokens': 8,
            'coverage': {
                'swa': {'suffix': '.swa_cache', 'layers': [[0, 43]]},
                'ring': {'suffix': '.compressor.state_cache', 'layers': [[2, 3], [8, 9], [14, 15]]},
                'main': {'suffix': '.attn', 'layers': [[2, 3], [8, 9], [14, 15], [20, 21]]},
                'index': {'suffix': '.indexer.k_cache', 'layers': [[2, 3], [8, 9], [14, 15], [20, 21]]}},
            'actions': [{'run': runs[0], 'mode': 'observe'}]}, runs


def metric_totals(raw):
    totals = collections.defaultdict(float)
    for name, _, value in METRIC.findall(raw):
        totals[name] += float(value)
    return totals


def spec_deltas(before_raw, after_raw):
    a, b = metric_totals(before_raw), metric_totals(after_raw)
    return {name: b.get(name, 0.0) - a.get(name, 0.0) for name in EXPECTED_SPEC}


def check_spec(deltas, label):
    bad = {k: v for k, v in deltas.items() if v != EXPECTED_SPEC[k]}
    if bad:
        raise RuntimeError(f'{label}: spec-decode deltas {deltas} differ from one 8-token verify {EXPECTED_SPEC}')


def markers(log_text):
    """(control-loaded lines, {run: observe line count}) from a container log."""
    loaded = [line for line in log_text.splitlines() if '[DS41-STALE-PROBE] control loaded, 1 actions' in line]
    observed = collections.Counter(m.group(1) for m in re.finditer(
        r'\[DS41-STALE-PROBE\] observe /cache/ds41-stale-state/([A-Za-z0-9_-]+)-(?:dusty|toby|rusty|kirby)\.pt layers=\d+',
        log_text))
    failures = [line for line in log_text.splitlines() if 'DS41-STALE-PROBE' in line
                and any(marker in line for marker in ('Error', 'observe-failed', 'control-rejected'))]
    return loaded, observed, failures


def request_body(entry):
    return {'model': MODEL, 'messages': entry['messages'], 'temperature': 0, 'max_tokens': 8,
            'logprobs': True, 'top_logprobs': 20, 'chat_template_kwargs': {'thinking': False},
            'cache_salt': uuid.uuid4().hex}


def signature(result):
    choice = result['choices'][0]
    return hashlib.sha256(json.dumps(choice['logprobs'], sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------
# Local identity (no node contact)
# --------------------------------------------------------------------------

def local_identity():
    helper = load_module('claude_stale_state_probe', ROOT / 'claude_stale_state_probe.py')
    lock_bytes = (ROOT / 'stale-probe.lock.json').read_bytes()
    lock = json.loads(lock_bytes)
    for name, expected in lock['inputs'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Local diagnostic input differs from the lock: ' + name)
    build = json.loads((ROOT / 'receipts/stale-probe-build-receipt.json').read_text())
    pin = json.loads((ROOT / 'candidate.json').read_text())
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    if (build['lock_sha256'] != lock_sha or pin['image_id'] != build['image_id']
            or pin.get('diagnostic', {}).get('kind') != 'stale-state-probe'
            or pin['diagnostic']['lock_sha256'] != lock_sha):
        raise RuntimeError('candidate.json is not the built stale-state probe image')
    original = corpus_entry(*ORIGINAL)
    prime = corpus_entry(*PRIME)
    original_sha = helper.token_sha256(original['tokenize']['tokens'])
    prime_sha = helper.token_sha256(prime['tokenize']['tokens'])
    if original_sha == prime_sha:
        raise RuntimeError('Prime and original prompts hash identically')
    return helper, pin, lock_sha, original, prime, original_sha, prime_sha


def preflight(_args):
    helper, pin, lock_sha, _, _, original_sha, prime_sha = local_identity()
    control, runs = build_control(original_sha, 'PREFLIGHT')
    helper.parse_control(control)
    print(json.dumps({'image_id': pin['image_id'], 'lock_sha256': lock_sha, 'original_prompt_sha256': original_sha,
                      'prime_prompt_sha256': prime_sha, 'control_example': control}, indent=1))
    print('STALE-CAPTURE-PREFLIGHT-PASS', flush=True)
    return 0


# --------------------------------------------------------------------------
# Node operations (run / diff only)
# --------------------------------------------------------------------------

def ssh(node, command, *, stdin=None, timeout=60):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node, command],
                            input=stdin, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{node}: {command!r} failed: {result.stderr.decode(errors="replace")[-400:]}')
    return result.stdout.decode()


def container_identity(node, pin, kit):
    info = json.loads(ssh(node, f'podman inspect {NAME}'))[0]
    env = dict(item.split('=', 1) for item in info['Config']['Env'])
    problems = []
    if not info['State']['Running']:
        problems.append('not running')
    if info['Image'].removeprefix('sha256:') != pin['image_id']:
        problems.append('image ' + info['Image'])
    if info['Config']['Labels'].get('local-inference.ds41.kit.sha256') != kit:
        problems.append('kit label')
    if env.get('B12X_DYNAMIC_DETERMINISTIC_OUTPUT') != '1' or env.get('B12X_DENSE_SPLITK_TURBO') != '0' \
            or 'VLLM_DS41_L2_PREFETCH' in env:
        problems.append('profile environment')
    if problems:
        raise RuntimeError(f'{node}: serving identity mismatch: {problems}')
    return {'StartedAt': info['State']['StartedAt'], 'RestartCount': info['RestartCount'], 'Id': info['Id']}


def logs_since(node, since):
    return ssh(node, f'podman logs --since {shlex.quote(since)} {NAME} 2>&1', timeout=120)


def disarm(nodes, stamp, out):
    results = {}
    for node in nodes:
        try:
            ssh(node, f'if [ -e {CONTROL} ]; then mv -T {CONTROL} {HOST_CACHE}/ds41-stale-control.{stamp}.disarmed; fi')
            results[node] = 'disarmed'
        except Exception as error:  # noqa: BLE001 - report every rank
            results[node] = f'FAILED: {error}'
    (out / 'disarm.json').write_text(json.dumps(results, indent=2) + '\n')
    print('DISARM', results, flush=True)
    return results


def post(path, body, timeout=600):
    request = urllib.request.Request(BASE_URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def one_request(label, entry, out, cache_metrics):
    body = request_body(entry)
    before = cache_metrics.idle_snapshot(BASE_URL)
    start = time.monotonic()
    result = post('/v1/chat/completions', body)
    elapsed = time.monotonic() - start
    (out / f'{label}.json').write_text(json.dumps({'request': body, 'response': result, 'elapsed_s': elapsed},
                                                  indent=1) + '\n')
    length = int(entry['tokenize']['count'])
    if result['usage']['prompt_tokens'] != length:
        raise RuntimeError(f'{label}: prompt token count {result["usage"]["prompt_tokens"]} != {length}')
    hits = cache_metrics.finish(BASE_URL, before, length, out / f'{label}-cache.json')
    if hits != 0:
        raise RuntimeError(f'{label}: prefix cache hit {hits}')
    after = json.loads((out / f'{label}-cache.json').read_text())['after_samples'][-1]
    deltas = spec_deltas(before['raw'], after['raw'])
    check_spec(deltas, label)
    choice = result['choices'][0]
    code = entry['retrieval_code'] if 'retrieval_code' in entry else '739184'
    if choice['finish_reason'] != 'stop' or (choice['message'].get('content') or '').strip() != code:
        raise RuntimeError(f'{label}: wrong or incomplete answer {choice["message"]!r}')
    row = {'label': label, 'elapsed_s': elapsed, 'signature': signature(result), 'spec_deltas': deltas}
    print(json.dumps(row), flush=True)
    return row


def run(args):
    helper, pin, lock_sha, original, prime, original_sha, prime_sha = local_identity()
    sys.path.insert(0, str(ROOT))
    import cache_metrics                                           # task-tree helper, unchanged
    from runtime import kit_digest                                 # task-tree kit identity
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    control, runs = build_control(original_sha, stamp)
    helper.parse_control(control)
    payload = (json.dumps(control, indent=1, sort_keys=True) + '\n').encode()
    control_sha = hashlib.sha256(payload).hexdigest()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'control.json').write_bytes(payload)
    report = {'stamp': stamp, 'runs': runs, 'image_id': pin['image_id'], 'lock_sha256': lock_sha,
              'control_sha256': control_sha, 'original_prompt_sha256': original_sha,
              'prime_prompt_sha256': prime_sha, 'requests': [], 'status': 'started'}

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    kit = kit_digest()
    identities = {node: container_identity(node, pin, kit) for node in NODES}
    report['identities_before'] = identities
    cache_metrics.idle_snapshot(BASE_URL)
    for node in NODES:
        existing = ssh(node, f'ls -1 {CONTROL} {SNAPSHOTS}/{runs[0]}-{node}.pt {SNAPSHOTS}/{runs[1]}-{node}.pt '
                             f'2>/dev/null || true').strip()
        if existing:
            raise RuntimeError(f'{node}: control or snapshot already present: {existing}')
    # Node clocks, not the workstation's, bound the log window; taken before deployment.
    since = {node: ssh(node, 'date -u +%Y-%m-%dT%H:%M:%SZ').strip() for node in NODES}
    report['log_since'] = since
    deployed = []
    try:
        for node in NODES:
            tmp = f'{HOST_CACHE}/.ds41-stale-control.{stamp}.tmp'
            # A disconnected SSH client can report failure after mv succeeded.
            # Include the attempted rank in cleanup before issuing the write.
            deployed.append(node)
            ssh(node, f'umask 077; cat > {tmp} && mv -T {tmp} {CONTROL} && sha256sum {CONTROL}', stdin=payload)
            remote_sha = ssh(node, f'sha256sum {CONTROL}').split()[0]
            if remote_sha != control_sha:
                raise RuntimeError(f'{node}: deployed control digest {remote_sha} != {control_sha}')
        report['deployed'] = deployed
        save()
        # Arming probe: the first prime cannot match (its prompt digest differs), but it makes
        # every rank run a real step, which is when the helper reads the control file.
        report['requests'].append(one_request('prime-0', prime, out, cache_metrics))
        for node in NODES:
            loaded, observed, failures = markers(logs_since(node, since[node]))
            if len(loaded) != 1 or observed or failures:
                raise RuntimeError(f'{node}: not armed after the first prime (loaded={len(loaded)}, '
                                   f'observed={dict(observed)}, failures={failures[:2]})')
        report['armed'] = True
        save()
        report['requests'].append(one_request('prime-1', prime, out, cache_metrics))
        for index, run_name in enumerate(runs):
            if index:
                # Async scheduling may prepare an unused step after the response.
                # Arm exactly one capture for each separately issued request.
                cache_metrics.idle_snapshot(BASE_URL)
                control['actions'] = [{'run': run_name, 'mode': 'observe'}]
                helper.parse_control(control)
                payload = (json.dumps(control, indent=1, sort_keys=True) + '\n').encode()
                control_sha = hashlib.sha256(payload).hexdigest()
                (out / f'control-{index}.json').write_bytes(payload)
                for node in NODES:
                    tmp = f'{HOST_CACHE}/.ds41-stale-control.{stamp}.{index}.tmp'
                    ssh(node, f'umask 077; cat > {tmp} && mv -T {tmp} {CONTROL}', stdin=payload)
                    if ssh(node, f'sha256sum {CONTROL}').split()[0] != control_sha:
                        raise RuntimeError(f'{node}: rearmed control digest mismatch')
            report['requests'].append(one_request(f'original-{index}', original, out, cache_metrics))
            for node in NODES:
                loaded, observed, failures = markers(logs_since(node, since[node]))
                expected = {r: 1 for r in runs[:index + 1]}
                if dict(observed) != expected or failures or len(loaded) != index + 1:
                    raise RuntimeError(f'{node}: markers after original-{index}: observed={dict(observed)} '
                                       f'expected={expected} loaded={len(loaded)} failures={failures[:2]}')
                listing = ssh(node, f'ls -l {SNAPSHOTS}/{run_name}-{node}.pt')
                if f'{run_name}-{node}.pt' not in listing:
                    raise RuntimeError(f'{node}: snapshot {run_name} missing')
            save()
        signatures = [r['signature'] for r in report['requests'] if r['label'].startswith('original-')]
        report['reproduced'] = signatures[0] != signatures[1]
        report['status'] = 'captured' if report['reproduced'] else 'captured-NOT-REPRODUCED'
    except BaseException:
        report['status'] = 'failed'
        raise
    finally:
        report['disarm'] = disarm(deployed or NODES, stamp, out)
        if any(value != 'disarmed' for value in report['disarm'].values()):
            report['status'] = 'failed-disarm'
        try:
            report['identities_after'] = {node: container_identity(node, pin, kit) for node in NODES}
            if report['identities_after'] != identities:
                report['status'] = 'failed-container-changed'
        except Exception as error:  # noqa: BLE001
            report['identities_after'] = repr(error)
            report['status'] = 'failed-identity-after'
        save()
    collect(out, runs)
    save()
    print(f"STALE-CAPTURE-{report['status'].upper()}", out, flush=True)
    return 0 if report['status'] == 'captured' else 1


def collect(out, runs):
    target = out / 'snapshots'
    target.mkdir(exist_ok=True)
    digests = {}
    for node in NODES:
        for run_name in runs:
            name = f'{run_name}-{node}.pt'
            subprocess.run(['scp', '-q', f'{node}:{SNAPSHOTS}/{name}', str(target / name)], check=True)
            digests[name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
    (out / 'snapshots.sha256.json').write_text(json.dumps(digests, indent=2) + '\n')


def diff(args):
    out = Path(args.out)
    report = json.loads((out / 'report.json').read_text())
    if report.get('status') not in ('captured', 'captured-NOT-REPRODUCED'):
        raise RuntimeError('capture did not complete; nothing to diff')
    image, runs = report['image_id'], report['runs']
    results = {}
    for node in NODES:
        busy = ssh(node, 'podman ps -q').strip()
        if busy and not args.alongside_serving:
            raise RuntimeError(f'{node}: containers running; stop serving or pass --alongside-serving')
        a, b = (shlex.quote(f'/data/{name}-{node}.pt') for name in runs)
        command = (f'podman run --rm --pull=never --network=none --cpus=2 --memory=16g '
                   f'-v "$HOME/.cache/vllm-jj-ds41-tp4/ds41-stale-state:/data:ro" '
                   f'--entrypoint /opt/venv/bin/python {shlex.quote(image)} {HELPER_IN_IMAGE} diff {a} {b}')
        text = ssh(node, command, timeout=1800)
        (out / f'diff-{node}.json').write_text(text)
        results[node] = json.loads(text)
        print(node, json.dumps({k: (len(v) if isinstance(v, (dict, list)) else v)
                                for k, v in results[node].items()}), flush=True)
    print('STALE-CAPTURE-DIFF-DONE', out, flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('preflight')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--out', required=True)
    diff_parser = sub.add_parser('diff')
    diff_parser.add_argument('--out', required=True)
    diff_parser.add_argument('--alongside-serving', action='store_true')
    args = parser.parse_args()
    return {'preflight': preflight, 'run': run, 'diff': diff}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
