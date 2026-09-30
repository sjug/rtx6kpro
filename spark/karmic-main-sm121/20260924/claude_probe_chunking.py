"""Fixed-input chunking diagnostic for the frozen 524288-token dual-needle family (DS4.1 TP4).

Supplementary evidence only: not an acceptance gate, not an independent model reference, and not a
causal precision claim. The same token-identical prompts saved by the sensitivity receipts
(053026Z, 050535Z) are replayed cold on a separately approved boot of the clean router candidate
parent e06df11a whose only differences from the authorized B2 contract are the scheduler's
long-prefill threshold (DS41_PREFILL_THRESHOLD=7936 or 4096, --long-prefill-token-threshold) and
the KV pin DS41_CHUNKING_BLOCKS=81389 (this family's cached selections). The prepared capacity
(max_num_batched_tokens 8192) is unchanged, so every B12X key and the eight-plan regime stay at the
pinned 045954Z snapshot; the threshold only moves the prefill chunk grid.

Requests, in order (each cold with a unique cache_salt, the frozen gate request otherwise):
  control-1, frozen-516096, fresh1-524288, frozen-532480, frozen-524287, control-2
Both controls must share the full response signature (within-boot determinism); their answers may
be wrong. Every row is contrasted with its saved 8192-chunking row (answer, first-token and
late-code top-20 margins, signature) under a historical, non-causal scope.

Gate (fail closed, every rank): router-candidate image and labels, current reviewed kit, repaired
determinism environment, no prefetch override, no DS41_DECISION_ROW_BLOCKS, DS41_PREFILL_THRESHOLD
equal to --threshold, DS41_CHUNKING_BLOCKS=81389, the boot log's effective non-default args
read by key pattern on the head rank (long_prefill_token_threshold, num_gpu_blocks_override,
max_num_batched_tokens; workers start no API server), the rendered profile JSON line runtime.py prints on
every rank (effective argv and environment), 81389 serving blocks on every rank, the B2 selection namespace, and the pinned regime before and after. Saved prompts must
match their receipts' sha256 and re-count to their recorded token totals on this boot.

Never: build, launch, restart, stop, cache or image changes, deletion on a node, or any change to
the launch contract, start driver or runtime manifest. GLM nodes are never contacted.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import uuid

from cache_metrics import idle_snapshot
import claude_probe_needle_sensitivity as sens
import claude_run_decision_capture as drv
from prepare_decision_reference import B2_IMAGE, PROMPT_SHA
from runtime import kit_digest

ROOT = Path(__file__).resolve().parent
NODES = drv.NODES
NAME, REMOTE, SELECTION = drv.NAME, drv.REMOTE, drv.SELECTION
THRESHOLDS = (7936, 4096)
CAPACITY, BLOCKS = 8192, 81389
REGIME_SNAPSHOT = 'receipts/needle-sensitivity-20260926T045954Z/regime-pre.json'
REGIME_SNAPSHOT_SHA256 = 'd8793ca37b7bdc70e84b23dc9f63472a8b27c1fe5c413bd579bfda944227f2f4'
ALIGNMENT_RECEIPT = 'receipts/needle-sensitivity-20260926T053026Z'
SENSITIVITY_RECEIPT = 'receipts/needle-sensitivity-20260926T050535Z'
ARGS_PREFIX = 'non-default args: '
ARGS_KEYS = ('long_prefill_token_threshold', 'num_gpu_blocks_override', 'max_num_batched_tokens', 'max_num_seqs',
             'enable_chunked_prefill')
PROFILE_PREFIX = '{"container_resource_args"'
SCOPE = ('fixed-input chunking diagnostic: token-identical saved prompts replayed under a different prefill '
         'chunk grid with the prepared capacity and all B12X keys unchanged; contrasts are historical '
         'observations against the 8192-chunking boot, not a causal precision claim, not an acceptance gate, '
         'not an independent model reference; the historical 524288 finding stays unresolved')


# ---- pure helpers (CPU-tested) ------------------------------------------------------------------

def plan():
    """Saved prompts to replay, in order: (label, tokens, source receipt and variant label)."""
    align = ALIGNMENT_RECEIPT
    return [{'label': 'control-1', 'tokens': 524288, 'source': (align, 'control-1')},
            {'label': 'frozen-516096', 'tokens': 516096, 'source': (align, 'frozen-516096')},
            {'label': 'fresh1-524288', 'tokens': 524288, 'source': (SENSITIVITY_RECEIPT, 'fresh-1')},
            {'label': 'frozen-532480', 'tokens': 532480, 'source': (align, 'frozen-532480')},
            {'label': 'frozen-524287', 'tokens': 524287, 'source': (align, 'frozen-524287')},
            {'label': 'control-2', 'tokens': 524288, 'source': (align, 'control-1')}]


def verify_prompt(item, content, construction):
    digest = hashlib.sha256(content.encode()).hexdigest()
    if digest != construction['content_sha256'] or construction['rounds'][-1]['prompt_tokens'] != item['tokens']:
        raise RuntimeError(f'{item["label"]}: saved prompt differs from its construction record')
    return digest


def load_prompt(root, item):
    """The saved prompt text, verified against its receipt, plus that receipt's response row."""
    receipt, label = item['source']
    construction = json.loads((root / receipt / 'construction.json').read_text())[label]
    content = (root / receipt / label / 'content.txt').read_text()
    digest = verify_prompt(item, content, construction)
    row = json.loads((root / receipt / label / 'response-row.json').read_text())
    if row['prompt_tokens'] != item['tokens'] or row['content_sha256'] != digest:
        raise RuntimeError(f'{item["label"]}: saved response row does not belong to the saved prompt')
    historical = {'boot_receipt': receipt, 'variant': label, 'prompt_tokens': row['prompt_tokens'],
                  'content': row['content'], 'answer_shape': row['answer_shape'], 'correct': row['correct'],
                  'first_token_margin': row['margins']['first_token']['margin'],
                  'late_code_margin': (row['margins']['late_code_after_separator'] or {}).get('margin'),
                  'response_signature': row['response_signature']}
    return dict(item, content=content, content_sha256=digest, hex=construction['hex'], historical=historical)


def parse_non_default_args(log_text):
    """Scalar settings from the API server's startup line (api_utils.log_non_default_args), read by
    key pattern only: the line is a Python repr with enum and object reprs, never evaluated. The line is
    printed by the head rank's API server only (headless workers start no API server)."""
    lines = {line[line.index(ARGS_PREFIX):].strip() for line in log_text.splitlines() if ARGS_PREFIX in line}
    if len(lines) != 1:
        raise ValueError(f'boot log must carry exactly one non-default args line, found {len(lines)}')
    line = lines.pop()
    args = {}
    for key in ARGS_KEYS:
        match = re.search(r"'" + re.escape(key) + r"': (-?\d+|True|False|None)(?=[,}])", line)
        if match:
            value = match.group(1)
            args[key] = {'True': True, 'False': False, 'None': None}.get(value, None) if not value.lstrip('-').isdigit() else int(value)
    return args


def check_args(args, threshold):
    expected = {'long_prefill_token_threshold': threshold, 'num_gpu_blocks_override': BLOCKS,
                'max_num_batched_tokens': CAPACITY, 'max_num_seqs': 4, 'enable_chunked_prefill': True}
    return [f'effective {key}={args.get(key)!r}, approved {value!r}' for key, value in expected.items()
            if args.get(key) != value]


def parse_profile(log_text):
    """The rendered launch profile runtime.py prints as one JSON line on every rank before exec."""
    lines = {line.strip() for line in log_text.splitlines() if line.startswith(PROFILE_PREFIX)}
    if len(lines) != 1:
        raise ValueError(f'container log must carry exactly one rendered profile line, found {len(lines)}')
    profile = json.loads(lines.pop())
    if not isinstance(profile.get('model'), list) or not isinstance(profile.get('env'), dict):
        raise ValueError('rendered profile line lacks model argv or env')
    return profile


def check_profile(profile, node, threshold):
    """The effective vLLM argv and environment of one rank against the approved chunking contract."""
    argv = profile['model']
    problems = []
    if profile.get('node') != node:
        problems.append(f'rendered profile is for {profile.get("node")!r}, not {node}')
    expected = {'--long-prefill-token-threshold': str(threshold), '--num-gpu-blocks-override': str(BLOCKS),
                '--max-num-batched-tokens': str(CAPACITY), '--max-num-seqs': '4'}
    for flag, value in expected.items():
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != value:
            problems.append(f'effective argv {flag} is not {value}')
    if '--enable-chunked-prefill' not in argv:
        problems.append('effective argv lacks --enable-chunked-prefill')
    env = profile['env']
    if env.get('DS41_PREFILL_THRESHOLD') != str(threshold) or env.get('DS41_CHUNKING_BLOCKS') != str(BLOCKS):
        problems.append('rendered environment lacks the approved chunking controls')
    if 'DS41_DECISION_ROW_BLOCKS' in env:
        problems.append('rendered environment carries the capture KV pin')
    return problems


def verify_boot_log(node, log_text, threshold):
    """Problems with one rank's container log: serving blocks (all ranks), the rendered profile (all ranks),
    and the API server's non-default args (head rank; also checked on a worker if it ever prints one)."""
    problems = []
    try:
        blocks = sens.serving_blocks(log_text)
        if blocks != BLOCKS:
            problems.append(f'boot allocated {blocks} serving blocks, approved {BLOCKS}')
    except ValueError as error:
        problems.append(str(error))
    try:
        problems += check_profile(parse_profile(log_text), node, threshold)
    except ValueError as error:
        problems.append(str(error))
    if node == NODES[0] or ARGS_PREFIX in log_text:
        try:
            problems += check_args(parse_non_default_args(log_text), threshold)
        except ValueError as error:
            problems.append(str(error))
    return problems


def check_env(node, container, image, reference, kits, threshold):
    """The sensitivity gate plus the approved chunking environment; identity on success."""
    problems, identity = sens.check_container(node, container, image, reference, kits)
    env = drv.env_map(container['Config']['Env'])
    if env.get('DS41_PREFILL_THRESHOLD') != str(threshold):
        problems.append(f'DS41_PREFILL_THRESHOLD is {env.get("DS41_PREFILL_THRESHOLD")!r}, probe expects {threshold}')
    if env.get('DS41_CHUNKING_BLOCKS') != str(BLOCKS):
        problems.append(f'DS41_CHUNKING_BLOCKS is {env.get("DS41_CHUNKING_BLOCKS")!r}, approved {BLOCKS}')
    identity.update(threshold=threshold, chunking_blocks=BLOCKS)
    return problems, identity


def contrast(row, historical):
    def delta(current, past):
        return None if current is None or past is None else current - past
    late = row['margins']['late_code_after_separator'] or {}
    return {'historical_boot': historical['boot_receipt'], 'historical_answer_shape': historical['answer_shape'],
            'historical_correct': historical['correct'], 'historical_first_token_margin': historical['first_token_margin'],
            'historical_late_code_margin': historical['late_code_margin'],
            'first_token_margin_delta': delta(row['margins']['first_token']['margin'], historical['first_token_margin']),
            'late_code_margin_delta': delta(late.get('margin'), historical['late_code_margin']),
            'answer_changed': row['answer_shape'] != historical['answer_shape'] or row['correct'] != historical['correct'],
            'signature_equal': row['response_signature'] == historical['response_signature']}


def summarize(rows, threshold):
    controls = [r for r in rows if r['label'].startswith('control-')]
    return {'threshold': threshold, 'capacity': CAPACITY, 'blocks': BLOCKS,
            'controls_identical': len(controls) == 2 and len({r['response_signature'] for r in controls}) == 1,
            'variants': [{k: r[k] for k in ('label', 'hex', 'prompt_tokens', 'content', 'answer_shape', 'correct',
                                             'cached_tokens', 'elapsed_s', 'response_signature')}
                         | {'first_token_margin': r['margins']['first_token']['margin'],
                            'late_code_margin': (r['margins']['late_code_after_separator'] or {}).get('margin'),
                            'contrast': r['contrast']}
                         for r in rows],
            'scope': SCOPE}


# ---- remote plumbing (thin, untested) -----------------------------------------------------------

def preflight(out, reference, b2_cache_path, threshold):
    identities, blocks, effective = {}, {}, {}
    kits = {kit_digest(), reference['kit_sha256']}
    for node in NODES:
        running = drv.ssh(node, 'podman ps -q').split()
        container = drv.ssh_json(node, 'podman inspect ' + NAME)[0]
        image = drv.ssh_json(node, 'podman image inspect ' + B2_IMAGE)[0]
        (out / f'{node}-container.json').write_text(json.dumps(container, indent=2) + '\n')
        (out / f'{node}-image.json').write_text(json.dumps(image, indent=2) + '\n')
        (out / f'{node}-meminfo.txt').write_text(drv.ssh(node, 'cat /proc/meminfo'))
        problems, identity = check_env(node, container, image, reference, kits, threshold)
        if len(running) != 1 or not container['Id'].startswith(running[0]):
            problems.append('another container runs on the rank host')
        if identity['cache_dir'] and identity['cache_dir'] + '/' + SELECTION != b2_cache_path:
            problems.append('selection namespace differs from the B2 tuning snapshot')
        log = drv.ssh(node, f'podman logs {NAME} 2>&1 | grep -F -e "serving blocks" -e {shlex.quote(ARGS_PREFIX)}'
                            f' -e {shlex.quote(PROFILE_PREFIX)} || true')
        (out / f'{node}-boot-args.log').write_text(log)
        problems += verify_boot_log(node, log, threshold)
        if not problems:
            blocks[node] = sens.serving_blocks(log)
            effective[node] = {'profile_model': parse_profile(log)['model'],
                               'api_server_args': parse_non_default_args(log) if ARGS_PREFIX in log else None}
        if problems:
            raise RuntimeError(f'{node}: ' + '; '.join(problems))
        identities[node] = identity
        print('PREFLIGHT-OK', node, 'threshold', threshold, 'serving_blocks', blocks[node], flush=True)
    (out / 'effective-args.json').write_text(json.dumps(effective, indent=2, sort_keys=True) + '\n')
    return identities, BLOCKS


def run(a):
    reference, spec, _, b2_cache_path = sens.local_inputs()
    snapshot = ROOT / REGIME_SNAPSHOT
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != REGIME_SNAPSHOT_SHA256:
        raise RuntimeError('pinned regime snapshot changed')
    prompts = [load_prompt(ROOT, item) for item in plan()]
    if prompts[0]['content_sha256'] != PROMPT_SHA or prompts[0]['content'] != spec['messages'][0]['content']:
        raise RuntimeError('saved control prompt is not the frozen historical input')
    stamp = drv.stamp_now()
    out = ROOT / 'receipts' / f'chunking-{a.threshold}-{stamp}'
    out.mkdir(exist_ok=False)
    now = datetime.datetime.now(datetime.timezone.utc)
    since, podman_since = now.strftime('%Y-%m-%d %H:%M:%S UTC'), now.strftime('%Y-%m-%dT%H:%M:%SZ')
    observers = []
    try:
        for node in NODES:
            stream = (out / f'{node}-telemetry.log').open('x')
            process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', node,
                                        f'echo OBSERVER_PID=$$; exec bash {REMOTE}/observe.sh'],
                                       stdout=stream, stderr=subprocess.STDOUT)
            observers.append((node, process, stream))
        execute(a, reference, spec, prompts, b2_cache_path, snapshot, out)
    finally:
        for node, process, stream in observers:
            try:
                stream.flush()
                lines = (out / f'{node}-telemetry.log').read_text().splitlines()
                first = lines[0] if lines else ''
                if first.startswith('OBSERVER_PID=') and first.split('=', 1)[1].isdigit():
                    pid = first.split('=', 1)[1]
                    drv.ssh(node, f'if ps -p {pid} -o args= | grep -Fq "{REMOTE}/observe.sh"; then kill -TERM {pid}; fi')
                process.wait(timeout=30)
            except Exception as error:
                (out / f'{node}-observer-cleanup.txt').write_text(str(error) + '\n')
                process.terminate()
                process.wait(timeout=30)
            finally:
                stream.close()
        for node in NODES:
            command = (f'podman logs --since {shlex.quote(podman_since)} {NAME} 2>&1; p=$?; '
                       f'journalctl -k --since {shlex.quote(since)} --no-pager; '
                       'j=$?; grep -E "MemAvailable|SwapFree" /proc/meminfo; test "$p" = 0 && test "$j" = 0')
            try:
                result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, command],
                                        capture_output=True, text=True, timeout=120)
                (out / f'{node}-final.log').write_text(result.stdout + result.stderr)
                (out / f'{node}-collection.exit').write_text(str(result.returncode) + '\n')
            except Exception as error:
                (out / f'{node}-collection-error.txt').write_text(str(error) + '\n')


def execute(a, reference, spec, prompts, b2_cache_path, snapshot, out):
    print('RECEIPTS', out, flush=True)
    (out / 'reference.json').write_bytes(sens.REFERENCE.read_bytes())
    (out / 'regime-reference.json').write_bytes(snapshot.read_bytes())
    pre = out / 'pre'
    pre.mkdir()
    identities, blocks = preflight(pre, reference, b2_cache_path, a.threshold)
    (out / 'identities.json').write_text(json.dumps({'blocks': blocks, 'nodes': identities}, indent=2, sort_keys=True) + '\n')
    idle_snapshot(a.base_url)
    selection_reference, comparison = sens.comparison_reference(reference, snapshot)
    comparison.update(series='chunking', threshold=a.threshold, capacity=CAPACITY, scope=SCOPE)
    (out / 'comparison-scope.json').write_text(json.dumps(comparison, indent=2) + '\n')
    sens.snapshot_selections(identities, blocks, selection_reference, out, 'pre')
    construction = {}
    for item in prompts:
        encoding = sens.post(a.base_url, '/tokenize', sens.tokenize_body(sens.user(item['content'])), 600)
        tokens = encoding['tokens']
        if encoding['count'] != item['tokens'] or len(tokens) != item['tokens']:
            raise RuntimeError(f'{item["label"]}: saved prompt counts {encoding["count"]} tokens on this boot, recorded {item["tokens"]}')
        construction[item['label']] = {'source': item['source'], 'tokens': item['tokens'], 'hex': item['hex'],
                                       'content_sha256': item['content_sha256'],
                                       'token_ids_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                                       'historical': item['historical']}
        print('VERIFIED', item['label'], json.dumps(construction[item['label']]), flush=True)
    (out / 'construction.json').write_text(json.dumps(construction, indent=2) + '\n')
    if a.verify_only:
        print('CHUNKING-VERIFIED', out, flush=True)
        return
    rows = []
    for item in prompts:
        trial_dir = out / item['label']
        trial_dir.mkdir()
        (trial_dir / 'content.txt').write_text(item['content'])
        result, elapsed, cached = sens.send(a.base_url, sens.request_body(item['content'], uuid.uuid4().hex),
                                            item['tokens'], trial_dir)
        row = sens.response_row(result, spec['expected'], item['tokens'], reference)
        row.update(label=item['label'], hex=item['hex'], elapsed_s=elapsed, cached_tokens=cached,
                   content_sha256=item['content_sha256'], source=item['source'])
        row['contrast'] = contrast(row, item['historical'])
        rows.append(row)
        (trial_dir / 'response-row.json').write_text(json.dumps(row, indent=2) + '\n')
        print('VARIANT', json.dumps(row), flush=True)
    post_dir = out / 'post'
    post_dir.mkdir()
    identities_after, blocks_after = preflight(post_dir, reference, b2_cache_path, a.threshold)
    if identities_after != identities or blocks_after != blocks:
        raise RuntimeError('serving identity changed during the probe')
    sens.snapshot_selections(identities, blocks, selection_reference, out, 'post')
    summary = summarize(rows, a.threshold)
    summary['comparison'] = comparison
    summary.update(image_id=B2_IMAGE, kit_sha256=identities[NODES[0]]['kit_sha256'],
                   kit_equals_b2_reference=identities[NODES[0]]['kit_equals_b2_reference'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print('CHUNKING-COMPLETE', json.dumps(summary), flush=True)
    if not summary['controls_identical']:
        raise RuntimeError('the two frozen controls differ on this boot; contrasts are not interpretable')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-url', default='http://dusty:8000')
    parser.add_argument('--threshold', type=int, choices=THRESHOLDS, required=True,
                        help='the approved long-prefill threshold this boot was started with')
    parser.add_argument('--verify-only', action='store_true',
                        help='preflight, regime and saved-prompt verification only; no completion request')
    run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
