"""Alternate traced/untraced cold requests on one diagnostic boot. Never a qualification gate."""
import argparse
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

from cache_metrics import idle_snapshot
from claude_act_analyze import analyze, load
from runtime import kit_digest

ROOT = Path(__file__).resolve().parent
NODES = ('dusty', 'toby', 'rusty', 'kirby')
CACHE = '/home/jugs/.cache/vllm-jj-ds41-tp4'
TRIGGER = CACHE + '/ds41-act-trace.json'


def ssh(node, command):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, command], text=True, timeout=60)


def identities(out, label, kind='activation-digest'):
    pin = json.loads((ROOT / 'candidate.json').read_text())
    if pin.get('diagnostic', {}).get('kind') != kind:
        raise RuntimeError('Not the activation diagnostic')
    digest = kit_digest()
    result = {}
    for node in NODES:
        raw = ssh(node, 'podman inspect ds41-flash-karmic-main-tp4')
        (out / f'{node}-identity-{label}.json').write_text(raw)
        info = json.loads(raw)[0]
        if (not info['State']['Running'] or info['Image'].removeprefix('sha256:') != pin['image_id']
                or info['Config']['Labels'].get('local-inference.ds41.kit.sha256') != digest):
            raise RuntimeError('Unexpected candidate on ' + node)
        result[node] = [info['Id'], info['Image'], info['RestartCount'], info['State']['StartedAt'],
                        info['Config']['Env'], info['Config']['Cmd']]
    return result


def check_coverage(records, report, mode='sparse'):
    if report['counts'].get('covered', 0) < 3 or report['verdict'] == 'no-coverage':
        raise RuntimeError('No aligned model coverage')
    if any(r.get('dropped_before', 0) or r.get('overflow', False) for r in records):
        raise RuntimeError('Dropped or overflowed trace records')
    required = {f'layers.{layer}:{member}' for layer in (0, 1, 2, 8, 14, 19, 20, 39)
                for member in range(5)} | {'layers.1.engram', 'layers.14.engram'}
    if mode == 'layer0':
        required = {f'layers.0:{member}' for member in range(5)} | {
            '>layers.0.ffn:0', 'layers.0.ffn', 'layers.0.attn', 'layers.0.ffn.experts'}
    elif mode == 'all-layers':
        # Pinned checkpoint text_config.num_hidden_layers == 40, verified on dusty.
        required = {name for layer in range(40) for name in (
            f'>layers.{layer}.ffn:0', f'layers.{layer}.ffn', f'layers.{layer}.attn')}
    elif mode == 'all-ffn-branches':
        required = {name for layer in range(40) for name in (
            f'>layers.{layer}.ffn:0', f'layers.{layer}.ffn',
            f'layers.{layer}.ffn.gate:0', f'layers.{layer}.ffn.shared_experts')}
    elif mode in ('moe-seams', 'moe-inputs', 'moe-capture'):
        required = {f'layers.{layer}.ffn.experts#{seam}' for layer in range(40)
                    for seam in ('shared', 'routed', 'pre_reduce', 'post_reduce')}
        if mode == 'moe-capture':
            required |= {f'layers.{layer}.ffn.experts#{seam}' for layer in range(40)
                         for seam in ('input', 'logits', 'topk_weights', 'topk_ids', 'input_after')}
            if any(not isinstance(r.get('capture'), dict)
                   or type(r['capture'].get('generation')) is not int
                   or r['capture']['generation'] < 1
                   or r['capture'].get('state') == 'error' for r in records):
                raise RuntimeError('Capture is not armed cleanly on every record')
        if mode == 'moe-inputs':
            required |= {name for layer in range(40) for name in (
                f'>layers.{layer}.ffn:0', f'layers.{layer}.ffn.gate:0')}
        if any(r.get('seam_runners') != 40 for r in records):
            raise RuntimeError('Expected all 40 MoE seam runners')
    for rank in range(4):
        matching = [r for r in records if r['rank'] == rank]
        if not matching or any(not required <= set(r['names'])
                               or not {'<output>', '<output>:0'} & set(r['names']) for r in matching):
            raise RuntimeError('Incomplete actual hook coverage on rank ' + str(rank))


def complete_records(records, expected):
    return all(sum(r['rank'] == rank for r in records) == expected for rank in range(4))


def capture_downloads(status_lines, arm):
    downloads = []
    for entry in status_lines:
        if entry.get('arm') != arm or entry.get('state') != 'saved':
            continue
        rank, node = entry.get('rank'), entry.get('node')
        if type(rank) is not int or not 0 <= rank < 4 or node != NODES[rank]:
            raise RuntimeError('Capture rank/node mismatch')
        path = Path(entry.get('path', ''))
        if path.parent != Path('/cache/ds41-act-trace') or not re.fullmatch(
                re.escape(node) + r'-[0-9]+-g[0-9]+-capture\.pt', path.name):
            raise RuntimeError('Unsafe capture receipt path')
        downloads.append((node, path.name))
    return sorted(set(downloads))


def settle_capture(read_incomplete, flush, *, timeout=60, clock=time.monotonic, sleep=time.sleep):
    """A final-step latch needs another forward, outside the traced row window.

    Retry at most three flushes in case a receipt was copied just before the
    writer set ready. Never hide an error or wait without a deadline.
    """
    deadline, next_flush, attempts = clock() + timeout, clock(), 0
    while True:
        pending = read_incomplete()
        if not pending or any('error' in row['states'] for row in pending):
            return pending
        now = clock()
        if now >= deadline:
            return pending
        if attempts < 3 and now >= next_flush and any('fetching' not in row['states'] for row in pending):
            flush(attempts)
            attempts += 1
            next_flush = clock() + 1
        sleep(.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--cycles', type=int, default=10)
    parser.add_argument('--mode', choices=('sparse', 'layer0', 'all-layers', 'all-ffn-branches', 'moe-seams', 'moe-inputs', 'moe-capture'), default='sparse')
    args = parser.parse_args()
    if args.blocks < 2 or args.cycles < 3:
        parser.error('At least two blocks and three cycles required')
    args.out.mkdir(parents=True, exist_ok=False)
    kind = 'activation-digest-moe-seams' if args.mode in ('moe-seams', 'moe-inputs') else 'activation-digest'
    if args.mode == 'moe-capture':
        kind = 'activation-digest-moe-capture'
    before = identities(args.out, 'before', kind)
    idle_snapshot('http://dusty:8000')
    for node in NODES:
        if ssh(node, f'if test -e {TRIGGER}; then printf present; fi').strip():
            raise RuntimeError('Preexisting trigger: ' + node)
    arm = 'localize-' + str(time.time_ns())
    trigger = args.out / 'trigger.json'
    spec = {'arm': arm, 'min_tokens': 100, 'max_tokens': 1100,
        'post': [f'layers.{i}' for i in (0, 1, 2, 8, 13, 14, 19, 20, 39)]
                + ['layers.1.engram', 'layers.14.engram'], 'pre': []}
    if args.mode == 'layer0':
        spec.update(post=['layers.0', 'layers.0.attn', 'layers.0.ffn',
                          'layers.0.ffn.gate', 'layers.0.ffn.experts',
                          'layers.0.ffn.experts.routed_experts',
                          'layers.0.ffn.shared_experts',
                          'layers.0.ffn.shared_experts.gate_up_proj',
                          'layers.0.ffn.shared_experts.down_proj'],
                    pre=['layers.0.ffn'])
    elif args.mode == 'all-layers':
        spec.update(post=['layers.*.attn', 'layers.*.ffn'], pre=['layers.*.ffn'])
    elif args.mode == 'all-ffn-branches':
        spec.update(post=['layers.*.ffn', 'layers.*.ffn.gate', 'layers.*.ffn.shared_experts'],
                    pre=['layers.*.ffn'])
    elif args.mode in ('moe-seams', 'moe-inputs', 'moe-capture'):
        spec.update(post=['layers.39'], pre=[], moe=['layers.*.ffn.experts'])
        if args.mode == 'moe-inputs':
            spec.update(post=['layers.39', 'layers.*.ffn.gate'], pre=['layers.*.ffn'])
        if args.mode == 'moe-capture':
            spec['capture'] = {'rows': [254, 256],
                               'moe': ['layers.[0-9].ffn.experts', 'layers.1[0-9].ffn.experts']}
    trigger.write_text(json.dumps(spec) + '\n')
    (args.out / 'manifest.json').write_text(json.dumps({
        'arm': arm, 'blocks': args.blocks, 'cycles': args.cycles, 'mode': args.mode,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')

    def disarm():
        # Move only this run's own trigger; retain it as evidence instead of deleting it.
        for node in NODES:
            script = ('import json,pathlib; p=pathlib.Path(' + repr(TRIGGER) + '); '
                      'exists=p.exists(); '
                      'assert not exists or json.loads(p.read_text())["arm"]==' + repr(arm) + '; '
                      'p.rename(str(p)+' + repr('.' + arm + '.retained') + ') if exists else None')
            ssh(node, shlex.join(['python3', '-c', script]))

    def collect(block):
        directory = args.out / f'traces-{block:02d}'
        directory.mkdir()
        for node in NODES:
            (directory / node).mkdir()
        expected = (block // 2 + 1) * args.cycles * 5
        deadline = time.monotonic() + 60
        while True:
            for node in NODES:
                subprocess.run(['scp', f'{node}:{CACHE}/ds41-act-trace/*.jsonl',
                                str(directory / node) + '/'], check=True)
            try:
                records = [r for r in load(list(directory.glob('*/*.jsonl'))) if r.get('arm') == arm]
                complete = complete_records(records, expected)
            except json.JSONDecodeError:
                complete = False  # the asynchronous writer may be appending a line
            if complete:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('Missing/extra trace records after bounded writer drain')
            time.sleep(2)
        analyzer = analyze
        extra = {}
        status_lines = []
        if args.mode in ('moe-seams', 'moe-inputs', 'moe-capture'):
            stem = 'claude-moe-capture' if args.mode == 'moe-capture' else 'claude-moe-seams'
            analyzer_file = stem + '-act_analyze.py'
            expected_sha = json.loads((ROOT / (stem + '.lock.json')).read_text())['inputs'][analyzer_file]
            if hashlib.sha256((ROOT / analyzer_file).read_bytes()).hexdigest() != expected_sha:
                raise RuntimeError('Analyzer drift from diagnostic lock')
            module_spec = importlib.util.spec_from_file_location(
                'moe_seam_analysis', ROOT / analyzer_file)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            analyzer = module.analyze
            if args.mode == 'moe-capture':
                status_lines = module.load_status(list(directory.glob('*/*.jsonl')))
                extra['status_lines'] = status_lines
        report = analyzer(records, arm=arm, ranks=list(range(4)), **extra)
        report['scope'] = 'cumulative traced blocks on this boot and arm'
        report['expected_records_per_rank'] = expected
        (directory / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
        check_coverage(records, report, args.mode)
        if args.mode == 'moe-capture':
            def refresh_capture():
                nonlocal status_lines
                for node in NODES:
                    subprocess.run(['scp', f'{node}:{CACHE}/ds41-act-trace/*.jsonl',
                                    str(directory / node) + '/'], check=True)
                status_lines = module.load_status(list(directory.glob('*/*.jsonl')))
                return module.capture_summary(records, status_lines, arm)[1]

            def flush_capture(attempt):
                # Above the largest captured graph (32), below min_tokens (100):
                # execute Python _begin to drain the latch without a trace row.
                ids = json.loads((ROOT / 'receipts/determinism-corpus.json').read_text())['254']['tokenize']['tokens'][:64]
                if len(ids) != 64 or any(type(i) is not int for i in ids):
                    raise RuntimeError('Invalid short flush prompt')
                body = {'model': 'DeepSeek-V4.1-Flash', 'prompt': ids, 'temperature': 0,
                        'max_tokens': 1, 'cache_salt': uuid.uuid4().hex}
                receipt = directory / f'capture-flush-{attempt}.json'
                receipt.write_text(json.dumps({'request': body}, indent=2) + '\n')
                req = urllib.request.Request('http://dusty:8000/v1/completions',
                    json.dumps(body).encode(), {'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=3600) as response:
                    result = json.load(response)
                receipt.write_text(json.dumps({'request': body, 'response': result}, indent=2) + '\n')
                if result['usage']['prompt_tokens'] != 64 or len(result['choices']) != 1:
                    raise RuntimeError('Unexpected capture-flush response')

            if report['capture_incomplete']:
                settle_capture(refresh_capture, flush_capture)
                report = analyzer(records, arm=arm, ranks=list(range(4)), status_lines=status_lines)
                report['scope'] = 'cumulative traced blocks on this boot and arm'
                report['expected_records_per_rank'] = expected
                (directory / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
            downloaded = []
            for node, name in capture_downloads(status_lines, arm):
                target = directory / node / name
                subprocess.run(['scp', f'{node}:{CACHE}/ds41-act-trace/{name}', str(target)], check=True)
                downloaded.append({'node': node, 'file': str(target), 'bytes': target.stat().st_size,
                                   'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
            (directory / 'capture-downloads.json').write_text(json.dumps(downloaded, indent=2) + '\n')
            if report['capture_incomplete']:
                raise RuntimeError('Capture latched or failed without a complete save; inspect analysis.json')
        print('ACTIVATION-LOCALIZATION', json.dumps(report), flush=True)

    try:
        for block in range(args.blocks):
            traced = block % 2 == 0
            idle_snapshot('http://dusty:8000')
            if traced:
                for node in NODES:
                    subprocess.run(['scp', str(trigger), node + ':' + TRIGGER], check=True)
            else:
                disarm()
            # Keep complete request receipts and use the already-validated interleaved driver.
            command = [sys.executable, '-u', str(ROOT / 'probe_interleaved_repeatability.py'),
                       '--out', str(args.out / f'block-{block:02d}'), '--cycles', str(args.cycles),
                       '--lengths', '254,256,258,271,1023', '--seed', str(20260925 + block)]
            print('LOCALIZATION-BLOCK', block, 'traced' if traced else 'untraced', flush=True)
            with (args.out / f'block-{block:02d}.log').open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            if traced:
                collect(block)
    finally:
        disarm()
        if identities(args.out, 'after', kind) != before:
            raise RuntimeError('Diagnostic restarted or changed')
    print('ACTIVATION-LOCALIZATION-DIAGNOSTIC-COMPLETE', flush=True)


if __name__ == '__main__':
    main()
