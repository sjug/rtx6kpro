"""Decision-row capture driver for the approved diagnostic capture boots (DS4.1 TP4).

Ranks: dusty 0, toby 1, rusty 2, kirby 3. The GLM cluster is never contacted.

Modes (--mode):
  historical  the 80927-block boot with the B2 eight-plan regime as validity reference and the 8192
              chunk grid without a prefill threshold (the retained 041910Z configuration).
  matched     the user-approved matched-grid captures: 81389 blocks (the family whose eight plans were
              pinned in receipts/needle-sensitivity-20260926T045954Z/regime-pre.json), an explicit
              --chunk-rows of 8192 or 4096 equal to the boot's DS41_PREFILL_THRESHOLD, prepared
              capacity 8192 unchanged. Validity is equality with the pinned regime; the audit receives
              it as --regime-reference and reports the difference from B2 explicitly, never silently.

Phases (--phase):
  preflight  read-only checks of the running capture boot: image, labels, KV pin, idle endpoint,
             no consumed token. No arming, no request.
  capture    preflight; require two identical unarmed cold controls; arm the capture trigger
             on every rank host; replay the frozen historical 524288-token input once more;
             retain the complete response and the
             cache accounting; wait for the four per-rank capture receipts; snapshot the live
             selection files and decode the eight-plan regime; gather ranks 1..3 capture files to
             dusty over the switched 200G fabric (never the management NIC; originals stay on the
             nodes); assemble expected/observed for claude_decision_row_audit.py.
  audit      after the serving containers are stopped and retained: needle token spans and the
             offline audit on dusty in CPU-only containers of the capture image (no GPU device,
             no network), results copied back into the receipt directory.

Never: build, launch, restart, stop, cache or image changes, deletion on a node, or any change to
the launch contract, start driver or runtime manifest. Fail closed on every identity mismatch.
The audit's verdict is attribution evidence only; the original 524K failure stays a failure.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import urllib.request
import uuid

from cache_metrics import idle_snapshot, finish
from launch_contract import (NODES as CONTRACT_NODES, NCCL_GEOMETRY, NCCL_GEOMETRY_ENV, NCCL_TUNING_UNSET,
                             STANDARD_NCCL_ARM)
from prepare_decision_reference import exact_regime, signature, B2_IMAGE, PROMPT_SHA, build_reference
from runtime import kit_digest

ROOT = Path(__file__).resolve().parent
NODES = ('dusty', 'toby', 'rusty', 'kirby')                              # rank order
NAME = 'ds41-flash-karmic-main-tp4'
REMOTE = '/home/jugs/git/ds41-r38/karmic-main-20260924'
HOST_CACHE = '/home/jugs/.cache/vllm-jj-ds41-tp4'                        # mounted at /cache in the container
TRIGGER = HOST_CACHE + '/claude-decision-row.json'
OUT_DIR = HOST_CACHE + '/claude-decision-row'
SELECTION = 'preparation/6115b03c7a814701d5610b3e6aecf82e30fdbb46a122d70441228538b9bd1e5e.json'
B2_SELECTION_MANIFEST = ROOT / 'receipts/router-b2-tuning-20260926/manifest.json'
REFERENCE = ROOT / 'receipts/decision-row-b2-reference.json'
BUILD = ROOT / 'receipts/decision-row-build-receipt.json'
LOCK = ROOT / 'claude-decision-row.lock.json'
DIAGNOSTIC_KIND = 'decision-row-capture'
INPUT = ROOT.parents[1] / 'ds41/r38/receipts/20260916/admission-k7-1m-u80/needle-524288-input.json'
INPUT_SHA = '1b89c7f4d1e1047f6cf7ebe6000d6aff242981cbdbe8d8ce2a742ff61a7d6db6'
ENVELOPE = ROOT / 'receipts/mla-conformance-full-20260926T025241Z/result.envelope.json'
MODEL_DIR = ('/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/'
             'fb2764a5cf321eaa5070ca8f9e892818f477c16d')
BLOCKS, PROMPT_TOKENS = 80927, 524288
BLOCKS_LINE = f'b12x autotuning uses 32 temporary KV blocks before allocating {BLOCKS} serving blocks'
# Matched-grid captures (user approved 2026-09-26): the 81389-block family whose eight plans were
# pinned in the sensitivity series, an explicit chunk grid, the launcher's DS41_PREFILL_THRESHOLD
# equal to that grid, and the prepared capacity 8192 unchanged.
MATCHED_BLOCKS, CHUNKS, CAPACITY = 81389, (8192, 4096), 8192
PINNED_REGIME = ROOT / 'receipts/needle-sensitivity-20260926T045954Z/regime-pre.json'
PINNED_REGIME_SHA256 = 'd8793ca37b7bdc70e84b23dc9f63472a8b27c1fe5c413bd579bfda944227f2f4'
TOKEN = re.compile(r'^[a-z0-9-]{8,64}$')
AUDIT_FILES = ('claude_decision_row_audit.py', 'claude-pinned-b12x-compressed_reference.py',
               'claude-ds41-attention-config.json', 'claude_needle_spans.py')
FABRIC_DEV, FABRIC_SELF = 'enp1s0f0np0', CONTRACT_NODES['dusty'][1]


# ---- pure helpers (CPU-tested) ------------------------------------------------------------------

def token_for(stamp, chunk_rows=None):
    """Historical tokens keep their spelling; matched-grid tokens name the chunk so captures never collide."""
    token = 'decision-row-' + ('' if chunk_rows is None else f'm{chunk_rows}-') + stamp.lower()
    if not TOKEN.match(token):
        raise ValueError('capture token outside the helper grammar')
    return token


def trigger_payload(token, chunk_rows=8192):
    if chunk_rows not in CHUNKS:
        raise ValueError('chunk_rows must be 8192 or 4096')
    return {'token': token, 'prompt_tokens': PROMPT_TOKENS, 'chunk_rows': chunk_rows}


def profile_for(mode, chunk_rows=None, geometry=None, nccl_arm=None):
    """The boot this driver expects: historical (80927 blocks, B2 regime, 8192 grid, no threshold) or
    matched (81389 blocks, pinned 045954Z regime, explicit grid equal to the launcher's threshold).
    `geometry` (user-approved 2026-09-26, window-capture kind only) is the NCCL geometry diagnostic
    tree-simple-1ch: the launch contract's four explicit NCCL overrides on top of the matched profile.
    It changes the reduction order of every TP all-reduce, so its outputs are a separate numerical
    arm: same-boot control equality still gates, cross-boot instrumentation baselines do not apply,
    and cross-grid invariance is a prediction for one boot's tree, not a cross-boot guarantee."""
    if geometry is not None and (geometry != NCCL_GEOMETRY or mode != 'matched'):
        raise ValueError('NCCL geometry is tree-simple-1ch on the matched profiles only')
    if nccl_arm is not None and (nccl_arm != STANDARD_NCCL_ARM or mode != 'matched' or geometry is not None):
        raise ValueError('The standard NCCL arm is standard-upstream on the matched profiles without tree geometry')
    if mode == 'historical':
        if chunk_rows not in (None, 8192):
            raise ValueError('historical mode is the 8192 grid only')
        return {'mode': mode, 'blocks': BLOCKS, 'chunk_rows': 8192, 'threshold': None,
                'regime_source': 'b2-reference', 'tag': '', 'geometry': None, 'nccl_arm': None}
    if mode == 'matched':
        if chunk_rows not in CHUNKS:
            raise ValueError('matched mode needs an explicit chunk grid of 8192 or 4096')
        tag = (f'matched{chunk_rows}-' + (f'geometry-{geometry}-' if geometry else '')
               + (f'nccl-{nccl_arm}-' if nccl_arm else ''))
        return {'mode': mode, 'blocks': MATCHED_BLOCKS, 'chunk_rows': chunk_rows, 'threshold': chunk_rows,
                'regime_source': 'pinned-045954Z', 'tag': tag, 'geometry': geometry, 'nccl_arm': nccl_arm}
    raise ValueError(f'unknown mode {mode!r}')


def check_geometry_env(env, profile):
    """The NCCL geometry variables must be exactly present on a geometry boot and absent otherwise."""
    problems = []
    expected = dict(NCCL_GEOMETRY_ENV, DS41_NCCL_GEOMETRY=profile['geometry']) if profile.get('geometry') else {}
    for name in sorted(set(NCCL_GEOMETRY_ENV) | set(NCCL_TUNING_UNSET) | {'DS41_NCCL_GEOMETRY'}):
        if name in expected:
            if env.get(name) != expected[name]:
                problems.append(f'NCCL geometry variable {name} is not {expected[name]!r}')
        elif name in env:
            problems.append(f'unexpected NCCL tuning variable {name} on this boot')
    if env.get('DS41_NCCL_ARM') != profile.get('nccl_arm'):
        problems.append(f'NCCL arm selector is {env.get("DS41_NCCL_ARM")!r}, expected {profile.get("nccl_arm")!r}')
    return problems


NCCL_WARNING_PATTERN = re.compile(r'NCCL WARN.*(NCCL_ALGO|NCCL_PROTO|NCHANNELS|[Ii]nvalid|not supported|ignor)')


def nccl_geometry_warnings(log_text):
    """NCCL only warns on unsupported ALGO/PROTO/channel settings; such lines fail the geometry preflight."""
    # Recorded unchanged in the successful non-geometry boot: this is a NIC
    # speed-query capability warning, not rejection of NCCL_PROTO or NCCL_ALGO.
    port_query = 'NCCL WARN Call to ibv_query_port_speed failed with error Protocol not supported errno 93'
    return [line for line in log_text.splitlines() if NCCL_WARNING_PATTERN.search(line)
            and not line.rstrip().endswith(port_query)]


def blocks_line(blocks):
    return f'b12x autotuning uses 32 temporary KV blocks before allocating {blocks} serving blocks'


def pinned_regime(raw, reference):
    """The pinned four-rank snapshot: exact sha256, all ranks equal, same plan keys as B2; returns
    (regime, per-plan differences from the B2 reference)."""
    if hashlib.sha256(raw).hexdigest() != PINNED_REGIME_SHA256:
        raise RuntimeError('pinned regime snapshot changed')
    ranks = json.loads(raw)
    if set(ranks) != set(NODES) or any(value != ranks[NODES[0]] for value in ranks.values()):
        raise RuntimeError('pinned regime snapshot must hold four agreeing ranks')
    regime = ranks[NODES[0]]
    if set(regime) != set(reference['regime']):
        raise RuntimeError('pinned regime snapshot does not cover the eight B2 plans')
    differences = {name: {'b2': reference['regime'][name], 'pinned': value}
                   for name, value in regime.items() if value != reference['regime'][name]}
    return regime, differences


def check_boot_profile(rendered, node, profile):
    """The rendered launch profile runtime.py logs on every rank, against the expected boot."""
    argv, env, problems = rendered.get('model') or [], rendered.get('env') or {}, []
    if rendered.get('node') != node:
        problems.append(f'rendered profile is for {rendered.get("node")!r}, not {node}')

    def value(flag):
        return argv[argv.index(flag) + 1] if argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv) else None
    if value('--num-gpu-blocks-override') != str(profile['blocks']):
        problems.append(f'effective argv --num-gpu-blocks-override is not {profile["blocks"]}')
    if value('--max-num-batched-tokens') != str(CAPACITY):
        problems.append(f'effective argv --max-num-batched-tokens is not {CAPACITY}')
    threshold = value('--long-prefill-token-threshold')
    if profile['threshold'] is None:
        if threshold is not None or 'DS41_PREFILL_THRESHOLD' in env:
            problems.append('historical boot carries a prefill threshold')
    elif threshold != str(profile['threshold']) or env.get('DS41_PREFILL_THRESHOLD') != str(profile['threshold']):
        problems.append(f'effective prefill threshold is not {profile["threshold"]}')
    if env.get('DS41_DECISION_ROW_BLOCKS') != str(profile['blocks']) or 'DS41_CHUNKING_BLOCKS' in env:
        problems.append('rendered environment does not carry the expected capture pin alone')
    problems += check_geometry_env(env, profile)
    return problems


def check_api_args(args, profile):
    """The head API server's non-default args (scalars read by key) against the expected boot."""
    expected = {'num_gpu_blocks_override': profile['blocks'], 'max_num_batched_tokens': CAPACITY,
                'long_prefill_token_threshold': profile['threshold'], 'max_num_seqs': 4, 'enable_chunked_prefill': True}
    return [f'API server {key}={args.get(key)!r}, expected {value!r}' for key, value in expected.items()
            if args.get(key) != value]


def env_map(entries):
    return dict(entry.split('=', 1) for entry in entries if '=' in entry)


def check_container(node, container, image, build, lock_sha, profile=None):
    """Problems with one rank's running container against the capture build; identity on success.
    `profile` (profile_for) fixes the KV pin and the prefill threshold; default historical."""
    profile = profile or profile_for('historical')
    problems = []
    rank = NODES.index(node)
    if not container['State'].get('Running'):
        problems.append('not running')
    if container['Image'].removeprefix('sha256:') != build['image_id']:
        problems.append('container image is not the capture build')
    if image['Id'].removeprefix('sha256:') != build['image_id']:
        problems.append('image identity differs from the build receipt')
    labels = image['Config']['Labels']
    if labels.get('local-inference.ds41.diagnostic.kind') != DIAGNOSTIC_KIND:
        problems.append('image is not the ' + DIAGNOSTIC_KIND.removesuffix('-capture') + ' capture diagnostic')
    if labels.get('local-inference.ds41.diagnostic.lock.sha256') != lock_sha or build['lock_sha256'] != lock_sha:
        problems.append('capture lock differs from the build')
    env = env_map(container['Config']['Env'])
    if env.get('DS41_DECISION_ROW_BLOCKS') != str(profile['blocks']):
        problems.append(f'KV pin DS41_DECISION_ROW_BLOCKS={profile["blocks"]} absent from the container')
    if 'DS41_CHUNKING_BLOCKS' in env:
        problems.append('router chunking control present on a capture boot')
    if profile['threshold'] is None:
        if 'DS41_PREFILL_THRESHOLD' in env:
            problems.append('prefill threshold present on a historical capture boot')
    elif env.get('DS41_PREFILL_THRESHOLD') != str(profile['threshold']):
        problems.append(f'DS41_PREFILL_THRESHOLD is not the matched grid {profile["threshold"]}')
    if env.get('DS41_NODE') != node:
        problems.append('DS41_NODE differs from the rank host')
    for name, value in {'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '1', 'B12X_DENSE_SPLITK_TURBO': '0'}.items():
        if env.get(name) != value:
            problems.append('repaired runtime environment differs: ' + name)
    if 'VLLM_DS41_L2_PREFETCH' in env:
        problems.append('unexpected prefetch override')
    problems += check_geometry_env(env, profile)
    if profile.get('geometry') and DIAGNOSTIC_KIND not in ('window-capture', 'indexer-capture', 'precision-capture'):
        problems.append('NCCL geometry diagnostic is approved for the window-capture kind only')
    if profile.get('nccl_arm') and DIAGNOSTIC_KIND != 'precision-capture':
        problems.append('the standard NCCL arm is approved for the precision-capture kind only')
    kit = container['Config']['Labels'].get('local-inference.ds41.kit.sha256')
    if not kit or env.get('DS41_KIT_SHA256') != kit:
        problems.append('kit digest label and environment differ')
    cache_dir = env.get('B12X_COMPILE_CACHE_DIR')
    if not cache_dir or not cache_dir.startswith('/cache/'):
        problems.append('B12X selection namespace is not under /cache')
    identity = {'rank': rank, 'kit_sha256': kit, 'cache_dir': cache_dir,
                'source_trees': {'vllm': labels.get('vllm.source-tree'), 'b12x': labels.get('b12x.source-tree')}}
    return problems, identity


def blocks_pinned(log_text, blocks=BLOCKS):
    return blocks_line(blocks) in log_text


def selection_host_path(cache_dir):
    return HOST_CACHE + cache_dir.removeprefix('/cache') + '/' + SELECTION


def host_path(container_path):
    if not container_path.startswith('/cache/claude-decision-row/') or '..' in Path(container_path).parts:
        raise ValueError('capture path outside the cache mount')
    return HOST_CACHE + container_path.removeprefix('/cache')


def request_body(spec, salt):
    """The frozen gate request (qualify_original_needle.py) with a fresh salt: cold by construction."""
    return {'model': 'DeepSeek-V4.1-Flash', 'messages': spec['messages'],
            'chat_template_kwargs': spec['chat_template_kwargs'], 'temperature': 0,
            'max_tokens': 64, 'logprobs': True, 'top_logprobs': 20, 'cache_salt': salt}


def response_row(result, spec, reference):
    choice = result['choices'][0]
    observed = signature(result)
    return {'prompt_tokens': result['usage']['prompt_tokens'],
            'content': choice['message'].get('content'), 'finish_reason': choice['finish_reason'],
            'correct': choice['finish_reason'] == 'stop' and (choice['message'].get('content') or '').strip() == spec['expected'],
            'response_signature': observed, 'signature_equals_b2': observed == reference['response_signature']}


def parse_consumed(text, token, rank):
    record = json.loads(text)
    expected_name = f'-rank{rank}-{token}.pt'
    if (not record['file'].endswith(expected_name)
            or not re.fullmatch('[0-9a-f]{64}', record['sha256'])):
        raise ValueError(f'rank {rank}: consumed receipt names another capture')
    return {'container_file': record['file'], 'host_file': host_path(record['file']),
            'sha256': record['sha256'], 'problems': record['problems']}


def observed_identity(build, identities, node_regimes, response_signature, chunk_rows=None):
    kits = {i['kit_sha256'] for i in identities.values()}
    trees = {json.dumps(i['source_trees'], sort_keys=True) for i in identities.values()}
    regs = {json.dumps(r, sort_keys=True) for r in node_regimes.values()}
    if len(kits) != 1 or len(trees) != 1:
        raise RuntimeError('kit digest or source trees differ across ranks')
    if len(regs) != 1:
        raise RuntimeError('decoded eight-plan regimes differ across ranks')
    identity = {'image_id': build['image_id'], 'kit_sha256': kits.pop(), 'source_trees': json.loads(trees.pop()),
                'regime': json.loads(regs.pop()), 'response_signature': response_signature}
    if chunk_rows is not None:
        identity['chunk_rows'] = chunk_rows
    return identity


def expected_identity(reference):
    return {key: reference[key] for key in ('image_id', 'kit_sha256', 'source_trees', 'regime', 'response_signature')}


def matched_control(rows, build, identities, node_regimes, chunk_rows=None):
    if len(rows) != 2 or len({row['response_signature'] for row in rows}) != 1:
        raise RuntimeError('Two unarmed cold controls must match before arming')
    return observed_identity(build, identities, node_regimes, rows[0]['response_signature'], chunk_rows)


def regime_reference_record(profile, regime, differences):
    """What the audit receives as --regime-reference in matched mode: the pinned regime, its provenance
    and its per-plan differences from B2, so the audit can say which regime it validated against."""
    return {'mode': 'matched-pinned-regime', 'source': str(PINNED_REGIME.relative_to(ROOT)),
            'sha256': PINNED_REGIME_SHA256, 'blocks': profile['blocks'], 'chunk_rows': profile['chunk_rows'],
            'regime': regime, 'regime_difference_from_b2': differences,
            'scope': ('the eight plans of the 81389-block family pinned in the sensitivity series; validity is '
                      'equality with this regime, B2 kernel-plan equivalence is reported, not claimed')}


def gather_script(remote_dir, captures):
    """bash for dusty: pull ranks 1..3 over the fabric addresses (route checked), verify every sha256."""
    lines = ['set -euo pipefail', f'mkdir -p {shlex.quote(remote_dir + "/captures")}']
    for node, record in captures.items():
        source, digest = record['host_file'], record['sha256']
        dest = remote_dir + '/captures/' + Path(source).name
        if node == 'dusty':
            lines.append(f'cp --reflink=auto {shlex.quote(source)} {shlex.quote(dest)}')
        else:
            address = CONTRACT_NODES[node][1]
            lines.append(f'ip -j route get {address} | python3 -c "import json,sys; r=json.load(sys.stdin)[0]; '
                         f'assert r.get(\'dev\')==\'{FABRIC_DEV}\' and r.get(\'prefsrc\')==\'{FABRIC_SELF}\', r"')
            lines.append('rsync --whole-file -e ' + shlex.quote('ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com')
                         + f' {shlex.quote(address + ":" + source)} {shlex.quote(dest)}')
        lines.append(f'echo "{digest}  {dest}" | sha256sum -c -')
        lines.append(f'echo GATHERED {node} {dest}')
    return '\n'.join(lines) + '\n'


def cpu_container(image, gate_dir, argv, *, memory, hf_mount=None):
    """CPU-only run of the capture image: no GPU device, no network, bounded memory."""
    command = ['podman', 'run', '--rm', '--pull=never', '--network=none', '-e', 'PYTHONUNBUFFERED=1',
               '-v', gate_dir + ':/gate:rw']
    if hf_mount:
        command += ['-v', hf_mount + ':/root/.cache/huggingface:ro',
                    '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1']
    if memory != 'none':
        command += ['--memory=' + memory]
    return command + ['--entrypoint', '/opt/venv/bin/python', image, '-u', *argv]


def audit_argv(capture_names, *, spans, control=False, regime_reference=False):
    argv = ['/gate/claude_decision_row_audit.py', *['/gate/captures/' + n for n in capture_names],
            '--expected', '/gate/expected.json', '--observed', '/gate/observed.json',
            '--envelope', '/gate/envelope.json', '--json', '/gate/audit.json']
    return (argv + (['--spans', '/gate/spans.json'] if spans else [])
            + (['--control', '/gate/control.json'] if control else [])
            + (['--regime-reference', '/gate/regime-reference.json'] if regime_reference else []))


# ---- remote plumbing (thin, untested) -----------------------------------------------------------

def ssh(node, command, timeout=120):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, command], text=True, timeout=timeout)


def ssh_json(node, command):
    return json.loads(ssh(node, command))


def stamp_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def local_inputs(profile=None):
    """Frozen inputs plus the regime this run must match: B2's own regime (historical) or the pinned
    045954Z snapshot (matched); the B2 reference is always carried for the audit's B2 comparisons."""
    profile = profile or profile_for('historical')
    reference = json.loads(REFERENCE.read_text())
    if reference != build_reference():
        raise RuntimeError('B2 reference no longer matches its retained source receipts')
    if reference['image_id'] != B2_IMAGE or reference['proposed_capture_blocks'] != BLOCKS:
        raise RuntimeError('B2 decision reference is not for this image or capacity')
    if profile['mode'] == 'matched':
        regime, differences = pinned_regime(PINNED_REGIME.read_bytes(), reference)
        profile['regime'], profile['regime_difference_from_b2'] = regime, differences
    else:
        profile['regime'], profile['regime_difference_from_b2'] = reference['regime'], {}
    raw = INPUT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise RuntimeError('Historical input file identity changed')
    spec = json.loads(raw)
    if hashlib.sha256(spec['messages'][0]['content'].encode()).hexdigest() != PROMPT_SHA:
        raise RuntimeError('Historical prompt identity changed')
    build = json.loads(BUILD.read_text())
    lock_sha = hashlib.sha256(LOCK.read_bytes()).hexdigest()
    if (Path(build['directory']) / 'BUILD-OK').read_text().strip() != build['image_id']:
        raise RuntimeError('Capture image installation gate missing')
    if build['lock_sha256'] != lock_sha:
        raise RuntimeError('Capture lock changed since the build')
    manifest = json.loads(B2_SELECTION_MANIFEST.read_text())
    return reference, spec, build, lock_sha, manifest['cache_path']


def check_precision_marker(log, rank):
    lines = [line for line in log.splitlines() if 'DS41-PRECISION-APPLIED' in line]
    expected = f'DS41-PRECISION-APPLIED rank={rank} allow_bf16_reduced_precision_reduction=False '
    return [] if len(lines) == 1 and expected in lines[0] else ['missing, duplicate or incorrect worker precision marker']


def preflight(out, build, lock_sha, b2_cache_path, token, profile=None):
    profile = profile or profile_for('historical')
    identities = {}
    digest = kit_digest()
    lock = json.loads(LOCK.read_text())
    from claude_probe_chunking import parse_profile, parse_non_default_args, ARGS_PREFIX, PROFILE_PREFIX
    for node in NODES:
        running = ssh(node, 'podman ps -q').split()
        container = ssh_json(node, 'podman inspect ' + NAME)[0]
        image = ssh_json(node, 'podman image inspect ' + build['image_id'])[0]
        (out / f'{node}-container.json').write_text(json.dumps(container, indent=2) + '\n')
        (out / f'{node}-image.json').write_text(json.dumps(image, indent=2) + '\n')
        (out / f'{node}-pre-arm-meminfo.txt').write_text(ssh(node, 'cat /proc/meminfo'))
        problems, identity = check_container(node, container, image, build, lock_sha, profile)
        if identity['kit_sha256'] != digest:
            problems.append('container does not run the current reviewed runtime kit')
        if identity['source_trees'] != lock['trees']:
            problems.append('source-tree labels differ from the capture lock')
        if len(running) != 1 or not container['Id'].startswith(running[0]):
            problems.append('another container runs on the rank host')
        if identity['cache_dir'] and identity['cache_dir'] + '/' + SELECTION != b2_cache_path:
            problems.append('selection namespace differs from the B2 tuning snapshot')
        line = blocks_line(profile['blocks'])
        log = ssh(node, f'podman logs {NAME} 2>&1 | grep -F -e {shlex.quote(line)} -e {shlex.quote(ARGS_PREFIX)}'
                        f' -e {shlex.quote(PROFILE_PREFIX)} || true')
        (out / f'{node}-boot-args.log').write_text(log)
        if line not in log:
            problems.append(f'boot log does not report the {profile["blocks"]} serving-block allocation')
        try:
            problems += check_boot_profile(parse_profile(log), node, profile)
        except ValueError as error:
            problems.append(str(error))
        if node == NODES[0] or ARGS_PREFIX in log:
            try:
                problems += check_api_args(parse_non_default_args(log), profile)
            except ValueError as error:
                problems.append(str(error))
        if profile.get('geometry'):
            warnings = nccl_geometry_warnings(ssh(node, f'podman logs {NAME} 2>&1 | grep -F "NCCL WARN" || true'))
            (out / f'{node}-nccl-warnings.log').write_text(''.join(w + '\n' for w in warnings))
            if warnings:
                problems.append('NCCL warned about the requested geometry; unsupported settings are not accepted')
        if DIAGNOSTIC_KIND == 'precision-capture':
            marker = ssh(node, f'podman logs {NAME} 2>&1 | grep -F DS41-PRECISION-APPLIED || true')
            (out / f'{node}-precision-marker.log').write_text(marker)
            problems += check_precision_marker(marker, NODES.index(node))
        consumed = ssh(node, f'ls {shlex.quote(OUT_DIR)} 2>/dev/null | grep -F {shlex.quote(token)} || true').strip()
        if consumed:
            problems.append('capture token already consumed')
        trigger_mtime = ssh(node, f'if test -e {shlex.quote(TRIGGER)}; then stat -c %Y {shlex.quote(TRIGGER)}; fi').strip()
        started = datetime.datetime.fromisoformat(container['State']['StartedAt'].replace('Z', '+00:00')).timestamp()
        if trigger_mtime and float(trigger_mtime) >= started:
            problems.append('trigger was written after this boot; unarmed controls are not assured')
        if problems:
            raise RuntimeError(f'{node}: ' + '; '.join(problems))
        identities[node] = identity
        print('PREFLIGHT-OK', node, flush=True)
    return identities


def arm(token, armed_log, chunk_rows=8192):
    code = ('import json,os,sys,time; p=sys.argv[1]; t=p+".tmp"; '
            'open(t,"w").write(json.dumps(json.loads(sys.argv[2]))+"\\n"); os.replace(t,p); '
            'print(json.dumps({"path":p,"mtime":os.stat(p).st_mtime,"unix":time.time()}))')
    payload = json.dumps(trigger_payload(token, chunk_rows))
    for node in NODES:
        line = ssh(node, shlex.join(['python3', '-c', code, TRIGGER, payload])).strip()
        armed_log[node] = json.loads(line)
        print('ARMED', node, line, flush=True)


def send(base_url, body, out):
    before = idle_snapshot(base_url)
    (out / 'request.json').write_text(json.dumps(body, indent=1) + '\n')
    (out / 'before.json').write_text(json.dumps(before, indent=2) + '\n')
    start = time.monotonic()
    request = urllib.request.Request(base_url + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=3600) as response:
        result = json.load(response)
    elapsed = time.monotonic() - start
    (out / 'trial.json').write_text(json.dumps({'request': body, 'response': result, 'elapsed_s': elapsed}, indent=1) + '\n')
    if result['usage']['prompt_tokens'] != PROMPT_TOKENS:
        raise RuntimeError('Historical token count changed')
    cached = finish(base_url, before, PROMPT_TOKENS, out / 'trial-cache.json')
    if cached != 0:
        raise RuntimeError('Historical replay was not cold; the capture is not comparable')
    return result, elapsed


def wait_captures(token, timeout_s, out):
    captures, deadline = {}, time.monotonic() + timeout_s
    while len(captures) < len(NODES):
        for rank, node in enumerate(NODES):
            if node in captures:
                continue
            lines = ssh(node, f'podman logs {NAME} 2>&1 | grep -F CLAUDE-DECISION-ROW || true')
            (out / f'{node}-helper.log').write_text(lines)
            if any(f'{kind} token={token} ' in lines for kind in ('aborted', 'incomplete decision forward', 'error')):
                raise RuntimeError(f'{node}: capture helper aborted; no receipt will arrive')
            path = f'{OUT_DIR}/{token}-rank{rank}.consumed'
            text = ssh(node, f'cat {shlex.quote(path)} 2>/dev/null || true')
            if text.strip():
                captures[node] = parse_consumed(text, token, rank)
                (out / 'partial-captures.json').write_text(json.dumps(captures, indent=2) + '\n')
                print('CAPTURED', node, json.dumps(captures[node]), flush=True)
        if len(captures) < len(NODES):
            if time.monotonic() > deadline:
                raise RuntimeError('capture receipts missing on: ' + ', '.join(n for n in NODES if n not in captures))
            time.sleep(5)
    for node in NODES:
        lines = ssh(node, f'podman logs {NAME} 2>&1 | grep -F CLAUDE-DECISION-ROW || true')
        (out / f'{node}-helper.log').write_text(lines)
    return captures


def snapshot_selections(identities, out, profile=None):
    profile = profile or profile_for('historical')
    node_regimes = {}
    for node, identity in identities.items():
        text = ssh(node, 'cat ' + shlex.quote(selection_host_path(identity['cache_dir'])), timeout=300)
        (out / f'{node}-selection.json').write_text(text)
        payload = json.loads(text)
        node_regimes[node] = exact_regime(payload, profile['blocks'])
    (out / 'regime.json').write_text(json.dumps(node_regimes, indent=2, sort_keys=True) + '\n')
    return node_regimes


def phase_capture(a, preflight_only=False):
    profile = profile_for(a.mode, a.chunk_rows, getattr(a, 'nccl_geometry', None), getattr(a, 'nccl_arm', None))
    reference, spec, build, lock_sha, b2_cache_path = local_inputs(profile)
    stamp = stamp_now()
    token = token_for(stamp, None if profile['mode'] == 'historical' else profile['chunk_rows'])
    out = ROOT / 'receipts' / f'decision-row-{profile["tag"]}{"preflight" if preflight_only else "capture"}-{stamp}'
    out.mkdir(exist_ok=False)
    now = datetime.datetime.now(datetime.timezone.utc)
    since = now.strftime('%Y-%m-%d %H:%M:%S UTC')
    podman_since = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    observers = []
    try:
        if not preflight_only:
            for node in NODES:
                stream = (out / f'{node}-telemetry.log').open('x')
                process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', node,
                    f'echo OBSERVER_PID=$$; exec bash {REMOTE}/observe.sh'],
                    stdout=stream, stderr=subprocess.STDOUT)
                observers.append((node, process, stream))
        execute_capture(a, preflight_only, reference, spec, build, lock_sha, b2_cache_path, token, out, profile)
    finally:
        for node, process, stream in observers:
            try:
                stream.flush()
                lines = (out / f'{node}-telemetry.log').read_text().splitlines()
                first = lines[0] if lines else ''
                if first.startswith('OBSERVER_PID=') and first.split('=', 1)[1].isdigit():
                    pid = first.split('=', 1)[1]
                    ssh(node, f'if ps -p {pid} -o args= | grep -Fq "{REMOTE}/observe.sh"; '
                              f'then kill -TERM {pid}; fi')
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


def execute_capture(a, preflight_only, reference, spec, build, lock_sha, b2_cache_path, token, out, profile=None):
    profile = profile or profile_for('historical')
    print('RECEIPTS', out, 'mode', profile['mode'], 'chunk_rows', profile['chunk_rows'], 'blocks', profile['blocks'], flush=True)
    (out / 'reference.json').write_bytes(REFERENCE.read_bytes())
    (out / 'expected.json').write_text(json.dumps(expected_identity(reference), indent=2, sort_keys=True) + '\n')
    (out / 'profile.json').write_text(json.dumps(profile, indent=2, sort_keys=True) + '\n')
    if profile['mode'] == 'matched':
        record = regime_reference_record(profile, profile['regime'], profile['regime_difference_from_b2'])
        (out / 'regime-reference.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    before_controls = out / 'pre-control'
    before_controls.mkdir()
    identities = preflight(before_controls, build, lock_sha, b2_cache_path, token, profile)
    (out / 'identities.json').write_text(json.dumps(identities, indent=2, sort_keys=True) + '\n')
    idle_snapshot(a.base_url)
    node_regimes = snapshot_selections(identities, out, profile)
    if any(regime != profile['regime'] for regime in node_regimes.values()):
        raise RuntimeError(f'Pre-request kernel selections differ from the {profile["regime_source"]} regime; do not arm')
    if preflight_only:
        print('DECISION-PREFLIGHT-COMPLETE', out, flush=True)
        return
    controls = []
    for index in range(2):
        trial_dir = out / f'unarmed-{index + 1}'
        trial_dir.mkdir()
        result, elapsed = send(a.base_url, request_body(spec, uuid.uuid4().hex), trial_dir)
        row = response_row(result, spec, reference)
        row['elapsed_s'] = elapsed
        controls.append(row)
        (trial_dir / 'response-row.json').write_text(json.dumps(row, indent=2) + '\n')
        print('UNARMED-CONTROL', index + 1, json.dumps(row), flush=True)
    # Recheck identities and selections after the controls, before any trigger is written.
    identities = preflight(out, build, lock_sha, b2_cache_path, token, profile)
    node_regimes = snapshot_selections(identities, out, profile)
    if any(regime != profile['regime'] for regime in node_regimes.values()):
        raise RuntimeError('Controls changed kernel selections; do not arm')
    control = matched_control(controls, build, identities, node_regimes, profile['chunk_rows'])
    (out / 'control.json').write_text(json.dumps(control, indent=2, sort_keys=True) + '\n')
    armed = {}
    arm(token, armed, profile['chunk_rows'])
    (out / 'armed.json').write_text(json.dumps({'token': token, 'nodes': armed}, indent=2) + '\n')
    result, elapsed = send(a.base_url, request_body(spec, uuid.uuid4().hex), out)
    row = response_row(result, spec, reference)
    row['elapsed_s'] = elapsed
    (out / 'response-row.json').write_text(json.dumps(row, indent=2) + '\n')
    print('RESPONSE', json.dumps(row), flush=True)
    captures = wait_captures(token, a.capture_timeout, out)
    (out / 'captures.json').write_text(json.dumps(captures, indent=2, sort_keys=True) + '\n')
    node_regimes = snapshot_selections(identities, out, profile)
    observed = observed_identity(build, identities, node_regimes, row['response_signature'], profile['chunk_rows'])
    (out / 'observed.json').write_text(json.dumps(observed, indent=2, sort_keys=True) + '\n')
    remote_dir = REMOTE + '/receipts/' + out.name
    script = gather_script(remote_dir, captures)
    (out / 'gather.sh').write_text(script)
    gathered = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'dusty', 'bash -s'],
                                       input=script, text=True, timeout=1800)
    (out / 'gather.log').write_text(gathered)
    if gathered.count('GATHERED ') != len(NODES):
        raise RuntimeError('gather did not confirm all four capture files on dusty')
    summary = {'token': token, 'remote_dir': remote_dir, 'image_id': build['image_id'],
               'captures': {n: Path(c['host_file']).name for n, c in captures.items()},
               'capture_problems': {n: c['problems'] for n, c in captures.items()},
               'mode': profile['mode'], 'chunk_rows': profile['chunk_rows'], 'blocks': profile['blocks'],
               'regime_source': profile['regime_source'], 'nccl_geometry': profile.get('geometry'),
               'nccl_arm': profile.get('nccl_arm'),
               'regime_equals_reference': observed['regime'] == profile['regime'],
               'regime_equals_b2': observed['regime'] == reference['regime'],
               'regime_difference_from_b2': profile['regime_difference_from_b2'],
               'signature_equals_b2': row['signature_equals_b2'], 'correct': row['correct'],
               'signature_equals_control': row['response_signature'] == control['response_signature'],
               'control_equals_b2': control['response_signature'] == reference['response_signature'],
               'scope': (f'diagnostic capture of the frozen 524K input at {profile["blocks"]} blocks on the '
                         f'{profile["chunk_rows"]}-row chunk grid ({profile["mode"]} mode, {profile["regime_source"]} '
                         'regime); attribution evidence only, no qualification claim; audit pending')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print('DECISION-CAPTURE-COMPLETE', json.dumps(summary), flush=True)
    if (not summary['regime_equals_reference'] or not summary['signature_equals_control']
            or any(summary['capture_problems'].values())):
        raise RuntimeError('Capture retained but differs from its same-boot control or is invalid: inspect summary.json')


def phase_audit(a):
    out = a.receipt
    attempt = out / ('audit-attempt-' + stamp_now())
    attempt.mkdir(exist_ok=False)
    summary = json.loads((out / 'summary.json').read_text())
    if not a.while_serving and ssh('dusty', 'podman ps -q').strip():
        raise RuntimeError('dusty still runs a container; stop and retain the boot first, or pass --while-serving')
    image = summary['image_id']
    info = ssh_json('dusty', 'podman image inspect ' + image)[0]
    if info['Id'].removeprefix('sha256:') != image:
        raise RuntimeError('capture image missing on dusty')
    remote_dir = summary['remote_dir']
    kit = {n: ROOT / n for n in AUDIT_FILES}
    kit.update({'expected.json': out / 'expected.json', 'observed.json': out / 'observed.json',
                'envelope.json': ENVELOPE, 'needle-input.json': INPUT})
    has_control = (out / 'control.json').is_file()
    if has_control:
        kit['control.json'] = out / 'control.json'
    has_regime_reference = (out / 'regime-reference.json').is_file()
    if has_regime_reference:
        record = json.loads((out / 'regime-reference.json').read_text())
        if record.get('sha256') != PINNED_REGIME_SHA256 or record.get('mode') != 'matched-pinned-regime':
            raise RuntimeError('regime-reference.json is not the pinned matched regime record')
        kit['regime-reference.json'] = out / 'regime-reference.json'
    files = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in kit.items()}
    for name, path in kit.items():
        subprocess.run(['scp', str(path), f'dusty:{remote_dir}/{name}'], check=True)
        actual = ssh('dusty', 'sha256sum ' + shlex.quote(remote_dir + '/' + name)).split()[0]
        if actual != files[name]:
            raise RuntimeError('Audit input transfer differs: ' + name)
    names = [summary['captures'][n] for n in NODES]
    ssh('dusty', ' && '.join(f'test -s {shlex.quote(remote_dir + "/captures/" + n)}' for n in names))
    runs = {}
    if not a.no_spans:
        argv = ['/gate/claude_needle_spans.py', '--input', '/gate/needle-input.json',
                '--model-dir', MODEL_DIR, '--out', '/gate/spans.json']
        runs['spans'] = cpu_container(image, remote_dir, argv, memory=a.audit_memory,
                                      hf_mount='/home/jugs/.cache/huggingface')
    runs['audit'] = cpu_container(image, remote_dir, audit_argv(names, spans=not a.no_spans, control=has_control,
                                                                regime_reference=has_regime_reference), memory=a.audit_memory)
    (attempt / 'audit-invocation.json').write_text(json.dumps({'image_id': image, 'files': files, 'runs': runs}, indent=2) + '\n')
    for label, command in runs.items():
        with (attempt / f'{label}.log').open('x') as log:
            process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
        (attempt / f'{label}.exit-code').write_text(str(code) + '\n')
        # Deviation/invalid-capture are useful results, not grounds to lose the receipt.
        remote_result = remote_dir + '/' + label + '.json'
        available = ssh('dusty', f'if test -s {shlex.quote(remote_result)}; then echo present; fi').strip()
        if available == 'present':
            result_path = attempt / (label + '.json')
            subprocess.run(['scp', 'dusty:' + remote_result, str(result_path)], check=True)
            (out / (label + '.json')).write_bytes(result_path.read_bytes())
            if label == 'audit':
                report = json.loads((out / 'audit.json').read_text())
                print('AUDIT-VERDICT', report['verdict'], flush=True)
        if code:
            raise SystemExit(f'{label} failed with {code}; inspect {attempt / (label + ".log")}')
    for name in (['spans.json'] if not a.no_spans else []) + ['audit.json']:
        subprocess.run(['scp', f'dusty:{remote_dir}/{name}', str(out / name)], check=True)
    report = json.loads((out / 'audit.json').read_text())
    print('DECISION-AUDIT-COMPLETE', report['verdict'], json.dumps(report.get('summary'), default=str), flush=True)


def main(argv=None):
    global BUILD, LOCK, DIAGNOSTIC_KIND
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kind', choices=('decision-row', 'window', 'indexer', 'precision'), default='decision-row',
                        help='Separate installation-gated capture image and lock; same cold controls')
    parser.add_argument('--phase', choices=('preflight', 'capture', 'audit'), default='preflight')
    parser.add_argument('--base-url', default='http://dusty:8000')
    parser.add_argument('--capture-timeout', type=int, default=900, help='seconds to wait for the four receipts')
    parser.add_argument('--receipt', type=Path, help='audit: the capture receipt directory')
    parser.add_argument('--while-serving', action='store_true',
                        help='audit: allow a CPU-only container beside a running boot on dusty (host memory risk)')
    parser.add_argument('--audit-memory', default='24g', help='podman --memory bound for the CPU containers, or none')
    parser.add_argument('--no-spans', action='store_true', help='audit: skip tokenizer-based needle spans')
    parser.add_argument('--mode', choices=('historical', 'matched'), default='historical',
                        help='historical: 80927 blocks, B2 regime, 8192 grid; matched: 81389 blocks, pinned 045954Z '
                             'regime, explicit --chunk-rows equal to the boot threshold')
    parser.add_argument('--chunk-rows', type=int, choices=CHUNKS, help='matched: the prefill chunk grid of this boot')
    parser.add_argument('--nccl-geometry', choices=(NCCL_GEOMETRY,),
                        help='user-approved NCCL geometry diagnostic boot (window kind, matched mode only)')
    parser.add_argument('--nccl-arm', choices=(STANDARD_NCCL_ARM,),
                        help='precision kind, matched mode: ordinary launch NCCL settings as an explicit arm')
    a = parser.parse_args(argv)
    if a.nccl_arm and (a.kind != 'precision' or a.mode != 'matched' or a.nccl_geometry):
        parser.error('The standard NCCL arm is approved for --kind precision --mode matched without --nccl-geometry')
    if a.nccl_geometry and (a.kind not in ('window', 'indexer', 'precision') or a.mode != 'matched'):
        parser.error('NCCL geometry is approved for --kind window --mode matched only')
    if a.kind in ('indexer', 'precision'):
        if a.mode != 'matched' or (a.nccl_geometry != NCCL_GEOMETRY and not (a.kind == 'precision' and a.nccl_arm)):
            parser.error('Indexer capture requires the approved matched geometry arm; '
                         'precision alternatively --nccl-arm standard-upstream')
        BUILD = ROOT / f'receipts/{a.kind}-build-receipt.json'
        LOCK = ROOT / f'ds41-{a.kind}.lock.json'
        DIAGNOSTIC_KIND = a.kind + '-capture'
    elif a.kind == 'window':
        if a.mode != 'matched':
            parser.error('Window capture is approved only for the matched profiles')
        BUILD = ROOT / 'receipts/window-build-receipt.json'
        LOCK = ROOT / 'claude-window.lock.json'
        DIAGNOSTIC_KIND = 'window-capture'
    else:
        BUILD = ROOT / 'receipts/decision-row-build-receipt.json'
        LOCK = ROOT / 'claude-decision-row.lock.json'
        DIAGNOSTIC_KIND = 'decision-row-capture'
    if a.mode == 'matched' and a.chunk_rows is None:
        parser.error('--chunk-rows is required in matched mode')
    if a.mode == 'historical' and a.chunk_rows not in (None, 8192):
        parser.error('historical mode is the 8192 grid only')
    if a.phase == 'audit':
        if a.receipt is None:
            parser.error('--receipt is required for the audit phase')
        phase_audit(a)
    else:
        phase_capture(a, preflight_only=a.phase == 'preflight')


if __name__ == '__main__':
    main()
