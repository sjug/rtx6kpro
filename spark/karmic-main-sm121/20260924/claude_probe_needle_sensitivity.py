"""Content/length sensitivity probe for the frozen 524288-token dual-needle input (DS4.1 TP4).

Supplementary evidence only. It never replaces the frozen acceptance gate (the historical 524288
finding stays unresolved; passes on one boot do not re-qualify it), is not an independent model reference, and changes nothing on the nodes beyond ordinary cold
chat-completion and /tokenize requests to the already-authorized B2 serving contract: router
candidate parent e06df11a with the normal automatic KV profile, no capture hooks, no KV pin.

Variants (each one cold request with a unique cache_salt, thinking False, temperature 0,
max_tokens 64, top-20 logprobs; every other request field is the frozen gate request):
  control-1      the frozen input unchanged, 524288 tokens (same-boot control)
  fresh-1/2      the frozen input with only the 32-hex archive identity replaced by a fresh uuid4
                 hex; the filler run before the late marker is adjusted by the identity's token
                 delta so the late marker keeps its recorded token position 366986 and the prompt
                 has exactly 524288 tokens
  frozen-524285  the frozen input with only the filler run after the late marker shortened so the
                 prompt has exactly 524285 tokens (identity, codes, late position unchanged)
  frozen-524291  the same with the run lengthened to exactly 524291 tokens
  control-2      the frozen input unchanged again, last (drift bracket)
  --series alignment (separate explicit mode; the default series above is unchanged):
  control-1, frozen-524287, frozen-524289, frozen-524032 (block aligned, chunk not), frozen-516096
  and frozen-532480 (both chunk and block aligned), fresh1-524285 and fresh1-524291 (the identity
  that failed at 524288 in receipt 050535Z, with its recorded +1 prefix compensation held fixed),
  control-2. Only the filler run after the late marker varies. Caution: margins that correlate with
  length alignment are suggestive, not proof, because changing the length also changes positions and
  content after the marker. A causal chunking test would hold the token ids fixed and change the chunk
  size, which is a separately approved launcher change outside this probe.
Token counts are measured with the serving /tokenize endpoint on the same chat path the gate uses
(qualify.py's construction), never estimated. Construction is verified before any completion runs,
and every constructed prompt is written verbatim to its trial directory (content.txt) with its sha256.

Identity gate (fail closed, every rank): one running container of image e06df11a with the B2
source-tree labels, the current reviewed runtime kit, the repaired determinism environment, no
prefetch override, no DS41_DECISION_ROW_BLOCKS pin, the B2 selection namespace, a boot log that
reports the automatic serving-block allocation, and an eight-plan regime equal to the B2 reference.

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
import time
import urllib.request
import uuid

from cache_metrics import idle_snapshot, finish
from prepare_decision_reference import exact_regime, signature, B2_IMAGE, PROMPT_SHA, build_reference
import claude_run_decision_capture as drv
from runtime import kit_digest

ROOT = Path(__file__).resolve().parent
NODES = drv.NODES
NAME, REMOTE, HOST_CACHE, SELECTION = drv.NAME, drv.REMOTE, drv.HOST_CACHE, drv.SELECTION
INPUT, INPUT_SHA, REFERENCE, B2_SELECTION_MANIFEST = drv.INPUT, drv.INPUT_SHA, drv.REFERENCE, drv.B2_SELECTION_MANIFEST
FROZEN_TOKENS, LATE_POSITION = 524288, 366986
FILLER = ' filler'
CONTENT = re.compile(r'(Archive identity )([0-9a-f]{32})(\. Memorize this unique retrieval code: 739184\.\nArchive:\n)'
                     r'((?: filler)+)(\nThe late retrieval code is 482617\.\n)((?: filler)+)(\nArchive ends\. .*)', re.S)
BLOCKS_LINE = re.compile(r'b12x autotuning uses \d+ temporary KV blocks before allocating (\d+) serving blocks')
HEX = re.compile(r'^[0-9a-f]{32}$')
MAX_FIT_ROUNDS = 8
MAX_PROMPT = 600000 - 64                                  # launch contract max-model-len minus max_tokens
ALIGNMENT_LENGTHS = (524287, 524289, 524032, 516096, 532480)
FRESH1 = {'hex': '94e8bd2ae83f4340bdcc38c52b3ad98a', 'n1_delta': 1,
          'receipt': 'receipts/needle-sensitivity-20260926T050535Z/construction.json', 'label': 'fresh-1'}
SCOPE = ('supplementary content/length sensitivity evidence on the authorized B2 serving contract; '
         'not an acceptance gate, not an independent model reference; the historical 524288 finding stays unresolved')
ALIGNMENT_CAUTION = ('length-alignment correlation is suggestive, not proof: a length change moves positions and '
                     'content after the marker; a causal chunking test holds token ids fixed and changes the chunk '
                     'size, a separately approved launcher change')


# ---- pure helpers (CPU-tested) ------------------------------------------------------------------

def parse_content(content):
    """The frozen prompt's parts: identity hex, filler runs before/after the late marker, fixed text."""
    match = CONTENT.fullmatch(content)
    if not match:
        raise ValueError('prompt does not have the historical dual-needle structure')
    before, after = match.group(4), match.group(6)
    if len(before) % len(FILLER) or len(after) % len(FILLER):
        raise ValueError('filler runs are not whole " filler" pieces')
    return {'head': match.group(1), 'hex': match.group(2), 'code': match.group(3),
            'n1': len(before) // len(FILLER), 'late': match.group(5), 'n2': len(after) // len(FILLER),
            'tail': match.group(7)}


def build_content(parts, hex_, n1, n2):
    if not HEX.match(hex_) or n1 < 1 or n2 < 1:
        raise ValueError('invalid identity hex or filler counts')
    return parts['head'] + hex_ + parts['code'] + FILLER * n1 + parts['late'] + FILLER * n2 + parts['tail']


def before_late(parts, hex_, n1):
    """The historical before_late string: everything up to the newline before the late marker."""
    return parts['head'] + hex_ + parts['code'] + FILLER * n1


def fresh_hex(frozen_hex):
    value = uuid.uuid4().hex
    if value == frozen_hex or not HEX.match(value):
        raise RuntimeError('fresh identity collides with the frozen one')
    return value


def plan(frozen_hex, fresh_hexes):
    """Ordered variants: [label, hex, target_tokens]; controls bracket the run."""
    if len(fresh_hexes) not in (2, 3) or len(set(fresh_hexes)) != len(fresh_hexes) or frozen_hex in fresh_hexes:
        raise ValueError('two or three distinct fresh identities are required')
    if any(not HEX.match(h) for h in fresh_hexes):
        raise ValueError('fresh identities must be 32 lowercase hex characters')
    variants = [['control-1', frozen_hex, FROZEN_TOKENS], ['fresh-1', fresh_hexes[0], FROZEN_TOKENS],
                ['frozen-524285', frozen_hex, FROZEN_TOKENS - 3], ['fresh-2', fresh_hexes[1], FROZEN_TOKENS],
                ['frozen-524291', frozen_hex, FROZEN_TOKENS + 3]]
    if len(fresh_hexes) == 3:
        variants.append(['fresh-3', fresh_hexes[2], FROZEN_TOKENS])
    variants.append(['control-2', frozen_hex, FROZEN_TOKENS])
    return variants


def plan_alignment(frozen_hex, fresh1_hex):
    """Alignment series: frozen identity at aligned and unaligned lengths, the failing fresh identity at the
    two unaligned lengths that passed for the frozen one, controls bracketing; [label, hex, target]."""
    if not HEX.match(fresh1_hex) or fresh1_hex == frozen_hex:
        raise ValueError('fresh1 identity must be a distinct 32-hex identity')
    variants = [['control-1', frozen_hex, FROZEN_TOKENS]]
    variants += [[f'frozen-{n}', frozen_hex, n] for n in ALIGNMENT_LENGTHS]
    variants += [['fresh1-524285', fresh1_hex, FROZEN_TOKENS - 3], ['fresh1-524291', fresh1_hex, FROZEN_TOKENS + 3]]
    variants.append(['control-2', frozen_hex, FROZEN_TOKENS])
    for label, _, target in variants:
        if not LATE_POSITION + 1000 < target <= MAX_PROMPT:
            raise ValueError(f'{label}: target {target} outside the serving limit or the needle depth')
    return variants


def fresh1_from_receipt(path):
    """The failing fresh identity and its prefix compensation, exactly as receipt 050535Z constructed them."""
    record = json.loads(Path(path).read_text())[FRESH1['label']]
    last = record['rounds'][-1]
    if (record['hex'] != FRESH1['hex'] or record['changed'] != {'hex': True, 'n1_delta': FRESH1['n1_delta'], 'n2_delta': 0}
            or last['prefix_tokens'] != LATE_POSITION or last['prompt_tokens'] != FROZEN_TOKENS):
        raise RuntimeError('fresh1 identity or compensation differs from the retained receipt')
    return FRESH1['hex'], FRESH1['n1_delta']


def tokenize_body(messages):
    """Exactly qualify.py's /tokenize request: the serving chat path with the gate's template kwargs."""
    return {'model': 'DeepSeek-V4.1-Flash', 'messages': messages,
            'chat_template_kwargs': {'thinking': False}, 'add_generation_prompt': True}


def user(content):
    return [{'role': 'user', 'content': content}]


def fit(parts, hex_, target, count, n1_fixed=None):
    """Filler counts (n1, n2) so the rendered prompt has exactly `target` tokens with the late marker
    at LATE_POSITION, measured by `count(messages)` (the serving /tokenize path). Only n1 moves for a
    fresh identity (compensating its token delta), only n2 for a length change of the frozen identity.
    With `n1_fixed` (a fresh identity whose compensation is already recorded) n1 is held there and only
    n2 moves. Returns the construction record; raises unless both counts are exact."""
    n1, n2 = parts['n1'], parts['n2']
    rounds = []
    fixed_prefix = hex_ == parts['hex'] or n1_fixed is not None
    if n1_fixed is not None:
        if hex_ == parts['hex']:
            raise ValueError('n1_fixed applies to a fresh identity only')
        n1 = n1_fixed
    if fixed_prefix:
        n2 += target - FROZEN_TOKENS                       # length change only; prefix untouched
    for _ in range(MAX_FIT_ROUNDS):
        prefix = count(user(before_late(parts, hex_, n1)))
        total = count(user(build_content(parts, hex_, n1, n2)))
        rounds.append({'n1': n1, 'n2': n2, 'prefix_tokens': prefix, 'prompt_tokens': total})
        if prefix == LATE_POSITION and total == target:
            break
        if prefix != LATE_POSITION:
            if fixed_prefix:
                raise RuntimeError(f'fixed prefix measured {prefix} tokens, recorded {LATE_POSITION}')
            n1 += LATE_POSITION - prefix                   # fresh identity: keep the late marker in place
        else:
            if not fixed_prefix:
                raise RuntimeError(f'fresh identity prefix fits but the prompt has {total} tokens, not {target}')
            n2 += target - total                           # fixed prefix: only the run after the marker moves
        if n1 < 1 or n2 < 1:
            raise RuntimeError('filler adjustment left no filler')
    else:
        raise RuntimeError(f'could not fit {hex_} to {target} tokens in {MAX_FIT_ROUNDS} rounds: {rounds}')
    content = build_content(parts, hex_, n1, n2)
    changed = {'hex': hex_ != parts['hex'], 'n1_delta': n1 - parts['n1'], 'n2_delta': n2 - parts['n2']}
    if n1_fixed is not None:
        if changed['n1_delta'] != n1_fixed - parts['n1']:
            raise RuntimeError('fixed prefix compensation moved')
    elif changed['hex'] and changed['n2_delta']:
        raise RuntimeError('fresh identity must not change the filler after the late marker')
    elif not changed['hex'] and changed['n1_delta']:
        raise RuntimeError('length variant must not change the prefix')
    return {'hex': hex_, 'n1': n1, 'n2': n2, 'target_tokens': target, 'rounds': rounds, 'changed': changed,
            'content_sha256': hashlib.sha256(content.encode()).hexdigest(), 'content': content}


def request_body(content, salt):
    """The frozen gate request with only the content and salt substituted (drv.request_body)."""
    return drv.request_body({'messages': user(content), 'chat_template_kwargs': {'thinking': False}}, salt)


def choice_margin(top_logprobs, expected_text):
    """Margin (nats) of the top-20 token that starts `expected_text` over the best other token."""
    entries = [(e['token'], e['logprob']) for e in top_logprobs]
    starting = [(t, lp) for t, lp in entries if t and expected_text.startswith(t)]
    if not starting:
        return {'expected_token': None, 'expected_logprob': None, 'competitor': entries[0][0] if entries else None,
                'competitor_logprob': entries[0][1] if entries else None, 'margin': None}
    token, logprob = max(starting, key=lambda pair: len(pair[0]))
    others = [(t, lp) for t, lp in entries if t != token]
    competitor, competitor_lp = max(others, key=lambda pair: pair[1]) if others else (None, None)
    return {'expected_token': token, 'expected_logprob': logprob, 'competitor': competitor,
            'competitor_logprob': competitor_lp, 'margin': None if competitor_lp is None else logprob - competitor_lp}


def margins(result, expected):
    """First-token margin for the whole expected answer and, after the emitted ", ", for the late code."""
    records = result['choices'][0]['logprobs']['content']
    first = choice_margin(records[0]['top_logprobs'], expected)
    late = None
    late_code = expected.split(', ')[1]
    for index in range(len(records) - 1):
        if records[index]['token'] == ',' and records[index + 1]['token'] == ' ' and index + 2 < len(records):
            late = choice_margin(records[index + 2]['top_logprobs'], late_code)
            late['position'] = index + 2
            break
    return {'first_token': first, 'late_code_after_separator': late}


def response_row(result, expected, target, reference):
    choice = result['choices'][0]
    content = (choice['message'].get('content') or '').strip()
    return {'prompt_tokens': result['usage']['prompt_tokens'], 'target_tokens': target,
            'content': choice['message'].get('content'), 'finish_reason': choice['finish_reason'],
            'correct': choice['finish_reason'] == 'stop' and content == expected,
            'answer_shape': answer_shape(content, expected),
            'margins': margins(result, expected), 'response_signature': signature(result),
            'signature_equals_b2': signature(result) == reference['response_signature']}


def answer_shape(content, expected):
    codes = expected.split(', ')
    parts = [p.strip() for p in content.split(',')]
    names = []
    for part in parts:
        if part == codes[0]:
            names.append('early_code')
        elif part == codes[1]:
            names.append('late_code')
        elif HEX.match(part):
            names.append('identity')
        else:
            names.append('other')
    return '+'.join(names)


def serving_blocks(log_text):
    values = {int(v) for v in BLOCKS_LINE.findall(log_text)}
    if len(values) != 1:
        raise ValueError(f'boot log must report exactly one serving-block allocation, found {sorted(values)}')
    return values.pop()


def check_container(node, container, image, reference, kits):
    """Problems with one rank's container against the authorized B2 contract; identity on success.
    `kits` are the accepted runtime-kit digests: the current reviewed kit and the retained B2 kit."""
    problems = []
    if not container['State'].get('Running'):
        problems.append('not running')
    if container['Image'].removeprefix('sha256:') != B2_IMAGE or image['Id'].removeprefix('sha256:') != B2_IMAGE:
        problems.append('container is not the router candidate parent e06df11a')
    if reference['image_id'] != B2_IMAGE:
        problems.append('B2 reference is not for the router candidate parent')
    labels = image['Config']['Labels']
    if labels.get('local-inference.ds41.diagnostic.kind') != 'router-stage-release-candidate':
        problems.append('image is not the router stage-release candidate')
    trees = {'vllm': labels.get('vllm.source-tree'), 'b12x': labels.get('b12x.source-tree')}
    if trees != reference['source_trees']:
        problems.append('source-tree labels differ from the B2 reference')
    env = drv.env_map(container['Config']['Env'])
    if 'DS41_DECISION_ROW_BLOCKS' in env:
        problems.append('KV pin present; this probe needs the automatic KV profile')
    if env.get('DS41_NODE') != node:
        problems.append('DS41_NODE differs from the rank host')
    for name, value in {'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '1', 'B12X_DENSE_SPLITK_TURBO': '0'}.items():
        if env.get(name) != value:
            problems.append('repaired runtime environment differs: ' + name)
    if 'VLLM_DS41_L2_PREFETCH' in env:
        problems.append('unexpected prefetch override')
    label_kit = container['Config']['Labels'].get('local-inference.ds41.kit.sha256')
    if not label_kit or env.get('DS41_KIT_SHA256') != label_kit:
        problems.append('kit digest label and environment differ')
    if label_kit not in kits:
        problems.append('container runs neither the current reviewed runtime kit nor the retained B2 kit')
    cache_dir = env.get('B12X_COMPILE_CACHE_DIR')
    if not cache_dir or not cache_dir.startswith('/cache/'):
        problems.append('B12X selection namespace is not under /cache')
    identity = {'rank': NODES.index(node), 'image_id': B2_IMAGE, 'kit_sha256': label_kit,
                'kit_equals_b2_reference': label_kit == reference['kit_sha256'],
                'cache_dir': cache_dir, 'source_trees': trees}
    return problems, identity


def summarize(rows):
    controls = [r for r in rows if r['label'].startswith('control-')]
    signatures = {r['response_signature'] for r in controls}
    return {'controls_identical': len(controls) == 2 and len(signatures) == 1,
            'control_equals_b2': all(r['signature_equals_b2'] for r in controls),
            'variants': [{k: r[k] for k in ('label', 'hex', 'target_tokens', 'prompt_tokens', 'content',
                                             'answer_shape', 'correct', 'cached_tokens', 'elapsed_s')}
                         | {'first_token_margin': r['margins']['first_token']['margin'],
                            'late_code_margin': (r['margins']['late_code_after_separator'] or {}).get('margin')}
                         for r in rows],
            'scope': SCOPE}


# ---- remote plumbing (thin, untested) -----------------------------------------------------------

def local_inputs():
    reference = json.loads(REFERENCE.read_text())
    if reference != build_reference():
        raise RuntimeError('B2 reference no longer matches its retained source receipts')
    raw = INPUT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise RuntimeError('Historical input file identity changed')
    spec = json.loads(raw)
    content = spec['messages'][0]['content']
    if hashlib.sha256(content.encode()).hexdigest() != PROMPT_SHA:
        raise RuntimeError('Historical prompt identity changed')
    if (spec['prompt_tokens'] != FROZEN_TOKENS or spec['late_marker_prefix_tokens_including_template'] != LATE_POSITION
            or spec['chat_template_kwargs'] != {'thinking': False}):
        raise RuntimeError('Historical input spec differs from this probe\'s constants')
    parts = parse_content(content)
    if build_content(parts, parts['hex'], parts['n1'], parts['n2']) != content:
        raise RuntimeError('Prompt structure round trip failed')
    manifest = json.loads(B2_SELECTION_MANIFEST.read_text())
    return reference, spec, parts, manifest['cache_path']


def post(base_url, path, body, timeout):
    request = urllib.request.Request(base_url + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def preflight(out, reference, b2_cache_path):
    identities, blocks = {}, {}
    kits = {kit_digest(), reference['kit_sha256']}
    for node in NODES:
        running = drv.ssh(node, 'podman ps -q').split()
        container = drv.ssh_json(node, 'podman inspect ' + NAME)[0]
        image = drv.ssh_json(node, 'podman image inspect ' + B2_IMAGE)[0]
        (out / f'{node}-container.json').write_text(json.dumps(container, indent=2) + '\n')
        (out / f'{node}-image.json').write_text(json.dumps(image, indent=2) + '\n')
        (out / f'{node}-meminfo.txt').write_text(drv.ssh(node, 'cat /proc/meminfo'))
        problems, identity = check_container(node, container, image, reference, kits)
        if len(running) != 1 or not container['Id'].startswith(running[0]):
            problems.append('another container runs on the rank host')
        if identity['cache_dir'] and identity['cache_dir'] + '/' + SELECTION != b2_cache_path:
            problems.append('selection namespace differs from the B2 tuning snapshot')
        log = drv.ssh(node, f'podman logs {NAME} 2>&1 | grep -F "serving blocks" || true')
        try:
            blocks[node] = serving_blocks(log)
        except ValueError as error:
            problems.append(str(error))
        if problems:
            raise RuntimeError(f'{node}: ' + '; '.join(problems))
        identities[node] = identity
        print('PREFLIGHT-OK', node, 'serving_blocks', blocks[node], flush=True)
    if len(set(blocks.values())) != 1:
        raise RuntimeError(f'serving-block allocation differs across ranks: {blocks}')
    return identities, blocks[NODES[0]]


def snapshot_selections(identities, blocks, reference, out, tag):
    regimes = {}
    for node, identity in identities.items():
        text = drv.ssh(node, 'cat ' + shlex.quote(drv.selection_host_path(identity['cache_dir'])), timeout=300)
        (out / f'{node}-selection-{tag}.json').write_text(text)
        regimes[node] = exact_regime(json.loads(text), blocks)
    (out / f'regime-{tag}.json').write_text(json.dumps(regimes, indent=2, sort_keys=True) + '\n')
    if any(regime != reference['regime'] for regime in regimes.values()):
        raise RuntimeError(f'eight-plan regime at {blocks} blocks differs from the B2 reference ({tag})')
    return regimes


def comparison_reference(reference, path):
    """Explicitly pin a separately reviewed same-boot regime, never accept arbitrary drift."""
    if path is None:
        return reference, {'scope': 'B2 regime matched', 'regime_equals_b2': True}
    raw = path.read_bytes()
    ranks = json.loads(raw)
    if set(ranks) != set(NODES):
        raise ValueError('regime reference must contain exactly all four ranks')
    regime = ranks[NODES[0]]
    if set(regime) != set(reference['regime']) or any(value != regime for value in ranks.values()):
        raise ValueError('regime reference is incomplete or ranks disagree')
    changes = {name: {'b2': reference['regime'][name], 'current': value}
               for name, value in regime.items() if value != reference['regime'][name]}
    return dict(reference, regime=regime), {
        'scope': 'same-boot sensitivity only; not B2 numerical equivalence or causal precision A/B',
        'regime_reference_path': str(path.resolve()),
        'regime_reference_sha256': hashlib.sha256(raw).hexdigest(),
        'regime_equals_b2': not changes, 'regime_difference_from_b2': changes}


def send(base_url, body, target, out):
    before = idle_snapshot(base_url)
    (out / 'request.json').write_text(json.dumps(body, indent=1) + '\n')
    (out / 'before.json').write_text(json.dumps(before, indent=2) + '\n')
    start = time.monotonic()
    result = post(base_url, '/v1/chat/completions', body, 3600)
    elapsed = time.monotonic() - start
    (out / 'trial.json').write_text(json.dumps({'request': body, 'response': result, 'elapsed_s': elapsed}, indent=1) + '\n')
    if result['usage']['prompt_tokens'] != target:
        raise RuntimeError(f'served prompt has {result["usage"]["prompt_tokens"]} tokens, constructed {target}')
    cached = finish(base_url, before, target, out / 'trial-cache.json')
    if cached != 0:
        raise RuntimeError('request was not cold; the variant is not comparable')
    return result, elapsed, cached


def run(a):
    reference, spec, parts, b2_cache_path = local_inputs()
    stamp = drv.stamp_now()
    out = ROOT / 'receipts' / f'needle-sensitivity-{stamp}'
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
        execute(a, reference, spec, parts, b2_cache_path, out)
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


def execute(a, reference, spec, parts, b2_cache_path, out):
    print('RECEIPTS', out, flush=True)
    (out / 'reference.json').write_bytes(REFERENCE.read_bytes())
    pre = out / 'pre'
    pre.mkdir()
    identities, blocks = preflight(pre, reference, b2_cache_path)
    (out / 'identities.json').write_text(json.dumps({'blocks': blocks, 'nodes': identities}, indent=2, sort_keys=True) + '\n')
    idle_snapshot(a.base_url)
    selection_reference, comparison = comparison_reference(reference, a.regime_reference)
    (out / 'comparison-scope.json').write_text(json.dumps(comparison, indent=2) + '\n')
    snapshot_selections(identities, blocks, selection_reference, out, 'pre')
    fixed = {}
    if a.series == 'alignment':
        hex1, delta1 = fresh1_from_receipt(ROOT / FRESH1['receipt'])
        variants = plan_alignment(parts['hex'], hex1)
        fixed[hex1] = parts['n1'] + delta1
        comparison['series'] = 'alignment'
        comparison['alignment_caution'] = ALIGNMENT_CAUTION
        (out / 'comparison-scope.json').write_text(json.dumps(comparison, indent=2) + '\n')
    else:
        variants = plan(parts['hex'], [fresh_hex(parts['hex']) for _ in range(a.fresh)])

    def count(messages):
        return post(a.base_url, '/tokenize', tokenize_body(messages), 600)['count']

    constructed = {}
    for label, hex_, target in variants:
        if hex_ == parts['hex'] and target == FROZEN_TOKENS:
            record = {'hex': hex_, 'n1': parts['n1'], 'n2': parts['n2'], 'target_tokens': target, 'rounds': [],
                      'changed': {'hex': False, 'n1_delta': 0, 'n2_delta': 0}, 'content_sha256': PROMPT_SHA,
                      'content': spec['messages'][0]['content']}
            record['rounds'].append({'n1': parts['n1'], 'n2': parts['n2'],
                                     'prompt_tokens': count(user(record['content']))})
            if record['rounds'][0]['prompt_tokens'] != FROZEN_TOKENS:
                raise RuntimeError('frozen input no longer tokenizes to 524288 on this boot')
        else:
            record = fit(parts, hex_, target, count, n1_fixed=fixed.get(hex_))
        constructed[label] = record
        print('CONSTRUCTED', label, json.dumps({k: v for k, v in record.items() if k != 'content'}), flush=True)
    (out / 'construction.json').write_text(json.dumps(
        {label: {k: v for k, v in r.items() if k != 'content'} for label, r in constructed.items()}, indent=2) + '\n')
    if a.construct_only:
        print('NEEDLE-SENSITIVITY-CONSTRUCTED', out, flush=True)
        return
    rows = []
    for label, hex_, target in variants:
        trial_dir = out / label
        trial_dir.mkdir()
        record = constructed[label]
        (trial_dir / 'content.txt').write_text(record['content'])
        result, elapsed, cached = send(a.base_url, request_body(record['content'], uuid.uuid4().hex), target, trial_dir)
        row = response_row(result, spec['expected'], target, reference)
        row.update(label=label, hex=hex_, elapsed_s=elapsed, cached_tokens=cached, construction=record['changed'],
                   content_sha256=record['content_sha256'])
        rows.append(row)
        (trial_dir / 'response-row.json').write_text(json.dumps(row, indent=2) + '\n')
        print('VARIANT', json.dumps(row), flush=True)
    post_dir = out / 'post'
    post_dir.mkdir()
    identities_after, blocks_after = preflight(post_dir, reference, b2_cache_path)
    if identities_after != identities or blocks_after != blocks:
        raise RuntimeError('serving identity changed during the probe')
    snapshot_selections(identities, blocks, selection_reference, out, 'post')
    summary = summarize(rows)
    summary['comparison'] = comparison
    summary['series'] = a.series
    summary.update(blocks=blocks, image_id=B2_IMAGE, kit_sha256=identities[NODES[0]]['kit_sha256'],
                   kit_equals_b2_reference=identities[NODES[0]]['kit_equals_b2_reference'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print('NEEDLE-SENSITIVITY-COMPLETE', json.dumps(summary), flush=True)
    if not summary['controls_identical']:
        raise RuntimeError('the two frozen controls differ on this boot; variant comparisons are not interpretable')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-url', default='http://dusty:8000')
    parser.add_argument('--fresh', type=int, choices=(2, 3), default=2, help='fresh identities at 524288 tokens')
    parser.add_argument('--series', choices=('sensitivity', 'alignment'), default='sensitivity',
                        help='alignment: frozen identity at aligned/unaligned lengths plus the recorded failing '
                             'fresh identity at 524285/524291; only the filler after the late marker varies')
    parser.add_argument('--construct-only', action='store_true',
                        help='preflight and /tokenize construction only; no completion request')
    parser.add_argument('--regime-reference', type=Path,
                        help='explicit reviewed four-rank regime snapshot for a separately labeled same-boot experiment')
    run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
