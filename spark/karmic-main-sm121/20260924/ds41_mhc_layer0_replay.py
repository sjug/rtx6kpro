r"""Standalone mHC operator comparison on one idle GB10: passing vs release configurations against fp64.

Runs inside the exact precision release image (no model load, no serving, no collective). Section A
uses real inputs: the frozen prompt's final 8192-row chunk at layer 0, where DS4.1 feeds the token
embeddings as a 2-D residual through the broadcast fn (hc_attn_fn.view(24, 4, H).sum(1), exactly as
DeepseekV4Model.finalize_mhc_broadcast_weights) with incoming pre_mix [1, 0, 0, 0]. Section B runs the
tf32 mhc.pre.expanded configurations on the first engram layer's real weights with synthetic
four-stream residuals (no real residual exists without a full forward), and is labeled synthetic.

Contract: each plan is declared exactly as vllm/models/deepseek_v4_1/b12x_layers.py does (caps
max_tokens 8192, hidden 5120, split_k 80; invocation lagged_mix, norm weight, config eps and Sinkhorn
iterations) and its unpinned query must rebuild the recorded serving key before any pinned
configuration (B12X Plan override) is run. Both configurations run twice and must repeat bitwise.

Faithfulness gate (section A only): with the passing configuration the last 128 rows of y must equal
the retained layer-0 window capture's hidden_in bit for bit, positions 524160..524287. At layer 0 y is
RMSNorm(embedding), so the gate proves the token ids, embedding rows, norm weight and eps; it cannot
distinguish the configurations, which differ only in post, comb and pre_out. No interpretation is
written unless the gate passes. Information readout (section A, gate passed): both configurations'
post, comb and pre_out drive one fixed float32 reference of layer 0's post_pre consumer on the captured
rows (captured wo_reduced as the attention output, embedding streams as the residual), counting bf16
elements of the new residual and of the layer-0 FFN input that differ: the first activations the plan
can move. The served post_pre plan is not run; its own configuration is held out of this comparison.

Reference: fp64 of B12X's own lagged reference (tests/norm/test_mhc_lagged.py _lagged_reference), for
the served broadcast fn and, as information, for the full fn over four identical streams. Verdicts:
  not-faithful                  gate failed; nothing else is interpreted
  configs-bit-identical         the two configurations agree bit for bit on every output
  reduction-order-level         outputs differ within every B12X bound, reference errors within 4x
  asymmetric-error-review       both within tolerance, one configuration's error over 4x the other's
  outside-operator-tolerance    a configuration exceeds B12X's own tolerance against the reference
  outside-config-parity         both within reference tolerance, but the two configurations differ by more
                                than B12X's own cross-backend parity bound (bf16 bit-equal, fp32 2e-6)
Operator-level only: no statement about the served first token or a numerical policy follows.

Execution (one idle GB10, no serving container on it; user-run):
  T=/home/jugs/git/ds41-r38/karmic-main-20260924   K=<node copy of this kit dir>   S=$(date -u +%Y%m%dT%H%M%SZ)
  R=$T/receipts/decision-row-matched8192-nccl-standard-upstream-capture-20260926T233036Z
  mkdir $T/mhc-layer0-cache-$S        # fresh /cache inside the task tree; never an existing cache
  # PYTHONPATH is the serving value (launch_contract.render); -P keeps /gate off sys.path. The image
  # already places every JIT cache under /cache/jit/<release namespace>; no cache variable is overridden.
  podman run --rm --pull=never --network=none --device nvidia.com/gpu=all --ipc=private --shm-size=8g \
    -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/opt/jovian-judgement/vllm:/opt/jovian-judgement/b12x \
    -v $HOME/.cache/huggingface:/hf:ro -v $K:/gate:ro -v $R/captures:/capture:ro \
    -v $T/mhc-layer0-tokens.json:/tokens.json:ro -v $T/mhc-layer0-cache-$S:/cache:rw -v $T/receipts:/receipts:rw \
    --entrypoint python3 1989e16daf38d2966b03a2e2abcb878263fb7d17183c8d6c5b084a51fbed7e8f \
    -P /gate/ds41_mhc_layer0_replay.py --manifest /gate/ds41-mhc-layer0-inputs.json --manifest-sha256 <sha> \
    --tokens /tokens.json --tokens-sha256 <from tokenize> \
    --capture /capture/dusty-rank0-decision-row-m8192-20260926t233036z-window.pt \
    --snapshot /hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/fb2764a5cf321eaa5070ca8f9e892818f477c16d \
    --out /receipts/mhc-layer0-replay-$S
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Python puts the script directory first on sys.path; drop it so /gate can never shadow the image's
# PYTHONPATH (vLLM tree, then B12X tree). The command also runs python3 -P. The one kit helper this
# harness needs is loaded by explicit path with a hash check (helper()).
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != HERE]
import argparse
import hashlib
import json
import math
import re

OUTPUTS = ('post', 'comb', 'pre_out', 'y')
ERROR_RATIO = 4.0
TENSOR_PATTERNS = {
    'embed': r'^(model\.)?(embed|embed_tokens)\.weight$',
    'layer_fn': r'^(model\.)?layers\.{layer}\.hc_attn_fn$',
    'layer_scale': r'^(model\.)?layers\.{layer}\.hc_attn_scale$',
    'layer_base': r'^(model\.)?layers\.{layer}\.hc_attn_base$',
    'layer_norm': r'^(model\.)?layers\.{layer}\.attn_norm\.weight$',
    'ffn_norm': r'^(model\.)?layers\.{layer}\.ffn_norm\.weight$',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- pure logic (locally tested)

def locate(names, kind, layer=None):
    pattern = re.compile(TENSOR_PATTERNS[kind].format(layer=layer))
    hits = sorted(n for n in names if pattern.match(n))
    if len(hits) != 1:
        raise RuntimeError(f'{kind} (layer {layer}) matched {len(hits)} checkpoint tensors: {hits[:4]}')
    return hits[0]


def check_config(config, manifest):
    if config.get('model_type') == 'deepseek_v41':
        config = config['text_config']
        if config.get('model_type') != 'deepseek_v41_text':
            raise RuntimeError('Unexpected nested text model type')
    need = {'hidden_size': 5120, 'hc_mult': 4}
    for key, value in need.items():
        if config.get(key) != value:
            raise RuntimeError(f'config {key}={config.get(key)!r}, expected {value}')
    engram = list(config.get('engram_layer_ids') or [])
    if 0 in engram:
        raise RuntimeError('Layer 0 is an engram layer: its residual is engram-modified and not rebuildable here')
    return {'rms_eps': float(config['rms_norm_eps']), 'hc_eps': float(config['hc_eps']),
            'sinkhorn_iters': int(config['hc_sinkhorn_iters']), 'engram_layer_ids': engram}


def final_chunk(tokens, manifest):
    if len(tokens) != manifest['prompt_tokens']:
        raise RuntimeError('Token ids are not the frozen prompt length')
    return tokens[-manifest['chunk_rows']:]


def lagged_reference(residual4, fn, scale, base, incoming, weight, *, rms_eps, hc_eps, sinkhorn_iters):
    """B12X tests/norm/test_mhc_lagged.py _lagged_reference, evaluated in float64."""
    import torch
    f64 = torch.float64
    flat = residual4.flatten(1).to(f64)
    mixes = flat @ fn.to(f64).T * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    scale, base = scale.to(f64), base.to(f64)
    predicted = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + base[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    collapsed = (incoming.to(f64).unsqueeze(-1) * residual4.to(f64)).sum(1).to(torch.bfloat16).to(f64)
    y = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + rms_eps) * weight.to(f64)
    return {'post': post, 'comb': comb, 'pre_out': predicted, 'y': y}


def references(residual, fn_served, fn_full, scale, base, pre_mix, norm, params):
    """(served reference, full-fn information reference). A 2-D residual is the single broadcast stream
    DS4.1 feeds at layer 0 with incoming mix [1, 0, 0, 0]; its full-fn form is four identical streams."""
    import torch
    params = {k: params[k] for k in ('rms_eps', 'hc_eps', 'sinkhorn_iters')}
    if residual.ndim == 2:
        expected = torch.zeros_like(pre_mix)
        expected[:, 0] = 1
        if not torch.equal(pre_mix, expected):
            raise RuntimeError('A 2-D residual is only valid with incoming mix [1, 0, 0, 0]')
        served = lagged_reference(residual[:, None, :], fn_served, scale, base, pre_mix[:, :1], norm, **params)
        full = lagged_reference(residual[:, None, :].expand(-1, 4, -1), fn_full, scale, base, pre_mix, norm, **params)
        return served, full
    reference = lagged_reference(residual, fn_served, scale, base, pre_mix, norm, **params)
    return reference, reference


def broadcast_fn(fn, hidden):
    return fn.detach().view(-1, 4, hidden).sum(dim=1)


def error_stats(got, want, tolerance):
    import torch
    got64, want64 = got.to(torch.float64), want.to(torch.float64)
    diff = (got64 - want64).abs()
    bound = tolerance['atol'] + tolerance['rtol'] * want64.abs()
    return {'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
            'outside_tolerance': int((diff > bound).sum()), 'elements': int(diff.numel())}


def compare_outputs(a, b, parity):
    """Passing vs release; parity is B12X's cross-configuration bound (bf16 bit-equal, fp32 2e-6)."""
    import torch
    out = {}
    for name in OUTPUTS:
        equal = torch.equal(a[name], b[name])
        a64, b64 = a[name].to(torch.float64), b[name].to(torch.float64)
        diff = (a64 - b64).abs()
        bound = parity['bf16' if a[name].dtype == torch.bfloat16 else 'fp32']
        out[name] = {'bit_equal': equal, 'changed_elements': int((a[name] != b[name]).sum()),
                     'max_abs': float(diff.max()),
                     'outside_parity': int((diff > bound['atol'] + bound['rtol'] * b64.abs()).sum())}
    return out


def verdict(section):
    """Faithfulness outranks everything, including repeatability."""
    if section.get('gate') is not None and section['gate'].get('passed') is not True:
        return 'not-faithful'
    if not all(section['repeat_bit_equal'].values()):
        return 'nonrepeatable'
    if all(section['config_diff'][n]['bit_equal'] for n in OUTPUTS):
        return 'configs-bit-identical'
    errors = section['errors']
    if any(errors[c][n]['outside_tolerance'] for c in errors for n in OUTPUTS):
        return 'outside-operator-tolerance'
    if any(section['config_diff'][n]['outside_parity'] for n in OUTPUTS):
        return 'outside-config-parity'
    for name in OUTPUTS:
        p, r = errors['passing'][name]['max_abs'], errors['release'][name]['max_abs']
        low, high = min(p, r), max(p, r)
        if high > 0 and (low == 0 or high / low > ERROR_RATIO):
            return 'asymmetric-error-review'
    return 'reduction-order-level'


def may_continue(section_a):
    """Read the fields, not the verdict: the gate ran and passed, and both configurations repeated bitwise."""
    gate_result = section_a.get('gate')
    return (isinstance(gate_result, dict) and gate_result.get('passed') is True
            and bool(section_a.get('repeat_bit_equal')) and all(v is True for v in section_a['repeat_bit_equal'].values()))


def gate(y_passing, capture, expected_positions):
    import torch
    layer = capture['layers'][0]
    positions_ok = torch.equal(layer['positions'], torch.arange(*expected_positions, dtype=torch.int64))
    rows = layer['hidden_in']
    tail = y_passing[-rows.shape[0]:].cpu()
    equal = torch.equal(tail, rows)
    return {'passed': bool(positions_ok and equal), 'positions_ok': bool(positions_ok),
            'hidden_in_bit_equal': bool(equal), 'changed_elements': int((tail != rows).sum()),
            'rows': int(rows.shape[0])}


def propagate(post, comb, pre_out, x, embed_rows, ffn_norm, rms_eps):
    """Layer 0's post_pre consumer of one configuration's outputs, on the captured rows (float32, as
    B12X's own references): residual = bf16(post * x + comb^T streams) over the four identical embedding
    streams the wrapper expands a 2-D residual into (b12x_layers.py pre residual_out); FFN input =
    bf16(RMSNorm(bf16(sum pre_out * residual)) * ffn_norm), attention pre_out being the lagged mix."""
    import torch
    streams = embed_rows[:, None, :].expand(-1, 4, -1)
    residual = (post.float().unsqueeze(-1) * x.unsqueeze(1).float()
                + (comb.float().unsqueeze(-1) * streams.unsqueeze(2).float()).sum(dim=1)).to(torch.bfloat16)
    collapsed = (pre_out.float().unsqueeze(-1) * residual.float()).sum(1).to(torch.bfloat16).float()
    y = (collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + rms_eps) * ffn_norm.float())
    return {'residual': residual, 'ffn_in': y.to(torch.bfloat16)}


def propagation(outputs, x, embed_rows, ffn_norm, rms_eps):
    """Bf16 elements of layer 0's new residual and FFN input that differ between the configurations on
    the captured rows. Information only: the evaluator is fixed, so the count isolates the pre plan."""
    import torch
    rows = x.shape[0]
    got = {side: propagate(*(o[n][-rows:].cpu() for n in ('post', 'comb', 'pre_out')), x, embed_rows, ffn_norm,
                           rms_eps) for side, o in outputs.items()}
    out = {}
    for name in ('residual', 'ffn_in'):
        a, b = got['passing'][name], got['release'][name]
        out[name] = {'changed_elements': int((a != b).sum()), 'elements': int(a.numel()),
                     'rows_changed': int((a != b).flatten(1).any(1).sum()),
                     'max_abs': float((a.double() - b.double()).abs().max())}
    return out


# ---------------------------------------------------------------- B12X execution (GPU only)

def declare(caps_rows, hidden, params, expanded, config=None):
    from b12x.norm import mhc
    from b12x.norm.mhc._tuning import MhcConfig
    from b12x.preparation import FrozenMapping
    import torch
    caps = mhc.Caps(device=torch.device('cuda'), max_tokens=caps_rows, hidden_size=hidden, split_k=hidden // 64)
    invocation = FrozenMapping({'lagged_mix': True, 'has_norm_weight': True, 'rms_eps': params['rms_eps'],
                                'hc_eps': params['hc_eps'], 'sinkhorn_iters': params['sinkhorn_iters'],
                                'norm_eps': params['rms_eps'], 'operation': 'pre', 'has_fn_bf16': False,
                                'expanded_residual': expanded})
    override = None if config is None else MhcConfig.from_config(FrozenMapping(config))
    return mhc.plan(caps, invocation=invocation, override=override)


def plain(value):
    """FrozenMapping and nested mappings as plain JSON-comparable values."""
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


_HELPER = {}


def helper(expected_sha256=None, path=None):
    """claude_compare_selections loaded from this kit by path (never via sys.path), hash-checked when asked."""
    import importlib.util
    path = Path(path or HERE / 'claude_compare_selections.py')
    data = path.read_bytes()
    if expected_sha256 is not None and sha(data) != expected_sha256:
        raise RuntimeError('Kit helper differs from the manifest: ' + path.name)
    if path not in _HELPER:
        spec = importlib.util.spec_from_file_location('ds41_mhc_layer0_selection_helper', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _HELPER[path] = module
    return _HELPER[path]


def source_problems(source, environ, path_entries, read):
    """Before any GPU work: the image is the release lock's source runtime and imports resolve to it.
    read(path) returns bytes or None; nothing here imports vLLM or B12X."""
    problems = []
    if environ.get('PYTHONPATH') != source['pythonpath']:
        problems.append(f"PYTHONPATH is {environ.get('PYTHONPATH')!r}, not the serving {source['pythonpath']!r}")
    entries = list(dict.fromkeys(str(Path(p).resolve()) for p in path_entries if p))
    roots = source['pythonpath'].split(':')
    if [e for e in entries if e in roots] != roots:
        problems.append('sys.path does not hold the serving source roots in serving order')
    else:
        packages = [k for k, e in enumerate(entries) if e.endswith(('site-packages', 'dist-packages'))]
        if packages and max(entries.index(r) for r in roots) > min(packages):
            problems.append('an installed-package directory precedes the source roots on sys.path')
    if str(HERE) in entries or any(Path(e) == HERE for e in entries):
        problems.append('kit directory is on sys.path')
    lock = read(source['image_lock'])
    if lock is None or sha(lock) != source['lock_sha256']:
        problems.append('image release lock is missing or differs from the manifest')
    elif json.loads(lock)['trees'] != source['trees']:
        problems.append('image release lock trees differ')
    for component, files in source['required'].items():
        for rel, digest in files.items():
            data = read(f"{source['root']}/{component}/{rel}")
            if data is None or sha(data) != digest:
                problems.append(f'{component}/{rel} is missing or differs from the release lock')
    return problems


def module_files(modules):
    """Do not resolve Torch's synthetic registry filenames relative to the image cwd."""
    import torch
    registries = {'torch.ops': torch.ops, 'torch.classes': torch.classes}
    files = {}
    for name, module in modules.items():
        if name in registries and module is registries[name]:
            continue
        file = vars(module).get('__file__') if module is not None else None
        files[name] = str(Path(file).resolve()) if file else None
    return files


def loaded_problems(source, modules, lock_after, read):
    """Every loaded vllm*/b12x* module comes from the source root; every loaded .py there is a lock file with
    its hash; nothing is loaded from the kit except this harness and its helper. modules: {name: file}."""
    problems, unlisted = [], []
    root = source['root'].rstrip('/') + '/'
    allowed_kit = {str(HERE / Path(__file__).name), str(HERE / 'claude_compare_selections.py')}
    for name, file in sorted(modules.items()):
        if not file:
            continue
        top = name.split('.')[0]
        if file.startswith(str(HERE) + '/') and file not in allowed_kit:
            problems.append(f'{name} loaded from the kit: {file}')
        if top in ('vllm', 'b12x') and not file.startswith(root):
            problems.append(f'{name} loaded outside the source root: {file}')
        if not file.startswith(root):
            continue
        component, _, rel = file[len(root):].partition('/')
        entry = lock_after.get(component, {}).get(rel)
        if file.endswith('.py'):
            data = read(file)
            if entry is None or data is None or sha(data) != entry['sha256']:
                problems.append(f'{component}/{rel} is not the release lock source')
        elif entry is None:
            unlisted.append(f'{component}/{rel}')
    return problems, unlisted


def contract_check(encoded_query, invocation, expanded, expected_key, rows=8192):
    """The declared plan must equal the query and invocation whose key matched the live serving records."""
    selections = helper()
    MHC_QUERY, mhc_invocation, mhc_key = selections.MHC_QUERY, selections.mhc_invocation, selections.mhc_key
    want_invocation = mhc_invocation('pre', expanded)
    want_query = dict(MHC_QUERY, max_tokens=rows, **want_invocation)
    if plain(encoded_query) != plain(want_query):
        raise RuntimeError('Declared mHC query differs from the serving contract')
    if plain(invocation) != plain(want_invocation):
        raise RuntimeError('Declared mHC invocation differs from the serving contract')
    key = mhc_key('pre', rows, expanded)
    if key != expected_key:
        raise RuntimeError('Serving contract no longer rebuilds the recorded selection key')
    return key


def run(plan, residual, fn, scale, base, norm, pre_mix, params):
    import torch
    from b12x.norm import mhc
    from b12x.norm.mhc import _impl
    from b12x.preparation import PreparationSession, PreparedCall
    rows, hidden = residual.shape[0], residual.shape[-1]

    def outputs():
        return {'out': torch.empty((rows, 4, hidden), dtype=torch.bfloat16, device='cuda'),
                'y': torch.empty((rows, hidden), dtype=torch.bfloat16, device='cuda'),
                'post': torch.empty((rows, 4), dtype=torch.float32, device='cuda'),
                'comb': torch.empty((rows, 4, 4), dtype=torch.float32, device='cuda'),
                'pre_out': torch.empty((rows, 4), dtype=torch.float32, device='cuda')}

    def scratch(specs):
        specs = list(specs)
        if len(specs) != 1:
            raise RuntimeError('Unexpected mHC scratch layout')
        return torch.empty(specs[0].shape, dtype=specs[0].dtype, device='cuda')
    kwargs = dict(rms_eps=params['rms_eps'], hc_eps=params['hc_eps'], sinkhorn_iters=params['sinkhorn_iters'],
                  norm_weight=norm, norm_eps=params['rms_eps'], pre_mix=pre_mix)

    def prepare(state):
        buffers, space = outputs(), scratch(state.scratch_specs())
        binding = state.bind(scratch=space, tokens=rows, **buffers)
        return PreparedCall(run=lambda: _impl._b12x_mhc_pre_impl(residual, fn, scale, base, **kwargs,
                                                                  binding=binding, _state=state),
                            owners=(space, binding))
    results = []
    with PreparationSession(device=torch.device('cuda'), autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name='mhc-layer0-pre', prepare_call=prepare),))
        for _ in range(2):
            buffers = outputs()
            binding = mhc.bind(plan, scratch=scratch(plan.scratch_specs()), tokens=rows, **buffers)
            mhc.run_pre(residual, fn, scale, base, **kwargs, binding=binding)
            torch.cuda.synchronize()
            results.append({name: buffers[name].clone() for name in OUTPUTS})
    repeat = all(torch.equal(results[0][n], results[1][n]) for n in OUTPUTS)
    return results[0], repeat


def section(label, residual, fn_served, fn_full, scale, base, norm, pre_mix, params, manifest, key_name, expanded,
            capture=None):
    import torch
    from b12x.norm.mhc._tuning import TUNING
    base_plan = declare(manifest['chunk_rows'], 5120, params, expanded)
    rebuilt = contract_check(TUNING.encode_query(base_plan.query), base_plan.invocation, expanded,
                             manifest['keys'][key_name], manifest['chunk_rows'])
    outputs, repeats = {}, {}
    for side in ('passing', 'release'):
        plan = declare(manifest['chunk_rows'], 5120, params, expanded, manifest['configs'][side][key_name])
        outputs[side], repeats[side] = run(plan, residual, fn_served, scale, base, norm, pre_mix, params)
    served_ref, full_ref = references(residual, fn_served, fn_full, scale, base, pre_mix, norm, params)
    tol = manifest['tolerances']
    errors = {side: {n: error_stats(outputs[side][n].cpu(), served_ref[n].cpu(), tol['reference'][n])
                     for n in OUTPUTS} for side in outputs}
    info = {side: {n: error_stats(outputs[side][n].cpu(), full_ref[n].cpu(), tol['reference'][n])
                   for n in OUTPUTS} for side in outputs}
    result = {'label': label, 'key': rebuilt, 'repeat_bit_equal': repeats,
              'config_diff': compare_outputs(outputs['passing'], outputs['release'], tol['config_parity']),
              'errors': errors, 'errors_vs_full_fn_information': info, 'gate': None}
    if capture is not None:
        first, last = manifest['capture']['positions']
        result['gate'] = gate(outputs['passing']['y'], capture, (first, last + 1))
    result['verdict'] = verdict(result)
    return result, outputs


def main():
    import torch
    from safetensors import safe_open
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--tokens-sha256', required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True, help='DS4.1 checkpoint snapshot directory')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    for path, expected in ((a.manifest, a.manifest_sha256), (a.tokens, a.tokens_sha256)):
        if sha(path.read_bytes()) != expected:
            raise RuntimeError(f'Input digest mismatch: {path}')
    manifest = json.loads(a.manifest.read_text())
    if sha(a.capture.read_bytes()) != manifest['capture']['sha256']:
        raise RuntimeError('Capture digest mismatch')
    import os

    def read(path):
        try:
            return Path(path).read_bytes()
        except OSError:
            return None
    source = manifest['source']
    helper(source['helper']['sha256'])
    problems = source_problems(source, os.environ, sys.path, read)
    if problems:
        raise RuntimeError('Source runtime is not the release contract: ' + '; '.join(problems))
    import b12x
    lock_after = json.loads(read(source['image_lock']))['after']

    def modules_check(stage):
        files = module_files(dict(sys.modules))
        found, unlisted = loaded_problems(source, files, lock_after, read)
        if found:
            raise RuntimeError(f'Loaded modules break the source contract ({stage}): ' + '; '.join(found[:10]))
        return {'stage': stage, 'unlisted_non_python': unlisted}
    identity = [modules_check('imports')]
    if not Path(b12x.__file__).resolve().is_relative_to(source['root'] + '/b12x/b12x'):
        raise RuntimeError('B12X does not import from the image source tree')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('A visible GB10 is required')
    a.out.mkdir(parents=True, exist_ok=False)
    params = check_config(json.loads((a.snapshot / 'config.json').read_text()), manifest)
    tokens = torch.tensor(final_chunk(json.loads(a.tokens.read_text()), manifest), dtype=torch.long)
    index = json.loads((a.snapshot / 'model.safetensors.index.json').read_text())['weight_map']

    def tensor(name):
        with safe_open(str(a.snapshot / index[name]), framework='pt', device='cpu') as f:
            return f.get_tensor(name)
    names = {kind: locate(index, kind, 0) for kind in TENSOR_PATTERNS if kind.startswith(('layer', 'ffn'))}
    names['embed'] = locate(index, 'embed')
    embed = tensor(names['embed'])
    if embed.dtype != torch.bfloat16 or embed.shape[1] != 5120:
        raise RuntimeError(f'Unexpected embedding {embed.dtype} {tuple(embed.shape)}')
    x = embed.index_select(0, tokens).cuda()
    fn = tensor(names['layer_fn']).float().cuda()
    scale, base = tensor(names['layer_scale']).float().cuda(), tensor(names['layer_base']).float().cuda()
    norm = tensor(names['layer_norm']).to(torch.bfloat16).cuda()
    pre_mix = torch.zeros((x.shape[0], 4), dtype=torch.float32, device='cuda')
    pre_mix[:, 0] = 1
    capture = torch.load(a.capture, map_location='cpu', weights_only=True)
    meta = capture.get('meta', {})
    if (capture.get('schema') != 'claude-window-v1' or meta.get('problems') or meta.get('rank') != 0
            or meta.get('prompt_tokens') != manifest['prompt_tokens'] or meta.get('chunk_rows') != manifest['chunk_rows']
            or tuple(capture['layers'][0]['wo_reduced'].shape) != (128, 5120)):
        raise RuntimeError('Capture is not the retained rank-0 layer-0 window of the frozen prompt')
    report = {'scope': 'operator-level comparison; no served-output or numerical-policy claim',
              'tensors': names, 'params': params, 'torch': torch.__version__, 'source_identity': identity,
              'harness_sha256': sha(Path(__file__).read_bytes())}
    report['A_layer0_real'], outs = section('A: layer-0 mhc.pre, real inputs', x, broadcast_fn(fn, 5120), fn, scale,
                                            base, norm, pre_mix, params, manifest, 'pre', False, capture)
    proceed = may_continue(report['A_layer0_real'])
    report['A_layer0_real']['interpreted'] = proceed
    if proceed:
        wo = capture['layers'][0]['wo_reduced']
        report['A_layer0_real']['propagation_information'] = propagation(
            outs, wo, x[-wo.shape[0]:].cpu(), tensor(locate(index, 'ffn_norm', 0)).to(torch.bfloat16),
            params['rms_eps'])
    torch.save({side: {n: v.cpu() for n, v in o.items() if n != 'y'} | {'y_tail': o['y'][-128:].cpu()}
                for side, o in outs.items()}, a.out / 'section-a-outputs.pt')
    if proceed and params['engram_layer_ids']:
        layer = params['engram_layer_ids'][0]
        lnames = {kind: locate(index, kind, layer) for kind in TENSOR_PATTERNS if kind.startswith('layer')}
        g = torch.Generator(device='cpu').manual_seed(20260927)
        rms = float(x.float().square().mean().sqrt())
        residual = (torch.randn((x.shape[0], 4, 5120), generator=g) * rms).to(torch.bfloat16).cuda()
        mix = torch.softmax(torch.randn((x.shape[0], 4), generator=g), dim=-1).cuda()
        lfn = tensor(lnames['layer_fn']).float().cuda()
        report['B_expanded_synthetic'], _ = section(
            f'B: layer-{layer} mhc.pre.expanded, real weights, SYNTHETIC residual', residual, lfn, lfn,
            tensor(lnames['layer_scale']).float().cuda(), tensor(lnames['layer_base']).float().cuda(),
            tensor(lnames['layer_norm']).to(torch.bfloat16).cuda(), mix, params, manifest, 'pre_expanded', True)
        report['B_expanded_synthetic']['tensors'] = lnames
    identity.append(modules_check('after-run'))
    (a.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('MHC-LAYER0-REPLAY', report['A_layer0_real']['verdict'],
          report.get('B_expanded_synthetic', {}).get('verdict'), flush=True)


if __name__ == '__main__':
    main()
