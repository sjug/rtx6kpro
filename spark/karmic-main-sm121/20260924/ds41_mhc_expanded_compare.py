r"""Real-input expanded mHC comparison: served vs standalone, both configurations and the autotuner's own
candidate population against float64, on one idle GB10 inside the exact precision release image.

Input: a claude-mhc-expanded-v2 capture of one engram layer on the final 8192-row chunk of the frozen
prompt, gathered into one directory: every rank's row-shard staging file (together every row of the real
inputs and small outputs; weights from rank 0, 128-row tails from the last rank), record (integer digests
of every captured tensor at full size, served Selection) and receipt. Nothing here is synthetic.

Gates, each required before anything is interpreted:
  G1 capture   every shard file's sha256 equals its receipt (hashed while streaming it in); shards tile
               the rows exactly; the reassembled row tensors and rank 0's weights reproduce the full-size
               device digests; every rank's digests, layer and served Selection equal rank 0's
  G2 contract  the served Selection is norm.mhc from the seeded cache, its query equals the DS4.1
               contract query and rebuilds the recorded 8192-row mhc.pre.expanded key, and its config is
               the manifest's configuration for the arm the capture ran on (--served passing|release)
  G3 faithful  the standalone plan pinned to the served configuration reproduces the served post, comb
               and pre_out bit for bit, the served full-size y digest and y tail, twice

Measurements (only after all gates pass):
  reference    passing and release configurations against float64 of B12X's lagged reference with
               B12X's own bounds (post 1e-6 long-K, fp32 4e-5/2e-5, bf16 y 0.008) and cross-configuration
               parity (bf16 bit-equal, fp32 2e-6). For BF16 y, the contract check uses
               upstream's float32 reference with its final BF16 cast. Unrounded FP64
               y errors are information only, retained separately with outlier values.
  population   every distinct reduction order the autotuner may choose for this query (B12X
               TUNING.choices on this device, one representative per backend, k-splits, tile_k,
               lagged_prepare and partials class), each against the same reference and against passing
  consumer     both configurations' outputs through one fixed float32 reference of the next fused
               post_pre on the final 128 rows (captured attention-output tail, residual, ffn_norm): bf16
               elements of the new residual and of the FFN input that differ (information)

Classification (operator level only; no served-output or numerical-policy claim):
  not-faithful              a gate failed; nothing else is interpreted
  contract-violation        the passing or release configuration exceeds a B12X reference bound
  population-violation      some other candidate exceeds a bound; the tuning space admits a
                            configuration outside B12X's own contract (reported, not attributed)
  release-outlier           both within bounds, but release's reference error or its distance from
                            passing exceeds 4x the population's largest (others' error or spread)
  passing-outlier           both within bounds, but passing's reference error exceeds 4x the population's
  within-valid-population   both within every bound and within the population's error and spread:
                            the difference is autotuner reduction-order rounding at this operator

Run (release image, no network, capture and kit read-only; see ds41_mhc_layer0_replay.py for mounts):
  python3 -P /gate/ds41_mhc_expanded_compare.py --manifest /gate/ds41-mhc-layer0-inputs.json \
    --manifest-sha256 <sha> --captures-dir /captures --token <token> --served passing|release --out /receipts/<new>
  The recorder (run_mhc_expanded_capture.py) prints the exact command with every path and digest.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != HERE]
import argparse
import hashlib
import importlib.util
import json

NODES = ('dusty', 'toby', 'rusty', 'kirby')
OUTPUTS = ('post', 'comb', 'pre_out', 'y')
INPUTS = ('residual', 'pre_mix', 'fn', 'scale', 'base', 'norm')
ROW_SHARDED = ('residual', 'pre_mix', 'positions', 'post', 'comb', 'pre_out')
WEIGHTS = ('fn', 'scale', 'base', 'norm', 'ffn_norm')
TAILS = ('y_tail', 'residual_out_tail', 'attn_out_tail')
STAGED_FULL = ROW_SHARDED + WEIGHTS
SCHEMA = 'claude-mhc-expanded-v2'
DIGEST_BLOCK_BYTES = 2 << 20
READ_SLICE = 16 << 20
TAIL = 128
RATIO = 4.0
CLASS_FIELDS = ('backend', 'projection_k_splits', 'projection_tile_k', 'lagged_prepare', 'partials_per_cta')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_replay(path=HERE / 'ds41_mhc_layer0_replay.py'):
    """The reviewed layer-0 harness by path (its reference, bounds, plan and source checks, unchanged)."""
    spec = importlib.util.spec_from_file_location('ds41_mhc_layer0_replay_shared', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- pure logic (locally tested)

def capture_problems(records, recomputed, file_shas):
    """G1. records: {node: per-rank json}; recomputed: {name: [lane0, lane1]} of the reassembled row tensors
    and rank 0's weights; file_shas: {node: (receipt sha256, streamed sha256)}."""
    problems = []
    if set(records) != set(NODES):
        problems.append(f'capture records for {sorted(records)}, expected all four ranks')
        return problems
    head = records['dusty']
    rows = head['chunk_rows']
    for rank, node in enumerate(NODES):
        record = records[node]
        if record['rank'] != rank or record.get('world') != len(NODES):
            problems.append(f'{node} is not rank {rank} of {len(NODES)}')
        per = rows // len(NODES)
        if record.get('shard') != [rank * per, (rank + 1) * per]:
            problems.append(f'{node} shard {record.get("shard")} does not tile the rows')
        expected, streamed = file_shas.get(node, (None, None))
        if expected is None or expected != streamed:
            problems.append(f'{node} staging file digest differs from its receipt')
    if any(r['schema'] != SCHEMA for r in records.values()) or len({r['token'] for r in records.values()}) != 1:
        problems.append(f'records are not one token of schema {SCHEMA}')
    for name in STAGED_FULL:
        if recomputed.get(name) != head['digests'].get(name):
            problems.append(f'reassembled {name} differs from its device digest')
    for node in NODES[1:]:
        for key in ('layer', 'plan', 'hidden', 'chunk_rows'):
            if records[node].get(key) != head.get(key):
                problems.append(f'{node} {key} differs from rank 0')
        differing = sorted(n for n in head['digests'] if records[node]['digests'].get(n) != head['digests'][n])
        if differing or set(records[node]['digests']) != set(head['digests']):
            problems.append(f'{node} differs from rank 0: {differing}')
    return problems


def assemble(shards):
    """{node: {name: tensor}} in rank order into full row tensors, rank 0's weights and the last rank's tails."""
    import torch
    first, last = shards[NODES[0]], shards[NODES[-1]]
    data = {name: torch.cat([shards[n][name] for n in NODES]) for name in ROW_SHARDED}
    data.update({name: first[name] for name in WEIGHTS})
    data.update({name: last[name] for name in TAILS})
    return data


def contract_problems(plan, want_query, want_config, rebuilt_key, expected_key, plain):
    """G2. plan: rank 0's recorded Selection {'component_id', 'source', 'query', 'config'}."""
    problems = []
    if rebuilt_key != expected_key:
        problems.append('contract no longer rebuilds the recorded mhc.pre.expanded key')
    if plan.get('component_id') != 'norm.mhc' or plan.get('source') != 'cached':
        problems.append(f"served selection is {plan.get('component_id')}/{plan.get('source')}, not norm.mhc from the seeded cache")
    if plain(plan.get('query')) != plain(want_query):
        problems.append('served query differs from the contract query')
    if plain(plan.get('config')) != plain(want_config):
        problems.append('served configuration is not the expected arm configuration')
    return problems


def faithful(served, replayed, repeat, y_digest):
    """G3: post, comb, pre_out bit-equal to the served outputs; the replayed y's digest equal to the served
    full-size y digest and its tail bit-equal to the served tail."""
    changed = {n: int((served[n].cpu() != replayed[n].cpu()).sum()) for n in ('post', 'comb', 'pre_out')}
    changed['y_tail'] = int((served['y_tail'].cpu() != replayed['y'][-TAIL:].cpu()).sum())
    y_equal = [int(v) for v in digest(replayed['y'].cpu())] == list(y_digest)
    return {'passed': bool(repeat) and not any(changed.values()) and y_equal, 'repeat_bit_equal': bool(repeat),
            'changed_elements': changed, 'y_full_digest_equal': y_equal}


def config_class(config):
    return tuple(config[f] for f in CLASS_FIELDS)


def representatives(candidates, required):
    """One candidate per reduction-order class, in enumeration order, excluding the required configs'
    classes (those run exactly as recorded). candidates: [config dict]."""
    seen = {config_class(c) for c in required}
    chosen = []
    for config in candidates:
        key = config_class(config)
        if key not in seen:
            seen.add(key)
            chosen.append(config)
    return chosen


def classify(result):
    """result: {'gates': {name: bool}, 'errors': {label: {output: stats}}, 'distance': {label: {output: max}}}
    with labels 'passing', 'release' and population labels."""
    if not all(result['gates'].values()):
        return 'not-faithful'
    errors, distance = result['errors'], result['distance']
    outside = lambda label: any(errors[label][n]['outside_tolerance'] for n in OUTPUTS)
    if outside('passing') or outside('release'):
        return 'contract-violation'
    population = [label for label in errors if label not in ('passing', 'release')]
    if any(outside(label) for label in population):
        return 'population-violation'
    if not population:
        return 'no-population'
    for name in OUTPUTS:
        if name == 'y':
            continue
        worst_error = max(errors[label][name]['max_abs'] for label in population)
        worst_spread = max(distance[label][name] for label in population)
        if errors['release'][name]['max_abs'] > RATIO * worst_error or distance['release'][name] > RATIO * max(worst_spread, 0.0):
            return 'release-outlier'
        if errors['passing'][name]['max_abs'] > RATIO * worst_error:
            return 'passing-outlier'
    return 'within-valid-population'


def own_entry_removed(files, own=str(HERE / Path(__file__).name), modules=None):
    """The shared loaded-module check admits only the replay and its helper from the kit; this script runs
    as __main__ from the kit. Remove it and multiprocessing's object-identical
    __mp_main__ alias; no other module may name this file."""
    modules = sys.modules if modules is None else modules
    files = dict(files)
    if files.get('__mp_main__') == own:
        if modules.get('__main__') is None or modules.get('__mp_main__') is not modules['__main__']:
            raise RuntimeError('Multiprocessing main is not an identity alias of this harness')
        del files['__mp_main__']
    if files.get('__main__') == own:
        del files['__main__']
    stray = sorted(n for n, f in files.items() if f == own)
    if stray:
        raise RuntimeError(f'This harness is also loaded as {stray}')
    return files


def consumer(post, comb, pre_out, x, residual, ffn_norm, rms_eps):
    """Float32 reference of the fused post_pre that consumes an engram layer's pre outputs: B12X's
    post_reference over the captured four-stream residual, then the lagged collapse with this layer's
    pre_out and the FFN RMSNorm, rounded where the kernel stores bf16."""
    import torch
    new = (post.float().unsqueeze(-1) * x.unsqueeze(1).float()
           + (comb.float().unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)).to(torch.bfloat16)
    collapsed = (pre_out.float().unsqueeze(-1) * new.float()).sum(1).to(torch.bfloat16).float()
    y = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + rms_eps) * ffn_norm.float()
    return {'residual': new, 'ffn_in': y.to(torch.bfloat16)}


def consumer_diff(a, b):
    out = {}
    for name in ('residual', 'ffn_in'):
        changed = a[name] != b[name]
        out[name] = {'changed_elements': int(changed.sum()), 'elements': int(changed.numel()),
                     'rows_changed': int(changed.flatten(1).any(1).sum()),
                     'max_abs': float((a[name].double() - b[name].double()).abs().max())}
    return out


# ---------------------------------------------------------------- GPU execution

def candidates_for(plan):
    from b12x.norm.mhc._tuning import TUNING
    from b12x.preparation.device import detect_device
    return [config.to_dict() for _, config in TUNING.choices(plan.query, detect_device().identity)]


def upstream_y_reference(residual, incoming, weight, rms_eps):
    """Pinned test_mhc_lagged._lagged_reference y, including BOTH BF16 stores."""
    import torch
    collapsed = (incoming.float().unsqueeze(-1) * residual.float()).sum(1).bfloat16()
    value = collapsed.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + rms_eps) * weight.float()
    return value.bfloat16()


def y_rounding_samples(got, exact, upstream, tolerance, limit=256):
    """Retain high-precision outliers, without treating final quantization as a contract defect."""
    import torch
    delta = (got.double() - exact.double()).abs()
    indices = torch.nonzero(delta > tolerance['atol'] + tolerance['rtol'] * exact.abs())
    rounded = exact.bfloat16()
    samples = []
    for row, col in indices[:limit].tolist():
        samples.append({'row': row, 'column': col, 'actual': float(got[row, col]),
                        'fp64': float(exact[row, col]), 'fp64_rounded_bf16': float(rounded[row, col]),
                        'upstream_bf16': float(upstream[row, col])})
    return {'unrounded_outliers': len(indices), 'samples_truncated': len(indices) > limit,
            'samples': samples,
            'different_from_rounded_fp64': int((got != rounded).sum())}


def layer_report(replay, data, plan_record, y_digest, manifest, served_side, params, chunk_rows, max_population):
    import torch
    tol = manifest['tolerances']
    configs = {'passing': manifest['configs']['passing']['pre_expanded'],
               'release': manifest['configs']['release']['pre_expanded']}
    cuda = {n: data[n].cuda() for n in INPUTS}
    inputs = (cuda['residual'], cuda['fn'], cuda['scale'], cuda['base'], cuda['norm'], cuda['pre_mix'])
    served = {n: data[n] for n in ('post', 'comb', 'pre_out', 'y_tail')}
    plan = replay.declare(chunk_rows, 5120, params, True, plan_record['config'])
    replayed, repeat = replay.run(plan, *inputs, params)
    gate = faithful(served, replayed, repeat, y_digest)
    report = {'faithful': gate}
    if not gate['passed']:
        return report
    reference, _ = replay.references(cuda['residual'], cuda['fn'], cuda['fn'], cuda['scale'], cuda['base'],
                                     cuda['pre_mix'], cuda['norm'], params)
    reference = {n: v.cpu() for n, v in reference.items()}
    outputs = {served_side: {n: v.cpu() for n, v in replayed.items()}}
    other = 'release' if served_side == 'passing' else 'passing'
    got, rep = replay.run(replay.declare(chunk_rows, 5120, params, True, configs[other]), *inputs, params)
    outputs[other] = {n: v.cpu() for n, v in got.items()}
    repeats = {served_side: repeat, other: rep}
    base_plan = replay.declare(chunk_rows, 5120, params, True)
    population = representatives(candidates_for(base_plan), list(configs.values()))
    report['population_classes'] = len(population)
    population = population[:max_population]
    report['population_run'] = len(population)
    report['population_truncated'] = report['population_run'] < report['population_classes']
    for index, config in enumerate(population):
        label = f'candidate-{index}'
        got, rep = replay.run(replay.declare(chunk_rows, 5120, params, True, config), *inputs, params)
        outputs[label], repeats[label] = {n: v.cpu() for n, v in got.items()}, rep
    report['population'] = {f'candidate-{i}': c for i, c in enumerate(population)}
    report['repeat_bit_equal'] = repeats
    report['errors'] = {label: {n: replay.error_stats(o[n], reference[n], tol['reference'][n]) for n in OUTPUTS}
                        for label, o in outputs.items()}
    upstream_y = upstream_y_reference(cuda['residual'], cuda['pre_mix'], cuda['norm'], params['rms_eps']).cpu()
    report['y_unrounded_fp64_errors'] = {label: stats['y'] for label, stats in report['errors'].items()}
    report['y_reference_contract'] = 'pinned upstream float32 collapse/RMSNorm with intermediate and final BF16 rounding; unchanged tolerances'
    for label, stats in report['errors'].items():
        stats['y'] = replay.error_stats(outputs[label]['y'], upstream_y, tol['reference']['y'])
    report['y_rounding_information'] = {
        side: y_rounding_samples(outputs[side]['y'], reference['y'], upstream_y, tol['reference']['y'])
        for side in ('passing', 'release')}
    report['distance'] = {label: {n: float((o[n].double() - outputs['passing'][n].double()).abs().max())
                                  for n in OUTPUTS} for label, o in outputs.items()}
    report['config_diff'] = replay.compare_outputs(outputs['passing'], outputs['release'], tol['config_parity'])
    x, residual = data['attn_out_tail'], data['residual'][-TAIL:]
    ffn = {side: consumer(outputs[side]['post'][-TAIL:], outputs[side]['comb'][-TAIL:], outputs[side]['pre_out'][-TAIL:],
                          x, residual, data['ffn_norm'], params['rms_eps']) for side in ('passing', 'release')}
    report['consumer_information'] = consumer_diff(ffn['passing'], ffn['release'])
    report['gates'] = {'faithful': True, 'all_repeat': all(repeats.values()),
                       'population_complete': not report['population_truncated']}
    report['classification'] = classify(report)
    return report


def main():
    import os
    import torch
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--captures-dir', type=Path, required=True,
                   help='the gathered captures/ directory: four shard .bin files, records and receipts')
    p.add_argument('--token', required=True)
    p.add_argument('--served', choices=('passing', 'release'), required=True)
    p.add_argument('--max-population', type=int, default=32)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if sha(a.manifest.read_bytes()) != a.manifest_sha256:
        raise RuntimeError('Manifest digest mismatch')
    manifest = json.loads(a.manifest.read_text())
    replay = load_replay()

    def read(path):
        try:
            return Path(path).read_bytes()
        except OSError:
            return None
    source = manifest['source']
    replay.helper(source['helper']['sha256'])
    problems = replay.source_problems(source, os.environ, sys.path, read)
    if problems:
        raise RuntimeError('Source runtime is not the release contract: ' + '; '.join(problems))
    import b12x  # noqa: F401
    lock_after = json.loads(read(source['image_lock']))['after']

    def modules_check(stage):
        found, unlisted = replay.loaded_problems(source, own_entry_removed(replay.module_files(sys.modules)),
                                                 lock_after, read)
        if found:
            raise RuntimeError(f'Loaded modules break the source contract ({stage}): ' + '; '.join(found[:10]))
        return {'stage': stage, 'unlisted_non_python': unlisted}
    identity = [modules_check('imports')]
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('A visible GB10 is required')
    a.out.mkdir(parents=True, exist_ok=False)
    records, shards, file_shas = {}, {}, {}
    for rank, node in enumerate(NODES):
        stem = a.captures_dir / f'{node}-rank{rank}-{a.token}'
        records[node] = json.loads(stem.with_suffix('.json').read_text())
        receipt = json.loads((a.captures_dir / f'{a.token}-rank{rank}.mhc-consumed').read_text())
        shards[node], streamed = load_staging(stem.with_suffix('.bin'), records[node]['staging'])
        file_shas[node] = (receipt.get('bin_sha256'), streamed)
    head = records['dusty']
    data = assemble(shards)
    recomputed = {n: [int(v) for v in digest(data[n])] for n in STAGED_FULL}
    report = {'scope': 'operator-level comparison on real captured inputs; no served-output or numerical-policy claim',
              'token': a.token, 'layer': head['layer'], 'served_side': a.served,
              'harness_sha256': sha(Path(__file__).read_bytes()), 'source_identity': identity,
              'replay_sha256': sha((HERE / 'ds41_mhc_layer0_replay.py').read_bytes()),
              'capture_memory': {n: r.get('memory') for n, r in records.items()}}
    g1 = capture_problems(records, recomputed, file_shas)
    selections = replay.helper()
    want_query = dict(selections.MHC_QUERY, max_tokens=manifest['chunk_rows'], **selections.mhc_invocation('pre', True))
    g2 = contract_problems(head['plan'], want_query, manifest['configs'][a.served]['pre_expanded'],
                           selections.mhc_key('pre', manifest['chunk_rows'], True), manifest['keys']['pre_expanded'],
                           replay.plain)
    report['G1_capture'], report['G2_contract'] = g1, g2
    report['classification'] = 'not-faithful'
    if not g1 and not g2:
        query = head['plan']['query']
        params = {'rms_eps': float(query['rms_eps']), 'hc_eps': float(query['hc_eps']),
                  'sinkhorn_iters': int(query['sinkhorn_iters'])}
        report['result'] = layer_report(replay, data, head['plan'], head['digests']['y'], manifest, a.served, params,
                                        manifest['chunk_rows'], a.max_population)
        report['classification'] = report['result'].get('classification', 'not-faithful')
    identity.append(modules_check('after-run'))
    (a.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('MHC-EXPANDED-COMPARE', report['layer'], report['classification'], flush=True)


def load_staging(path, staging):
    """Rank 0's raw staging file into one host buffer (read in place, hashed while streaming) and its
    tensors as views, exactly as the capture helper laid them out."""
    import ctypes
    import torch
    total = int(staging['total'])
    if Path(path).stat().st_size != total:
        raise RuntimeError('Staging file size differs from its layout')
    buffer = torch.empty(total, dtype=torch.uint8)
    view = memoryview((ctypes.c_char * total).from_address(buffer.data_ptr())).cast('B')
    hasher = hashlib.sha256()
    with open(path, 'rb') as stream:
        for first in range(0, total, READ_SLICE):
            piece = view[first:first + READ_SLICE]
            if stream.readinto(piece) != len(piece):
                raise RuntimeError('Short read of the staging file')
            hasher.update(piece)
    tensors = {}
    for name, entry in staging['tensors'].items():
        dtype = getattr(torch, entry['dtype'])
        tensors[name] = buffer[entry['offset']:entry['offset'] + entry['nbytes']].view(dtype).view(entry['shape'])
    return tensors, hasher.hexdigest()


def digest(tensor, out=None):
    """Identical to the capture helper's digest(): two wrapping int64 lanes over raw 32-bit words."""
    import torch
    device = tensor.device
    out = torch.zeros(2, dtype=torch.int64, device=device) if out is None else out
    rows = tensor if tensor.dim() >= 2 else tensor.reshape(-1)
    row_bytes = (rows[0].numel() if rows.dim() >= 2 else 1) * rows.element_size()
    per_block = max(1, DIGEST_BLOCK_BYTES // row_bytes)
    start = 0
    for first in range(0, rows.shape[0], per_block):
        raw = rows[first:first + per_block].contiguous().reshape(-1).view(torch.uint8)
        if raw.numel() % 4:
            raise ValueError('digest needs a whole number of 32-bit words')
        words = raw.view(torch.int32)
        out[0] += words.sum(dtype=torch.int64)
        weights = torch.arange(start, start + words.numel(), dtype=torch.int64, device=device).remainder_(65521).add_(1)
        out[1] += (words.to(torch.int64) * weights).sum()
        start += words.numel()
    return out


if __name__ == '__main__':
    main()
