#!/usr/bin/env python3
"""Production-shape conformance of the DS4.1 MXFP4 DSA indexer (score, select, candidates). Not an oracle.

Runs on ONE idle GPU inside the router or capture image (b12x a7d7d29b). It
reproduces the three production index roles of deepseek_v4_1/attention.py
_declare_index_plan exactly:
  full     layers 2, 8, 14   page 128 (ratio 2), full-width score
  produce  layer 20          page 256 (ratio 1), full-width score, candidate_topk_blocks 2048
  consume  layers 24..36     page 256, scores only layer 20's candidates (max_candidates 16384)
with num_q_heads 32, topk 512, page-table width 2344, and the decision chunk's
prefill index plan (256 rows: the final INDEX_CHUNK of the 8192-row chunk,
positions 524032..524287 of one request).

Exact production configs: validate the saved selection identity and assignment,
then compile that configuration as an explicit override with autotuning off.
A missing multi-candidate record raises LookupError. A single legal fixed
configuration is accepted as production does. The source, key, configuration
and execution selection of every plan are reported. No serving cache is mounted.

Inputs go through the production paths: index keys are written by
dsa_indexer.quantize_write_index_k_mxfp4 into a block-major pool of the
production block count and stride (offsets beyond 2 GiB); queries by
dsa_indexer.quantize_q_mxfp4. Then bind, score, select as vLLM forward_mqa.

Checks per role, distribution and row (reference = the published rounding
stages, lowest logical index wins exact ties, the layer-20 candidate block
rule; claude_decision_row_audit.py implements the same rules):
  * layout: decoding the written pages with the audit's MXFP4 page layout
    equals an independent E2M1 x UE8M0 quantization of the same keys;
  * scores: the kernel's full BF16 score row versus the reference
    (fraction bitwise equal, maximum difference in BF16 ulps);
  * selection: structure, exact deterministic equality, differences outside
    a one-ulp band at the rank-512 threshold;
  * candidates (produce): the same at block level;
  * tie exposure (the discriminator): entries tied at the threshold, how
    many selected entries entered only by the tie rule, how many would change
    under the opposite (highest-index) tie rule, and where tie-admitted
    entries sit (position quartiles).
Fail closed: non-finite scores, a layout mismatch, LookupError, structural
selection errors, or disagreement beyond the band make the run fail and are
listed.

Scope: synthetic inputs; it checks that the kernels compute what the MXFP4
format, the published rounding stages and the chosen tie rule define. It is
not a model oracle; it cannot show whether the published model would select
the same entries, and it does not bear on the original 524K answer by itself.

  python3 claude_conformance_dsa.py --selection router-namespace.json --blocks 80927 --out /writable/dsa.json
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
SCOPE = ('synthetic-input conformance of the MXFP4 indexer at production plans to the format, published '
         'rounding stages and the image tie rule; not a model oracle; says nothing by itself about the '
         'original 524K answer')
BLOCK_STRIDE, HEADS, TOPK, WIDTH = 227840, 32, 512, 2344
CANDIDATES, CANDIDATE_BLOCKS, CANDIDATE_BLOCK = 16384, 2048, 8
PROMPT_TOKENS, INDEX_CHUNK = 524288, 256
ROLES = {'full': {'page': 128, 'ratio': 2, 'max_candidates': 0, 'candidate_topk_blocks': 0},
         'produce': {'page': 256, 'ratio': 1, 'max_candidates': 0, 'candidate_topk_blocks': CANDIDATE_BLOCKS},
         'consume': {'page': 256, 'ratio': 1, 'max_candidates': CANDIDATES, 'candidate_topk_blocks': 0}}
DISTRIBUTIONS = ('gaussian', 'codebook', 'outlier', 'low-magnitude')
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- pure, device-agnostic reference (CPU-tested; mirrors claude_decision_row_audit) ----------

def encode_mxfp4(x, torch):
    """Independent E2M1 x UE8M0/32 encoder (rules of b12x tests/attention/test_dsa_indexer_mxfp4.py)."""
    groups = x.float().reshape(*x.shape[:-1], 4, 32)
    amax = groups.abs().amax(-1).clamp_min(6 * 2.0 ** -126)
    bits = (amax / 6).contiguous().view(torch.int32)
    scales = ((bits >> 23) + ((bits & 0x7FFFFF) != 0)).to(torch.uint8)
    scale = (scales.int() << 23).view(torch.float32)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device)
    pick = (groups.div(scale[..., None]).abs()[..., None] - lut[order]).abs().argmin(-1)
    code = (order[pick].to(torch.uint8) | (torch.signbit(groups).to(torch.uint8) << 3)).reshape(*x.shape[:-1], 128)
    return code[..., ::2] | (code[..., 1::2] << 4), scales


def decode_mxfp4(data, scales, torch):
    codes = torch.stack((data & 15, data >> 4), dim=-1).reshape(*data.shape[:-1], 128)
    lut = torch.tensor(E2M1, device=data.device)
    values = lut[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -values, values)
    return (values * torch.pow(2.0, scales.float() - 127.0).repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def key_rows(pages, page, count, torch):
    """Logical keys from pages laid out [page,64] data then [page,4] UE8M0 (the audit's layout)."""
    data = pages[:, :page * 64].reshape(-1, 64)
    scales = pages[:, page * 64:page * 68].reshape(-1, 4)
    return decode_mxfp4(data[:count], scales[:count], torch)


def bf16(x, torch):
    return x.to(torch.bfloat16).to(torch.float32)


def published_scores(q_data, q_scales, weights, keys, torch):
    """One query row: [n] = BF16(sum_h BF16(max(BF16(dot), 0) * w)) with an FP32 head sum
    (dsa_indexer/mxfp4.py _PagedScore). q_data [heads, 64], q_scales [heads, 4], weights [heads],
    keys [n, 128] (BF16 values; FP32 accepted). Row-at-a-time bounds memory to n x heads."""
    q = decode_mxfp4(q_data, q_scales, torch).float()                    # [heads, 128]
    dots = keys.float() @ q.t()                                          # [n, heads]
    per_head = bf16(torch.clamp(bf16(dots, torch), min=0.0) * weights.float()[None, :], torch)
    return bf16(per_head.sum(dim=1), torch)


def ranked(values, torch, *, lowest_index_first=True):
    """Stable descending order; exact ties resolved by ascending (or descending) position."""
    if lowest_index_first:
        return torch.sort(-values, stable=True).indices
    flipped = torch.sort(-values.flip(-1), stable=True).indices
    return values.shape[-1] - 1 - flipped


def bf16_ulp(x):
    x = abs(float(x))
    if x == 0.0 or math.isinf(x):
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(x)) - 7)


def select_reference(scores, pool, k, torch):
    """Deterministic top-k of `scores` restricted to `pool` (positions), lowest logical index first on ties
    whatever the order of `pool`."""
    pool = pool.sort().values
    order = ranked(scores[pool], torch)
    return pool[order[:k]]


def candidate_reference(scores, count, torch):
    """Layer 20: blocks of 8, block logit = max valid score, last visible block forced, top-2048 blocks."""
    blocks = (count + CANDIDATE_BLOCK - 1) // CANDIDATE_BLOCK
    padded = torch.full((blocks * CANDIDATE_BLOCK,), float('-inf'), device=scores.device)
    padded[:count] = scores[:count]
    logits = padded.view(blocks, CANDIDATE_BLOCK).amax(dim=1)
    logits[(count - 1) // CANDIDATE_BLOCK] = float('inf')
    chosen = ranked(logits, torch)[:min(CANDIDATE_BLOCKS, blocks)]
    positions = (chosen.sort().values[:, None] * CANDIDATE_BLOCK + torch.arange(CANDIDATE_BLOCK, device=scores.device))
    positions = positions.reshape(-1)
    return positions[positions < count], logits, chosen


def tie_exposure(scores, pool, selected, k, torch):
    """How much of the selection is decided by the tie rule at the rank-k threshold."""
    pool = pool.sort().values
    values = scores[pool]
    order = ranked(values, torch)
    threshold = float(values[order[k - 1]])
    tied = pool[values == threshold]
    chosen = set(int(i) for i in selected)
    admitted = sorted(int(i) for i in tied if int(i) in chosen)
    strictly_above = int((values > threshold).sum())
    opposite = set(int(i) for i in pool[ranked(values, torch, lowest_index_first=False)[:k]])
    quartiles = [0, 0, 0, 0]
    top = max(int(pool.max()) + 1, 1)
    for i in admitted:
        quartiles[min(3, 4 * i // top)] += 1
    return {'threshold': threshold, 'tie_population': int(tied.numel()), 'strictly_above': strictly_above,
            'tie_admitted': len(admitted), 'opposite_rule_changes': len(chosen ^ opposite) // 2,
            'tie_admitted_position_quartiles': quartiles}


def compare_row(production, reference, scores, threshold, *, pool=None):
    """Structure, exact equality and one-ulp band, as the audit defines them."""
    valid = [int(v) for v in production if v >= 0]
    problems = []
    if [int(v) for v in production[:len(valid)]] != valid or any(int(v) != -1 for v in production[len(valid):]):
        problems.append('padding is not a pure tail')
    if len(valid) != len(reference):
        problems.append(f'{len(valid)} entries, expected {len(reference)}')
    if len(set(valid)) != len(valid) or valid != sorted(valid):
        problems.append('duplicates or not ascending')
    allowed = set(range(scores.numel())) if pool is None else set(int(v) for v in pool)
    if any(v not in allowed for v in valid):
        problems.append('entry outside allowed pool')
    band = 1.01 * bf16_ulp(threshold)
    difference = set(valid) ^ set(int(v) for v in reference)
    outside = [i for i in difference if not (0 <= i < scores.numel()) or abs(float(scores[i]) - threshold) > band]
    return {'problems': problems, 'exact_equal': set(valid) == set(int(v) for v in reference),
            'outside_band': len(outside)}


def ulp_distance(a, b, torch):
    """Distance in BF16 representable steps between two BF16-valued FP32 tensors (sign-aware)."""
    ia = a.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    ib = b.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    ia = torch.where(ia < 0, -32768 - ia, ia)
    ib = torch.where(ib < 0, -32768 - ib, ib)
    return (ia - ib).abs()


def valid_scores_finite(scores, lengths, torch):
    """Only live scores must be finite; the kernel intentionally masks future positions to -inf."""
    if scores.ndim != 2 or lengths.numel() != scores.shape[0]:
        raise ValueError('Score/length geometry differs')
    if bool(((lengths < 0) | (lengths > scores.shape[1])).any()):
        raise ValueError('Score length outside the output width')
    for row, length in enumerate(lengths.tolist()):
        if not bool(torch.isfinite(scores[row, :length]).all()):
            raise FloatingPointError('non-finite live kernel scores')


def assert_repeat_equal(first, second, live_lengths, torch):
    for key in first:
        if key == 'scores':
            equal = all(torch.equal(first[key][r, :n], second[key][r, :n])
                        for r, n in enumerate(live_lengths.tolist()))
        else:
            equal = torch.equal(first[key], second[key])
        if not equal:
            raise FloatingPointError('repeated execution differs: ' + key)


def make_keys(distribution, count, seed, torch, device):
    g = torch.Generator(device=device).manual_seed(seed)
    if distribution == 'codebook':                                        # heavy exact ties
        book = torch.randn((64, 128), generator=g, device=device)
        pick = torch.randint(0, 64, (count,), generator=g, device=device)
        return book[pick].bfloat16()
    keys = torch.randn((count, 128), generator=g, device=device)
    if distribution == 'outlier':
        keys = torch.where(torch.rand(keys.shape, generator=g, device=device) < 0.01, keys * 30, keys)
    if distribution == 'low-magnitude':
        keys = keys * 1e-3
    return keys.bfloat16()


def choose_pages(blocks, count, generator, torch):
    high_floor = (2 ** 31) // BLOCK_STRIDE + 1
    if blocks <= high_floor + count:
        raise ValueError('pool too small to exercise offsets beyond 2 GiB')
    low = torch.randperm(high_floor - 1, generator=generator)[:count // 2] + 1
    high = torch.randperm(blocks - high_floor, generator=generator)[:count - count // 2] + high_floor
    return torch.cat((low, high))[torch.randperm(count, generator=generator)].long()


def selection_file(path, directory):
    """Private copy of the production selection file under its identity digest; returns the namespace."""
    payload = json.loads(Path(path).read_text())
    identity = payload['identity']
    name = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest() + '.json'
    shutil.copyfile(path, Path(directory) / name)
    return identity['namespace'], name


# ---- GPU execution (in-image only; untested outside the image) -------------------------------

def _config_dict(config):
    import dataclasses
    return dataclasses.asdict(config) if dataclasses.is_dataclass(config) else repr(config)


def production_config(plan, payload, device):
    from b12x.preparation import FrozenMapping, detect_device
    from b12x.preparation._cache import cache_identity, digest
    contract = plan.contract
    identity = cache_identity(payload['identity']['namespace'], device.index)
    if json.loads(json.dumps(identity)) != payload['identity']:
        raise LookupError('selection file identity differs from this device/namespace')
    configuration = contract.configure(plan.query, device=detect_device(device).identity,
                                       override=None, search=True)
    key = digest(dict(component=contract.component_id, query_schema=contract.query_schema_version,
                      config_schema=contract.config_schema_version, semantic_version=contract.semantic_version,
                      candidate_contract_version=contract.candidate_contract_version,
                      query=configuration.encoded_query.to_dict(), invocation=plan.invocation.to_dict(),
                      pin=None, dependencies=[]))
    record = payload['records'].get(key)
    if record is not None:
        assignment = FrozenMapping(record['assignment'])
        configuration.space.validate(assignment)
        config = contract._lower(configuration.query, configuration.device, assignment)
        if contract.config_payload(config).to_dict() != record['config']:
            raise LookupError('cached assignment no longer lowers to saved config')
        source, coverage = 'cached', record['coverage']
    else:
        iterator, found = contract.iterate(configuration), []
        while not iterator.done and len(found) < 2:
            try:
                item = iterator.step()
            except StopIteration:
                break
            if item is not None:
                found.append(item)
        if not (iterator.done and len(found) == 1):
            raise LookupError('no production selection record ' + key)
        config, source, coverage = found[0][1], 'fixed', None
    return config, dict(key=key, source=source, coverage=coverage,
                        config=contract.config_payload(config).to_dict())


def run_role(role, distribution, seed, *, torch, device, pool, blocks, session, selection_payload, shared=None):
    """One role on the decision chunk's final 256-row index chunk. `shared` carries the produce role's
    keys, page table and candidate outputs into the consume role (layers 24..36 share layer 20's cache)."""
    from b12x.attention import dsa_indexer
    from b12x.attention.dsa_indexer.mxfp4 import score_mxfp4, select_mxfp4
    from b12x.preparation import PreparedCall
    spec = ROLES[role]
    page, ratio = spec['page'], spec['ratio']
    rows, count = INDEX_CHUNK, PROMPT_TOKENS // ratio
    used = count // page
    view = pool.as_strided((blocks, page * 68), (BLOCK_STRIDE, 1), 0)
    g = torch.Generator().manual_seed(seed)
    caps = dsa_indexer.Caps(device=device, num_q_heads=HEADS, max_q_rows=rows, max_page_table_width=WIDTH,
                            topk=TOPK, mode='prefill', cache_format='mxfp4', page_size=page,
                            max_candidates=spec['max_candidates'], candidate_topk_blocks=spec['candidate_topk_blocks'])
    declared = dsa_indexer.plan(caps)
    config, production_selection = production_config(declared, selection_payload, device)
    plan = dsa_indexer.plan(caps, override=config)
    positions = torch.arange(PROMPT_TOKENS - rows, PROMPT_TOKENS)
    lengths = ((positions + 1) // ratio).int().to(device)
    table_pages = shared['table_pages'] if shared else choose_pages(blocks, used, g, torch)
    pages = torch.full((rows, WIDTH), -1, dtype=torch.int32)
    pages[:, :used] = table_pages.int()
    pages = pages.to(device)
    active = torch.tensor([WIDTH * page], dtype=torch.int32, device=device)
    score_width = None if role == 'consume' else min(WIDTH * page, count)
    q_mx = torch.zeros((rows, HEADS, 64), dtype=torch.uint8, device=device)          # finite priming input
    q_sc = torch.full((rows, HEADS, 4), 127, dtype=torch.uint8, device=device)
    weights = (torch.randn((rows, HEADS), generator=g) * 0.05).bfloat16().to(device)
    indices = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
    extra = {}
    if role == 'produce':
        extra = dict(candidate_output=torch.empty((rows, CANDIDATES), dtype=torch.int32, device=device),
                     candidate_output_lengths=torch.empty((rows,), dtype=torch.int32, device=device))
    elif role == 'consume':
        extra = dict(candidate_indices=shared['candidates'], candidate_lengths=shared['candidate_lengths'])
    common = dict(q_mxfp4=q_mx, q_scales=q_sc, query_weights=weights, index_k_cache=view, page_table=pages,
                  cache_lengths=lengths, active_width=active, score_width=score_width, output_indices=indices,
                  **extra)

    def prepare(state):                                                  # as vLLM _index_call
        scratch = [torch.empty(s.shape, dtype=s.dtype, device=device) for s in state.layout.scratch_specs()]
        binding = state.bind(scratch=scratch, **common)

        def run():
            score_mxfp4(binding, launchers=state._launchers)
            return select_mxfp4(binding, launchers=state._launchers)
        return PreparedCall(run=run, output=indices, owners=(scratch, binding))

    session.prepare((plan.request(name=f'dsa-{role}', prepare_call=prepare),))
    if (plan.selection.source != 'override' or
            plan.contract.config_payload(plan.selection.config) != plan.contract.config_payload(config)):
        raise RuntimeError('executed selection differs from verified production config')
    selection = {'source': plan.selection.source, 'config': _config_dict(plan.selection.config)}
    selection['production'] = production_selection
    if shared:
        keys = shared['keys']
    else:
        keys = make_keys(distribution, count, seed * 3 + 1, torch, device)
        slots = (table_pages.repeat_interleave(page) * page + torch.arange(page).repeat(used)).to(device)
        for start in range(0, count, INDEX_CHUNK):                       # production writer, 256 keys a call
            dsa_indexer.quantize_write_index_k_mxfp4(plan, keys[start:start + INDEX_CHUNK], index_k_cache=view,
                                                     slot_mapping=slots[start:start + INDEX_CHUNK])
    scale = 1e-3 if distribution == 'low-magnitude' else 1.0
    source = (torch.randn((rows, HEADS, 128), generator=g) * scale).bfloat16().to(device)
    dsa_indexer.quantize_q_mxfp4(plan, source, q_mxfp4=q_mx, q_scales=q_sc)
    scratch = [torch.empty(s.shape, dtype=s.dtype, device=device) for s in dsa_indexer.scratch_specs(plan, device=device)]
    live_lengths = shared['candidate_lengths'] if role == 'consume' else lengths

    def execute():
        indices.fill_(-2)
        if role == 'produce':
            extra['candidate_output'].fill_(-2)
            extra['candidate_output_lengths'].fill_(-2)
        binding = dsa_indexer.bind(plan, scratch=scratch, **common)
        current_scores = dsa_indexer.score(binding).clone()
        dsa_indexer.select(binding)
        torch.cuda.synchronize(device)
        outputs = {'scores': current_scores, 'indices': indices.clone()}
        if role == 'produce':
            outputs.update(candidates=extra['candidate_output'].clone(),
                           lengths=extra['candidate_output_lengths'].clone())
        for key, value in outputs.items():
            if key != 'scores' and bool((value == -2).any()):
                raise FloatingPointError('unwritten output canary: ' + key)
        valid_scores_finite(current_scores, live_lengths, torch)
        return outputs

    first = execute()
    for _ in range(2):
        repeated = execute()
        assert_repeat_equal(first, repeated, live_lengths, torch)
        del repeated
    scores = first['scores']
    # ---- reference over the bytes the kernels read ------------------------------------------
    written = view[table_pages.to(device)]
    decoded = key_rows(written, page, count, torch)
    oracle_data, oracle_scales = encode_mxfp4(keys, torch)
    layout_equal = bool(torch.equal(decoded, decode_mxfp4(oracle_data, oracle_scales, torch)))
    del oracle_data, oracle_scales
    decoded = decoded.float()                                            # BF16 values held once in FP32
    valid_scores_finite(scores, shared['candidate_lengths'] if role == 'consume' else lengths, torch)
    report = {'role': role, 'distribution': distribution, 'seed': seed, 'selection': selection,
              'layout_equal': layout_equal, 'identical_executions': 3, 'rows': []}
    for r in range(rows):
        visible = int(lengths[r])
        reference = published_scores(q_mx[r], q_sc[r], weights[r], decoded, torch)
        if not bool(torch.isfinite(reference).all()):
            raise FloatingPointError(f'{role}: non-finite reference scores')
        row = {'row': r, 'visible': visible}
        if role == 'consume':
            clen = int(shared['candidate_lengths'][r])
            pool_positions = shared['candidates'][r, :clen].long()
        else:
            pool_positions = torch.arange(visible, device=device)
            ulps = ulp_distance(scores[r, :visible].float(), reference[:visible], torch)
            row['score_bitwise_fraction'] = float((ulps == 0).float().mean())
            row['score_max_ulps'] = int(ulps.max())
        k = min(TOPK, int(pool_positions.numel()))
        expected = select_reference(reference, pool_positions, k, torch)
        threshold = float(reference[expected[-1]]) if k else float('-inf')
        row.update(compare_row(indices[r].tolist(), expected.tolist(), reference, threshold,
                               pool=pool_positions.tolist()))
        row['ties'] = tie_exposure(reference, pool_positions, indices[r][indices[r] >= 0], k, torch)
        if role == 'produce':
            positions_ref, _, _ = candidate_reference(reference, visible, torch)
            clen = int(extra['candidate_output_lengths'][r])
            produced = extra['candidate_output'][r, :clen]
            row['candidates_exact'] = bool(torch.equal(produced.long(), positions_ref.long()))
            row['candidates_length'] = (clen, int(positions_ref.numel()))
        report['rows'].append(row)
    rows_ = report['rows']
    report['summary'] = {
        'exact_rows': sum(r['exact_equal'] for r in rows_), 'rows': len(rows_),
        'structure_problem_rows': sum(bool(r['problems']) for r in rows_),
        'outside_band_rows': sum(r['outside_band'] > 0 for r in rows_),
        'min_score_bitwise_fraction': min((r['score_bitwise_fraction'] for r in rows_
                                           if 'score_bitwise_fraction' in r), default=None),
        'max_score_ulps': max((r['score_max_ulps'] for r in rows_ if 'score_max_ulps' in r), default=None),
        'max_tie_admitted': max(r['ties']['tie_admitted'] for r in rows_),
        'max_opposite_rule_changes': max(r['ties']['opposite_rule_changes'] for r in rows_),
        'candidate_exact_rows': sum(r.get('candidates_exact', True) for r in rows_),
    }
    out_shared = None
    if role == 'produce':
        out_shared = {'keys': keys, 'table_pages': table_pages, 'candidates': extra['candidate_output'],
                      'candidate_lengths': extra['candidate_output_lengths']}
    return report, out_shared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--selection', type=Path, required=True,
                        help='production router-namespace selection file (e.g. a post-boot snapshot)')
    parser.add_argument('--blocks', type=int, required=True)
    parser.add_argument('--distributions', default=','.join(DISTRIBUTIONS))
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args(argv)
    import torch
    from b12x.preparation import PreparationSession
    device = torch.device('cuda', torch.cuda.current_device())
    need = a.blocks * BLOCK_STRIDE
    if torch.cuda.mem_get_info(device)[0] < need + 8 * 1024 ** 3:
        raise SystemExit('insufficient free GPU memory for the production-shaped pool')
    pool = torch.zeros((need,), dtype=torch.uint8, device=device)
    failures, reports, started = [], [], time.time()
    with tempfile.TemporaryDirectory() as cache_dir:
        namespace, name = selection_file(a.selection, cache_dir)
        selection_payload = json.loads(a.selection.read_text())
        factory = lambda: PreparationSession(device=device, autotune=False,
                                             compile_workers=int(os.environ.get('B12X_STATE_COMPILE_WORKERS', '16')))
        for n, distribution in enumerate(a.distributions.split(',')):
            shared = None
            for role in ('full', 'produce', 'consume'):
                try:
                    with factory() as session:
                        report, produced = run_role(role, distribution, 9100 + 17 * n + len(role), torch=torch,
                                                    device=device, pool=pool, blocks=a.blocks, session=session,
                                                    selection_payload=selection_payload,
                                                    shared=shared if role == 'consume' else None)
                except LookupError as error:
                    failures.append(f'{role}/{distribution}: not a production plan ({error})')
                    break
                except FloatingPointError as error:
                    failures.append(f'{role}/{distribution}: {error}')
                    break
                if role == 'produce':
                    shared = produced
                s = report['summary']
                if not report['layout_equal']:
                    failures.append(f'{role}/{distribution}: written pages do not decode as the audit layout')
                if s['structure_problem_rows'] or s['outside_band_rows']:
                    failures.append(f'{role}/{distribution}: selection outside structure or band')
                if s['candidate_exact_rows'] != s['rows']:
                    failures.append(f'{role}/{distribution}: candidate production differs from the block rule')
                reports.append(report)
    out = {'scope': SCOPE, 'blocks': a.blocks, 'selection_file': str(a.selection), 'selection_name': name,
           'device': torch.cuda.get_device_name(device), 'failures': failures,
           'summaries': [{k: r[k] for k in ('role', 'distribution', 'selection', 'layout_equal', 'summary')}
                         for r in reports],
           'reports': reports, 'seconds': time.time() - started}
    a.out.write_text(json.dumps(out, indent=2, default=str) + '\n')
    print(json.dumps({k: out[k] for k in ('failures', 'summaries', 'scope')}, indent=2, default=str))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
