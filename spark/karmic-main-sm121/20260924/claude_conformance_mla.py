#!/usr/bin/env python3
"""Option-2 production-plan conformance for DS4.1 compressed sparse MLA (empirical envelope; not an oracle).

Runs on ONE idle GPU inside the router or capture image (b12x a7d7d29b). For
each of the 8 production plans of one decoded regime ({swa, draft, ratio1,
ratio2} x {extend 8192 rows, decode 60 rows}) it:
  1. builds Caps and invocation descriptors exactly as vLLM
     deepseek_v4_1/attention.py _declare_attention does, over a block-major
     pool of the production block count and block stride (227840 bytes), so
     physical offsets span the serving range (beyond 2 GiB);
  2. requires the resulting selection key to equal the production key from
     claude_decode_sparse_mla for (role, mode, blocks); otherwise it stops;
  3. executes the regime's recorded config as an explicit override
     (autotune off) through PreparationSession, the prepared state's
     scratch/bind/run (as b12x tests/attention/test_compressed_sparse_mla_v41.py),
     letting the prepared state supply cache_format and page sizes;
  4. models the decision chunk as production sees it: every row belongs to
     one request at consecutive positions ending at 524287, all rows share
     one request page table of unique physical blocks, SWA windows slide
     over the last 128 positions, and indexed selections are unique ascending
     logical entries of the visible range (stratified or clustered);
  5. compares sampled rows (always the last) with the pinned FP32 reference
     over the same records, physical slots resolved independently through
     the page table; and repeats each run on identical inputs requiring
     bitwise equality.

Fail closed: any non-finite kernel or reference output, any key mismatch,
any non-deterministic repeat, or degenerate reference rows beyond the
recorded count makes the run fail (exit 1) and no envelope is written.

Envelope: E[mode] = max(2 x p99.9, max) of per-head relative L2 over all
plans of that compute mode (also reported per plan), for
claude_decision_row_audit.py --envelope. Limits, recorded in the output:
empirical, on synthetic distributions (gaussian, wide-range, outlier,
sink-dominant) and finite samples; it describes kernel-vs-reference
distance under production plans only; it cannot reveal a bias shared by
kernel and reference, does not bound errors on real activations, and makes
no output correct.

  python3 claude_conformance_mla.py --regime decoded.json --blocks 80927 --out /writable/conformance.json
(decoded.json: claude_decode_sparse_mla.py --json output; the regime at --blocks is used.)
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
SCOPE = ('empirical conformance/sensitivity envelope on synthetic inputs at exact production plans; '
         'not a model oracle; cannot reveal a bias shared by kernel and reference; does not bound errors '
         'on real activations; no output is certified')
BLOCK_STRIDE, SWA_PAGE, HEADS, HEAD_DIM = 227840, 128, 16, 512
ROLES = {'swa': (128, 0), 'draft': (192, 0), 'ratio1': (128, 1), 'ratio2': (128, 2)}
MODES = {'extend': 8192, 'decode': 60}
PAGE_TABLE_WIDTH, PROMPT_TOKENS, TOPK = 2344, 524288, 512
SLOTS = ('swa.extend', 'swa.decode', 'draft.extend', 'draft.decode',
         'ratio1.extend', 'ratio1.decode', 'ratio2.extend', 'ratio2.decode')
DISTRIBUTIONS = ('gaussian', 'wide-range', 'outlier', 'sink-dominant')
DEGENERATE_NORM = 1e-12


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- pure planning (CPU-tested) ---------------------------------------------------------------

def regime_configs(decoded, blocks):
    """The 8 slot configs at `blocks` from claude_decode_sparse_mla --json output (all caches must agree)."""
    found = None
    for path, report in decoded.items():
        if path == 'relation':
            continue
        regime = report['regimes'].get(str(blocks))
        if regime is None:
            raise SystemExit(f'{path}: no decoded regime at {blocks} blocks')
        configs = regime['configs']
        if found is not None and configs != found:
            raise SystemExit('decoded caches disagree on the regime')
        found = configs
    if found is None or sorted(found) != sorted(SLOTS):
        raise SystemExit(f'regime at {blocks} blocks must have exactly the 8 production slots')
    return found


def role_facts(role, mode):
    width, ratio = ROLES[role]
    return {'role': role, 'mode': mode, 'rows': MODES[mode], 'swa_width': width, 'ratio': ratio,
            'indexed_width': 512 if ratio else 0, 'indexed_page': 256 // max(ratio, 1),
            'indexed_record_bytes': (256 // max(ratio, 1)) * 288 if ratio else 0,
            'swa_page_bytes': SWA_PAGE * 528}


def pool_layout(facts, blocks):
    """Byte offsets of the SWA and indexed views inside one block (block-major, stride 227840)."""
    indexed_offset = facts['swa_page_bytes']
    if indexed_offset + facts['indexed_record_bytes'] > BLOCK_STRIDE:
        raise ValueError('views do not fit one block stride')
    return {'blocks': blocks, 'stride': BLOCK_STRIDE, 'swa_offset': 0, 'indexed_offset': indexed_offset,
            'bytes': blocks * BLOCK_STRIDE}


def compile_workers():
    """Production policy (vLLM b12x_prepare): B12X_<STAGE>_COMPILE_WORKERS for the attention 'state' stage,
    else the session default B12X_COMPILE_WORKERS or 8. Worker count changes compile parallelism, not programs."""
    raw = os.environ.get('B12X_STATE_COMPILE_WORKERS')
    return int(raw) if raw is not None else int(os.environ.get('B12X_COMPILE_WORKERS', '8'))


def choose_pages(blocks, count, generator, torch):
    """Distinct physical blocks > 0 spanning the pool, half of them beyond the 2 GiB byte offset."""
    high_floor = (2 ** 31) // BLOCK_STRIDE + 1
    if blocks <= high_floor + count:
        raise ValueError('pool too small to exercise offsets beyond 2 GiB')
    low = torch.randperm(high_floor - 1, generator=generator)[:count // 2] + 1
    high = torch.randperm(blocks - high_floor, generator=generator)[:count - count // 2] + high_floor
    return torch.cat((low, high))[torch.randperm(count, generator=generator)].long()


def stratified_indices(visible, count, generator, torch, *, clustered=None):
    """Unique ascending logical entries, one per row: [rows] visible sizes -> [rows, count].

    Stratified: `count` equal strata of [0, visible) with one uniform draw each (covers the whole range,
    unique by construction, O(rows x count)). Clustered rows (bool mask): a contiguous run at a random start.
    Rows whose visible range is smaller than `count` take every entry, then -1.
    """
    visible = visible.long()
    rows = visible.numel()
    base = torch.arange(count).unsqueeze(0)
    width = (visible // count).clamp_min(1).unsqueeze(1)
    strat = base * width + (torch.rand(rows, count, generator=generator) * width).long()
    if clustered is not None:
        start = (torch.rand(rows, generator=generator) * (visible - count).clamp_min(0).add(1)).long().unsqueeze(1)
        strat = torch.where(clustered.unsqueeze(1), start + base, strat)
    small = visible < count
    strat = torch.where(small.unsqueeze(1), base.expand(rows, count), strat)
    return torch.where(base < visible.unsqueeze(1), strat, torch.full_like(strat, -1)).int()


def swa_windows(positions, width, swa_table, page):
    """Production _chunk: slots of logical positions [pos+1-width, pos] through the request SWA page table."""
    import torch
    logical = positions.long().unsqueeze(1) - (width - 1) + torch.arange(width).unsqueeze(0)
    valid = logical >= 0
    pages = swa_table[logical.clamp_min(0) // page]
    slots = torch.where(valid, pages * page + logical.clamp_min(0) % page, torch.full_like(logical, -1))
    lengths = valid.sum(dim=1)
    # production writes valid entries first (start = max(pos + 1 - width, 0))
    ordered = torch.sort(torch.where(valid, torch.arange(width).expand_as(logical), width), dim=1).indices
    return torch.gather(slots, 1, ordered).int(), lengths.int()


def envelope(per_head_errors):
    """{mode: E} with E = max(2 x p99.9, max) over all per-head relative L2 values of that mode."""
    out = {}
    for mode, values in per_head_errors.items():
        if values:
            out[mode] = max(2 * summarize(values)['p999'], max(values))
    return out


def summarize(values):
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]
    return {'count': len(ordered), 'median': pick(0.5), 'p99': pick(0.99), 'p999': pick(0.999), 'max': ordered[-1]}


def group_summaries(pairs):
    """[(distribution, value)] -> {distribution: summarize(values)} so plan-level tails can be attributed."""
    groups = {}
    for name, value in pairs:
        groups.setdefault(name, []).append(value)
    return {name: summarize(values) for name, values in sorted(groups.items())}


def relative_errors(got, expected, torch):
    """Per-head relative L2; raises on non-finite values; returns (errors, degenerate heads)."""
    if not bool(torch.isfinite(got).all()) or not bool(torch.isfinite(expected).all()):
        raise FloatingPointError('non-finite kernel or reference output')
    norms = expected.norm(dim=1)
    keep = norms > DEGENERATE_NORM
    rel = (got - expected).norm(dim=1)[keep] / norms[keep]
    return rel.tolist(), int((~keep).sum())


def row_sample(rows, count, generator, torch):
    """Deterministic sampled rows, always including the last (decision) row."""
    picked = torch.randperm(rows - 1, generator=generator)[:max(count - 1, 0)].tolist()
    return sorted({rows - 1, *picked})


# ---- GPU execution (in-image only; untested outside the image) -------------------------------

def fill_pages(view, pages, page, kind, scales, seed, reference, torch, *, outlier=False, chunk=64):
    """Write packed DS4.1 records for `pages` into the pool view, generated on the view's device in chunks."""
    g = torch.Generator(device=view.device).manual_seed(seed)
    record = 528 if kind == 'swa' else 288
    for start in range(0, pages.numel(), chunk):
        ids = pages[start:start + chunk].to(view.device)
        kv = torch.randn((ids.numel() * page, 512), generator=g, device=view.device) * scales.to(view.device)
        if outlier:
            mask = torch.rand(kv.shape, generator=g, device=view.device) < 0.01
            kv = torch.where(mask, kv * 30, kv)
        packed = reference.pack_deepseek_v41_cache_reference(kv.bfloat16(), page_size=page, cache_kind=kind)
        if packed.shape != (ids.numel(), page * record):
            raise RuntimeError('packed page geometry differs from the pool view')
        view[ids] = packed


def build_inputs(facts, layout, distribution, seed, torch, reference, swa_view, idx_view):
    g = torch.Generator().manual_seed(seed)
    rows = facts['rows']
    scales = torch.linspace(0.03, 0.7, 32).repeat_interleave(16)
    if distribution == 'wide-range':
        scales = torch.logspace(-3, 1, 32).repeat_interleave(16)
    positions = torch.arange(PROMPT_TOKENS - rows, PROMPT_TOKENS)
    swa_needed = (PROMPT_TOKENS + SWA_PAGE - 1) // SWA_PAGE
    swa_table = torch.zeros(swa_needed, dtype=torch.long)
    first = (PROMPT_TOKENS - rows - facts['swa_width']) // SWA_PAGE
    swa_pages = choose_pages(layout['blocks'], swa_needed - first, g, torch)
    swa_table[first:] = swa_pages
    fill_pages(swa_view, swa_pages, SWA_PAGE, 'swa', scales, seed * 7 + 1, reference, torch,
               outlier=distribution == 'outlier')
    swa_idx, swa_len = swa_windows(positions, facts['swa_width'], swa_table, SWA_PAGE)
    data = {'swa_indices': swa_idx, 'swa_lengths': swa_len}
    if facts['ratio']:
        page, ratio = facts['indexed_page'], facts['ratio']
        visible_max = PROMPT_TOKENS // ratio
        used_pages = (visible_max + page - 1) // page
        table = torch.full((PAGE_TABLE_WIDTH,), -1, dtype=torch.long)
        table[:used_pages] = choose_pages(layout['blocks'], used_pages, g, torch)
        fill_pages(idx_view, table[:used_pages], page, 'indexed', scales.flip(0) * 3, seed * 7 + 2, reference,
                   torch, outlier=distribution == 'outlier')
        visible = (positions + 1) // ratio
        clustered = torch.rand(rows, generator=g) < 0.25
        clustered[-1] = False
        logical = stratified_indices(visible, TOPK, g, torch, clustered=clustered)
        lengths = torch.full((rows,), TOPK, dtype=torch.int32)
        short = torch.rand(rows, generator=g) < (1 / 16)
        short[-1] = False                                      # the decision row is always full (512)
        lengths = torch.where(short, torch.randint(1, TOPK, (rows,), generator=g, dtype=torch.int32), lengths)
        logical = torch.where(torch.arange(TOPK).unsqueeze(0) < lengths.unsqueeze(1), logical,
                              torch.full_like(logical, -1))
        data.update(indexed_table=table.int().unsqueeze(0).expand(rows, -1).contiguous(),
                    indexed_logical=logical, indexed_lengths=lengths)
    q = torch.randn(rows, HEADS, HEAD_DIM, generator=g) * torch.linspace(0.05, 0.4, HEADS).view(1, HEADS, 1)
    sink = torch.linspace(-0.2, 0.15, HEADS) + (6.0 if distribution == 'sink-dominant' else 0.0)
    data.update(q=q.bfloat16(), sink=sink)
    return data


def run_plan(slot, config, blocks, *, trials, sample_rows, torch, device, pool, workers):
    from dataclasses import fields
    from b12x.attention import compressed_sparse_mla
    from b12x.attention.compressed_sparse_mla._tuning import TUNING, SparseMlaConfig
    from b12x.preparation import PreparationSession, PreparedCall
    decoder = _load('claude_conformance_decoder', 'claude_decode_sparse_mla.py')
    reference = _load('claude_conformance_reference', 'claude-pinned-b12x-compressed_reference.py')
    role, mode = slot.split('.')
    facts = role_facts(role, mode)
    layout = pool_layout(facts, blocks)
    swa_view = pool.as_strided((blocks, facts['swa_page_bytes']), (BLOCK_STRIDE, 1), layout['swa_offset'])
    idx_view = (pool.as_strided((blocks, facts['indexed_record_bytes']), (BLOCK_STRIDE, 1), layout['indexed_offset'])
                if facts['ratio'] else None)
    rows = facts['rows']
    q_buffer = torch.empty((rows, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(q_buffer)
    sink_buffer = torch.empty((HEADS,), dtype=torch.float32, device=device)
    caps = compressed_sparse_mla.Caps(
        device=device, num_q_heads=HEADS, max_q_rows=rows, max_width=facts['swa_width'] + facts['indexed_width'],
        swa_width=facts['swa_width'], indexed_width=facts['indexed_width'],
        swa_page_size=SWA_PAGE, indexed_page_size=facts['indexed_page'],
        max_page_table_width=PAGE_TABLE_WIDTH, mode=mode,
        decode_row_capacity=rows if mode == 'decode' else None, cache_format='deepseek_v41', use_cuda_graph=True)
    descriptor = lambda t: {'shape': tuple(t.shape), 'stride': tuple(t.stride()), 'alignment': 16,
                            'dtype': str(t.dtype).removeprefix('torch.')}
    invocation = compressed_sparse_mla.invocation_from_descriptors(
        q={'shape': (rows, HEADS, HEAD_DIM), 'stride': (HEADS * HEAD_DIM, HEAD_DIM, 1), 'alignment': 16,
           'dtype': 'bfloat16'},
        swa_cache=descriptor(swa_view), indexed_cache=descriptor(idx_view) if idx_view is not None else None,
        attn_sink_present=True, output_mode='provided')
    identity = compressed_sparse_mla.plan(caps, invocation=invocation)
    key = decoder.digest({'component': TUNING.component_id, 'query_schema': TUNING.query_schema_version,
                          'config_schema': TUNING.config_schema_version,
                          'semantic_version': TUNING.semantic_version,
                          'candidate_contract_version': TUNING.candidate_contract_version,
                          'query': TUNING.encode_query(identity.query), 'invocation': identity.invocation.to_dict(),
                          'pin': None, 'dependencies': []})
    if key != decoder.key(role, mode, blocks):
        raise SystemExit(f'{slot}: harness query is not the production query at {blocks} blocks')
    names = {f.name for f in fields(SparseMlaConfig)}
    if set(config) != names:
        raise SystemExit(f'{slot}: recorded config fields differ from SparseMlaConfig')
    execution = SparseMlaConfig(**config)
    plan = compressed_sparse_mla.plan(caps, invocation=invocation, override=execution)
    errors, degenerate, deterministic, scratch, seeds, tagged = [], 0, True, None, [], []
    with PreparationSession(device=device, autotune=False, compile_workers=workers) as session:
        for trial in range(trials):
            distribution = DISTRIBUTIONS[trial % len(DISTRIBUTIONS)]
            seed = 7000 + 131 * SLOTS.index(slot) + trial
            seeds.append({'trial': trial, 'seed': seed, 'distribution': distribution})
            data = build_inputs(facts, layout, distribution, seed, torch, reference, swa_view, idx_view)
            q_buffer.copy_(data['q'].to(device))
            sink_buffer.copy_(data['sink'].to(device))
            bind_args = dict(q=q_buffer, swa_indices=data['swa_indices'].to(device),
                             swa_lengths=data['swa_lengths'].to(device))
            run_args = dict(swa_k_cache=swa_view, sm_scale=512 ** -0.5, attn_sink=sink_buffer)
            if facts['ratio']:
                bind_args.update(indexed_indices=data['indexed_logical'].to(device),
                                 indexed_lengths=data['indexed_lengths'].to(device),
                                 indexed_page_table=data['indexed_table'].to(device))
                run_args.update(indexed_k_cache=idx_view)
            if scratch is None:
                def prepare(state, bind_args=bind_args, run_args=run_args):
                    (spec,) = state.scratch_specs()
                    trial_scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
                    trial_out = torch.empty_like(output)
                    binding = state.bind_for_preparation(scratch=trial_scratch, **bind_args)
                    return PreparedCall(run=lambda: state.run(binding, **run_args, out=trial_out),
                                        output=trial_out, owners=(trial_scratch, binding))
                session.prepare((plan.request(name=f'conformance-{slot}', prepare_call=prepare),))
                (spec,) = plan.scratch_specs()
                scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            binding = compressed_sparse_mla.bind(plan, scratch=scratch, **bind_args)
            output.fill_(float('nan'))
            compressed_sparse_mla.run(binding=binding, out=output, **run_args)
            first = output.clone()
            if not bool(torch.isfinite(first).all()):          # NaN canary: every position must be written
                raise FloatingPointError('non-finite or unwritten kernel output')
            output.fill_(float('nan'))
            compressed_sparse_mla.run(binding=binding, out=output, **run_args)
            deterministic &= bool(torch.equal(first, output))
            sample = row_sample(rows, sample_rows, torch.Generator().manual_seed(seed + 1), torch)
            for r in sample:
                kwargs = {}
                if facts['ratio']:
                    page = facts['indexed_page']
                    logical = data['indexed_logical'][r].long()
                    table = data['indexed_table'][r].long()
                    physical = torch.where(logical >= 0, table[logical.clamp_min(0) // page] * page + logical % page,
                                           torch.full_like(logical, -1)).int()
                    kwargs = dict(extra_k_cache=idx_view, extra_indices=physical[None].to(device),
                                  extra_topk_lengths=data['indexed_lengths'][r:r + 1].to(device),
                                  extra_page_size=page)
                expected = reference.compressed_sparse_mla_reference(
                    q_buffer[r:r + 1].float(), swa_view, bind_args['swa_indices'][r:r + 1],
                    bind_args['swa_lengths'][r:r + 1], sm_scale=512 ** -0.5, attn_sink=sink_buffer,
                    swa_page_size=SWA_PAGE, cache_format='deepseek_v41', **kwargs)[0].float()
                rel, zero = relative_errors(first[r].float(), expected, torch)
                errors.extend(rel)
                tagged.extend((distribution, value) for value in rel)
                degenerate += zero
    return {'slot': slot, 'config': config, 'key': key, 'compute_mode': execution.v41_compute_mode,
            'errors': summarize(errors), 'plan_envelope': envelope({'plan': errors}).get('plan'),
            'by_distribution': group_summaries(tagged), 'raw_tagged': tagged,
            'raw': errors, 'degenerate_heads': degenerate, 'deterministic': deterministic, 'seeds': seeds}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--regime', type=Path, required=True)
    parser.add_argument('--blocks', type=int, required=True)
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--sample-rows', type=int, default=48)
    parser.add_argument('--max-degenerate', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args(argv)
    if a.trials < 1 or a.sample_rows < 1:
        parser.error('--trials and --sample-rows must be positive')
    import torch
    configs = regime_configs(json.loads(a.regime.read_text()), a.blocks)
    device = torch.device('cuda', torch.cuda.current_device())
    need = a.blocks * BLOCK_STRIDE
    free = torch.cuda.mem_get_info(device)[0]
    if free < need + 8 * 1024 ** 3:
        raise SystemExit(f'insufficient free GPU memory for the production-shaped pool: need {need}, free {free}')
    pool = torch.zeros((need,), dtype=torch.uint8, device=device)
    workers = compile_workers()
    started = time.time()
    failures, results = [], []
    for slot in SLOTS:
        try:
            results.append(run_plan(slot, configs[slot], a.blocks, trials=a.trials, sample_rows=a.sample_rows,
                                    torch=torch, device=device, pool=pool, workers=workers))
        except FloatingPointError as error:
            failures.append(f'{slot}: {error}')
    per_mode, per_mode_distribution = {}, {}
    for row in results:
        per_mode.setdefault(row['compute_mode'], []).extend(row.pop('raw'))
        for name, value in row.pop('raw_tagged'):
            per_mode_distribution.setdefault(row['compute_mode'], []).append((name, value))
        if not row['deterministic']:
            failures.append(f'{row["slot"]}: repeat not bitwise identical')
        if row['degenerate_heads'] > a.max_degenerate:
            failures.append(f'{row["slot"]}: {row["degenerate_heads"]} degenerate reference heads')
    report = {'scope': SCOPE, 'blocks': a.blocks, 'block_stride': BLOCK_STRIDE, 'compile_workers': workers,
              'device': torch.cuda.get_device_name(device), 'trials': a.trials, 'sample_rows': a.sample_rows,
              'distributions': DISTRIBUTIONS, 'plans': results, 'failures': failures,
              'per_mode_samples': {m: len(v) for m, v in per_mode.items()},
              'per_mode_by_distribution': {m: group_summaries(v) for m, v in per_mode_distribution.items()},
              'envelope': None if failures else envelope(per_mode), 'seconds': time.time() - started}
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    if not failures:
        (a.out.with_suffix('.envelope.json')).write_text(json.dumps(report['envelope'], indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('envelope', 'failures', 'per_mode_samples', 'per_mode_by_distribution',
                                             'scope')}, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
