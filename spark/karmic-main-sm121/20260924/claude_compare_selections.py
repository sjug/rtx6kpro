#!/usr/bin/env python3
"""Compare B12X preparation selections of a parent boot (A) and a seeded candidate boot (B). Offline, stdlib only.

Inputs are the per-node selection cache files (the tuning JSON with
{"identity", "records"}) and, optionally, the same node's startup logs.
Selection keys are sha256 digests of {component, schemas, query, invocation,
pin, dependencies}; records carry only assignment/config/coverage/programs.
So the helper:

  * decodes the router projection (gemm.bf16_gemv, 384x5120 bf16 -> fp32)
    and every norm.mhc key by rebuilding the key from source (b12x a7d7d29b,
    vLLM 1794dcf1 DS4.1 invocation) and reports winners per plan;
  * compares every key present in both files: a config or assignment change
    is config drift (same key means same query); a programs change alone is
    expected when the package fingerprint differs and is not drift;
  * classifies keys new in B by config family. New keys in a family whose
    query carries KV capacity (compressed sparse MLA: max_kv_rows,
    page-table width, cache shapes; DSA indexer: max_k_rows, paged logits
    rows) are capacity-dependent bind queries only if the logs show a
    different serving KV block count than A; with the same count they
    should have been cache hits and are unexplained. New keys in any other
    family, and keys lost from the seed, are drift-class;
  * extracts serving KV blocks/tokens and per-component ready lines
    (measured/cached counts) from the logs.

Keys are opaque, so a new capacity-family key cannot be paired with the A
query it replaces; the report says so and gives the winner multisets.
Every verdict is conditional on the files being the ones each boot used.

  python3 claude_compare_selections.py --parent A-dusty-tuning.json --candidate B-dusty-tuning.json \
      [--seed B-seed.json] [--parent-before A-seed.json] [--parent-log A.log] [--candidate-log B.log] [--json out.json]
Exit 0 identical or capacity-only, 1 drift, 3 unresolved (see verdict).
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

GATE_ROWS_RANGE = range(1, 8193)
MHC_CODEGEN = {'_MHC_PDL': False, '_PREFILL_THREADS': 512, '_PREFILL_GRAM_THREADS': 1024,
               '_PREFILL_TMA_COMPUTE_WARPS': 8, '_PREFILL_TMA_THREADS': 288, '_PREFILL_TMA_TILE_M': 128,
               '_PREFILL_TMA_TILE_N': 16, '_PREFILL_TMA_TILE_K': 64, '_PREFILL_TMA_STAGES': 3,
               '_PREFILL_FINALIZE_THREADS': 256}
MHC_QUERY = dict(dtype='bfloat16', hidden_size=5120, split_k=80, has_norm_weight=False,
                 norm_weight_dtype='bfloat16', has_fn_bf16=False, collapse_weighted=False, lagged_mix=False,
                 expanded_residual=False, bf16x2_eligible=True, output_mode='provided', rms_eps=1e-6,
                 hc_eps=1e-6, sinkhorn_iters=20, norm_eps=0.0, block_k=256, block_h=512, smem_limit=101376,
                 controls={}, codegen=MHC_CODEGEN)
DS41_RMS_EPS = 1e-20
# Config field signature -> component (field sets from each component's *Config at a7d7d29b).
FAMILIES = {
    ('backend', 'rows_per_tile'): 'gemm.bf16_gemv',
    ('backend', 'fused_merge', 'mxfp4_score_kind'): 'attention.dsa_indexer',
    ('backend', 'large_m_unroll', 'load_path', 'split_k_slices', 'swap_ab', 'target_occupancy', 'tile_k',
     'tile_m', 'tile_n'): 'gemm (dense/block_fp8)',
    ('backend', 'lagged_prepare', 'partials_per_cta', 'projection_k_splits', 'projection_num_m_warps',
     'projection_num_n_warps', 'projection_num_stages', 'projection_tile_k', 'projection_tile_m',
     'projection_tile_n'): 'norm.mhc',
    ('max_chunks_per_row', 'single_pass', 'split_chunk_size', 'v41_compute_mode', 'v41_heads_per_block'):
        'attention.compressed_sparse_mla',
    ('backend', 'dynamic_route_mode', 'dynamic_tile_m', 'max_active_clusters', 'nvfp4_materialize_intermediate',
     'nvfp4_share_input', 'route_planner', 'w4a16_block_size_m', 'w4a16_pipeline_stages', 'w4a16_route_mode', 'w4a16_tile_config'):
        'moe.fused_moe',
    ('backend', 'decode_tile_n'): 'gemm.wo_projection',
    ('algorithm', 'backend', 'block_k', 'num_warps'): 'gemm.bf16_vocab_projection',
    ('tile_m', 'tile_n'): 'attention.varlen',
}
CAPACITY_FAMILIES = {'attention.compressed_sparse_mla', 'attention.dsa_indexer'}
READY = re.compile(r'b12x ready (\S+): (\d+)/(\d+) ready, (?:candidates \S+ prepared, )?(\d+) measured, (\d+) cached')
BLOCKS = re.compile(r'before allocating (\d+) serving blocks')
KV_TOKENS = re.compile(r'GPU KV cache size: ([\d,]+) tokens')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
                          .encode()).hexdigest()


def gate_key(rows):
    """session._choice_key for the DS4.1 router projection (same as claude_replay_gate_prefill.choice_key)."""
    query = {'source_dtype': 'bfloat16', 'weight_dtype': 'bfloat16', 'max_rows': int(rows), 'in_features': 5120,
             'out_features': 384, 'source_contiguous': True, 'source_aligned': True, 'weight_contiguous': True,
             'weight_aligned': True, 'output_dtype': 'float32', 'output_contiguous': True,
             'output_aligned': True, 'bias_dtype': None}
    return digest({'component': 'gemm.bf16_gemv', 'query_schema': 4, 'config_schema': 5, 'semantic_version': 1,
                   'candidate_contract_version': 3, 'query': query, 'invocation': {}, 'pin': None,
                   'dependencies': []})


def mhc_invocation(operation, expanded):
    if operation == 'post':
        return {'operation': 'post', 'output_mode': 'provided'}
    return {'lagged_mix': True, 'has_norm_weight': True, 'rms_eps': DS41_RMS_EPS, 'hc_eps': 1e-6,
            'sinkhorn_iters': 20, 'norm_eps': DS41_RMS_EPS, 'operation': operation, 'has_fn_bf16': False,
            'expanded_residual': expanded}


def mhc_key(operation, tokens, expanded=False):
    """session._choice_key for plan_mhc with the DS4.1 invocation (b12x_layers.py get_b12x_preparation_units)."""
    invocation = mhc_invocation(operation, expanded)
    query = dict(MHC_QUERY, max_tokens=int(tokens), **invocation)
    return digest({'component': 'norm.mhc', 'query_schema': 8, 'config_schema': 4, 'semantic_version': 1,
                   'candidate_contract_version': 13, 'query': query, 'invocation': invocation, 'pin': None,
                   'dependencies': []})


def _index():
    table = {gate_key(rows): ('router', rows) for rows in GATE_ROWS_RANGE}
    for tokens in GATE_ROWS_RANGE:
        for label, operation, expanded in (('mhc.pre', 'pre', False), ('mhc.pre.expanded', 'pre', True),
                                           ('mhc.post_pre', 'post_pre', False), ('mhc.post', 'post', False)):
            table[mhc_key(operation, tokens, expanded)] = (label, tokens)
    return table


_KNOWN = None


def known():
    global _KNOWN
    if _KNOWN is None:
        _KNOWN = _index()
    return _KNOWN


def family(record):
    return FAMILIES.get(tuple(sorted(record['config'])), 'unknown:' + ','.join(sorted(record['config'])))


def load(path):
    payload = json.loads(Path(path).read_text())
    if set(payload) != {'identity', 'records'}:
        raise SystemExit(f'{path}: not a selection cache file')
    return payload


def decode(records):
    """Router and mHC winners by plan, from one file."""
    table, out = known(), {}
    for key, record in records.items():
        if key in table:
            label, rows = table[key]
            out.setdefault(label, {})[rows] = {'key': key, 'config': record['config'],
                                               'assignment': record['assignment']}
    undecoded_mhc = [k for k, r in records.items() if family(r) == 'norm.mhc' and k not in table]
    return {label: dict(sorted(plans.items())) for label, plans in sorted(out.items())}, undecoded_mhc


def parse_log(path):
    text = Path(path).read_text(errors='replace')
    ready = {}
    for m in READY.finditer(text):
        ready[f'{m.group(1)}@{m.group(3)}'] = {'measured': int(m.group(4)), 'cached': int(m.group(5))}
    blocks = sorted({int(b) for b in BLOCKS.findall(text)})
    tokens = sorted({int(t.replace(',', '')) for t in KV_TOKENS.findall(text)})
    return {'serving_blocks': blocks, 'kv_tokens': tokens, 'ready': ready}


def compare(parent, candidate, *, seed=None, parent_before=None, parent_log=None, candidate_log=None):
    a, b = parent['records'], candidate['records']
    report = {'identity_equal': parent['identity'] == candidate['identity'],
              'counts': {'parent': len(a), 'candidate': len(b)}}
    drift, unresolved, notes = [], [], []
    if not report['identity_equal']:
        drift.append('cache identity differs: B could not have read the seeded records')
    if seed is not None:
        s = seed['records']
        report['seed'] = {'equals_parent': s == a, 'identity_equal': seed['identity'] == parent['identity'],
                          'config_equal_parent': {k: (r['config'], r['assignment']) for k, r in s.items()}
                          == {k: (r['config'], r['assignment']) for k, r in a.items()}}
        if not report['seed']['config_equal_parent']:
            drift.append('seed selections differ from the parent file')
    common = a.keys() & b.keys()
    config_drift = sorted(k for k in common if a[k]['config'] != b[k]['config']
                          or a[k]['assignment'] != b[k]['assignment'])
    report['common'] = {
        'keys': len(common), 'config_or_assignment_changed': len(config_drift),
        'programs_changed': sum(a[k]['programs'] != b[k]['programs'] for k in common),
        'coverage_changed': sum(a[k]['coverage'] != b[k]['coverage'] for k in common)}
    report['config_drift'] = [{'key': k, 'family': family(b[k]), 'decoded': known().get(k),
                               'parent': {f: a[k][f] for f in ('config', 'assignment')},
                               'candidate': {f: b[k][f] for f in ('config', 'assignment')}} for k in config_drift]
    if config_drift:
        drift.append(f'{len(config_drift)} same-query selections changed config/assignment')
    lost = sorted(a.keys() - b.keys())
    report['lost_from_parent'] = [{'key': k, 'family': family(a[k]), 'decoded': known().get(k)} for k in lost]
    if lost:
        drift.append(f'{len(lost)} parent selections absent from the candidate file (not seeded or rewritten)')

    logs = {}
    if parent_log:
        logs['parent'] = parse_log(parent_log)
    if candidate_log:
        logs['candidate'] = parse_log(candidate_log)
    report['logs'] = logs
    capacity = None
    if len(logs) == 2 and logs['parent']['serving_blocks'] and logs['candidate']['serving_blocks']:
        capacity = 'differs' if logs['parent']['serving_blocks'] != logs['candidate']['serving_blocks'] else 'same'
    report['capacity'] = capacity or 'unknown'

    new = sorted(b.keys() - a.keys())
    classes = Counter()
    rows = []
    for k in new:
        fam = family(b[k])
        if known().get(k) or fam not in CAPACITY_FAMILIES:
            kind = 'drift: new capacity-independent query'
        elif capacity == 'differs':
            kind = 'capacity-dependent bind query (conditional)'
        elif capacity == 'same':
            kind = 'unexplained: capacity family missed cache at equal capacity'
        else:
            kind = 'unresolved: capacity family, capacity unknown'
        classes[kind] += 1
        rows.append({'key': k, 'family': fam, 'decoded': known().get(k), 'class': kind, 'config': b[k]['config']})
    report['new_in_candidate'] = rows
    # Log "N measured" counts candidate measurements, not choices: compare with coverage.measured_count.
    report['new_measured_by_family'] = {
        f: sum(b[r['key']]['coverage'].get('measured_count', 0) for r in rows if r['family'] == f)
        for f in sorted({r['family'] for r in rows})}
    report['new_classes'] = dict(classes)
    for kind, n in classes.items():
        if kind.startswith('drift'):
            drift.append(f'{n} new capacity-independent selections in B')
        elif kind.startswith('unexplained'):
            unresolved.append(f'{n} capacity-family selections re-measured at equal KV capacity')
        elif kind.startswith('unresolved'):
            unresolved.append(f'{n} capacity-family selections new in B, KV capacity not established from logs')

    if parent_before is not None:
        a_new = [a[k] for k in a.keys() - parent_before['records'].keys()]
        winners = lambda recs: sorted(Counter((family(r), json.dumps(r['config'], sort_keys=True))
                                              for r in recs).items())
        report['boot_new_winners'] = {'parent_boot': winners(a_new), 'candidate_boot': winners([b[k] for k in new]),
                                      'note': 'unpaired: keys are opaque, capacities may differ'}

    decoded_a, undecoded_a = decode(a)
    decoded_b, undecoded_b = decode(b)
    winners = {}
    for label in sorted(set(decoded_a) | set(decoded_b)):
        plans = sorted(set(decoded_a.get(label, {})) | set(decoded_b.get(label, {})))
        winners[label] = {}
        for rows in plans:
            pa, pb = decoded_a.get(label, {}).get(rows), decoded_b.get(label, {}).get(rows)
            same = pa is not None and pb is not None and pa['config'] == pb['config'] \
                and pa['assignment'] == pb['assignment']
            winners[label][str(rows)] = {'parent': pa and pa['config'], 'candidate': pb and pb['config'],
                                         'same': same}
            if not same:
                drift.append(f'{label} m{rows}: winner differs or plan missing on one side')
    report['decoded'] = winners
    report['undecoded_mhc'] = {'parent': len(undecoded_a), 'candidate': len(undecoded_b)}
    if undecoded_a or undecoded_b:
        notes.append('some norm.mhc keys did not decode with the DS4.1 invocation (env controls or codegen differ?)')
    if 'router' in decoded_b:
        plans = decoded_b['router']
        report['router_plan_for_rows'] = {str(r): (r if r in plans else max(plans)) for r in (254, 256, 258)}

    if drift:
        verdict = 'drift'
    elif unresolved:
        verdict = 'unresolved'
    elif new:
        verdict = 'capacity-only (conditional)'
    else:
        verdict = 'identical-selections'
    report.update(verdict=verdict, issues=drift + unresolved, drift=drift, unresolved=unresolved, notes=notes)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--parent-before', type=Path)
    parser.add_argument('--parent-log', type=Path)
    parser.add_argument('--candidate-log', type=Path)
    parser.add_argument('--json', type=Path)
    a = parser.parse_args(argv)
    report = compare(load(a.parent), load(a.candidate), seed=a.seed and load(a.seed),
                     parent_before=a.parent_before and load(a.parent_before),
                     parent_log=a.parent_log, candidate_log=a.candidate_log)
    report['files'] = {k: str(v) for k, v in vars(a).items() if isinstance(v, Path)}
    text = json.dumps(report, indent=2, sort_keys=True)
    if a.json:
        a.json.write_text(text + '\n')
    summary = {k: report[k] for k in ('verdict', 'issues', 'notes', 'capacity', 'counts', 'common', 'new_classes')}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return {'identical-selections': 0, 'capacity-only (conditional)': 0, 'drift': 1}.get(report['verdict'], 3)


if __name__ == '__main__':
    sys.exit(main())
