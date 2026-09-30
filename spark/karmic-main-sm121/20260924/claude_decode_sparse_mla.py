#!/usr/bin/env python3
"""Decode DS4.1 compressed-sparse-MLA selection keys into (serving blocks, role, mode). Offline, stdlib only.

A selection key is sha256 of {component, schemas, query, invocation, pin,
dependencies} (b12x a7d7d29b session._choice_key). For
attention.compressed_sparse_mla the query and invocation are rebuilt from
vLLM 1794dcf1 deepseek_v4_1/attention.py _declare_attention and b12x
compressed_sparse_mla/_preparation.py _query, with the launch profile's
constants (TP4 -> 16 local heads, max_num_batched_tokens 8192, max_num_seqs 4,
7 DSpark tokens with parallel drafting -> decode rows 60, max_model_len
600000, block 256, SWA block 128, window 128, DSpark SWA width 192).

Each boot declares one plan per (role, mode); every layer of a role shares
it:
  swa        main layers with compress_ratio 0 (SWA only, width 128)
  draft      DSpark draft layers (SWA only, width 192)
  ratio1     compress_ratio 1 (indexed page 256 tokens, 73728 bytes)
  ratio2     compress_ratio 2 (indexed page 128 tokens, 36864 bytes)
  mode       extend (8192 rows) or decode (60 rows)
The live caches enter only through their descriptors: shape [blocks, page
bytes] and stride [block stride, 1] of the block-major allocation
(block stride 227840 bytes, SWA page 67584 bytes). All of this was
confirmed by exact key matches against retained caches, not assumed.
A key is decoded only on an exact sha256 match, so every label is exact.

  python3 claude_decode_sparse_mla.py CACHE.json [...] [--blocks N ...] [--scan LO HI] [--json out.json]
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

COMPONENT = 'attention.compressed_sparse_mla'
HEADS, HEAD_DIM = 16, 512
EXTEND_ROWS, DECODE_ROWS = 8192, 60
MAX_MODEL_LEN, BLOCK = 600000, 256
SWA_PAGE_TOKENS, SWA_PAGE_BYTES = 128, 128 * 528
BLOCK_STRIDE = 227840
ROLES = {  # role: (swa_width, compress_ratio, indexed page bytes)
    'swa': (128, 0, None),
    'draft': (192, 0, None),
    'ratio1': (128, 1, (BLOCK // 1) * 288),
    'ratio2': (128, 2, (BLOCK // 2) * 288),
}
MODES = (('extend', EXTEND_ROWS), ('decode', DECODE_ROWS))
SIGNATURE = ('max_chunks_per_row', 'single_pass', 'split_chunk_size', 'v41_compute_mode', 'v41_heads_per_block')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
                          .encode()).hexdigest()


def _descriptor(blocks, page_bytes, stride=BLOCK_STRIDE):
    return {'shape': [blocks, page_bytes], 'stride': [stride, 1], 'alignment': 16, 'dtype': 'uint8'}


def query_and_invocation(role, mode, blocks):
    width, ratio, indexed_bytes = ROLES[role]
    rows = dict(MODES)[mode]
    q = {'shape': [rows, HEADS, HEAD_DIM], 'stride': [HEADS * HEAD_DIM, HEAD_DIM, 1], 'alignment': 16,
         'dtype': 'bfloat16'}
    swa = _descriptor(blocks, SWA_PAGE_BYTES)
    indexed = _descriptor(blocks, indexed_bytes) if ratio else None
    invocation = {'q': q, 'swa_k_cache': swa, 'indexed_k_cache': indexed, 'attn_sink_present': True,
                  'return_lse': False, 'lse_scale': 'base2', 'output_mode': 'provided'}
    query = {
        'cache_format': 'deepseek_v41', 'layout': 'compressed_dsv4', 'mode': mode, 'q_dtype': 'bfloat16',
        'kv_dtype': 'uint8', 'num_q_heads': HEADS, 'qk_head_dim': HEAD_DIM, 'v_head_dim': HEAD_DIM,
        'swa_width': width, 'swa_page_size': SWA_PAGE_TOKENS, 'indexed_width': 512 if ratio else 0,
        'indexed_page_size': BLOCK // max(ratio, 1), 'query_rows': rows, 'max_batch': rows, 'max_kv_rows': 0,
        'max_page_table_width': -(-MAX_MODEL_LEN // BLOCK), 'max_q_chunks': None,
        'decode_row_capacity': rows if mode == 'decode' else None, 'use_cuda_graph': True,
        'q_shape': q['shape'], 'q_stride': q['stride'], 'q_alignment': 16,
        'swa_cache_shape': swa['shape'], 'swa_cache_stride': swa['stride'], 'swa_cache_alignment': 16,
        'indexed_cache_present': indexed is not None,
        'indexed_cache_shape': None if indexed is None else indexed['shape'],
        'indexed_cache_stride': None if indexed is None else indexed['stride'],
        'indexed_cache_alignment': None if indexed is None else 16,
        'attn_sink_present': True, 'return_lse': False, 'lse_scale': 'base2', 'output_mode': 'provided'}
    return query, invocation


def key(role, mode, blocks):
    query, invocation = query_and_invocation(role, mode, blocks)
    return digest({'component': COMPONENT, 'query_schema': 5, 'config_schema': 4, 'semantic_version': 1,
                   'candidate_contract_version': 5, 'query': query, 'invocation': invocation, 'pin': None,
                   'dependencies': []})


def is_sparse_mla(record):
    return tuple(sorted(record['config'])) == SIGNATURE


def decode(records, blocks=(), scan=(1, 200000)):
    """Map sparse-MLA keys to (blocks, role, mode). Explicit block counts first, then a scan."""
    targets = {k for k, r in records.items() if is_sparse_mla(r)}
    found = {}
    candidates = list(dict.fromkeys([*blocks, *range(*scan)])) if scan else list(blocks)
    for count in candidates:
        if len(found) == len(targets):
            break
        probe = key('swa', 'extend', count)
        if probe not in targets and count not in blocks:
            continue  # every boot declares the swa extend plan; skip counts it cannot explain
        for role in ROLES:
            for mode, _ in MODES:
                k = key(role, mode, count)
                if k in targets:
                    found[k] = (count, role, mode)
    return found, sorted(targets - set(found))


def regimes(records, found):
    """{blocks: {role.mode: config}} for every decoded capacity."""
    out = {}
    for k, (count, role, mode) in found.items():
        out.setdefault(count, {})[f'{role}.{mode}'] = records[k]['config']
    return {count: dict(sorted(v.items())) for count, v in sorted(out.items())}


def main_regime(regime):
    """Target-model precision choices only (draft layers do not produce the served logprobs)."""
    return tuple(sorted((slot, regime[slot]['v41_compute_mode'], regime[slot]['v41_heads_per_block'])
                        for slot in regime if not slot.startswith('draft.')))


def relate(boots, decoded):
    """boots: [(label, blocks, modal)]; decoded: {blocks: regime}. Test regime <-> modal one-to-one."""
    rows, missing = [], []
    for label, count, modal in boots:
        if count not in decoded:
            missing.append(label)
            continue
        rows.append({'boot': label, 'blocks': count, 'modal': modal, 'regime': main_regime(decoded[count]),
                     'draft': {s: decoded[count][s]['v41_compute_mode'] for s in decoded[count] if s.startswith('draft.')}})
    violations = []
    for i, x in enumerate(rows):
        for y in rows[i + 1:]:
            if (x['regime'] == y['regime']) != (x['modal'] == y['modal']):
                violations.append((x['boot'], y['boot'], 'same regime' if x['regime'] == y['regime'] else 'different regime',
                                   'same modal' if x['modal'] == y['modal'] else 'different modal'))
    return {'boots': rows, 'missing': missing, 'distinct_regimes': len({r['regime'] for r in rows}),
            'distinct_modals': len({r['modal'] for r in rows}), 'violations': violations,
            'consistent': not violations and not missing}


def precision(regime):
    return {slot: f"{c['v41_compute_mode']}/h{c['v41_heads_per_block']}" for slot, c in regime.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('caches', nargs='+', type=Path)
    parser.add_argument('--blocks', type=int, nargs='*', default=[])
    parser.add_argument('--scan', type=int, nargs=2, default=[1, 200000], metavar=('LO', 'HI'))
    parser.add_argument('--boot', action='append', default=[], metavar='LABEL=BLOCKS=MODAL',
                        help='relate a boot (serving blocks, modal response signature) to its decoded regime')
    parser.add_argument('--json', type=Path)
    a = parser.parse_args(argv)
    report = {}
    for path in a.caches:
        records = json.loads(path.read_text())['records']
        found, undecoded = decode(records, a.blocks, tuple(a.scan))
        report[str(path)] = {'sparse_mla_records': sum(map(is_sparse_mla, records.values())),
                             'decoded': len(found), 'undecoded': undecoded,
                             'regimes': {str(b): {'configs': r, 'precision': precision(r)}
                                         for b, r in regimes(records, found).items()}}
    if a.boot:
        merged = {}
        for path in a.caches:
            records = json.loads(path.read_text())['records']
            found, _ = decode(records, a.blocks, tuple(a.scan))
            for count, regime in regimes(records, found).items():
                if merged.setdefault(count, regime) != regime:
                    raise SystemExit(f'caches disagree on the regime at {count} blocks')
        boots = []
        for item in a.boot:
            label, count, modal = item.split('=')
            boots.append((label, int(count), modal))
        report['relation'] = relate(boots, merged)
    text = json.dumps(report, indent=2, sort_keys=True)
    if a.json:
        a.json.write_text(text + '\n')
    for path, r in report.items():
        if path == 'relation':
            print('relation consistent', r['consistent'], 'regimes', r['distinct_regimes'], 'modals',
                  r['distinct_modals'], 'violations', r['violations'], 'missing', r['missing'])
            continue
        print(path, 'decoded', r['decoded'], '/', r['sparse_mla_records'], 'undecoded', len(r['undecoded']))
        for b, reg in r['regimes'].items():
            print('  blocks', b, reg['precision'])
    ok = all(not r['undecoded'] for p, r in report.items() if p != 'relation')
    return 0 if ok and report.get('relation', {'consistent': True})['consistent'] else 1


if __name__ == '__main__':
    sys.exit(main())
