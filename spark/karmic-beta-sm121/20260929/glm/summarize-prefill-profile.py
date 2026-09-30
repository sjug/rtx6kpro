#!/usr/bin/env python3
"""Compare GPU kernel time between two torch-profiler prefill captures (one trace per rank).

Only GPU kernel events are summed; overlapping CPU and user annotations are ignored. The
window is the profiled request's forward passes: every kernel between the first and last
`execute_context_*` annotation that carries prefill tokens. Families are coarse name
classes; unmatched kernels stay visible in the per-name table.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re

FAMILIES = [
    ('nccl_allreduce', r'ncclDevKernel'),
    ('rocenante', r'b12xcommroce'),
    ('b12x_moe', r'b12xmoe|MoEDynamicKernel'),
    ('moe_router_other', r'vllm::moe::|marlin|moe_align|topk_softmax'),
    ('mhc', r'b12xnormmhc|mhc'),
    ('mla_prefill', r'mlaprefill|flash_mla|sparse_mla'),
    ('dsa_indexer', r'dsa_indexer|indexer|kpool'),
    ('kda', r'flash_kda|kda|causal_conv1d|layer_norm_gated|fwht'),
    ('bf16_cutlass80_gemm', r'cutlass_80_tensorop_bf16|cutlass_80_wmma'),
    ('nvjet_cublas_gemm', r'nvjet|gemvx|cutlass_80_simt|cublas'),
    ('b12x_dense_gemm', r'b12x_libdense|DenseGemm'),
    ('torch_elementwise', r'at::native'),
    ('triton_other', r'^_|triton'),
]


def family(name):
    for label, pattern in FAMILIES:
        if re.search(pattern, name):
            return label
    return 'other'


def load(path):
    with gzip.open(path, 'rt') as source:
        return json.load(source)['traceEvents']


def summarize(path):
    events = load(path)
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    steps = [e for e in events if e.get('cat') == 'user_annotation' and e.get('ph') == 'X'
             and e.get('name', '').startswith('execute_context')]
    by_name = defaultdict(lambda: [0, 0.0])
    fam = defaultdict(float)
    for e in kernels:
        by_name[e['name']][0] += 1
        by_name[e['name']][1] += e['dur']
        fam[family(e['name'])] += e['dur']
    span = (max(e['ts'] + e['dur'] for e in kernels) - min(e['ts'] for e in kernels)) if kernels else 0
    return {'path': str(path), 'kernels': len(kernels), 'kernel_us': sum(e['dur'] for e in kernels),
            'span_us': span, 'steps': sorted({e['name'] for e in steps}), 'step_count': len(steps),
            'families': dict(fam), 'by_name': {k: v for k, v in by_name.items()}}


def arm(root):
    traces = sorted(Path(root).rglob('*.json.gz'))
    traces = [t for t in traces if 'capture_traces' not in t.parts]
    return [summarize(t) for t in traces]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline')
    parser.add_argument('candidate')
    parser.add_argument('--top', type=int, default=25)
    args = parser.parse_args()
    a, b = arm(args.baseline), arm(args.candidate)
    report: dict = {'baseline': [{k: v for k, v in s.items() if k != 'by_name'} for s in a],
              'candidate': [{k: v for k, v in s.items() if k != 'by_name'} for s in b]}
    def total(side, key):
        out = defaultdict(float)
        for s in side:
            for k, v in s[key].items():
                out[k] += v[1] if isinstance(v, list) else v
        return out
    fa, fb = total(a, 'families'), total(b, 'families')
    ranks_a, ranks_b = max(len(a), 1), max(len(b), 1)
    report['families_ms_per_rank'] = {
        k: {'baseline': fa[k] / ranks_a / 1e3, 'candidate': fb[k] / ranks_b / 1e3,
            'delta_ms': (fb[k] / ranks_b - fa[k] / ranks_a) / 1e3}
        for k in sorted(set(fa) | set(fb), key=lambda k: -(fb[k] / ranks_b - fa[k] / ranks_a))}
    na, nb = total(a, 'by_name'), total(b, 'by_name')
    report['top_names_baseline'] = sorted(((v / ranks_a / 1e3, k[:160]) for k, v in na.items()), reverse=True)[:args.top]
    report['top_names_candidate'] = sorted(((v / ranks_b / 1e3, k[:160]) for k, v in nb.items()), reverse=True)[:args.top]
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
