#!/usr/bin/env python3
"""Summarize actual GPU kernels, not overlapping synthetic profiler annotations."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re


def family(name):
    if 'MoEDynamicKernelSilu' in name:
        return 'target_dynamic_moe'
    if 'flashinfergemmkernelscute_dsl' in name and ('bf16_fp4' in name or 'Bf16Fp4' in name):
        return 'flashinfer_draft_head'
    if 'marlin_moe_wna16' in name:
        return 'marlin_draft_moe'
    if 'b12xcommroce' in name:
        return 'rocenante'
    if 'cutlass_80_wmma_tensorop_bf16_' in name:
        return 'bf16_wmma_gemm'
    if 'gdn_decode_recover' in name:
        return 'gdn_recovery_family'
    return 'other'


def summarize(path):
    with gzip.open(path, 'rt') as source:
        events = json.load(source)['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    forwards = [e for e in events if e.get('cat') == 'user_annotation'
                and e.get('ph') == 'X' and e.get('name', '').startswith('execute_context_')]
    if len(forwards) != 5 or not kernels:
        raise ValueError('Expected five worker forwards and nonempty GPU kernel capture')
    labels = {e['name'] for e in forwards}
    if len(labels) != 1 or not re.fullmatch(r'execute_context_0\(0\)_generation_(1\(4\)|4\(16\))', next(iter(labels))):
        raise ValueError('Capture is not homogeneous C1 or C4 K3 decode')
    by_name = defaultdict(lambda: {'calls': 0, 'kernel_us': 0.0})
    families = defaultdict(float)
    for e in kernels:
        by_name[e['name']]['calls'] += 1
        by_name[e['name']]['kernel_us'] += e['dur']
        families[family(e['name'])] += e['dur']
    total = sum(e['dur'] for e in kernels)
    intervals = sorted((e['ts'], e['ts'] + e['dur']) for e in kernels)
    first, end = intervals[0]
    busy = 0.0
    for start, stop in intervals[1:]:
        if start > end:
            busy += end - first
            first, end = start, stop
        else:
            end = max(end, stop)
    busy += end - first
    span = max(e['ts'] + e['dur'] for e in kernels) - intervals[0][0]
    return {
        'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'forward_count': len(forwards), 'forward_label': next(iter(labels)),
        'kernel_count': len(kernels), 'summed_kernel_us': total,
        'gpu_kernel_span_us': span, 'gpu_busy_union_us': busy,
        'idle_between_kernels_us': span - busy,
        'caveat': 'Instrumented capture; kernel sums overlap streams and are not wall time. Idle does not attribute scheduler cost.',
        'families': {name: {'kernel_us': us, 'percent_of_kernel_sum': 100 * us / total}
                     for name, us in sorted(families.items(), key=lambda pair: -pair[1])},
        'top_kernels': [{'name': name, **data} for name, data in
                        sorted(by_name.items(), key=lambda pair: -pair[1]['kernel_us'])[:20]],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    paths = sorted(args.receipt.glob('*.pt.trace.json.gz'))
    if len(paths) != 8:
        raise SystemExit('Expected exactly eight traces, four ranks at C1 and C4')
    records = [summarize(path) for path in paths]
    if any(sum(label in r['forward_label'] for r in records) != 4
           for label in ('generation_1(4)', 'generation_4(16)')):
        raise SystemExit('Wrong C1/C4 trace inventory')
    print(json.dumps(records, indent=2))
