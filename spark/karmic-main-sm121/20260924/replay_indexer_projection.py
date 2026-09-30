"""Isolated GPU diagnostic. Never run alongside serving on the selected GPU.

Only the captured final 128 rows are replayed; other rows are zero. Failure to
match serving outputs means this is not yet a faithful serving reproducer.
No model is loaded and no serving process or persistent configuration is edited.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import torch

from compare_indexer_inputs import compare_captures, tensor_difference
from reference_indexer_projection import references


@contextmanager
def reduction_mode(enabled):
    backend = torch.backends.cuda.matmul
    original = backend.allow_bf16_reduced_precision_reduction
    try:
        if enabled is not None:
            backend.allow_bf16_reduced_precision_reduction = enabled
        yield
    finally:
        backend.allow_bf16_reduced_precision_reduction = original


def place_rows(hidden, rows, device):
    if hidden.ndim != 2 or rows < hidden.shape[0]:
        raise ValueError('Invalid replay geometry')
    out = torch.zeros((rows, hidden.shape[1]), dtype=hidden.dtype, device=device)
    out[-hidden.shape[0]:].copy_(hidden)
    return out


def load_pair(paths, digests):
    captures = []
    for path, expected in zip(paths, digests):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Capture digest mismatch: ' + str(path))
        captures.append(torch.load(path, map_location='cpu', weights_only=True))
    comparison = compare_captures(*captures)
    if not comparison['fields']['hidden_input']['bit_equal']:
        raise ValueError('Replay requires identical inputs')
    if any(c['layer']['plan']['selection']['config'].get('backend') != 'torch'
           for c in captures):
        raise ValueError('This replay only represents the captured torch backend')
    return captures


def run_pair(captures, reference, repeats, mode):
    weight = captures[0]['layer']['projection_weight'].cuda()
    results, outputs = {}, {}
    with reduction_mode(mode):
        for rows, capture in zip((8192, 4096), captures):
            x = place_rows(capture['layer']['hidden_input'], rows, 'cuda')
            out = torch.empty((rows, weight.shape[0]), dtype=weight.dtype, device='cuda')
            samples = []
            for _ in range(repeats):
                torch.mm(x, weight.T, out=out)
                torch.cuda.synchronize()
                samples.append(out[-128:].cpu().clone())
            outputs[rows] = samples[0]
            results[str(rows)] = {
                'sample_sha256': [hashlib.sha256(bytes(s.contiguous().view(torch.uint8).flatten().tolist())).hexdigest()
                                  for s in samples],
                'input_stride': list(x.stride()), 'weight_stride': list(weight.stride()),
                'output_stride': list(out.stride()),
                'input_alignment_mod256': x.data_ptr() % 256,
                'weight_alignment_mod256': weight.data_ptr() % 256,
                'output_alignment_mod256': out.data_ptr() % 256,
                'repeat_comparisons': [tensor_difference(samples[0], s) for s in samples[1:]],
                'serving_capture': tensor_difference(capture['layer']['raw_weights'], samples[0]),
                'fp64_round_once': tensor_difference(reference, samples[0]),
            }
    return {'arms': results, 'cross_grid': tensor_difference(outputs[8192], outputs[4096])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('left', type=Path)
    p.add_argument('right', type=Path)
    p.add_argument('--left-sha256', required=True)
    p.add_argument('--right-sha256', required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--test-no-reduced-precision', action='store_true')
    args = p.parse_args()
    if args.repeats < 2:
        p.error('At least two repeats are required')
    captures = load_pair((args.left, args.right), (args.left_sha256, args.right_sha256))
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('A visible SM121 GPU is required; no CPU fallback')
    layer = captures[0]['layer']
    _, refs = references(layer['hidden_input'], layer['projection_weight'])
    report = {'scope': 'isolated projection, not whole-model qualification',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'device': torch.cuda.get_device_name(),
              'capture_sha256': [args.left_sha256, args.right_sha256],
              'original_reduced_precision': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              'environment': {key: os.environ.get(key) for key in (
                  'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF',
                  'CUBLASLT_WORKSPACE_SIZE', 'CUBLAS_WORKSPACE_CONFIG',
                  'CUBLASLT_LOG_LEVEL', 'CUBLASLT_LOG_FILE',
                  'CUBLAS_LOGINFO_DBG', 'CUBLAS_LOGDEST_DBG')},
              'variants': {}}
    variants = [('baseline', None)]
    if args.test_no_reduced_precision:
        variants += [('no_reduced_precision', False), ('baseline_return', None)]
    for name, mode in variants:
        report['variants'][name] = run_pair(captures, refs['fp64_round_once'], args.repeats, mode)
        print('INDEXER-REPLAY-VARIANT ' + name, flush=True)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
