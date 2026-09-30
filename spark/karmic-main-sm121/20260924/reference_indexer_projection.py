"""CPU-only arithmetic references for saved, matched indexer captures.

These references do not emulate cuBLAS or establish final-answer causality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from compare_indexer_inputs import compare_captures, tensor_difference


def references(hidden, weight):
    x, w = hidden.double(), weight.double()
    reference = x @ w.T
    # Independent ordered reduction checks the FP64 BLAS result before rounding.
    ordered = torch.zeros_like(reference)
    for k in range(x.shape[1]):
        ordered += x[:, k:k+1] * w[:, k][None, :]
    if not torch.equal(reference.bfloat16(), ordered.bfloat16()):
        raise ValueError('FP64 reduction orders disagree after BF16 rounding')
    result = {'fp64_round_once': reference.bfloat16()}
    for block in (16, 64, 256, 512, 1024):
        total = torch.zeros_like(reference, dtype=torch.float32)
        for k in range(0, x.shape[1], block):
            total += (x[:, k:k+block] @ w[:, k:k+block].T).float()
        result[f'fp32_partial_{block}'] = total.bfloat16()
    return reference, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('left', type=Path)
    p.add_argument('right', type=Path)
    args = p.parse_args()
    captures = [torch.load(f, map_location='cpu', weights_only=True)
                for f in (args.left, args.right)]
    comparison = compare_captures(*captures)
    if not comparison['fields']['hidden_input']['bit_equal']:
        raise ValueError('Inputs differ')
    layer = captures[0]['layer']
    precise, variants = references(layer['hidden_input'], layer['projection_weight'])
    out = {'scope': 'CPU reference only; no runtime intervention or cuBLAS emulation',
           'files': {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in (args.left, args.right)}, 'arms': {}}
    for name, capture in zip(('8192', '4096'), captures):
        raw = capture['layer']['raw_weights']
        out['arms'][name] = {
            'references': {key: tensor_difference(value, raw)
                           for key, value in variants.items()},
            'max_abs_error_to_fp64': float((raw.double()-precise).abs().max()),
            'mean_abs_error_to_fp64': float((raw.double()-precise).abs().mean()),
        }
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
