"""Offline FP32 order study, not a runtime fix or whole-model correctness gate.

Run only on trusted four-rank window captures, on a CPU host or idle node.
The FP64 reference is not called exact without checking a sufficient exponent
bound. FP32 accumulation is explicitly tested, never assumed invariant.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch


def fp32_references(parts):
    for order in itertools.permutations(range(4)):
        result = parts[order[0]].float()
        for rank in order[1:]:
            result = result + parts[rank].float()
        yield 'sequential_' + ''.join(map(str, order)), result.to(torch.bfloat16)
    for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
        result = (parts[a].float() + parts[b].float()) + (parts[c].float() + parts[d].float())
        yield f'pair_{a}{b}_{c}{d}', result.to(torch.bfloat16)


def bit_equal(a, b):
    return a.contiguous().view(torch.int16) == b.contiguous().view(torch.int16)


def analyze(parts):
    if (len(parts) != 4 or any(p.dtype != torch.bfloat16 or p.device.type != 'cpu'
                              or p.ndim != 2 or p.shape != parts[0].shape
                              or not torch.isfinite(p).all() for p in parts)):
        raise ValueError('Four finite same-shape CPU BF16 matrices required')
    values = torch.stack([p.double() for p in parts])
    exponents = torch.frexp(values.abs())[1]
    nonzero = values != 0
    high = exponents.masked_fill(~nonzero, -3000).max(dim=0).values
    low = exponents.masked_fill(~nonzero, 3000).min(dim=0).values
    span = torch.where(nonzero.any(dim=0), high - low, 0)
    # BF16 needs at most 8 significant bits; four-term addition needs at most
    # two carry bits. Span <= 43 therefore fits the 53-bit FP64 significand,
    # including all partial sums. This sufficient condition is conservative.
    exact_bound = bool((span <= 43).all())
    reference = values.sum(dim=0).to(torch.bfloat16)
    first = None
    differing = torch.zeros_like(reference, dtype=torch.bool)
    wrong_reference = torch.zeros_like(reference, dtype=torch.bool)
    reports = {}
    for name, result in fp32_references(parts):
        if first is None:
            first = result
        unequal = ~bit_equal(result, first)
        mismatch = ~bit_equal(result, reference)
        differing |= unequal
        wrong_reference |= mismatch
        error = (result.double() - reference.double()).abs()
        reports[name] = {
            'nonfinite_output_elements': int((~torch.isfinite(result)).sum()),
            'bitwise_unequal_elements_to_first_order': int(unequal.sum()),
            'bitwise_unequal_elements_to_fp64_round_once': int(mismatch.sum()),
            'max_abs_to_fp64_round_once': float(error.max()) if torch.isfinite(error).all() else None,
        }
    return {
        'shape': list(reference.shape), 'orders_tested': len(reports),
        'max_nonzero_exponent_span': int(span.max()),
        'fp64_exactness_sufficient_bound_pass': exact_bound,
        'fp32_significand_sufficient_bound_pass': bool((span <= 14).all()),
        'all_orders_bit_equal_after_bf16_round': not bool(differing.any()),
        'all_orders_match_fp64_round_once': not bool(wrong_reference.any()),
        'elements_with_order_dependence': int(differing.sum()),
        'elements_mismatching_reference_in_any_order': int(wrong_reference.sum()),
        'rows_with_order_dependence': torch.nonzero(differing.any(dim=1)).flatten().tolist(),
        'orders': reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, nargs=4, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    captures = [torch.load(p, map_location='cpu', weights_only=True) for p in args.captures]
    meta = captures[0]['meta']
    for rank, capture in enumerate(captures):
        m = capture['meta']
        if (capture['schema'] != 'claude-window-v1' or m['rank'] != rank
                or m['tp_world_size'] != 4 or m['problems']):
            raise ValueError('Unexpected capture identity')
        for key in ('chunk_rows', 'kit_sha256', 'source_trees', 'prompt_tokens', 'row_position'):
            if m[key] != meta[key]:
                raise ValueError('Rank identity mismatch: ' + key)
    result = {str(layer): analyze([c['layers'][layer]['wo_partial'] for c in captures]) for layer in (0, 1)}
    print(json.dumps({
        'scope': 'FP32 sum order sensitivity on captured operands, not a serving qualification',
        'chunk_rows': meta['chunk_rows'], 'result': result,
        'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.captures},
    }, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
