"""CPU-only arithmetic analysis of trusted, matched four-rank window captures.

Reference enumeration can identify a rounding association, not prove which
communication backend or instruction sequence ran. No model or GPU execution.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch


def references(parts):
    if len(parts) != 4 or any(p.dtype != torch.bfloat16 for p in parts):
        raise ValueError('Four BF16 rank partials required')
    yield 'fp64_sum_round_once', sum(p.double() for p in parts).to(torch.bfloat16)
    for order in itertools.permutations(range(4)):
        value = parts[order[0]].clone()
        for rank in order[1:]:
            value = (value.float() + parts[rank].float()).to(torch.bfloat16)
        yield 'bf16_sequential_' + ''.join(map(str, order)), value
    for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
        x = (parts[a].float() + parts[b].float()).to(torch.bfloat16)
        y = (parts[c].float() + parts[d].float()).to(torch.bfloat16)
        yield f'bf16_pair_{a}{b}_{c}{d}', (x.float() + y.float()).to(torch.bfloat16)


def analyze(parts, outputs):
    result = {name: {'references': {}, 'exact_rows_explained_by_any_reference': []} for name in outputs}
    explained = {name: torch.zeros(parts[0].shape[0], dtype=torch.bool) for name in outputs}
    for ref_name, ref in references(parts):
        for name, got in outputs.items():
            equal = got.contiguous().view(torch.uint8) == ref.contiguous().view(torch.uint8)
            rows = equal.reshape(got.shape[0], -1).all(dim=1)
            explained[name] |= rows
            delta = got.double() - ref.double()
            result[name]['references'][ref_name] = {
                'exact_rows': torch.nonzero(rows).flatten().tolist(),
                'unequal_values': int((got != ref).sum()),
                'max_abs': float(delta.abs().max()),
                'relative_l2': float(delta.norm() / ref.double().norm().clamp_min(1e-30)),
            }
    for name in outputs:
        result[name]['exact_rows_explained_by_any_reference'] = torch.nonzero(explained[name]).flatten().tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', nargs=4, type=Path, required=True)
    parser.add_argument('--right', nargs=4, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    groups = [[torch.load(p, map_location='cpu', weights_only=True) for p in paths]
              for paths in (args.left, args.right)]
    for expected_chunk, group in zip((8192, 4096), groups):
        for rank, capture in enumerate(group):
            if (capture['schema'] != 'claude-window-v1' or capture['meta']['rank'] != rank
                    or capture['meta']['chunk_rows'] != expected_chunk or capture['meta']['problems']):
                raise ValueError('Unexpected capture identity')
    left, right = groups
    for a, b in zip(left, right):
        for key in ('kit_sha256', 'source_trees', 'prompt_tokens', 'row_position'):
            if a['meta'][key] != b['meta'][key]:
                raise ValueError('Unmatched ' + key)
    parts = [c['layers'][0]['wo_partial'] for c in left]
    for rank in range(4):
        if not torch.equal(parts[rank].view(torch.uint8), right[rank]['layers'][0]['wo_partial'].view(torch.uint8)):
            raise ValueError('Rank partials differ between grids')
    for group in groups:
        ref = group[0]['layers'][0]['wo_reduced']
        if any(not torch.equal(ref.view(torch.uint8), c['layers'][0]['wo_reduced'].view(torch.uint8)) for c in group):
            raise ValueError('Reduced outputs differ across ranks')
    result = analyze(parts, {'8192': left[0]['layers'][0]['wo_reduced'],
                             '4096': right[0]['layers'][0]['wo_reduced']})
    print(json.dumps({'scope': 'Layer 0 arithmetic association analysis, not backend attribution',
                      'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.left + args.right},
                      'result': result}, indent=2))


if __name__ == '__main__':
    main()
