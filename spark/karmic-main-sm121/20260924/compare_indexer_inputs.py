"""CPU-only comparison of layer-2 observation tensors, not an accuracy gate.

The caller must establish matched image, prompt, plans, rank, geometry and cold
controls before using this report. Equal rounded downstream tensors never prove
equal hidden inputs. A first observed difference does not identify its cause.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

FIELDS = ('hidden_input', 'kv_norm', 'q_rotated', 'index_query_rotated',
          'raw_weights', 'scaled_weights')


def tensor_difference(left, right):
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise ValueError('Expected tensors')
    if left.device.type != 'cpu' or right.device.type != 'cpu':
        raise ValueError('CPU captures only')
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('Shape or dtype differs')
    if left.ndim < 1 or left.numel() == 0 or not left.is_floating_point():
        raise ValueError('Expected nonempty floating tensor')
    a, b = left.contiguous(), right.contiguous()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite capture')
    bits = a.view(torch.uint8).reshape(a.shape[0], -1) != b.view(torch.uint8).reshape(b.shape[0], -1)
    rows = bits.any(dim=1).nonzero().flatten().tolist()
    diff = b.double() - a.double()
    norm = float(a.double().norm())
    return {'bit_equal': not rows, 'changed_rows': rows,
            'changed_bytes': int(bits.sum()),
            'changed_values': int((a != b).sum()),
            'max_abs': float(diff.abs().max()),
            'relative_l2': float(diff.norm()) / norm if norm else None,
            'shape': list(a.shape), 'dtype': str(a.dtype)}


def compare_tensors(left, right):
    """Compare mappings of the explicit hook arguments on the final 128 rows."""
    expected = torch.arange(524160, 524288, dtype=torch.int64)
    for entry in (left, right):
        positions = entry['positions']
        if (positions.device.type != 'cpu' or positions.dtype != torch.int64
                or not torch.equal(positions, expected)):
            raise ValueError('Not the final 128 positions of the exact 524K prompt')
        for key in FIELDS:
            if entry[key].shape[0] != 128:
                raise ValueError('Window length differs: ' + key)
        for key, shape in (('hidden_input', (128, 5120)),
                           ('raw_weights', (128, 32)), ('scaled_weights', (128, 32)),
                           ('projection_weight', (32, 5120))):
            if tuple(entry[key].shape) != shape or entry[key].dtype != torch.bfloat16:
                raise ValueError('Unexpected projection geometry or dtype: ' + key)
    weight = tensor_difference(left['projection_weight'], right['projection_weight'])
    if not weight['bit_equal']:
        raise ValueError('Different projection weights invalidate this input comparison')
    fields = {key: tensor_difference(left[key], right[key]) for key in FIELDS}
    if not fields['hidden_input']['bit_equal']:
        finding = 'hidden inputs differ; inspect the preceding compiled interval'
    elif not fields['raw_weights']['bit_equal']:
        finding = 'raw projection outputs differ on bit-identical captured inputs and weights'
    elif not fields['scaled_weights']['bit_equal']:
        finding = 'scaled outputs differ despite equal captured raw weights'
    else:
        finding = 'captured hidden input and index-weight path agree'
    return {'scope': 'descriptive observation, not an operator fault or answer-causality verdict',
            'finding': finding, 'fields': fields, 'projection_weight': weight,
            'limitations': ['capture covers the final 128 rows only',
                            'numerical plans and unarmed controls are checked separately',
                            'equal q or key products do not establish hidden-input equality']}


def compare_captures(left, right):
    from ds41_indexer_capture import validate
    lm, rm = validate(left), validate(right)
    if (lm['chunk_rows'], rm['chunk_rows']) != (8192, 4096):
        raise ValueError('Expected 8192 then 4096 captures')
    for key in ('rank', 'node', 'kit_sha256', 'source_trees', 'tp_world_size', 'hidden', 'heads', 'index_heads'):
        if lm.get(key) is None or lm[key] != rm.get(key):
            raise ValueError('Unmatched capture identity: ' + key)
    if (lm['tp_world_size'] != 4 or type(lm['rank']) is not int or lm['rank'] not in range(4)
            or lm['node'] != ('dusty', 'toby', 'rusty', 'kirby')[lm['rank']]):
        raise ValueError('Unexpected rank/world identity')
    result = compare_tensors(left['layer'], right['layer'])
    result['plans'] = {'left': left['layer']['plan'], 'right': right['layer']['plan']}
    result['identity'] = {k: lm[k] for k in ('rank', 'node', 'kit_sha256', 'source_trees')}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('left', type=Path)
    p.add_argument('right', type=Path)
    p.add_argument('--left-token', required=True)
    p.add_argument('--right-token', required=True)
    a = p.parse_args()
    captures = [torch.load(path, map_location='cpu', weights_only=True) for path in (a.left, a.right)]
    for path, token, capture in zip((a.left, a.right), (a.left_token, a.right_token), captures):
        meta = capture['meta']
        stem = f'{meta["node"]}-rank{meta["rank"]}-{token}'
        if (meta['generation'] != token or path.name != stem + '-indexer.pt'
                or meta['decision_row_file'] != stem + '.pt' or meta['window_file'] != stem + '-window.pt'):
            raise ValueError('Capture token or sibling names differ from the receipt')
    result = compare_captures(*captures)
    result['files'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.left, a.right)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
