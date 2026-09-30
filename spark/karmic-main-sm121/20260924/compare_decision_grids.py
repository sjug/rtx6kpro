"""CPU-only descriptive comparison of one rank's saved decision-row captures.

This is NOT a correctness gate or proof that the experiments are comparable.
The caller must separately verify frozen prompt/token hashes, image and kit
identities, numerical plans, KV capacity, cold-cache accounting, and two
unarmed controls matching the armed response on EACH boot. Captures do not
contain all that evidence. Per-operator conformance uses each operator's own
inputs, never the differences reported here. The earliest captured difference
is not necessarily the causal layer: prior KV writes and unobserved operations
are outside this comparison. No SSH, CUDA, serving, or cache mutation occurs.

Usage: python compare_decision_grids.py left-rank.pt right-rank.pt
Output is JSON on stdout. Only trusted local, weights-only captures are loaded.
The reviewed v3 capture hook supports explicit 8192 and 4096 chunk grids.
These comparison helpers do not themselves authorize or launch a live capture.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from claude_decision_row_audit import decode_mxfp4, load_reference


def tensor_delta(left, right):
    if left.device.type != 'cpu' or right.device.type != 'cpu':
        raise ValueError('offline CPU tensors required')
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('tensor shape/dtype mismatch')
    a, b = left.double(), right.double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('nonfinite tensor cannot be summarized as a valid difference')
    diff = b - a
    norm_a, norm_b = float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b))
    norm_diff = float(torch.linalg.vector_norm(diff))
    changed = int(torch.count_nonzero(left != right))
    return {'shape': list(left.shape), 'dtype': str(left.dtype), 'equal_values': changed == 0,
            'changed_elements': changed, 'elements': left.numel(),
            'left_l2': norm_a, 'right_l2': norm_b, 'difference_l2': norm_diff,
            'relative_l2_to_left': norm_diff / norm_a if norm_a else None,
            'max_abs': float(diff.abs().max()) if diff.numel() else 0.0}


def selection_delta(left, right):
    def unpack(t):
        if t.device.type != 'cpu' or t.ndim != 1 or t.is_floating_point():
            raise ValueError('logical selection must be a CPU integer vector')
        values = t.tolist()
        selected = [v for v in values if v >= 0]
        if len(set(selected)) != len(selected) or any(v < -1 for v in values):
            raise ValueError('invalid or duplicate logical indices')
        if values[:len(selected)] != selected:
            raise ValueError('selection padding must be a tail')
        return selected
    a, b = unpack(left), unpack(right)
    sa, sb = set(a), set(b)
    return {'equal_ordered': a == b, 'left_count': len(a), 'right_count': len(b),
            'intersection': len(sa & sb), 'jaccard': len(sa & sb) / len(sa | sb) if sa | sb else 1.0,
            'left_only': sorted(sa - sb), 'right_only': sorted(sb - sa)}


def swa_delta(left, right):
    """Compare active chronological SWA records, never physical slot addresses."""
    n = int(left['swa_len'])
    if n != int(right['swa_len']) or not 0 < n <= 128:
        raise ValueError('SWA active window mismatch')
    for entry in (left, right):
        records = entry['swa_records']
        if (records.device.type != 'cpu' or records.dtype != torch.uint8
                or records.ndim != 2 or records.shape[1] != 528 or records.shape[0] < n):
            raise ValueError('invalid SWA records')
    a, b = (entry['swa_records'][:n].contiguous() for entry in (left, right))
    ref = load_reference()
    decoded = [ref.unpack_deepseek_v41_cache_reference(t, page_size=1, cache_kind='swa')
               for t in (a, b)]
    return {'active_rows': n, 'raw_bytes': tensor_delta(a, b),
            'decoded_values': tensor_delta(*decoded),
            'raw_row_multiset_equal': Counter(bytes(row.tolist()) for row in a)
                                     == Counter(bytes(row.tolist()) for row in b),
            'changed_row_offsets': torch.nonzero(torch.any(a != b, dim=1)).flatten().tolist(),
            'scope': 'active records in captured read order; prior writes are not validated'}


def _validate(capture):
    if capture.get('schema') != 'claude-decision-row-v3':
        raise ValueError('unsupported capture schema')
    meta = capture['meta']
    rows = meta['chunk_rows']
    if (rows not in (8192, 4096) or meta['row_index'] != rows - 1
            or meta['batch_requests'] != 1 or meta['prompt_tokens'] != 524288
            or meta['row_position'] != 524287 or meta['problems']):
        raise ValueError('capture geometry or reported problems')
    if set(capture['layers']) != set(range(40)):
        raise ValueError('all 40 target layers required')
    for layer, entry in capture['layers'].items():
        expected_rows = 128 if layer >= 20 else rows
        if (entry['position'] != 524287 or entry['query_rows'] != expected_rows
                or entry['row'] != expected_rows - 1 or entry['ced_decoder'] != (layer >= 20)):
            raise ValueError(f'layer {layer}: wrong decision-row geometry')


def compare_rank(left, right):
    _validate(left)
    _validate(right)
    for key in ('rank', 'node', 'kit_sha256', 'source_trees', 'prompt_tokens', 'row_position'):
        if not left['meta'].get(key) and key != 'rank':
            raise ValueError(f'missing identity: {key}')
        if left['meta'].get(key) != right['meta'].get(key):
            raise ValueError(f'capture identity differs: {key}')
    layers, first = {}, None
    first_q, first_out, first_swa = None, None, None
    for layer in range(40):
        a, b = left['layers'][layer], right['layers'][layer]
        if a['plan'] != b['plan']:
            raise ValueError(f'layer {layer}: plans differ')
        record = {key: tensor_delta(a[key], b[key]) for key in ('q', 'out', 'attn_sink')}
        if first_q is None and not record['q']['equal_values']:
            first_q = layer
        if first_out is None and not record['out']['equal_values']:
            first_out = layer
        differs = any(not item['equal_values'] for item in record.values())
        if ('swa_records' in a) != ('swa_records' in b):
            raise ValueError(f'layer {layer}: SWA record presence differs')
        if 'swa_records' in a:
            record['swa'] = swa_delta(a, b)
            if not record['swa']['decoded_values']['equal_values']:
                if first_swa is None:
                    first_swa = layer
                differs = True
        if ('indexer' in a) != ('indexer' in b):
            raise ValueError(f'layer {layer}: indexer presence differs')
        if 'indexer' in a:
            x, y = a['indexer'], b['indexer']
            if x['cache_length'] != y['cache_length'] or x['page_size'] != y['page_size']:
                raise ValueError(f'layer {layer}: indexer geometry differs')
            indexer = {
                'query': tensor_delta(decode_mxfp4(x['q_data'], x['q_scales']),
                                      decode_mxfp4(y['q_data'], y['q_scales'])),
                'weights': tensor_delta(x['weights'], y['weights']),
                'topk': selection_delta(x['topk'], y['topk']),
                'key_pages_hash_equal': x['key_pages_sha256'] == y['key_pages_sha256']}
            differs |= (not indexer['query']['equal_values'] or not indexer['weights']['equal_values']
                        or not indexer['topk']['equal_ordered'] or not indexer['key_pages_hash_equal'])
            if ('candidates' in x) != ('candidates' in y):
                raise ValueError(f'layer {layer}: candidate presence differs')
            if 'candidates' in x:
                lengths = [int(z['candidate_len']) for z in (x, y)]
                if any(n < 0 or n > z['candidates'].numel() for n, z in zip(lengths, (x, y))):
                    raise ValueError(f'layer {layer}: invalid candidate length')
                indexer['candidate_lengths'] = lengths
                indexer['candidates'] = selection_delta(x['candidates'][:lengths[0]],
                                                         y['candidates'][:lengths[1]])
                differs |= not indexer['candidates']['equal_ordered']
            record['indexer'] = indexer
        if first is None and differs:
            first = layer
        layers[layer] = record
    return {'scope': 'descriptive-only; experiment comparability not established',
            'rank': left['meta']['rank'], 'left_chunk_rows': left['meta']['chunk_rows'],
            'right_chunk_rows': right['meta']['chunk_rows'],
            'first_differing_captured_layer': first,
            'first_differing_query_layer': first_q,
            'first_differing_output_layer': first_out,
            'first_differing_swa_values_layer': first_swa, 'layers': layers,
            'limitations': ['not an independent model oracle or a correctness verdict',
                            'earliest observed difference need not be the causal layer',
                            'earlier cache writes, full hidden states and logits are not compared',
                            'SWA read order is compared; earlier writes are not validated',
                            'selected indexed attention-cache record bytes are not compared',
                            'physical page addresses are deliberately not compared across boots',
                            'index key hashes are reported, not revalidated against bytes here']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    args = parser.parse_args()
    captures = [torch.load(p, map_location='cpu', weights_only=True) for p in (args.left, args.right)]
    result = compare_rank(*captures)
    result['files'] = {}
    for path in (args.left, args.right):
        with path.open('rb') as stream:
            result['files'][str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
