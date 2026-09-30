"""CPU-only descriptive analysis of final-window captures (schema claude-window-v1).

This is NOT a correctness gate or proof that two experiments are comparable.
The caller separately verifies frozen prompt/token hashes, image and kit
identities, numerical plans, KV capacity, cold-cache accounting and matched
unarmed controls on EACH boot. Nothing here is an independent model oracle.

Within one capture (each grid on its own):
  reference_rows   the pinned CPU attention reference on every captured row of a layer, from the row's own
                   captured query and the records its window read (page_size 1 over the 256 gathered records);
                   conformance of the kernel per row, on this grid's inputs
  packed_rows      the pinned CPU record encoder on kv_rot against the record captured at the row's write slot:
                   whether the packed write equals the reference packing of the rotated KV it was handed
  slot_rows        write_slots against the last row's chronological window and the page offsets of positions
  decision_consistency  the last captured row against the decision-row capture of the SAME boot (bit equality)
Across two captures of one rank (left grid, right grid), per layer and boundary
in producer order: hidden_in, kv_norm, kv_rot, records of the decision window
(decoded, by position), records of the preceding window, q, out, attn_sink,
wo_partial, wo_reduced. Per-row changed counts and relative L2 give the row
profile; the first differing boundary is the earliest OBSERVED difference,
never the causal operator: the compiled interval between wo_reduced of layer 0
and hidden_in of layer 1 (o-proj epilogue, mHC, FFN) is bracketed, not opened.
Across ranks: replicated tensors (hidden_in, kv_norm, kv_rot, wo_reduced,
positions) hashed for equality; per-rank tensors reported, not expected equal.

Usage:
  python claude-window-compare.py compare LEFT.pt RIGHT.pt [--decision-left D.pt --decision-right D.pt]
  python claude-window-compare.py ranks RANK0.pt RANK1.pt RANK2.pt RANK3.pt
Only trusted local, weights-only captures are loaded. No SSH, CUDA or serving.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claude_decision_row_audit import load_reference  # noqa: E402

SCHEMA, WINDOW, RECORDS, SWA_RECORD, KV_DIM, HEAD_DIM = 'claude-window-v1', 128, 256, 528, 512, 512
LAYERS = (0, 1)
SM_SCALE = 512 ** -0.5
REPLICATED = ('positions', 'hidden_in', 'kv_norm', 'kv_rot', 'wo_reduced')
PER_RANK = ('q', 'out', 'attn_sink', 'wo_partial')
ORDER = ('hidden_in', 'kv_norm', 'kv_rot', 'records_decision_window', 'records_preceding_window', 'q', 'out',
         'attn_sink', 'wo_partial', 'wo_reduced')
LIMITATIONS = [
    'not an independent model oracle or a correctness verdict; experiment comparability is established elsewhere',
    'earliest observed difference need not be the causal operator; o-proj epilogue, mHC and FFN are bracketed, not captured',
    'reference conformance uses each row\'s own captured inputs; it never explains a cross-grid difference',
    'records are compared by position through each capture\'s own slots; physical slot values may differ between boots',
    'the preceding window (positions prompt-255..prompt-128) is read by rows 0..126 only; its writers were not captured',
    'equal rounded outputs do not imply equal inputs, and unchanged rows do not exclude shape-dependent kernels',
]


def sha(tensor):
    flat = tensor.detach().contiguous().view(torch.uint8).flatten()
    return hashlib.sha256(bytes(flat.tolist()) if flat.numel() else b'').hexdigest()


def validate(capture):
    if capture.get('schema') != SCHEMA:
        raise ValueError('unsupported capture schema')
    meta = capture['meta']
    rows = meta['chunk_rows']
    if (rows not in (8192, 4096) or meta['prompt_tokens'] != 524288 or meta['row_position'] != 524287
            or meta['window_rows'] != WINDOW or meta['record_rows'] != RECORDS or meta['layers'] != list(LAYERS)
            or meta['batch_requests'] != 1 or meta['problems']):
        raise ValueError('capture geometry or reported problems')
    if set(capture['layers']) != set(LAYERS):
        raise ValueError('layers 0 and 1 required')
    expected = torch.arange(524288 - WINDOW, 524288, dtype=torch.int64)
    for lid, entry in capture['layers'].items():
        if entry['layer_id'] != lid or entry['chunk_rows'] != rows or entry['chunk_row_start'] != rows - WINDOW:
            raise ValueError(f'layer {lid}: wrong window geometry')
        if not torch.equal(entry['positions'], expected):
            raise ValueError(f'layer {lid}: positions are not the final {WINDOW}')
        shapes = {'hidden_in': (WINDOW, entry['hidden']), 'kv_norm': (WINDOW, KV_DIM), 'kv_rot': (WINDOW, KV_DIM),
                  'write_slots': (WINDOW,), 'q': (WINDOW, entry['heads'], HEAD_DIM),
                  'out': (WINDOW, entry['heads'], HEAD_DIM), 'attn_sink': (entry['heads'],),
                  'swa_indices': (WINDOW, entry['swa_width']), 'swa_lengths': (WINDOW,), 'record_slots': (RECORDS,),
                  'records': (RECORDS, SWA_RECORD), 'wo_partial': (WINDOW, entry['hidden']),
                  'wo_reduced': (WINDOW, entry['hidden'])}
        for name, shape in shapes.items():
            t = entry[name]
            if t.device.type != 'cpu' or tuple(t.shape) != shape:
                raise ValueError(f'layer {lid}: {name} has shape {tuple(t.shape)}, expected {shape}')
            if not t.is_floating_point() and name not in ('records',) and (t < -1).any():
                raise ValueError(f'layer {lid}: {name} has invalid negative entries')
            if t.is_floating_point() and not torch.isfinite(t.float()).all():
                raise ValueError(f'layer {lid}: {name} is not finite')
    return meta


def record_lookup(entry):
    """Physical slot -> first index among the 256 gathered records."""
    lookup = {}
    for index, slot in enumerate(entry['record_slots'].tolist()):
        lookup.setdefault(slot, index)
    return lookup


def row_windows(entry):
    """Per captured row: local record indices of the slots its window read, in captured (chronological) order."""
    lookup, rows = record_lookup(entry), []
    for r in range(WINDOW):
        n = int(entry['swa_lengths'][r])
        slots = entry['swa_indices'][r][:n].tolist()
        if any(s < 0 for s in slots):
            raise ValueError(f'row {r}: negative slot inside swa_length')
        try:
            rows.append(torch.tensor([lookup[s] for s in slots], dtype=torch.int32))
        except KeyError as missing:
            raise ValueError(f'row {r}: slot {missing.args[0]} is outside the gathered records') from None
    return rows


def decoded_records(entry, ref):
    return ref.unpack_deepseek_v41_cache_reference(entry['records'].contiguous(), page_size=1, cache_kind='swa')


def per_row(left, right):
    """Row profile of a difference: exact per-row comparison plus overall norms (descriptive only)."""
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('tensor shape/dtype mismatch')
    a, b = left.reshape(left.shape[0], -1).double(), right.reshape(right.shape[0], -1).double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('nonfinite tensor cannot be summarized')
    changed = (a != b).sum(dim=1)
    rows = torch.nonzero(changed).flatten().tolist()
    diff = b - a
    rel = diff.norm(dim=1) / a.norm(dim=1).clamp_min(1e-30)
    total = float(diff.norm())
    return {'shape': list(left.shape), 'dtype': str(left.dtype), 'equal_values': not rows,
            'changed_rows': rows, 'changed_row_count': len(rows),
            'first_changed_row': rows[0] if rows else None, 'last_changed_row': rows[-1] if rows else None,
            'changed_elements': int(changed.sum()), 'elements': int(a.numel()),
            'relative_l2_to_left': total / float(a.norm()) if float(a.norm()) else None,
            'max_abs': float(diff.abs().max()) if diff.numel() else 0.0,
            'row_relative_l2': [round(float(v), 8) for v in rel.tolist()],
            'row_changed_elements': changed.tolist()}


def reference_rows(entry, ref, sink=True):
    """Pinned reference on each captured row from its own query and records; per-row relative L2 to the kernel."""
    windows = row_windows(entry)
    records = entry['records'].contiguous()
    heads = entry['heads']
    rel_max, rel_median, cos_min, max_abs = [], [], [], []
    for r in range(WINDOW):
        idx = windows[r]
        q32 = entry['q'][r].float().reshape(1, heads, HEAD_DIM)
        expected = ref.compressed_sparse_mla_reference(
            q32, records, idx.reshape(1, -1), torch.tensor([idx.numel()], dtype=torch.int32),
            sm_scale=SM_SCALE, attn_sink=entry['attn_sink'].float() if sink else None, swa_page_size=1,
            cache_format='deepseek_v41').reshape(heads, -1)
        got = entry['out'][r].float()
        diff = got - expected
        rel = diff.norm(dim=1) / expected.norm(dim=1).clamp_min(1e-30)
        rel_max.append(float(rel.max()))
        rel_median.append(float(rel.median()))
        cos_min.append(float(torch.nn.functional.cosine_similarity(got, expected, dim=1).min()))
        max_abs.append(float(diff.abs().max()))
    return {'rows': WINDOW, 'window_lengths': [int(v) for v in entry['swa_lengths'].tolist()],
            'rel_l2_max_per_row': rel_max, 'rel_l2_median_per_row': rel_median,
            'cosine_min_per_row': cos_min, 'max_abs_per_row': max_abs,
            'rel_l2_max': max(rel_max), 'rel_l2_max_row': rel_max.index(max(rel_max)),
            'scope': 'same-input conformance per row on this grid; not a cross-grid explanation'}


def packed_rows(entry, ref):
    """Reference packing of kv_rot against the record captured at each row's write slot."""
    lookup = record_lookup(entry)
    packed = ref.pack_deepseek_v41_cache_reference(entry['kv_rot'].float(), page_size=1, cache_kind='swa')
    packed = packed.view(torch.uint8).reshape(-1, SWA_RECORD)
    equal_bytes, decoded_max_abs, missing = [], [], []
    decoded_all = decoded_records(entry, ref)
    decoded_packed = ref.unpack_deepseek_v41_cache_reference(packed.contiguous(), page_size=1, cache_kind='swa')
    for r, slot in enumerate(entry['write_slots'].tolist()):
        index = lookup.get(slot)
        if index is None:
            missing.append(r)
            equal_bytes.append(None)
            decoded_max_abs.append(None)
            continue
        equal_bytes.append(int((packed[r] == entry['records'][index]).sum()))
        decoded_max_abs.append(float((decoded_packed[r] - decoded_all[index]).abs().max()))
    return {'record_bytes': SWA_RECORD, 'equal_bytes_per_row': equal_bytes, 'decoded_max_abs_per_row': decoded_max_abs,
            'rows_not_in_gathered_records': missing,
            'rows_bit_equal': sum(1 for v in equal_bytes if v == SWA_RECORD),
            'scope': 'reference encoder of the captured rotated KV vs the record read back; descriptive'}


def slot_rows(entry):
    page = int(entry['page_size'])
    write, last_window = entry['write_slots'], entry['swa_indices'][WINDOW - 1][:int(entry['swa_lengths'][WINDOW - 1])]
    positions = entry['positions']
    return {'page_size': page,
            'write_slots_equal_last_row_window': bool(last_window.numel() == WINDOW and torch.equal(write, last_window)),
            'write_offsets_equal_position_offsets': bool(torch.equal(write % page, positions % page)),
            'write_slots_ascending': bool((write[1:] > write[:-1]).all()),
            'last_row_window_length': int(last_window.numel())}


def within(capture, ref):
    return {lid: {'reference': reference_rows(entry, ref), 'packed': packed_rows(entry, ref), 'slots': slot_rows(entry)}
            for lid, entry in sorted(capture['layers'].items())}


def decision_consistency(window, decision):
    """The last captured row against the decision-row capture (schema v3) of the same boot and rank."""
    if decision.get('schema') != 'claude-decision-row-v3':
        raise ValueError('decision-row capture schema')
    if decision['meta']['rank'] != window['meta']['rank'] or decision['meta']['chunk_rows'] != window['meta']['chunk_rows']:
        raise ValueError('decision-row capture is for another rank or grid')
    out = {}
    for lid, entry in sorted(window['layers'].items()):
        d = decision['layers'][lid]
        n = int(d['swa_len'])
        last = WINDOW - 1
        out[lid] = {'q_equal': torch.equal(entry['q'][last], d['q']), 'out_equal': torch.equal(entry['out'][last], d['out']),
                    'attn_sink_equal': torch.equal(entry['attn_sink'], d['attn_sink']),
                    'window_slots_equal': torch.equal(entry['swa_indices'][last][:n], d['swa_slots'][:n].to(torch.int64)),
                    'window_records_equal': torch.equal(entry['records'][WINDOW:WINDOW + n], d['swa_records'][:n]),
                    'position_equal': int(entry['positions'][last]) == int(d['position'])}
        out[lid]['all_equal'] = all(out[lid].values())
    return out


def compare_layer(a, b, ref):
    layers = {}
    decoded_a, decoded_b = decoded_records(a, ref), decoded_records(b, ref)
    lookup_a, lookup_b = record_lookup(a), record_lookup(b)

    def by_position(entry, decoded, lookup, row):
        n = int(entry['swa_lengths'][row])
        idx = [lookup[s] for s in entry['swa_indices'][row][:n].tolist()]
        return decoded[idx]

    for name in ('hidden_in', 'kv_norm', 'kv_rot'):
        layers[name] = per_row(a[name], b[name])
    for name, row in (('records_decision_window', WINDOW - 1), ('records_preceding_window', 0)):
        x, y = by_position(a, decoded_a, lookup_a, row), by_position(b, decoded_b, lookup_b, row)
        layers[name] = per_row(x, y) if x.shape == y.shape else {'equal_values': False, 'window_lengths': [x.shape[0], y.shape[0]]}
    for name in ('q', 'out'):
        layers[name] = per_row(a[name], b[name])
    layers['attn_sink'] = per_row(a['attn_sink'].reshape(1, -1), b['attn_sink'].reshape(1, -1))
    for name in ('wo_partial', 'wo_reduced'):
        layers[name] = per_row(a[name], b[name])
    layers['write_offsets_equal'] = bool(torch.equal(a['write_slots'] % a['page_size'], b['write_slots'] % b['page_size']))
    first = next((n for n in ORDER if not layers[n]['equal_values']), None)
    return {'boundaries': layers, 'first_differing_boundary': first}


def compare_rank(left, right, ref):
    validate(left)
    validate(right)
    for key in ('rank', 'node', 'kit_sha256', 'source_trees', 'prompt_tokens', 'row_position', 'tp_world_size'):
        if key != 'rank' and not left['meta'].get(key):
            raise ValueError(f'missing identity: {key}')
        if left['meta'].get(key) != right['meta'].get(key):
            raise ValueError(f'capture identity differs: {key}')
    layers, first = {}, None
    for lid in LAYERS:
        layers[lid] = compare_layer(left['layers'][lid], right['layers'][lid], ref)
        if first is None and layers[lid]['first_differing_boundary'] is not None:
            first = {'layer': lid, 'boundary': layers[lid]['first_differing_boundary']}
    return {'scope': 'descriptive-only; experiment comparability not established', 'rank': left['meta']['rank'],
            'left_chunk_rows': left['meta']['chunk_rows'], 'right_chunk_rows': right['meta']['chunk_rows'],
            'first_differing_boundary': first, 'boundary_order': list(ORDER), 'layers': layers,
            'within_left': within(left, ref), 'within_right': within(right, ref), 'limitations': LIMITATIONS}


def cross_rank(captures):
    metas = [c['meta'] for c in captures]
    if sorted(m['rank'] for m in metas) != list(range(len(captures))) or len({m['chunk_rows'] for m in metas}) != 1:
        raise ValueError('one capture per rank of one grid is required')
    report = {'chunk_rows': metas[0]['chunk_rows'], 'layers': {}}
    for lid in LAYERS:
        entry = {}
        for name in REPLICATED + PER_RANK:
            hashes = [sha(c['layers'][lid][name]) for c in captures]
            entry[name] = {'equal_across_ranks': len(set(hashes)) == 1, 'expected_equal': name in REPLICATED,
                           'sha256': hashes}
        report['layers'][lid] = entry
    report['replicated_all_equal'] = all(report['layers'][l][n]['equal_across_ranks'] for l in LAYERS for n in REPLICATED)
    return report


def load(path):
    return torch.load(Path(path), map_location='cpu', weights_only=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('compare')
    c.add_argument('left', type=Path)
    c.add_argument('right', type=Path)
    c.add_argument('--decision-left', type=Path)
    c.add_argument('--decision-right', type=Path)
    r = sub.add_parser('ranks')
    r.add_argument('captures', type=Path, nargs=4)
    a = parser.parse_args(argv)
    ref = load_reference()
    if a.command == 'compare':
        left, right = load(a.left), load(a.right)
        result = compare_rank(left, right, ref)
        for side, path, capture in (('left', a.decision_left, left), ('right', a.decision_right, right)):
            if path is not None:
                result['decision_consistency_' + side] = decision_consistency(capture, load(path))
        files = [a.left, a.right, a.decision_left, a.decision_right]
    else:
        result = cross_rank([validate(c) and c for c in map(load, a.captures)])
        files = a.captures
    result['files'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p is not None}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
