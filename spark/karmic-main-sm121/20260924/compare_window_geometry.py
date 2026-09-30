"""Descriptive comparison of an NCCL-geometry window arm against its matched baseline arm (CPU only).

The geometry arm (launch contract DS41_NCCL_GEOMETRY=tree-simple-1ch: NCCL_ALGO=allreduce:tree,
NCCL_PROTO=Simple, one channel) changes the reduction order of the TP all-reduces that the
runtime dispatches to PyNCCL, which on these boots are the prefill-sized calls above the
RoCEnante limit (VLLM_ROCE_ALLREDUCE_MAX_SIZE, 2 MB); RoCEnante-eligible small calls keep their
own fixed-order one-shot reduction and are unchanged. Its outputs are therefore a separate
numerical arm: `compare_window_baseline.py` is an
instrumentation-only equality gate and must NOT be applied to it as a pass requirement.
This module is that separate comparison. It never gates, never calls anything qualified,
and never touches nodes.

  pair     one geometry receipt against the baseline receipt of the same grid (same image, blocks,
           regime source, controls equal within their own boots): which decision-row fields differ
           per layer (from compare_window_baseline.compare, reported not required), and the
           window-level per-row profile at layers 0-1 with identity relaxed to rank, grid and trees,
           since the runtime kit digest legitimately changes with the contract
  grids    the two geometry receipts (8192 then 4096) with the same relaxed identity: the cross-grid
           prediction is that layer-0 reduced WO becomes equal within this boot pair; report it either way

Cautions (user review, 2026-09-26): NCCL builds its tree per boot, so cross-boot equality of
reduced tensors is not predicted even under the pinned geometry; NCCL splits chunks across two
complementary trees, so per-element order can still depend on buffer offset and partial
invariance is a possible outcome; NCCL only warns on unsupported settings, so the driver's
preflight, not this comparison, decides whether the geometry was actually in effect.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from compare_window_baseline import compare as baseline_fields  # noqa: E402

NODES = ('dusty', 'toby', 'rusty', 'kirby')
GEOMETRY = 'tree-simple-1ch'
RELAXED_IDENTITY = ('rank', 'prompt_tokens', 'row_position', 'tp_world_size')


def window_compare_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('claude_window_compare', ROOT / 'claude-window-compare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_geometry_pair(geometry, baseline):
    """Summaries of a geometry receipt and the baseline receipt it is compared with."""
    if geometry.get('nccl_geometry') != GEOMETRY:
        raise ValueError('Left receipt is not the NCCL geometry arm')
    if baseline.get('nccl_geometry') is not None:
        raise ValueError('Baseline receipt must be a non-geometry arm')
    for key in ('chunk_rows', 'image_id', 'blocks', 'mode', 'regime_source'):
        if geometry[key] != baseline[key]:
            raise ValueError('Unmatched ' + key)
    if geometry['mode'] != 'matched' or geometry['blocks'] != 81389:
        raise ValueError('Unexpected experiment envelope')
    for data in (geometry, baseline):
        if not data['signature_equals_control'] or not data['regime_equals_reference']:
            raise ValueError('Controls or numerical plans differ within a boot')
        if set(data['capture_problems']) != set(NODES) or any(data['capture_problems'].values()):
            raise ValueError('Capture failures')


def validate_geometry_grids(left, right):
    if (left['chunk_rows'], right['chunk_rows']) != (8192, 4096):
        raise ValueError('Requires the 8192 then 4096 geometry grids')
    for data in (left, right):
        if data.get('nccl_geometry') != GEOMETRY:
            raise ValueError('Both receipts must be the NCCL geometry arm')
    for key in ('image_id', 'blocks', 'mode', 'regime_source'):
        if left[key] != right[key]:
            raise ValueError('Unmatched ' + key)


def relaxed_window_compare(left, right, compare):
    """claude-window-compare.compare_rank with kit and trees checked separately, not as refusal of the kit."""
    if left['meta']['source_trees'] != right['meta']['source_trees']:
        raise ValueError('window captures come from different source trees')
    for key in RELAXED_IDENTITY:
        if left['meta'].get(key) != right['meta'].get(key):
            raise ValueError('window capture identity differs: ' + key)
    kits = (left['meta']['kit_sha256'], right['meta']['kit_sha256'])
    adjusted = dict(right, meta=dict(right['meta'], kit_sha256=left['meta']['kit_sha256']))
    report = compare.compare_rank(left, adjusted, compare.load_reference())
    report['kit_sha256'] = {'left': kits[0], 'right': kits[1], 'equal': kits[0] == kits[1],
                            'note': 'the runtime kit digest changes with the launch contract; trees must still match'}
    report['scope'] = 'descriptive-only; numerical arm versus its comparison partner, never a gate'
    return report


def decision_row_fields(old, new):
    """compare_window_baseline's per-layer field equality, reported rather than required."""
    report = baseline_fields(old, new)
    unequal = {layer: [k for k, v in fields.items() if not v] for layer, fields in report['layers'].items()
               if not all(fields.values())}
    return {'equal_observations': report['equal_observations'], 'unequal_fields_by_layer': unequal,
            'first_unequal_layer': min(unequal) if unequal else None,
            'scope': 'instrumentation-only baseline fields reported for a numerical arm; not a requirement'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pair')
    p.add_argument('--geometry-window', type=Path, required=True)
    p.add_argument('--baseline-window', type=Path, required=True)
    p.add_argument('--geometry-decision', type=Path, required=True)
    p.add_argument('--baseline-decision', type=Path, required=True)
    g = sub.add_parser('grids')
    g.add_argument('--left', type=Path, required=True)
    g.add_argument('--right', type=Path, required=True)
    a = parser.parse_args(argv)
    compare = window_compare_module()
    load = lambda path: torch.load(path, map_location='cpu', weights_only=True)
    if a.command == 'pair':
        result = {'window': relaxed_window_compare(load(a.baseline_window), load(a.geometry_window), compare),
                  'decision_row': decision_row_fields(load(a.baseline_decision), load(a.geometry_decision))}
        files = [a.geometry_window, a.baseline_window, a.geometry_decision, a.baseline_decision]
    else:
        result = {'window': relaxed_window_compare(load(a.left), load(a.right), compare)}
        files = [a.left, a.right]
    result['files'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
