"""Receipt loading for the precision image's standard-NCCL arm (host side, no node contact).

Standard-arm receipts record nccl_geometry None and nccl_arm 'standard-upstream'. The tree-arm
loaders reject them because they require tree-simple-1ch. These loaders reject tree receipts and
earlier non-geometry receipts, which carry no nccl_arm. Both grids must come from the one
precision build. Equality between grids is reported by the comparators, never required here.
"""
import json

from audit_window_geometry import load_receipt, ROOT
from compare_window_geometry import GEOMETRY, validate_geometry_grids
from launch_contract import STANDARD_NCCL_ARM

ARMS = (GEOMETRY, STANDARD_NCCL_ARM)
ENVELOPE = 'within-conformance-envelope'


def load_standard_receipt(path, *, root=ROOT):
    receipt = load_receipt(path, geometry=False, root=root)
    summary = receipt['summary']
    if summary.get('nccl_arm') != STANDARD_NCCL_ARM:
        raise ValueError(f'{receipt["path"].name}: not the standard NCCL arm')
    build = json.loads((root / 'receipts/precision-build-receipt.json').read_text())
    if summary['image_id'] != build['image_id']:
        raise ValueError(f'{receipt["path"].name}: not the precision build image')
    verdict = json.loads((receipt['path'] / 'audit.json').read_text())['verdict']
    if verdict != ENVELOPE:
        raise ValueError(f'{receipt["path"].name}: decision-row audit verdict {verdict!r}')
    return receipt


def validate_standard_grids(left, right):
    if (left['chunk_rows'], right['chunk_rows']) != (8192, 4096):
        raise ValueError('Requires the 8192 then 4096 standard-arm grids')
    for data in (left, right):
        if data.get('nccl_geometry') is not None or data.get('nccl_arm') != STANDARD_NCCL_ARM:
            raise ValueError('Both receipts must be the standard NCCL arm')
    for key in ('image_id', 'blocks', 'mode', 'regime_source'):
        if left[key] != right[key]:
            raise ValueError('Unmatched ' + key)
    if left['mode'] != 'matched' or left['blocks'] != 81389:
        raise ValueError('Unexpected experiment envelope')


def load_grids(left, right, arm, *, root=ROOT):
    """The 8192/4096 receipt pair for one explicit NCCL arm; the tree path is the existing one unchanged."""
    if arm == GEOMETRY:
        pair = [load_receipt(v, geometry=True, root=root) for v in (left, right)]
        validate_geometry_grids(pair[0]['summary'], pair[1]['summary'])
    elif arm == STANDARD_NCCL_ARM:
        pair = [load_standard_receipt(v, root=root) for v in (left, right)]
        validate_standard_grids(pair[0]['summary'], pair[1]['summary'])
    else:
        raise ValueError('Unknown NCCL arm: ' + repr(arm))
    return pair
