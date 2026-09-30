"""Compare retained decision-row observations across instrumentation revisions.

Not a model-correctness or experiment-identity gate. The driver must separately
validate source/image pins, cold controls and numerical plans. Physical page
addresses intentionally differ across boots; only gathered record contents are
compared. Trusted local captures only; CPU and weights_only loading.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from compare_decision_grids import _validate


def compare(old, new):
    _validate(old)
    _validate(new)
    for field in ('rank', 'node', 'prompt_tokens', 'row_position', 'chunk_rows'):
        if old['meta'][field] != new['meta'][field]:
            raise ValueError('Baseline geometry mismatch: ' + field)
    rows = {}
    for layer in range(40):
        a, b = old['layers'][layer], new['layers'][layer]
        for key in ('position', 'query_rows', 'row', 'ced_decoder', 'plan', 'swa_len'):
            if a[key] != b[key]:
                raise ValueError(f'Layer {layer} metadata mismatch: {key}')
        n = int(a['swa_len'])
        if not 0 < n <= 128:
            raise ValueError('Invalid active SWA length')
        fields = {}
        for key in ('q', 'out', 'attn_sink', 'swa_records'):
            x, y = a[key], b[key]
            if key == 'swa_records':
                if x.shape[0] < n or y.shape[0] < n:
                    raise ValueError('Truncated active SWA records')
                x, y = x[:n], y[:n]
            if x.device.type != 'cpu' or y.device.type != 'cpu':
                raise ValueError('CPU-only comparison')
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError(f'Layer {layer} tensor mismatch: {key}')
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError('Nonfinite capture')
            # uint8 views distinguish signed zeros and compare actual bytes.
            fields[key] = bool(torch.equal(x.contiguous().view(torch.uint8),
                                           y.contiguous().view(torch.uint8)))
        rows[layer] = fields
    return {'scope': 'observed perturbation check only, not qualification',
            'equal_observations': all(all(row.values()) for row in rows.values()),
            'layers': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old', type=Path)
    parser.add_argument('new', type=Path)
    args = parser.parse_args()
    result = compare(*(torch.load(p, map_location='cpu', weights_only=True)
                       for p in (args.old, args.new)))
    result['sha256'] = {}
    for path in (args.old, args.new):
        with path.open('rb') as stream:
            result['sha256'][str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    print(json.dumps(result, indent=2))
    if not result['equal_observations']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
