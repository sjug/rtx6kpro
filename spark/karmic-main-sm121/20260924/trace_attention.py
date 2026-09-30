"""Diagnostic-only snapshots at the existing opaque attention boundary.

Disabled unless /cache/ds41-trace-control.json exists. No CUDA graph tracing.
Copies can perturb a race; the external repeatability gate remains decisive.
"""
import json
import os
from pathlib import Path
import re

import torch

_active = None
_seen = set()
CONTROL = Path('/cache/ds41-trace-control.json')


def before(hidden, positions, prefix):
    global _active
    if torch.cuda.is_current_stream_capturing():
        return None
    match = re.search(r'layers\.(\d+)', prefix)
    if match is None:
        raise RuntimeError('DS41 trace cannot identify layer')
    layer = int(match.group(1))
    if layer == 0 and _active is None and CONTROL.is_file():
        control = json.loads(CONTROL.read_text())
        run = control['run']
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', run):
            raise RuntimeError('Unsafe trace run name')
        if hidden.shape[0] == control['rows'] and run not in _seen:
            free, _ = torch.cuda.mem_get_info(hidden.device)
            # Conservative full-row bound, before CED reduces the decoder rows.
            footprint = 40 * (2 * hidden.numel() * hidden.element_size()
                              + positions.numel() * positions.element_size())
            if free < 2 * footprint:
                print(f'[DS41-DIAG-TRACE-REFUSED] free={free} need={2 * footprint}', flush=True)
                return None
            _seen.add(run)
            _active = {'run': run, 'node': os.environ['DS41_NODE'], 'layers': []}
    if _active is None:
        return None
    expected = len(_active['layers'])
    if layer != expected:
        raise RuntimeError(f'DS41 trace layer order: {layer} != {expected}')
    row = {'layer': layer, 'prefix': prefix,
           'input': hidden.detach().clone(), 'positions': positions.detach().clone()}
    _active['layers'].append(row)
    return row


def after(row, output):
    global _active
    if row is None:
        return
    row['output'] = output.detach().clone()
    if row['layer'] != 39:
        return
    # Copies on the allocating stream preserve the snapshots until consumption.
    # Synchronization happens only after the last attention layer, not per layer.
    for entry in _active['layers']:
        for key in ('input', 'positions', 'output'):
            entry[key] = entry[key].cpu()
    directory = Path('/cache/ds41-attention-trace')
    directory.mkdir(exist_ok=True)
    path = directory / f"{_active['run']}-{_active['node']}.pt"
    if path.exists():
        raise RuntimeError(f'Trace already exists: {path}')
    torch.save(_active, path)
    print('[DS41-DIAG-TRACE] ' + str(path), flush=True)
    _active = None
