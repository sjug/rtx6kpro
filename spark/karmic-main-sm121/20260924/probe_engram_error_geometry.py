"""Offline diagnostic: compare tile-local error against individual K32 contributions.

This is hypothesis ranking, not an exact oracle or a serving operation.
"""
import argparse
import json
import torch
from safetensors import safe_open

torch.set_num_threads(1)
p = argparse.ArgumentParser()
p.add_argument('first')
p.add_argument('second')
p.add_argument('checkpoint')
a = p.parse_args()
traces = [torch.load(path, map_location='cpu', weights_only=True, mmap=True)
          for path in (a.first, a.second)]
ops = [next(r for r in t['operators'] if r['name'] == 'engram_projection') for t in traces]
for name in ('source',):
    if not torch.equal(ops[0]['inputs'][name], ops[1]['inputs'][name]):
        raise RuntimeError('Inputs differ')
for name in ('quantized_values', 'quantized_scale_rows', 'quantized_scale_mma'):
    if not torch.equal(ops[0]['outputs'][name], ops[1]['outputs'][name]):
        raise RuntimeError('Quantized inputs differ')
left, right = [r['outputs']['projected_kv'][..., 0].float() for r in ops]
delta = left - right
rows, cols = torch.where(delta != 0)
print('GEOMETRY', json.dumps({'rows': rows.unique().tolist(), 'cols': cols.unique().tolist()}))
xq = ops[0]['outputs']['quantized_values'].view(torch.float8_e4m3fn).float()
xs = ops[0]['outputs']['quantized_scale_rows'].view(torch.float8_e8m0fnu).float()[0]
xq = xq.reshape(left.shape[0], -1)
with safe_open(a.checkpoint, framework='pt', device='cpu') as handle:
    weight = handle.get_tensor('layers.1.engram.wkv.weight')
    scales = handle.get_tensor('layers.1.engram.wkv.scale')
for row_index in rows.unique()[::16].tolist():
    changed_cols = torch.where(delta[row_index] != 0)[0]
    # Separate each public N=128 tile, since independent CTAs may differ.
    for tile in (changed_cols // 128).unique().tolist():
        selected = changed_cols[changed_cols // 128 == tile]
        if len(selected) < 8:
            continue
        x = xq[row_index].reshape(-1, 32) * xs[row_index, :, None]
        w = weight[selected].float().reshape(len(selected), -1, 32)
        w *= scales[selected // 32].float()[:, :, None]
        target = delta[row_index, selected]
        for group in (8, 16, 32, 64, 128):
            contributions = torch.einsum('bk,cbk->bc', x.reshape(-1, group),
                                         w.reshape(len(selected), -1, group))
            alpha = (contributions @ target) / contributions.square().sum(1).clamp_min(1e-30)
            residual = (contributions * alpha[:, None] - target).square().sum(1).sqrt()
            residual /= target.square().sum().sqrt().clamp_min(1e-30)
            indices = residual.argsort()[:3]
            print('K-GROUP-FITS', json.dumps({'row': row_index, 'tile_n': tile,
                  'group': group, 'columns': selected.tolist(), 'max_delta': float(target.abs().max()),
                  'fits': [{'k_start': int(i) * group, 'alpha': float(alpha[i]), 'relative_l2': float(residual[i])}
                           for i in indices]}), flush=True)
