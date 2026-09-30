#!/usr/bin/env python3
"""Offline forensics of a captured nonmodal router projection (gate logits). CPU only.

Given a verified v3 capture (input hidden states and fp32 logits of one MoE
runner) and that layer's gate weight [384, 5120] (bf16), it computes the
fp64 reference logits = input @ weight.T and reports:
  * the error distribution and every 64x64 output tile (the B12X bf16_gemv
    prefill kernel's tile_m x tile_n CTA tile) whose error exceeds --tol;
  * for each affected row, the best single-substitution explanation: one
    8/16/32/64-wide K chunk of either operand (A = activation row, B =
    weight) replaced by the same offset in K-tile kt-2, kt-1, kt+1 or kt+2,
    or by zeros. kt+2 and kt-2 are the next and previous occupants of the
    same stage of the kernel's 2-stage TMA pipeline.
An exact fit (residual at fp32 accumulation noise) is data-level evidence of
which staged bytes the kernel consumed. It does not by itself say why.

The weight comes from a local raw file (--weight-bin, little-endian bf16,
row-major) with its expected sha256 (--weight-sha256), or from the pinned
Hugging Face revision by HTTP range read (--fetch; about 4 MB per layer).

    python3 claude_gate_capture_forensics.py CAPTURE.pt --fetch --json out.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import urllib.request

import torch

REPO = 'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/fb2764a5cf321eaa5070ca8f9e892818f477c16d/'
TILE_M = TILE_N = TILE_K = 64
STAGES = 2


def _range(url, first, last):
    request = urllib.request.Request(url, headers={'Range': f'bytes={first}-{last}'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch_gate(layer):
    with urllib.request.urlopen(REPO + 'model.safetensors.index.json', timeout=120) as response:
        shard = json.load(response)['weight_map'][f'layers.{layer}.ffn.gate.weight']
    url = REPO + shard
    size = struct.unpack('<Q', _range(url, 0, 7))[0]
    header = json.loads(_range(url, 8, 8 + size - 1))
    info = header[f'layers.{layer}.ffn.gate.weight']
    if info['dtype'] != 'BF16':
        raise SystemExit(f'unexpected gate dtype {info["dtype"]}')
    start, end = info['data_offsets']
    data = _range(url, 8 + size + start, 8 + size + end - 1)
    if len(data) != end - start:
        raise SystemExit('short range read')
    return data, tuple(info['shape']), shard


def weight_tensor(data, shape):
    return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).view(*shape)


def tiles(err, tol):
    rows, cols = err.shape
    found = []
    for i in range((rows + TILE_M - 1) // TILE_M):
        for j in range((cols + TILE_N - 1) // TILE_N):
            block = err[i * TILE_M:(i + 1) * TILE_M, j * TILE_N:(j + 1) * TILE_N]
            if block.max() > tol:
                bad = block > tol
                found.append({'m_tile': i, 'n_tile': j, 'max_abs': block.max().item(),
                              'elements': int(bad.sum()),
                              'rows': [i * TILE_M + r for r in torch.nonzero(bad.any(1)).flatten().tolist()],
                              'cols': [j * TILE_N + c for c in torch.nonzero(bad.any(0)).flatten().tolist()]})
    return found


def explain_row(x_row, w_cols, target, top=3):
    """Best single chunk substitution for one output row over a column block."""
    k = x_row.numel()
    candidates = []
    for size in (8, 16, 32, 64):
        for start in range(0, k, size):
            chunk = slice(start, start + size)
            for label, offset in (('kt-2', -2 * TILE_K), ('kt-1', -TILE_K), ('kt+1', TILE_K),
                                  ('kt+2', 2 * TILE_K), ('zero', None)):
                if offset is None:
                    source_a = torch.zeros(size, dtype=x_row.dtype)
                    source_b = torch.zeros(w_cols.shape[0], size, dtype=x_row.dtype)
                else:
                    other = start + offset
                    if not 0 <= other <= k - size:
                        continue
                    source_a, source_b = x_row[other:other + size], w_cols[:, other:other + size]
                for operand, predicted in (('A', w_cols[:, chunk] @ (source_a - x_row[chunk])),
                                           ('B', (source_b - w_cols[:, chunk]) @ x_row[chunk])):
                    candidates.append({'residual': (target - predicted).norm().item(), 'operand': operand,
                                       'chunk': size, 'k_start': start, 'k_tile': start // TILE_K,
                                       'k_offset_in_tile': start % TILE_K, 'source': label})
    candidates.sort(key=lambda c: c['residual'])
    return candidates[:top]


def analyze(x, logits, weight, *, tol=1e-4, fit_tol=1e-4):
    x, logits, weight = x.double(), logits.double(), weight.double()
    reference = x @ weight.T
    diff = logits - reference
    err = diff.abs()
    affected = tiles(err, tol)
    rows = []
    for tile in affected:
        cols = slice(tile['cols'][0] // TILE_N * TILE_N, tile['cols'][0] // TILE_N * TILE_N + TILE_N)
        for r in tile['rows']:
            target = diff[r, cols]
            best = explain_row(x[r], weight[cols], target)
            rows.append({'row': r, 'n_tile': tile['n_tile'], 'target_norm': target.norm().item(),
                         'best': best, 'exact': best[0]['residual'] <= fit_tol})
    return {'max_abs_error': err.max().item(), 'median_abs_error': err.median().item(),
            'p99_abs_error': torch.quantile(err.flatten().float(), 0.99).item(),
            'affected_tiles': affected, 'row_fits': rows,
            'all_rows_exact': bool(rows) and all(r['exact'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('capture', type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--fetch', action='store_true')
    source.add_argument('--weight-bin', type=Path)
    parser.add_argument('--weight-sha256')
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    saved = torch.load(args.capture, map_location='cpu', weights_only=True)
    runner = saved['runner']
    layer = int(runner.split('.')[1])
    if args.fetch:
        data, shape, shard = fetch_gate(layer)
    else:
        data, shape, shard = args.weight_bin.read_bytes(), (384, 5120), str(args.weight_bin)
    digest = hashlib.sha256(data).hexdigest()
    if args.weight_sha256 and digest != args.weight_sha256:
        raise SystemExit('gate weight sha256 differs')
    report = analyze(saved['tensors']['input'], saved['tensors']['logits'], weight_tensor(data, shape), tol=args.tol)
    report.update(capture=str(args.capture), runner=runner, epoch=saved['meta']['epoch'], rows=saved['meta']['rows'],
                  rank=saved['rank'], node=saved['node'], weight={'shard': shard, 'sha256': digest, 'shape': list(shape)})
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(text + '\n')
    print(text)
    sys.exit(0)


if __name__ == '__main__':
    main()
