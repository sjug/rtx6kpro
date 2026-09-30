"""CPU fitter adapter for faulty columns beyond the captured weight prefix.

The full checkpoint is accepted only after matching the captured prefix and
every logical packed weight scale. No CUDA API is used.
"""
import argparse
import sys
import torch
from safetensors import safe_open
import claude_fit_error_structure as fit

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--trace', required=True)
p.add_argument('--observed-key', default='dense_replay_synchronized')
p.add_argument('--reference-key', default='projected_kv')
p.add_argument('--block')
a = p.parse_args()
trace = torch.load(a.trace, map_location='cpu', weights_only=True, mmap=True)
row = fit._projection_row(trace)
out = row['outputs']
with safe_open(a.checkpoint, framework='pt', device='cpu') as f:
    weight = f.get_tensor('layers.1.engram.wkv.weight')
    scales = f.get_tensor('layers.1.engram.wkv.scale')
prefix = out['weight_prefix']
if not torch.equal(weight[:prefix.shape[0]].view(torch.uint8), prefix):
    raise RuntimeError('Checkpoint weight prefix mismatch')
logical = fit.physical_scales_to_compact(out['weight_scale_mma'], weight.shape[0], weight.shape[1] // 32)
if not torch.equal(logical, scales.view(torch.uint8).repeat_interleave(32, dim=0)):
    raise RuntimeError('Checkpoint weight scale mismatch')
print('CHECKPOINT-PREFIX-AND-ALL-SCALES-MATCH', flush=True)
out['weight_prefix'] = weight.view(torch.uint8)
# Adapt the existing fitter in memory; no new trace is written or falsified.
original_load = fit._load_any
fit._load_any = lambda path: trace if path == a.trace else original_load(path)
sys.argv = ['claude_fit_error_structure.py', 'trace', '--trace', a.trace,
            '--observed-key', a.observed_key, '--reference-key', a.reference_key]
if a.block:
    sys.argv += ['--block', a.block]
raise SystemExit(fit.main())
