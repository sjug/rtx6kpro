"""Isolated fixed-input repeatability for the observed 384-row MXFP8 tactics."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from b12x.gemm import DenseGemmConfig, block_fp8_linear as bfl
from b12x.preparation import PreparationSession, PreparedCall

p = argparse.ArgumentParser()
p.add_argument('--quick', action='store_true')
p.add_argument('--poison', action='store_true')
p.add_argument('--shared-down', action='store_true', help='Test the previously omitted K576 shared-expert down projection')
p.add_argument('--engram256', action='store_true', help='Reproduce the actual Engram capacity-256 tactic')
p.add_argument('--trace', help='Use captured Engram projection source instead of synthetic rows')
p.add_argument('--checkpoint', help='Use actual layer-1 Engram weights from this pinned snapshot')
p.add_argument('--repeats', type=int, default=64)
p.add_argument('--selected-only', action='store_true')
a = p.parse_args()
if a.repeats < 2:
    raise RuntimeError('At least two repeats required')
torch.manual_seed(20260925)
device = torch.device('cuda')
shapes = [(5120, 1792, 128)] if a.quick else [(5120, 1792, 128), (5120, 1152, 64), (6144, 25600, 128)]
if a.shared_down:
    shapes = [(576, 5120, 64)]
if a.engram256:
    shapes = [(6144, 25600, 64)]
if a.trace and not a.engram256:
    raise RuntimeError('--trace requires --engram256')
report = []
for k, n, tile_m in shapes:
    if a.checkpoint:
        if not a.engram256 or not a.trace:
            raise RuntimeError('Actual checkpoint requires Engram trace replay')
        from safetensors import safe_open
        with safe_open(str(Path(a.checkpoint) / 'model-00047-of-00048.safetensors'), framework='pt', device='cpu') as handle:
            weight = handle.get_tensor('layers.1.engram.wkv.weight')
            scales = handle.get_tensor('layers.1.engram.wkv.scale')
        if weight.shape != (n, k) or weight.dtype != torch.float8_e4m3fn or scales.dtype != torch.float8_e8m0fnu:
            raise RuntimeError('Unexpected checkpoint tensor contract')
        print('CHECKPOINT-TENSORS', json.dumps({name: hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
              for name, x in [('weight', weight), ('scale', scales)]}), flush=True)
        weight, scales = weight.to(device), scales.to(device)
    else:
        weight = (torch.randn((n, k), dtype=torch.bfloat16, device=device) / 8).to(torch.float8_e4m3fn)
        scale_bytes = ((torch.arange(n // 32, device=device)[:, None]
                        + 2 * torch.arange(k // 32, device=device)[None, :]) % 7 + 123).to(torch.uint8)
        scales = scale_bytes.view(torch.float8_e8m0fnu)
    packed = bfl.pack_weight(weight, scales, block_size=(32, 32))
    source_base = torch.randn((512, k), dtype=torch.bfloat16, device=device)
    trace_rows = None
    captured_outputs = []
    if a.trace:
        trace = torch.load(a.trace, map_location='cpu', weights_only=True)
        row, = (r for r in trace['operators'] if r['name'] == 'engram_projection')
        source = row['inputs']['source']
        if source.ndim != 2 or source.shape[1] != k or source.dtype != torch.bfloat16:
            raise RuntimeError('Unexpected captured projection source')
        trace_rows = source.shape[0]
        source_base[:trace_rows].copy_(source.to(device))
        if a.checkpoint:
            for suffix in ('-0-toby.pt', '-1-toby.pt'):
                path = str(a.trace).replace('-0-toby.pt', suffix)
                sample = torch.load(path, map_location='cpu', weights_only=True)
                op, = (r for r in sample['operators'] if r['name'] == 'engram_projection')
                captured_outputs.append(op['outputs']['projected_kv'].to(device))
                del sample
        del trace, source
    variants = ([(256, False, 64, 64), (384, False, 128, 64), (512, False, 128, 64)]
                if a.shared_down else [(384, True, tile_m, 64), (384, False, 128, 64),
                                       (512, True, tile_m, 64), (384, True, 64, 128)])
    if a.engram256:
        variants = [(256, True, 64, 128), (256, False, 128, 64)]
        if a.selected_only:
            variants = variants[:1]
    for capacity, swap, tm, tn in variants:
        config = DenseGemmConfig(backend='cutedsl', tile_m=tm, tile_n=tn, tile_k=128,
                                load_path='tma', swap_ab=swap, split_k_slices=1,
                                large_m_unroll=True, target_occupancy=None)
        plan = bfl.plan(bfl.Caps(device=device, max_tokens=capacity, in_features=k,
                                out_features=n, block_size=(32, 32)), override=config)
        spec, = plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        source = source_base[:capacity]
        output = torch.empty((capacity, n, 1), dtype=source.dtype, device=device)

        def prepare(state):
            binding = state.bind(scratch=scratch, source=source, packed_weight=packed, output=output)
            return PreparedCall(run=lambda: state.run_binding(binding))

        with PreparationSession(device=device, autotune=False, compile_workers=20) as session:
            session.prepare((plan.request(name='block32-tactic', prepare_call=prepare),))
            session.freeze()
            lengths = (128, 255, 256, 257, 300, 384, 385) if a.shared_down else ((300,) if a.quick else (64, 65, 257, 300, 320, 384))
            if a.engram256:
                lengths = (trace_rows,) if trace_rows else (128, 129, 160, 192, 224, 255, 256)
            for m in lengths:
                if m > capacity:
                    continue
                out = output[:m]
                binding = bfl.bind(plan, scratch=scratch, source=source_base[:m], packed_weight=packed, output=out)
                snapshots = []
                quantized_reference = None
                quantized_changes = []
                mma_layout_mismatches = []
                for repeat in range(a.repeats):
                    if a.poison:
                        scratch.fill_((0, 127, 255)[repeat % 3])
                    out.fill_(float('nan'))
                    bfl.run(binding=binding)
                    torch.cuda.synchronize()
                    if not torch.isfinite(out).all():
                        raise RuntimeError('Nonfinite output')
                    snapshots.append(out.clone())
                    if a.engram256:
                        logical_quantized = (binding.x_q.values.contiguous().view(torch.uint8),
                                             binding.x_q.scale_rows.contiguous().view(torch.uint8))
                        rr = torch.arange(m, device=device)[:, None]
                        gg = torch.arange(k // 32, device=device)[None, :]
                        mma_scales = binding.x_q.scale_mma.view(torch.uint8)[rr % 32, (rr // 32) % 4, rr // 128, gg % 4, gg // 4, 0]
                        if not torch.equal(mma_scales, binding.x_q.scale_rows[0].view(torch.uint8)):
                            mma_layout_mismatches.append(repeat)
                        logical_quantized += (mma_scales,)
                        if quantized_reference is None:
                            quantized_reference = tuple(x.clone() for x in logical_quantized)
                        elif any(not torch.equal(x, y) for x, y in zip(logical_quantized, quantized_reference)):
                            quantized_changes.append(repeat)
                differences = [i for i, x in enumerate(snapshots) if not torch.equal(x, snapshots[0])]
                stacked = torch.stack(snapshots).float()
                spread = stacked.amax(0) - stacked.amin(0)
                changed_rows = torch.nonzero(spread.reshape(m, -1).amax(1) > 0).flatten().cpu().tolist()
                row = {'k': k, 'n': n, 'm': m, 'capacity': capacity, 'swap': swap,
                       'tile_m': tm, 'tile_n': tn, 'repeats': a.repeats,
                       'scratch_poison': a.poison,
                       'different_from_first': len(differences), 'max_spread': float(spread.max()),
                       'changed_rows': changed_rows}
                if a.engram256:
                    row.update(logical_quantized_changes=quantized_changes, mma_layout_mismatches=mma_layout_mismatches,
                               source_trace=a.trace,
                               first_output_sha256=hashlib.sha256(snapshots[0].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
                if a.checkpoint:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    dx = binding.x_q.values.view(m, k).float() * binding.x_q.scale_rows[0].view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=1)
                    dw = weight.float() * scales.float().repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
                    reference = (dx @ dw.T).unsqueeze(-1)
                    row.update(captured_exact=[sum(torch.equal(x, saved) for x in snapshots) for saved in captured_outputs],
                               reference_max_abs=[float((snapshots[i].float() - reference).abs().max()) for i in (0, -1)],
                               reference_bf16_equal_fraction=float((snapshots[0] == reference.bfloat16()).float().mean()))
                    del dx, dw, reference
                report.append(row)
                print(json.dumps(row), flush=True)
                del snapshots, stacked, spread
print('BLOCK32-TACTIC-PROBE-COMPLETE', flush=True)
if a.engram256 and any(row['different_from_first'] or row['logical_quantized_changes'] or row['mma_layout_mismatches'] for row in report):
    raise SystemExit(1)
