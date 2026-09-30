"""Isolate ordered versus atomic split-K on the same generated dense operands.

Run separately with B12X_DENSE_SPLITK_TURBO=1 and =0. This is an operator
discriminator, not a claim that these exact shapes caused the serving failure.
"""
import json
import os
import torch
from b12x.gemm import DenseGemmConfig, block_fp8_linear as bfl
from b12x.preparation import PreparationSession, PreparedCall

torch.manual_seed(20260925)
turbo = os.environ['B12X_DENSE_SPLITK_TURBO']
if turbo not in ('0', '1'):
    raise RuntimeError('Explicit binary turbo setting required')
device = torch.device('cuda')
for m in (1, 4, 8):
    k, n = 5120, 1792
    source = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    weight = (torch.randn((n, k), dtype=torch.bfloat16, device=device) / 8).to(torch.float8_e4m3fn)
    scale_bytes = ((torch.arange(n // 32, device=device)[:, None]
                    + 2 * torch.arange(k // 32, device=device)[None, :]) % 7 + 123).to(torch.uint8)
    scales = scale_bytes.view(torch.float8_e8m0fnu)
    packed = bfl.pack_weight(weight, scales, block_size=(32, 32))
    config = DenseGemmConfig(backend='cutedsl', tile_m=16, tile_n=64, tile_k=128,
        load_path='tma', swap_ab=False, split_k_slices=2,
        large_m_unroll=False, target_occupancy=None)
    plan = bfl.plan(bfl.Caps(device=device, max_tokens=m, in_features=k,
                            out_features=n, block_size=(32, 32)), override=config)
    output = torch.empty((m, n, 1), dtype=source.dtype, device=device)
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)

    def prepare(state):
        binding = state.bind(scratch=scratch, source=source, packed_weight=packed, output=output)
        return PreparedCall(run=lambda: state.run_binding(binding))

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name='dense-repeatability', prepare_call=prepare),))
        policy = plan.prepared.state.dense.lowering.policy
        if policy.split_k_slices != 2 or policy.split_k_atomic_bf16 != (turbo == '1'):
            raise RuntimeError('Requested split-K arm not engaged')
        session.freeze()
        binding = bfl.bind(plan, scratch=scratch, source=source, packed_weight=packed, output=output)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            bfl.run(binding=binding)
        outputs = []
        for _ in range(100):
            output.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            if not torch.isfinite(output).all():
                raise RuntimeError('Non-finite result')
            outputs.append(output.clone())
        distinct = sum(not torch.equal(row, outputs[0]) for row in outputs[1:])
        spread = (torch.stack(outputs).float().amax(0) - torch.stack(outputs).float().amin(0)).amax().item()
        print(json.dumps({'m': m, 'k': k, 'n': n, 'turbo': turbo,
                          'repeats': 100, 'different_from_first': distinct,
                          'max_abs_spread': spread}), flush=True)
        if turbo == '0' and distinct:
            raise RuntimeError('Ordered reducer still varies')
        graph.reset()
print('DENSE-DISCRIMINATOR-COMPLETE', flush=True)
