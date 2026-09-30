"""Exercise the real dependency enumeration with an isolated dynamic plan.

The large expert GEMM compiler is mocked, but the missing ordered-reduction
compiler and its program identities are real. This tests the exact missing seam
without allocating a model's weights. Full boot remains the integration gate.
"""
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from b12x._lib.compile_plan import program_keys
from b12x.moe.fused_moe import _preparation as p, _impl
from b12x.moe._shared.kernels.w4a16.kernel import compile_w4a16_topk_sum

plan = NS(implementation='dynamic', quant_mode='w4a8_mx', routed_rows=24,
          num_topk=8, weight_E=4, k=5120, n=576, max_rows=24,
          activation='silu', deterministic_output=True, swiglu_limit=None,
          swiglu_alpha=None, swiglu_beta=None, dtype=torch.bfloat16,
          decode_config=NS(nvfp4_materialize_intermediate=False))
caps = NS(w4a16_fast_math=True, weight_plan=NS(trellis_bits=0, intermediate_hadamard=False))
scratch = NS(_prewarmed_fused_launches=(), _prewarmed_topk_sum_launches=(),
             _mixed_trellis_launches=(), launch_plan=plan)
dynamic = dict(n=576, w4a8_repacked=True, w4a8_n64_repacked=True, direct_routing=False,
               external_route_plan=False, share_input_across_experts=False, planned_tile_m=32)
with patch.object(p, '_dynamic_program_arguments', return_value=dynamic), \
     patch.object(_impl, '_get_dynamic_kernel', return_value=(None, None)):
    declared = set(program_keys(p._program_carriers(scratch, caps,
                        input_scale_count=1, intermediate_scale_count=1)))
    for fp32 in (False, True):
        required = set(program_keys(compile_w4a16_topk_sum(
            m=3, topk=8, hidden_size=5120, element_dtype='bf16', float32_output=fp32)))
        if not required or not required.issubset(declared):
            raise RuntimeError(f'UNDECLARED-DETERMINISTIC-MOE-REDUCTION fp32={fp32}')
print('DETERMINISTIC-MOE-DEPENDENCY-PASS')
