#!/usr/bin/env python3
"""Probe whether r18p can retain A8 and source-native W4A16 together."""

import torch

from b12x.moe._shared.execution import PreparedWeightLayout
from b12x.moe.fused_moe import plan_weights


COMMON = {
    "source_format": "fp4_e8m0_k32",
    "activation": "silu",
    "params_dtype": torch.bfloat16,
    "num_experts": 256,
    "hidden_size": 4096,
    "intermediate_size": 1024,
    "w13_layout": "w31",
}


for modes in (("w4a8_mx",), ("w4a16",), ("w4a8_mx", "w4a16")):
    for layout in (None, PreparedWeightLayout.SOURCE_NATIVE):
        try:
            plan = plan_weights(
                **COMMON,
                quant_modes=modes,
                w4a16_layout=layout,
            )
        except Exception as exc:
            print("FAIL", modes, layout, type(exc).__name__, str(exc))
            continue
        print(
            "PASS",
            modes,
            layout,
            sorted(plan.quant_modes),
            plan.required_weight_layout("w4a8_mx") if "w4a8_mx" in modes else None,
            plan.required_weight_layout("w4a16") if "w4a16" in modes else None,
            plan.discards_source_parameters,
            plan.storage_policy,
        )
