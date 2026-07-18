#!/usr/bin/env python3
"""Compile and admit the exact Grid48 kernel on one SM121/48-SM device."""

from __future__ import annotations

import json

import torch

from b12x.moe.fused.w4a16.kernel import (
    clear_w4a16_kernel_cache,
    compile_w4a16_hybrid_mapped_grid48,
    w4a16_hybrid_mapped_grid48_mapping_proof,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Grid48 validation requires a CUDA device")

    device = int(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    capability = (int(props.major), int(props.minor))
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)
    if capability != (12, 1) or sms != 48:
        raise RuntimeError(
            "Grid48 validation requires exact SM121/48-SM hardware, got "
            f"SM{capability[0]}{capability[1]}/{sms} SMs"
        )

    proof = w4a16_hybrid_mapped_grid48_mapping_proof()
    assert proof["grid_x"] == 48
    assert proof["fc1_per_cta_counts"] == (3,) * 32 + (2,) * 16
    assert proof["fc2_per_cta_counts"] == (16,) * 48

    clear_w4a16_kernel_cache()
    launch = compile_w4a16_hybrid_mapped_grid48(
        size_m=4,
        hidden_size=6144,
        intermediate_size=512,
        nv_num_experts=64,
        nf_num_experts=192,
        top_k=8,
        activation="silu",
        element_dtype="bf16",
        fast_math=True,
        sms=sms,
        max_shared_mem=max_shared_mem,
        force_tile_config=(64, 256, 64, 256),
    )
    resources = dict(launch.codegen_resource_metadata)
    assert launch.grid_x == 48
    assert launch.blocks_per_sm == 1
    assert launch.cta_threads == 256
    assert launch.workspace_words == 194
    assert launch.shared_memory_bytes == 45_184
    assert launch.fc1_waves == 3
    assert launch.fc2_waves == 16
    assert 1 <= launch.registers_per_thread <= 255
    assert launch.local_memory_bytes == 0
    assert launch.spill_store_bytes == 0
    assert launch.spill_load_bytes == 0
    assert resources["candidate"] == "w4a16_hybrid_mapped_grid48"
    assert resources["architecture"] == "sm121"
    assert resources["resident_capacity_ctas"] == 48
    assert resources["whole_grid_resident"] is True
    assert resources["one_cta_per_sm"] is True
    assert resources["shared_memory_bytes_exact"] == 45_184
    assert int(resources["register_capacity_ctas"]) >= 1

    print(
        json.dumps(
            {
                "status": "PASS",
                "device": props.name,
                "capability": capability,
                "sms": sms,
                "grid_x": launch.grid_x,
                "cta_threads": launch.cta_threads,
                "blocks_per_sm": launch.blocks_per_sm,
                "fc1_waves": launch.fc1_waves,
                "fc2_waves": launch.fc2_waves,
                "workspace_words": launch.workspace_words,
                "dynamic_shared_memory_bytes": launch.shared_memory_bytes,
                "registers_per_thread": launch.registers_per_thread,
                "local_memory_bytes": launch.local_memory_bytes,
                "generated_code_bytes": launch.generated_code_bytes,
                "kernel_symbol": resources["kernel_symbol"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
