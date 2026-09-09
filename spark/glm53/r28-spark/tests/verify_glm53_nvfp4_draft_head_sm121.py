#!/usr/bin/env python3
"""Exercise the R28 GLM NVFP4 proposal head on a physical SM121 GPU."""

from __future__ import annotations

if not __debug__:
    raise RuntimeError("R28 verification requires Python assertions enabled")

import torch
import torch.nn.functional as F
from vllm.models.glm5next.nvidia.mtp_draft_head import (
    QuantizedDraftHead,
    supports_nvfp4_draft_head,
)


def main() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda", 0)
    capability = torch.cuda.get_device_capability(device)
    assert capability == (12, 1), capability
    assert supports_nvfp4_draft_head(capability)

    generator = torch.Generator().manual_seed(26)
    source = torch.nn.Linear(
        4096, 8192, bias=False, device=device, dtype=torch.bfloat16
    )
    source.tp_size = 4
    source.shard_indices = object()
    with torch.no_grad():
        source.weight.copy_(
            torch.randn(
                source.weight.shape,
                generator=generator,
                dtype=torch.float32,
            ).to(device=device, dtype=torch.bfloat16)
            / 64.0
        )

    draft = QuantizedDraftHead(source, "nvfp4")
    # Aligned 8192 x 4096: two weights per byte, one scale byte per 16
    # weights, plus the scalar FP32 global output scale. No layout padding.
    expected_storage = source.weight.numel() // 2 + source.weight.numel() // 16 + 4
    assert draft.storage_bytes == expected_storage, (
        draft.storage_bytes,
        expected_storage,
    )

    for rows in (1, 4, 32):
        hidden = torch.randn(
            (rows, source.in_features), generator=generator, dtype=torch.float32
        ).to(device=device, dtype=torch.bfloat16)
        reference = F.linear(hidden, source.weight)
        actual = draft(hidden)
        torch.cuda.synchronize(device)

        assert actual.shape == reference.shape
        assert torch.isfinite(actual).all()
        actual_f32 = actual.float()
        reference_f32 = reference.float()
        relative_rmse = (
            torch.linalg.vector_norm(actual_f32 - reference_f32)
            / torch.linalg.vector_norm(reference_f32)
        ).item()
        cosine = F.cosine_similarity(
            actual_f32.flatten(), reference_f32.flatten(), dim=0
        ).item()
        assert torch.isfinite(reference).all()
        # R27 fixed-seed receipts: RMSE 0.0949-0.0956, cosine 0.9954-0.9956.
        assert relative_rmse < 0.12, (rows, relative_rmse)
        assert cosine > 0.99, (rows, cosine)
        print(
            f"rows={rows} relative_rmse={relative_rmse:.6f} cosine={cosine:.6f}",
            flush=True,
        )

    print("GLM R28 NVFP4 proposal head SM121: PASS", flush=True)


if __name__ == "__main__":
    main()
