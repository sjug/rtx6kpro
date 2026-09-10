#!/usr/bin/env python3
"""Verify the JJ r29 Spark source, native, and SM121 runtime contract."""

from __future__ import annotations

if not __debug__:
    raise RuntimeError("R29 verification requires Python assertions enabled")

import argparse
import hashlib
import importlib
import importlib.metadata
import os
import runpy
import subprocess
from pathlib import Path

from contracts import require_sm121a
from verify_native import check_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--vllm-spark-tree", required=True)
    parser.add_argument("--vllm-subtree", required=True)
    parser.add_argument("--b12x-tree", required=True)
    parser.add_argument("--b12x-subtree", required=True)
    parser.add_argument("--lmcache-version", required=True)
    parser.add_argument("--flashkda-sha256", required=True)
    parser.add_argument("--stable-native-sha256", required=True)
    parser.add_argument("--source-lock-sha256", required=True)
    return parser.parse_args()


def git_output(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def assert_ldd_clean(path: Path) -> None:
    output = subprocess.check_output(["ldd", str(path)], text=True)
    assert "not found" not in output, f"unresolved dependency in {path}:\n{output}"


def main() -> None:
    args = parse_args()
    assert importlib.metadata.version("vllm") == args.vllm_version
    assert importlib.metadata.version("b12x") == "1.3.0"
    assert importlib.metadata.version("lmcache") == args.lmcache_version
    source_lock = Path("/opt/glm53-flash/source.lock")
    assert source_lock.is_file()
    assert (
        hashlib.sha256(source_lock.read_bytes()).hexdigest() == args.source_lock_sha256
    )

    import b12x
    import torch
    import vllm
    from vllm.model_executor.models.registry import ModelRegistry

    vllm_root = Path("/opt/jovian-judgement/vllm")
    b12x_root = Path("/opt/jovian-judgement/b12x")
    assert Path(vllm.__file__).resolve().is_relative_to(vllm_root)
    assert Path(b12x.__file__).resolve().is_relative_to(b12x_root)
    assert git_output(vllm_root, "write-tree") == args.vllm_spark_tree
    assert (
        git_output(vllm_root, "rev-parse", f"{args.vllm_spark_tree}:vllm")
        == args.vllm_subtree
    )
    assert git_output(b12x_root, "write-tree") == args.b12x_tree
    assert (
        git_output(b12x_root, "rev-parse", f"{args.b12x_tree}:b12x")
        == args.b12x_subtree
    )
    subprocess.run(["git", "-C", str(vllm_root), "diff", "--quiet"], check=True)
    subprocess.run(["git", "-C", str(b12x_root), "diff", "--quiet"], check=True)

    # R29 source markers: widened scalar restore, leading-instruction
    # recurrent checkpoint renderer, and the SM121 draft-head overlay.
    checkpoint = (vllm_root / "vllm/v1/worker/gpu/boundary_checkpoint.py").read_text()
    assert "block = tl.full((), block, tl.int64)" in checkpoint
    assert "slot = tl.full((), slot, tl.int64)" in checkpoint
    assert (
        vllm_root / "tests/v1/worker/test_boundary_checkpoint_restore_scalars.py"
    ).is_file()
    renderer = (vllm_root / "vllm/renderers/online_renderer.py").read_text()
    assert "recurrent_instruction_boundary" in renderer
    draft_head = (
        vllm_root / "vllm/models/glm5next/nvidia/mtp_draft_head.py"
    ).read_text()
    assert "_SUPPORTED_NVFP4_CAPABILITIES = frozenset(((12, 0), (12, 1)))" in draft_head

    # B12X invariants carried from r26 plus the r29 loader sources, which stay
    # inert because the launchers keep InstantTensor and an empty VLLM_PLUGINS.
    packed_linear = b12x_root / "b12x/gemm/blockscaled/_linear.py"
    assert "scale_physical_base[:, -1].fill_(127)" not in packed_linear.read_text()
    dense_gemm = (b12x_root / "b12x/_lib/dense_gemm.py").read_text()
    assert "Persistent CTAs also reuse sC across multiple stores." in dense_gemm
    assert (b12x_root / "b12x/policy/_profiles/data/nvidia.gb10.48sm.json.gz").is_file()
    assert (b12x_root / "b12x/loader/_direct.c").is_file()
    assert os.environ.get("VLLM_PLUGINS", "") == ""

    native_objects = sorted((vllm_root / "vllm").rglob("*.abi3.so"))
    assert len(native_objects) == 10, native_objects
    for native_object in native_objects:
        assert_ldd_clean(native_object)
    flashkda_object = vllm_root / "vllm/_flashkda_C.abi3.so"
    assert flashkda_object.is_file()
    assert (
        hashlib.sha256(flashkda_object.read_bytes()).hexdigest() == args.flashkda_sha256
    )
    cubins = subprocess.check_output(
        ["cuobjdump", "--list-elf", str(flashkda_object)], text=True
    )
    require_sm121a(cubins)
    importlib.import_module("vllm._flashkda_C")
    assert torch.ops._flashkda_C.get_workspace_size(16_384, 1, 1) > 0

    assert (vllm_root / "vllm/_rust_tool_parser.abi3.so").is_file()
    assert os.access(vllm_root / "vllm/vllm-rs", os.X_OK)
    triton_source = vllm_root / "vllm/third_party/triton_kernels"
    assert (triton_source / "matmul_ogs.py").is_file()
    triton_spec = importlib.util.find_spec("triton_kernels")
    assert triton_spec is not None and triton_spec.submodule_search_locations
    assert triton_source.resolve() in {
        Path(location).resolve() for location in triton_spec.submodule_search_locations
    }

    for module in (
        "lmcache",
        "lmcache.cuda_ops",
        "lmcache.lmcache_native",
        "lmcache.lmcache_fs",
        "lmcache.lmcache_redis",
    ):
        imported = importlib.import_module(module)
        module_file = getattr(imported, "__file__", None)
        if module_file and module_file.endswith(".so"):
            assert_ldd_clean(Path(module_file))
    assert Path("/opt/lmcache/lib/liblmcache_cumem_shareable.so").is_file()
    check_binary("/opt/lmcache/lib/liblmcache_cumem_shareable.so")
    check_binary(importlib.import_module("lmcache.cuda_ops").__file__, cuda=True)
    assert os.environ.get("LMCACHE_ENABLED") == "0"

    from b12x.comm import roce
    from b12x.comm.roce import _proxy

    assert roce.API_VERSION == 1
    assert os.environ.get("B12X_ROCE_CACHE_DIR") == "/opt/jovian-judgement/b12x-roce"
    proxy_source = b12x_root / "b12x/comm/roce/_roce_proxy.c"
    proxy_digest = hashlib.sha256(proxy_source.read_bytes()).hexdigest()[:16]
    proxy_path = Path(os.environ["B12X_ROCE_CACHE_DIR"]) / (
        f"roce_proxy-{proxy_digest}.so"
    )
    assert proxy_path.is_file()
    assert_ldd_clean(proxy_path)
    assert _proxy.load().roce_abi_version() == 3

    from vllm.distributed.device_communicators.b12x_roce_all_reduce import (
        REQUIRED_B12X_ROCE_API_VERSION,
        B12xRoceAllReduce,
    )

    assert REQUIRED_B12X_ROCE_API_VERSION == roce.API_VERSION
    assert B12xRoceAllReduce.backend_name == "B12X_ROCENANTE"

    supported = ModelRegistry.get_supported_archs()
    for architecture in (
        "Glm5NextForCausalLM",
        "Glm5NextForConditionalGeneration",
        "DFlash2DraftModel",
        "Qwen3_8FlashNextForConditionalGeneration",
    ):
        assert architecture in supported

    from vllm.platforms import current_platform

    assert current_platform.__class__.__name__ == "NvmlCudaPlatform"
    assert torch.cuda.get_device_capability() == (12, 1)
    from vllm.models.glm5next.nvidia.mtp_draft_head import (
        supports_nvfp4_draft_head,
    )

    assert supports_nvfp4_draft_head(torch.cuda.get_device_capability())
    importlib.import_module("vllm._C_stable_libtorch")
    importlib.import_module("vllm._moe_C_stable_libtorch")
    stable = vllm_root / "vllm/_C_stable_libtorch.abi3.so"
    assert hashlib.sha256(stable.read_bytes()).hexdigest() == args.stable_native_sha256
    check_binary(stable, cuda=True)
    # R29's incompatible schema must be registered from the rebuilt extension.
    schema = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.default._schema
    assert "q_out" in str(schema) and "q_head_padded" not in str(schema)
    assert len(schema.returns) == 0
    # Invoke the new upstream regression directly: a skipped pytest selection
    # cannot masquerade as proof that the incompatible schema executes.
    upstream_test = runpy.run_path(str(vllm_root / "tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py"))
    upstream_test["test_quant_insert_writes_caller_owned_q_out"]()
    print("R29 caller-owned Q output numerical regression: PASS", flush=True)
    hidden_states = torch.linspace(
        -2.0, 2.0, steps=512, device="cuda", dtype=torch.bfloat16
    ).reshape(4, 128)
    weight = torch.linspace(0.5, 1.5, steps=128, device="cuda", dtype=torch.bfloat16)
    result = torch.empty_like(hidden_states)
    epsilon = 1e-6
    torch.ops._C.rms_norm(result, hidden_states, weight, epsilon)
    torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    reference = (
        hidden_states.float()
        * torch.rsqrt(
            hidden_states.float().square().mean(dim=-1, keepdim=True) + epsilon
        )
        * weight.float()
    )
    torch.testing.assert_close(result.float(), reference, rtol=0.02, atol=0.02)

    print("JJ r29 Spark runtime contract: PASS")


if __name__ == "__main__":
    main()
