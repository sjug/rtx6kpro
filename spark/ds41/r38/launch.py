"""In-container identity, native-op and io_uring admission before vLLM startup."""

import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from contract import require, runtime_environment, serve_command, settings, verify_kit


def preflight():
    import b12x
    import torch
    import vllm

    require(Path(vllm.__file__).resolve().is_relative_to("/opt/jovian-judgement/vllm/vllm"),
            "Wrong imported vLLM tree")
    require(Path(b12x.__file__).resolve().is_relative_to("/opt/jovian-judgement/b12x"),
            "Wrong imported B12X tree")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one visible GPU required")
    require(torch.cuda.get_device_capability() == (12, 1), "SM121 is required")
    for name in ("vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch"):
        module = importlib.import_module(name)
        linkage = subprocess.check_output(["ldd", module.__file__], text=True, timeout=30)
        require("not found" not in linkage, f"Missing native dependency: {name}")
    nccl = Path(os.environ["VLLM_NCCL_SO_PATH"])
    require(nccl.is_file() and str(nccl) in os.environ["LD_PRELOAD"].replace(":", " ").split(),
            "Image-owned NCCL preload contract mismatch")
    importlib.import_module("vllm.models.deepseek_v4_1.nvidia.model")
    x = torch.arange(512, dtype=torch.float32, device="cuda").reshape(2, 256).to(torch.bfloat16) / 256
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x, torch.ones(256, dtype=x.dtype, device=x.device), 1e-6)
    ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(out.float(), ref, atol=0.02, rtol=0.02)
    torch.cuda.synchronize()
    from b12x.loader._native import load

    native = load()
    require(native.ABI_VERSION == 1, "Wrong B12X loader ABI")
    linkage = subprocess.check_output(["ldd", native.__file__], text=True, timeout=30)
    require("not found" not in linkage and "liburing.so" in linkage, "Disk loader lacks liburing linkage")
    from disk_probe import verify as verify_disk
    from parse_cli import verify as verify_cli

    verify_disk(native)
    verify_cli()
    print("DS41_SM121_NATIVE_AND_IO_URING_PREFLIGHT_PASS", flush=True)


def main():
    require(os.environ["DS41_KIT_SHA256"] == verify_kit(), "Host/container runner digest mismatch")
    cfg = settings(os.environ)
    node = os.environ["DS41_NODE"]
    expected = runtime_environment(cfg, node)
    for key, value in expected.items():
        require(os.environ.get(key) == value, f"Runtime environment drift: {key}")
    preflight()
    cmd = serve_command(cfg, node)
    print(json.dumps({"profile": "ds41-r38-tp4-dcp1-disk-engram", "node": node, "settings": cfg}), flush=True)
    print(shlex.join(cmd), flush=True)
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    main()
