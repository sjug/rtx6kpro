"""Frozen R38 DS4.1 admission contract, shared by host and container renders."""

import hashlib
import json
import re
from pathlib import Path

IMAGE = "localhost/voipmonitor/vllm:jj-r38-spark-sm121"
IMAGE_ID = "ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5"
REVISION = "fb2764a5cf321eaa5070ca8f9e892818f477c16d"
REPO_DIR = "models--deepseek-ai--DeepSeek-V4.1-Flash"
MODEL = f"/root/.cache/huggingface/hub/{REPO_DIR}/snapshots/{REVISION}"
NAME = "ds41-flash-jj-r38-tp4"
ROOT = Path(__file__).resolve().parent
NODES = {"dusty": (0, "10.11.11.7"), "toby": (1, "10.11.11.6"),
         "rusty": (2, "10.11.11.5"), "kirby": (3, "10.11.11.8")}
MASTER = NODES["dusty"][1]
HCAS = "rocep1s0f0,roceP2p1s0f0"
INTERFACES = "enp1s0f0np0,enP2p1s0f0np0"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_kit(root=ROOT):
    manifest = root / "runtime-files.sha256"
    expected_files = {"contract.py", "launch.py", "disk_probe.py", "parse_cli.py", "run_node.py", "run-node.sh",
                      "verify_model.py", "model-manifest.json"}
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(name in expected_files and name not in entries,
                f"Unexpected or duplicate runtime manifest entry: {name}")
        require(sha256(root / name) == digest, f"Runtime file changed: {name}")
        entries[name] = digest
    require(set(entries) == expected_files, "Incomplete runtime file manifest")
    return sha256(manifest)


def settings(env):
    """Only documented scalar overrides are accepted, never arbitrary CLI text."""
    fixed = {"TP_SIZE": "4", "NNODES": "4", "DCP_SIZE": "1",
             "ENGRAM_TABLE_MEMORY": "disk", "MASTER_ADDR": MASTER,
             "MODEL_REVISION": REVISION, "IMAGE": IMAGE, "EXPECTED_IMAGE_ID": IMAGE_ID,
             "NCCL_IB_HCA": HCAS, "NCCL_SOCKET_IFNAME": INTERFACES,
             "NCCL_PROTO": "LL,Simple", "VLLM_ENABLE_ROCE_ALLREDUCE": "0",
             "VLLM_ENABLE_PCIE_ALLREDUCE": "0", "LMCACHE_ENABLED": "0"}
    for key, value in fixed.items():
        require(env.get(key, value) == value, f"{key} is fixed to {value} in this candidate")
    for key in ("KV_CACHE_MEMORY_BYTES", "EXTRA_VLLM_ARGS", "NCCL_MIN_NCHANNELS",
                "NCCL_MAX_NCHANNELS", "NCCL_MIN_CTAS", "NCCL_MAX_CTAS",
                "NCCL_LAUNCH_ORDER_IMPLICIT"):
        require(not env.get(key), f"{key} is not part of this admission contract")
    cfg = {}
    for key, default, low, high in (
        ("MAX_MODEL_LEN", 131072, 1, 1048576), ("MAX_NUM_SEQS", 4, 1, 4),
        ("MAX_NUM_BATCHED_TOKENS", 4096, 1, 4096), ("DSPARK_TOKENS", 7, 0, 7),
        ("PORT", 8000, 1024, 65535), ("MASTER_PORT", 25000, 1024, 65535),
    ):
        value = str(env.get(key, default))
        require(re.fullmatch(r"0|[1-9][0-9]*", value) is not None, f"Invalid {key}")
        cfg[key] = int(value)
        require(low <= cfg[key] <= high, f"{key} must be between {low} and {high}")
    value = env.get("GPU_MEMORY_UTILIZATION", "0.85")
    require(re.fullmatch(r"0\.[0-9]+", value) is not None, "Invalid memory utilization")
    require(0 < float(value) <= 0.85, "Memory utilization must not exceed 0.85")
    cfg["GPU_MEMORY_UTILIZATION"] = value
    cfg["MAX_CUDAGRAPH_CAPTURE_SIZE"] = cfg["MAX_NUM_SEQS"] * (1 + cfg["DSPARK_TOKENS"])
    cfg["JIT_MONITOR_MODE"] = env.get("JIT_MONITOR_MODE", "error")
    require(cfg["JIT_MONITOR_MODE"] in ("error", "warn"), "JIT monitoring cannot be disabled")
    require(cfg["JIT_MONITOR_MODE"] == "error" or cfg["DSPARK_TOKENS"] == 0,
            "JIT warn is only admitted for the untimed target-only correctness control")
    # Diagnostic passthrough added 2026-09-16 for the approved deterministic-MoE
    # arm of the 524K nondeterminism investigation. B12X reads this variable in
    # b12x/moe/fused_moe/_impl.py (_dynamic_deterministic_output_enabled) and
    # switches expert-output accumulation from bf16 global atomics to one-writer
    # stores plus a fixed-order top-k reduction. It is set only when given
    # explicitly, accepts 0 or 1 only, and changes no other control. It is a
    # diagnostic setting, not part of the qualified serving profile.
    deterministic = env.get("B12X_DYNAMIC_DETERMINISTIC_OUTPUT")
    if deterministic is not None:
        require(deterministic in ("0", "1"), "B12X_DYNAMIC_DETERMINISTIC_OUTPUT must be 0 or 1")
        cfg["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"] = deterministic
    return cfg


def runtime_environment(cfg, node):
    require(node in NODES, f"Unknown node {node}")
    _, ip = NODES[node]
    values = {key: str(value) for key, value in cfg.items()}
    values.update({
        "DS41_NODE": node, "MASTER_ADDR": MASTER, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONOPTIMIZE": "0",
        "TMPDIR": "/container-tmp", "CUDA_VISIBLE_DEVICES": "0",
        "CUTE_DSL_ARCH": "sm_121a", "TORCH_CUDA_ARCH_LIST": "12.1a",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_HOST_IP": ip,
        "NCCL_DEBUG": "WARN", "NCCL_IB_DISABLE": "0", "NCCL_IB_HCA": HCAS,
        "NCCL_IB_GID_INDEX": "3", "NCCL_IB_TC": "106", "NCCL_IB_MERGE_NICS": "0",
        "NCCL_IB_SUBNET_AWARE_ROUTING": "1", "NCCL_PROTO": "LL,Simple",
        "NCCL_SOCKET_IFNAME": INTERFACES, "GLOO_SOCKET_IFNAME": "enp1s0f0np0",
        "VLLM_ENABLE_PCIE_ALLREDUCE": "0", "VLLM_ENABLE_ROCE_ALLREDUCE": "0",
        "VLLM_PCIE_DMA_FP8": "0", "B12X_PCIE_DMA_FP8": "0",
        "LMCACHE_ENABLED": "0", "VLLM_PLUGINS": "", "OMP_NUM_THREADS": "8",
        "B12X_PRINT_COMPILE_PROGRESS": "1", "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_USE_AOT_COMPILE": "1", "VLLM_USE_STANDALONE_COMPILE": "1",
        "VLLM_USE_MEGA_AOT_ARTIFACT": "1", "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "1", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD": "1024", "CUDA_DEVICE_MAX_CONNECTIONS": "32",
        "SAFETENSORS_FAST_GPU": "1", "INSTANTTENSOR_BACKEND": "BUFFERED",
        "INSTANTTENSOR_BUFFER_SIZE": "1342177280", "INSTANTTENSOR_CONCURRENCY": "1",
        "INSTANTTENSOR_IO_DEPTH": "3", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    return values


def serve_command(cfg, node):
    rank, _ = NODES[node]
    command = ["/opt/venv/bin/python", "-m", "vllm.entrypoints.cli.main", "serve", MODEL]
    options = {
        "served-model-name": "DeepSeek-V4.1-Flash", "host": "0.0.0.0", "port": cfg["PORT"],
        "dtype": "bfloat16", "tensor-parallel-size": 4, "pipeline-parallel-size": 1,
        "decode-context-parallel-size": 1, "distributed-executor-backend": "mp",
        "nnodes": 4, "node-rank": rank, "master-addr": MASTER, "master-port": cfg["MASTER_PORT"],
        "load-format": "instanttensor", "safetensors-load-strategy": "lazy",
        "block-size": 256, "swa-block-size": 128,
        "gpu-memory-utilization": cfg["GPU_MEMORY_UTILIZATION"],
        "max-model-len": cfg["MAX_MODEL_LEN"], "max-num-seqs": cfg["MAX_NUM_SEQS"],
        "max-num-batched-tokens": cfg["MAX_NUM_BATCHED_TOKENS"],
        "max-cudagraph-capture-size": cfg["MAX_CUDAGRAPH_CAPTURE_SIZE"],
        "compilation-config": json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"]}),
        "max-parallel-prefills": 1, "prefill-policy": "round-robin", "decode-refill-target": "auto",
        "prefill-compute-share": "0.4", "generation-config": "vllm",
        "override-generation-config": json.dumps({"temperature": 1.0, "top_p": 0.95}),
        "engram-config": json.dumps({"cpu_offload": False, "table_memory": "disk",
                                     "disk_resident_scales": False, "disk_prefetch_max_tokens": 0,
                                     "projection_tp": False}),
        "attention-backend": "B12X", "linear-backend": "b12x", "moe-backend": "b12x",
        "jit-monitor-mode": cfg["JIT_MONITOR_MODE"], "tokenizer-mode": "deepseek_v41",
        "reasoning-parser": "deepseek_v41", "tool-call-parser": "deepseek_v41",
        "mm-processor-cache-gb": 0,
    }
    if cfg["DSPARK_TOKENS"]:
        options["speculative-config"] = json.dumps({
            "method": "dspark", "num_speculative_tokens": cfg["DSPARK_TOKENS"],
            "draft_tensor_parallel_size": 4, "attention_backend": "B12X",
            "draft_sample_method": "greedy", "rejection_sample_method": "standard",
            "enable_adaptive_verification": True, "adaptive_verification_cost_scale": 1.0,
        })
    for key, value in options.items():
        command += [f"--{key}", str(value)]
    command += ["--enable-prefix-caching", "--enable-chunked-prefill", "--async-scheduling",
                "--no-scheduler-reserve-full-isl", "--disable-custom-all-reduce",
                "--enable-auto-tool-choice", "--enable-prompt-tokens-details",
                "--enable-force-include-usage", "--enable-request-id-headers",
                "--default-chat-template-kwargs.thinking=true",
                "--default-chat-template-kwargs.reasoning_effort=max"]
    if rank:
        command.append("--headless")
    return command
