#!/usr/bin/env python3
"""Parse the rendered command with the image's real vLLM argument parser."""
import json
import sys

import vllm.platforms

# Parser-only CPU container: DeviceConfig's default requires a platform even
# though no engine or model is instantiated. This does not alter the serve.
vllm.platforms.current_platform.device_type = "cpu"

from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser

args = make_arg_parser(FlexibleArgumentParser()).parse_args(json.load(sys.stdin))
if not __debug__:
    raise RuntimeError("Assertions required")
assert args.default_chat_template_kwargs["reasoning_effort"] == "max"
assert args.default_chat_template_kwargs["thinking"] is True
assert args.tensor_parallel_size == 2 and args.decode_context_parallel_size == 1
assert args.max_num_seqs == 4 and args.max_model_len == 524288
assert args.gpu_memory_utilization == 0.85 and args.max_num_batched_tokens == 4096
assert args.disable_custom_all_reduce
assert args.speculative_config["method"] == "dspark"
assert args.speculative_config["num_speculative_tokens"] == 3
assert args.speculative_config["draft_sample_method"] == "probabilistic"
assert args.speculative_config["model"].endswith("6821d6ad3681a4b137b066b76094fa82ebd0a380")
assert args.served_model_name == ["DeepSeek-V4-Flash-Vision-Exp"]
assert args.nnodes == 2 and args.master_addr == "10.11.1.1"
assert args.load_format == "instanttensor" and args.kv_cache_memory_bytes is None
print(json.dumps({"status": "PARSED-VISION-CLI-PASS", "rank": args.node_rank,
                  "headless": args.headless, "defaults": args.default_chat_template_kwargs,
                  "speculative": args.speculative_config}, sort_keys=True))
