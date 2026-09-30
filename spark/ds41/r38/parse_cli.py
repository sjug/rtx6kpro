"""Parse the profile with the image's parser, without starting an engine."""

import json

from contract import MASTER, NODES, require, serve_command, settings


def verify(cpu_only=False):
    import vllm.platforms

    if cpu_only:
        # Only for a separate CPU-only gate process, never for the real serve.
        vllm.platforms.current_platform.device_type = "cpu"
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    for node, (rank, _) in NODES.items():
        for tokens in (0, 7):
            cfg = settings({"DSPARK_TOKENS": str(tokens)})
            args = make_arg_parser(FlexibleArgumentParser()).parse_args(serve_command(cfg, node)[4:])
            require(args.tensor_parallel_size == 4 and args.decode_context_parallel_size == 1,
                    "Parsed parallelism differs")
            require(args.nnodes == 4 and args.node_rank == rank and args.master_addr == MASTER,
                    "Parsed rank map differs")
            require(args.headless == bool(rank), "Parsed headless mode differs")
            require(args.engram_config.table_memory == "disk" and not args.engram_config.cpu_offload
                    and not args.engram_config.disk_resident_scales
                    and args.engram_config.disk_prefetch_max_tokens == 0
                    and not args.engram_config.projection_tp, "Parsed Engram placement differs")
            require(args.block_size == 256 and args.swa_block_size == 128, "Parsed KV geometry differs")
            require(args.tokenizer_mode == args.reasoning_parser == args.tool_call_parser == "deepseek_v41",
                    "Parsed tokenizer/parser identity differs")
            require(args.default_chat_template_kwargs == {"thinking": True, "reasoning_effort": "max"},
                    "Parsed reasoning defaults differ")
            require(args.max_num_seqs == 4 and args.max_model_len == 131072
                    and args.gpu_memory_utilization == 0.85 and args.kv_cache_memory_bytes is None,
                    "Parsed admission envelope differs")
            require(args.max_cudagraph_capture_size == 4 * (tokens + 1), "Parsed capture cap differs")
            require(args.disable_custom_all_reduce and args.load_format == "instanttensor",
                    "Parsed loader/transport differs")
            spec = args.speculative_config
            require((spec is None) if tokens == 0 else (
                spec["method"] == "dspark" and spec["num_speculative_tokens"] == tokens
                and spec["draft_tensor_parallel_size"] == 4), "Parsed DSpark differs")
    print(json.dumps({"status": "DS41_IMAGE_CLI_PARSE_PASS", "renders": 8}), flush=True)


if __name__ == "__main__":
    verify(cpu_only=True)
