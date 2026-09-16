# DSv4 Vision Exp on R38 Spark

The user explicitly authorized qualification on rusty/toby on 2026-09-15.
Both nodes were idle and already held the pinned Vision Exp checkpoint.
Nous, Qwen and GLM configuration are outside this DSv4 qualification.

Candidate image ID:
ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5.
Tag: localhost/voipmonitor/vllm:jj-r38-spark-sm121.
Model: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp.
Revision: 6821d6ad3681a4b137b066b76094fa82ebd0a380.

Carry forward the qualified R32 TP2/DCP1 profile: DSpark K3 probabilistic,
524288 context, four sequences, 4096 scheduled tokens, utilization 0.85,
FP8 KV, block256, NCCL LL/Simple over the f1 direct-pair rails, no channel
pin, no RoCEnante or LMCache. Maximum reasoning remains the default.
The image source launcher is byte-identical to R32 and keeps SHA256
3542a3f6663503dc8697c85e27e5a0a67b80c68c9ffc76143a39403f2bf0d18c.
Only image identity and deployment paths change in the wrappers. The older
unused VLLM_USE_B12X_FP8_GEMM variable is retained for profile fidelity;
actual dense dispatch must be read from the boot, not inferred from it.

Verify both rendered roles with the candidate's real CPU CLI parser, then
verify the existing checkpoint files. Transfer the existing Docker archive
from rusty to toby on switched 200G without compression or conversion.
First meaningful completion precedes semantic, vision, tool, mixed-load,
16K/524K retrieval, the standard run_bench.sh grid, and post-grid semantics.
Record native imports, actual backends, per-rank KV, JIT and timestamps.
Use ../qualify.py unchanged, with new receipt directories. Historical R32
receipts remain immutable; performance comparisons use the same checkpoint.

R32 image remains available on both nodes as the qualified fallback.
No existing containers were present at the initial live inventory.

## Execution so far

Both roles passed the R38 image's real CLI parser, including maximum
reasoning, DSpark K3 and the complete Spark envelope. Both local snapshots
passed link/file-size and configuration checks; this execution did not
rehash all 167.8 GB of weights or download another checkpoint.

R38 was copied to toby using the existing Docker archive over 10.11.11.5
to 10.11.11.6; transfer and image identity checks passed at 15:06:53 EDT.
First correct completion passed at 15:15:24. Three repetitions of default
maximum arithmetic, non-thinking, image colors and image follow-up passed,
as did the tool round trip, four simultaneous mixed requests and 16K needle.
The 524,000-token needle also returned exactly 739184 with a normal stop,
in 336.510 seconds. Full correctness admission passed at 15:21:39 EDT.

The controller then failed its pre-benchmark logging check: it required the
selected-backend line on the worker as well as the head, but that line is
rank-zero-only. This was not a model failure. The corrected check requires
the actual startup commands on both ranks to disable custom all-reduce and
the head to report exactly `['PYNCCL']`. Both checks pass on the saved logs.
The original failed status and successful correctness receipts are retained.
`benchmark.sh` resumes from that admission without a restart or repeating
the long needle. It writes separate benchmark telemetry and exit status.
The canonical grid and post-grid checks completed at 15:40:35 EDT with
exit0 under ds4-r38-benchmark.service. R38 remains serving on both nodes.
See QUALIFICATION.md for the complete results and single-boot comparison
limits, and receipts/r32-vs-r38.json for the numerical comparison.

Boot KV: rusty 12.73 GiB, toby 12.81 GiB; engine effective capacity 1,323,142
tokens at 524288 maximum length (2.52 full-length sequences). Historical R32
reported 1,165,720 tokens, but this is a boot capacity comparison, not speed.

Observed backend logs match the same entries in the R32 receipt:
`DeepGemmFp8BlockScaledMMKernel for Fp8LinearMethod`,
`B12X_MXFP4_MXFP8` MoE, FlashAttention for vision, and `['PYNCCL']`.
The initial linear selection log is not evidence that every dense projection
uses that implementation after weight processing; no backend speed claim is
made from this line alone.
