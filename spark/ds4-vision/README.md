# DeepSeek V4 Flash Vision migration, 2026-09-10

User-authorized replacement of `DeepSeek-V4-Flash-0731` on rusty/toby.
Qwen on dusty/kirby, GLM on sparky/buddy/rocky/lucky, and nous are out of
scope for serving changes.

## Frozen candidate

- Image: `localhost/voipmonitor/vllm:jj-r32-spark-sm121`
- Image ID: `74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c`
- Model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
- Revision: `6821d6ad3681a4b137b066b76094fa82ebd0a380`
- 48 weight shards, 167,819,404,368 bytes. Whole repository: 167,831,848,791 bytes.
- TP2/DCP1, fixed probabilistic DSpark K3, four sequences, 4096 scheduled
  tokens, 524288 context, 0.85 utilization, FULL_AND_PIECEWISE graphs at
  1/2/4/8/12/16, FP8 KV, block256. No fixed KV allocation or external cache.
- Published shared-image DGLIN dense selection. Boot selected
  `FlashInferCutlassMxfp8LinearKernel`, B12X experts and sparse attention,
  and FlashAttention for vision.
- NCCL-only baseline, LL/Simple, f1 direct-pair rails, no channel pin.
  RoCEnante optimization is not part of this checkpoint migration.
- Maximum reasoning by default. Only the canonical Vision served name.

The upstream source launcher is used byte-exact, with its SHA checked at
launch. The host wrapper supplies the Spark envelope and positional maximum
reasoning override. Both target and draft use the immutable local snapshot.
CPU-only rendered commands passed the actual R32 argument parser for both
ranks. Those checks are not GPU or model qualification.

## Staging

Read-only inventory found no Vision repository under the actual HF cache
mounts on any of the eight Sparks. Nous has the model but only a management
NIC, so no model was copied from nous over that network. The pinned model
was downloaded once to rusty, then copied to toby over switched 200G.

The existing Docker archive from dusty was transferred over
10.11.11.7 to 10.11.11.5/6 with AES128-GCM and compression disabled. All
receiver archive digests and image IDs match. No conversion or rebuild.

The first HTTP download used four workers. A native Xet attempt exceeded
the download container's 2 GiB cgroup limit and exited137. It did not stop
the inference container. Download resumed with HTTP, 16 file workers and
the same 2 GiB limit. No HF cache files were deleted. The source verifier
checks every file size and every published LFS SHA256 before peer transfer.
The receiver repeats those checks. Four independent rsync blob lanes use
the ARM crypto cores and preserve snapshot symlinks. No compression or
`--delete` is used.

## Cutover and acceptance

Both nodes passed file-size and LFS SHA256 validation. Cutover started at
2026-09-10 19:30:07 EDT, worker first and head last for both stop and start.
`ds4-0731-tp2` and its corrected-r34 image are preserved, stopped, as rollback.
No other deployment was stopped or reconfigured.

First correct completion passed at 19:37:01 EDT. Default-maximum arithmetic,
nonthinking arithmetic, image colors and image follow-ups passed three
repetitions each; the tool round trip and four simultaneous mixed image/text
requests also passed. The 16,384-token retrieval returned the exact code.
The 524,000-token retrieval also returned the exact code with normal stop
in 413.06 seconds, with 16,128 cached prompt tokens from the preceding 16K
probe. Admission, the full 15-cell standard benchmark and post-benchmark
semantic checks passed. Vision remains active on both nodes.

This boot reports 81.11 GiB model loading memory on the worker, available
KV memory of 11.67 GiB on rusty and 11.21 GiB on toby, and an engine-level
effective KV capacity of 1,165,720 tokens at the 524,288-token envelope
(reported maximum full-length concurrency 2.22). These are boot measurements,
not a promise that four maximum-length requests can reside simultaneously.

The native preflight verifies source-first imports, coherent NCCL, SM121,
and executes vLLM RMSNorm. First meaningful completion comes before timing.
Qualification checks maximum/default arithmetic, nonthinking replies,
deterministic image colors, image conversation continuation, a complete
tool round trip, concurrent mixed image/text requests, and exact retrieval
at 16K and near the configured524K limit. Receipts include actual text,
finish reasons and token accounting. Performance, if run, uses the standard
`llm-inference-bench/run_bench.sh`; a changed checkpoint and K5-to-K3 change
are not an isolated engine A/B.

## Runtime caveats

The wrapper currently passes `VLLM_USE_B12X_FP8_GEMM=0`, an older compatibility
setting that R32 warns is unknown. It does not select the dense backend in
this image. The absence of `--linear-backend b12x` and the observed CUTLASS
kernel establish the actual DGLIN path. Keep this launched profile frozen
through qualification; remove the unused setting in a future wrapper cleanup.

Some shapes compiled on first use during admission. Benchmark measurements
were checked against timestamped JIT warnings: the two benchmark-time
compilations both preceded their cells' readiness events. Geometric-mean
aggregate output is 34.48 / 57.80 / 79.75 tok/s at C1 / C2 / C4. Full raw
numbers, acceptance-normalized steps and scope limits are in
[QUALIFICATION.md](QUALIFICATION.md).
