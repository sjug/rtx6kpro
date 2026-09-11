# Vision-Exp on rusty/toby, 2026-09-10

## Scope and identity

Replacement of DeepSeek-V4-Flash-0731 on this pair only. No serving changes
on dusty/kirby, the four GLM nodes, or nous. The old corrected-r34 containers,
images and HF snapshot remain available as stopped rollback.

- Endpoint: `http://rusty:8000/v1`
- Served ID: `DeepSeek-V4-Flash-Vision-Exp`, without aliases
- Image: `localhost/voipmonitor/vllm:jj-r32-spark-sm121`
- Image ID: `74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c`
- Checkpoint: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
- Revision: `6821d6ad3681a4b137b066b76094fa82ebd0a380`
- Containers: `ds4-vision-jj-r32-tp2` on rusty and toby
- Configuration: TP2/DCP1, probabilistic DSpark K3, maximum reasoning,
  524288-token context, four sequences, 4096 scheduled tokens, utilization
  0.85, FP8 KV, FULL_AND_PIECEWISE graphs, NCCL LL/Simple over the f1 pair
  rails. No channel cap, RoCEnante, LMCache or fixed KV-byte allocation.

`runtime-files.sha256` pins the deployed host wrapper, in-container wrapper,
native preflight and model verifier. The wrappers use the image's source
launcher without modifying it; that launcher's digest is checked before exec.

During the benchmark, the host runner's idle probe was hardened to reject
a failed `podman ps` command explicitly. Its head and worker render hashes
are unchanged, and neither the container command nor either mounted runtime
file changed. `receipts/runtime-files-at-cutover.sha256` preserves the original
launch-time inventory; the top-level inventory names the hardened runner.

## Staging and boot

No Spark had this checkpoint in its actual mounted HF cache. Nous had it but
no 200G path, so the model was downloaded once to rusty instead of copied over
the public NIC. Both nodes independently verified all published LFS SHA256
values and file lengths. The 48 weight shards total 167819404368 bytes.

The existing R32 Docker archive was copied from dusty over the switched 200G
fabric without conversion or compression. Source and both receiver image IDs
match. The model was copied rusty to toby over that same switched fabric,
preserving the cache's relative symlinks. No HF cache was cleared.

Cutover began at 19:30:07 EDT. Stop and start were worker-first, then head.
First correct completion passed at 19:37:01 EDT. Native source imports,
SM121 capability, coherent NCCL 2.31.2 loading and vLLM RMSNorm execution
passed before model startup. The backend list was exactly `['PYNCCL']`.

Boot selected FlashInfer CUTLASS MXFP8 dense linears, B12X experts and sparse
attention, and FlashAttention for vision. Model loading on the worker
reported 81.11 GiB. Available KV memory was 11.67 GiB on rusty and 11.21 GiB
on toby. Engine effective KV capacity was 1165720 tokens at this context
envelope, with a reported 2.22 maximum-length concurrency. This is not a
four-full-length-request residency claim.

## Meaningful output admission

Full responses, elapsed times, finish reasons and token usage are retained
under `receipts/semantic/`. Admission precedes performance timing.

| Check | Result |
|---|---|
| Default maximum-reasoning arithmetic | 3/3 exact `333` |
| Nonthinking arithmetic | 3/3 exact `133` |
| Image colors, order not supplied in prompt | 3/3 exact `red, blue` |
| Image conversation continuation | 3/3 exact `blue` |
| Tool call and return | Exact multiply arguments, final `391` |
| Four concurrent mixed image/text requests | 4/4 exact answers |
| 16384-token retrieval | Exact `739184`, normal stop |
| 524000-token retrieval | Exact `739184`, normal stop, 413.06 seconds |

The deterministic image is 512 by 256 pixels. These are bounded functional
checks, not an OCR, general vision-accuracy or maximum-image-load benchmark.
The needle uses repeated filler and one known code; it establishes that case,
not broad long-context reasoning accuracy.

The longer needle reused 16128 prompt tokens from the preceding 16K probe.
Its 413.06-second latency is therefore a partial-prefix-hit measurement,
not an entirely cold 524K prefill result. Both requests report the exact
requested prompt-token counts.

## Performance

The standard `run_bench.sh` sweep completed at 19:58:37 EDT: 15/15 cells,
30 seconds each, no request errors, underfill, capacity-limit or warmup-timeout
flags. One boot and one sweep, not a repeatability study. Output includes
reasoning tokens. Context labels below are the harness's nominal targets;
its standard estimated token targeting was retained.

| Nominal context | C1 output tok/s | C2 aggregate tok/s | C4 aggregate tok/s |
|---|---:|---:|---:|
| Short | 32.28 | 58.43 | 79.48 |
| 16K | 36.91 | 57.93 | 81.42 |
| 32K | 34.28 | 58.65 | 79.14 |
| 64K | 36.07 | 58.44 | 79.69 |
| 128K | 33.09 | 55.60 | 79.04 |
| Geometric mean | 34.48 | 57.80 | 79.75 |

| Nominal context | C1 normalized steps/s | C2 normalized steps/s | C4 normalized steps/s |
|---|---:|---:|---:|
| Short | 16.37 | 27.13 | 39.99 |
| 16K | 15.98 | 25.70 | 39.08 |
| 32K | 16.26 | 25.94 | 36.58 |
| 64K | 15.73 | 25.88 | 36.31 |
| 128K | 15.85 | 26.68 | 37.25 |
| Geometric mean | 16.04 | 26.26 | 37.81 |

Geometric-mean effective acceptance lengths: 2.150 / 2.201 / 2.109 at
C1 / C2 / C4. These are the harness's acceptance-normalized aggregate
step metrics, not a claim that C4 has that many separate batched GPU
launches each second.

Prefill scouts, one sample each, client prompt tokens divided by TTFT:

| Nominal context | Actual prompt tokens | TTFT seconds | Prompt tok/s |
|---|---:|---:|---:|
| 8K | 8198 | 5.194 | 1578 |
| 16K | 16141 | 7.078 | 2280 |
| 32K | 32064 | 14.299 | 2242 |
| 64K | 63914 | 33.766 | 1893 |
| 128K | 127609 | 81.210 | 1571 |

Two JIT warnings occurred on both ranks, outside timed decode windows:
`_topk_topp_kernel` at 19:52:20, before C2/short readiness at 19:52:24;
`call_extra_pertok` at 19:54:49, before C2/32K readiness at 19:54:53.
Neither rank logged an ERROR or Traceback during the benchmark. Kernel
journal extracts from cutover through the late benchmark contained no
NVRM, Xid or OOM entries. This is not a dedicated thermal-soak result.

Raw result, under the separate benchmark-results repository:
`runs/deepseek-v4-flash/vision-exp/2026-09-jj-r32-vision-spark-admission/throughput/20260910T194509-0400__jj-r32-sm121-tp2-dspark-k3-dglin-524k__r01.json`

SHA256: `11634125bd26c28eaeb9af052c0b405cb502b665a3129f5bcf6fed5dabd6a4ee`.
Benchmark version 0.4.29; script SHA256 is pinned in `benchmark.sh`.

The campaign has no matched baseline: checkpoint, runtime, backend and
K5-to-K3 speculation all differ from the old 0731 deployment. These results
do not establish an isolated engine improvement or checkpoint quality parity.

## Status

Vision remains active on both nodes. Functional admission, long-context
retrieval, the full standard throughput sweep, and a second short semantic
battery after benchmarking all passed. Only Vision-Exp is advertised by the
endpoint. The previous 0731 containers are stopped and preserved as rollback.

Post-benchmark receipts are in `receipts/post-benchmark-semantic/`; no server
restart was required. Nothing was committed or pushed as part of this task.
