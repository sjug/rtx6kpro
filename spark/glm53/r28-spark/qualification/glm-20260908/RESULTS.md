# GLM R28 rollout and qualification, 2026-09-08

Status: **R28 aligned MTP3 qualification passed at 19:20:25 EDT**. R28 is left
serving on sparky, buddy, rocky and lucky. Qwen and nous were untouched.
R27 containers remain stopped and available for rollback. No image rebuild,
source commit, push or upstream filing was performed in this rollout.

## Identity and profile

- R28 `cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1`,
  tag `localhost/voipmonitor/vllm:jj-r28-spark-sm121`.
- Previous GLM image `ef669fa1cde3e99936c02575eca8f610990bb6c64dd1bbfcb04fd42dde87afae`,
  container `glm53-flash-nvfp4-jj-r27-spark-tp4` on each node. Containers are
  stopped, not removed; restart workers before head for rollback.
- Same NVFP4 checkpoint revision `46aaae8a82032f77100f2f03e9cc11b391df3b4d`,
  44 resolved shards, 198042331512 safetensors bytes. Checked on all four nodes.
- TP4/DCP1, MTP3/BF16 head, aligned policy, FP8 KV, native 1048576 context,
  max sequences 8, batch 4096, utilization 0.85, no fixed KV-byte override.
- FlashKDA prefill, RoCEnante plus PyNCCL, NCCL_PROTO=LL,Simple, f0 switched
  rails only. No experimental RTX tuning, loader switch or fairness change.

## Transfer

The existing Docker archive (33327663616 bytes) was pulled from kirby
10.11.11.8 to sparky 10.11.11.1, then sent concurrently to buddy .2,
rocky .4, lucky .3. All payloads used enp1s0f0np0 on the switched 200G rail.
Rsync whole-file/inplace/partial, AES128-GCM, compression disabled. No
conversion or recompression. SHA256 on every receiver:
`4b915899489f238a6f2e0f578b6d5d0dfdad0cf00818f7f99219a60d460f23b4`.
All archive copies verified by 18:39:35 EDT. Image IDs, layers, manifest type
and labels are compared with kirby's source after podman load.

R27 stopped on all four nodes by 18:43:40. Every image ID, Docker v2 manifest
type, rootfs layer list and label set matched the source at 18:44:29.
Workers started buddy, rocky, lucky, then sparky. Current candidate runner
SHA256 `f9e781dcebc22a50bc08904361cbc523902ee361d29a280151b97e0b79fb30b3`;
launcher `b5c1cdc827148257d96a0d60a22276dd524fb098cbd52e94aaef252f8a1f3b1e`.

Kirby had no outgoing SSH identity and sparky's key was not authorized there.
Authenticated public host keys were scoped to a transfer-only known-hosts
file. A short-lived agent with the existing workstation login key enabled the
private copy. No authorized_keys or global SSH configuration was changed.

## Qualification

Correct completion before any timing, all-rank profile checks, semantic x3,
concurrent fresh/extension/repeat and HOL probes, frozen prefix pairs/triples,
exact native retrieval through 1048000 tokens, then run_bench.sh's unchanged
15-cell grid and post-benchmark smoke. All-rank GPU clock telemetry is saved.
Prefix and native retrieval times are cache-dependent, not cold-prefill A/Bs.

Reference: R27 aligned `20260906T222747-0400__jj-r27-aligned-sm121-tp4-dcp1-mtp3-native1m__r01.json`.
Single-boot historical comparison; no assumption of Qwen's gain transferring
to GLM. Auto-policy controls are not part of this aligned rollout.

Raw data: `aligned-mtp3/` and private benchmark campaign
`glm-5.3-flash/nvfp4/2026-09-jj-r28-vs-r27`.

### Admission progress

The API started at 18:49:19. The first readiness request used the inherited
Qwen-style `enable_thinking=false` parameter and returned correct arithmetic
with a literal closing think marker in content. Qualification stopped before
timing. The checkpoint template always opens thinking and accepts low/high/max
reasoning effort, not that disabling convention. The original receipt and
driver log are preserved. A supported `reasoning_effort=low` request returned
content exactly `333`, reasoning separately, and finish_reason stop. The
readiness driver now uses that request, retaining its exact-answer assertion.
The unchanged GLM semantic diagnostic passed all 15 checks before resuming.
This does not qualify non-thinking mode. Legacy concurrency/cache probes retain
their original request bytes for comparison; their timing is not evidence that
the unsupported thinking-disable parameter works.

Boot reported 40.47 GiB available KV memory and 6,206,382 effective KV tokens
at the 1,048,576-token envelope (18:47:41). This is the engine's group-aware
capacity figure, not a flat allocation inferred from benchmark metrics.

Semantic admission after the readiness correction passed 15/15 as well.
The legacy concurrency probe completed every request: c4 identical 124.6
tok/s versus distinct 108.0/107.8; repeats and extensions batched normally.
HOL repeat TTFT 1.1 s, fresh arrivals 0.5/0.3 s while the two long requests
ran for about 16 s. This timer accepts nonempty reasoning or content, unlike
the newer Qwen content-only timer; it proves admission progress, not necessarily
time to first final-answer text.

All six frozen pair requests passed exact retrieval and isolated counter checks.
The unaligned long extension reused 126976 tokens, the aligned extension
129024, and the short extension zero, matching R27 aligned's behavior.
Both short triples also passed, with zero reuse. The 262000-token predecessor
processed cold in 92.24 s with deliberate 16-token truncation; its 64-token
repeat stopped correctly in 2.72 s and reused 258048 tokens. The 1M extension
stopped correctly after 389.32 s and reused 258048 tokens, versus R27 aligned
386.73 s at the same reuse count (about +0.7% elapsed, not a material change).

The inherited native-context helper also sent the unsupported thinking-disable
parameter. It retrieved the needle at all four lengths, but the driver's
stricter exact-content gate rejected reasoning mixed into content. These raw
receipts remain in native-context.jsonl. A GLM-specific probe reuses the same
exact-length prompt construction, sends no thinking-disable override, and
requires exact answer content and normal stop. It passed 2048, 2049, 262000,
1048000 tokens at 19:07:49, with clean separated reasoning. These final
retrievals were warm cache tests, not cold throughput samples. A local helper
import path typo stopped one resume before issuing requests; it was corrected
without changing the server. The full driver log retains both attempts.

The standard 15-cell grid started at 19:07:49 using the unchanged R27 benchmark
protocol. Receipt basename:
`20260908T190749-0400__jj-r28-aligned-sm121-tp4-dcp1-mtp3-native1m__r01.json`.

R27 reference JSON SHA256:
`6e109f5e7dd7f45e85e0ce6ed13f667f510637b14fcf4d8b16b632ec0113787f`.

## Completed standard grid and decision

All 15 cells completed with zero errors, warmup timeouts, capacity-limited or
underfilled cells. Comparison code verified matching benchmark protocol fields
and complete grids. No JIT warnings, ERROR lines or tracebacks occurred during
the 19:07:49 to 19:20:25 benchmark window on any rank. All four final container
inspections report the exact R28 image, running and not OOMKilled. The final
supported-reasoning completion returned exactly 333 with normal stop.

Geometric means across contexts 0, 16K, 32K, 64K and 128K:

| Concurrency | R27 steps/s | R28 steps/s | Change | R27 aggregate tok/s | R28 aggregate tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 19.742 | 19.938 | +0.99% | 50.131 | 51.691 |
| 2 | 31.198 | 31.175 | -0.07% | 80.820 | 81.519 |
| 4 | 45.427 | 45.118 | -0.68% | 122.258 | 119.860 |

Acceptance-length geometric means changed +2.10%, +0.94%, -1.29% at c1/c2/c4.
That explains much of the difference between output-token and engine-step
changes; these are not fixed-token or repeated-boot acceptance measurements.

| Prefill context | R27 tok/s | R28 tok/s | Change |
| --- | ---: | ---: | ---: |
| 8K | 2786 | 2776 | -0.36% |
| 16K | 2974 | 2896 | -2.62% |
| 32K | 2966 | 2950 | -0.54% |
| 64K | 2958 | 2941 | -0.57% |
| 128K | 2882 | 2839 | -1.49% |

Verdict: no material performance regression demonstrated in this historical
single-boot comparison, but no Qwen-sized gain either. GLM is effectively at
R27 performance. Keep the qualified R28 aligned profile serving. The small
prefill differences are single-scout observations, not a repeatable slowdown
claim. This does not qualify auto checkpoints, LMCache, DFlash2 or an NVFP4
draft head.

The benchmark's built-in hardware summary samples the client workstation,
not the remote cluster; do not use it for Spark power or thermals. Dedicated
per-node GPU logs are the authority. During the benchmark, mean SM clocks were
2459.8/2479.1/2514.2/2435.0 MHz (sparky/buddy/rocky/lucky), 756 samples each,
with zero active-throttle samples. The earlier long-1M interval did include
intermittent throttle flags, preserved in the same raw traces.

R28 raw result SHA256:
`3a502970e06e0005c138f1e0fc0af1fecf584aaacee0c670e48d0ba7cf971f36`.
Full per-cell and prefill deltas are in `aligned-mtp3/grid-comparison.json`.
The raw benchmark stays in the private benchmark results repository; its
catalog refresh/check completed with 210 results and zero warnings.
