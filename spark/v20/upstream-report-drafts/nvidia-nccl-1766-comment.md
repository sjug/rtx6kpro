# DRAFT — do not post

Repository: `NVIDIA/nccl`

Target: comment on issue #1766, not a new issue and not issue #2353

We can reproduce the same multi-communicator progress class with two ranks on separate NVIDIA GB10 systems connected by two 200 Gbit/s RoCE rails. The workload uses two `ncclAllGather` operations on different communicators and different non-blocking streams, with the same host launch order on both ranks: a side-stream CKV gather, a current-stream indexer gather, and then a current-stream wait on the CKV event. The host enqueues 75 model-layer iterations before synchronizing.

This reproduces identically with NVIDIA NCCL `v2.31.2-1` at `7b83616df3ae082a1f32bb74c27458bfe8153a13` and with our fork built from that release. Automatic selection creates 64 channels per communicator and the run does not complete within 300 seconds. Each of the following completes with patterned output validation on all three production-shaped cases:

| Configuration | Result |
|---|---|
| automatic 64 channels, implicit ordering disabled | HANG, timeout at 300 seconds |
| automatic 64 channels, `NCCL_LAUNCH_ORDER_IMPLICIT=1` | PASS in 11 seconds |
| exact four CTAs/channels, implicit ordering disabled | PASS in 8 seconds |
| `NCCL_LAUNCH_ORDER_IMPLICIT=1` plus exact four | PASS, both ranks and all outputs validated |

Native GDB stacks from the unmodified official 2.31.2-1 automatic-64 arm do not match the proxy-pool exhaustion in #2353. Neither rank has an application thread in `ncclLocalOpAppend`/`uploadProxyOps`, and neither has a proxy thread in `ncclProxyGetPostedOps`. The application had finished enqueueing and was waiting in CUDA synchronization. Each rank had both communicator proxy threads in active progress code; one rank showed `recvProxyProgress -> progressOps -> ncclProxyProgress` on one communicator while the other proxy spun at `ncclProxyProgress`.

This makes #1766's launch-order guidance the closest existing explanation. It also means we did not port #2353's proposed flush-before-block patch: its required stack signature is absent here.

One production-shaped result should remain separate from the progress reproducer. Automatic 64 channels plus implicit ordering completed the weight-free test, but a full model-serving arm near the 121 GiB GB10 UMA limit produced repeated `NVRM: NV_ERR_NO_MEMORY` and systemd memory-pressure notifications before the engine and hosts became unresponsive. There were no Xids or kernel lockup/RCU/hung-task reports. We therefore continue to cap the communicator resource footprint even when implicit ordering is enabled; the serving incident may combine launch-order progress with an independent automatic-64 allocation/resource problem.

Questions for NVIDIA:

1. Does this stack and 64-versus-4 behavior fit the cross-communicator ordering/resource case described in #1766?
2. Is process-wide `NCCL_LAUNCH_ORDER_IMPLICIT=1` plus a four-CTA/channel resource cap the recommended contract for this topology until the application can enforce a deterministic schedule itself? Our serving-qualified cap currently uses the deprecated `NCCL_MIN_NCHANNELS=4` / `NCCL_MAX_NCHANNELS=4`; should we move to `NCCL_MIN_CTAS=4` / `NCCL_MAX_CTAS=4` only after repeating the serving qualification?
3. Which communicator memory/resource counters should we capture to separate the automatic-64 UMA failure from the progress failure?

We can attach the weight-free Python reproducer, exact commands, full `NCCL_DEBUG=INFO` logs, process maps, library digest, and both native stack dumps. The tested official library SHA-256 is `1e9696ffeee4074005f99b9172c78803c5c92d10cb53f958550daaea2109aae8`.
