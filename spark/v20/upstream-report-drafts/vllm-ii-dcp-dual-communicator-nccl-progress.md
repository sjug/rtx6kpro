# DRAFT — do not file

Repository: `local-inference-lab/vllm`

Proposed title: `[Bug][II GLM DCP] CKV-prefetch/indexer overlap can deadlock across NCCL communicators`

## Summary

The GLM DCP prefill path overlaps two PyNCCL all-gathers on separate communicators: `_dcp_gather_ckv` launches the CKV history gather on a side stream and a dedicated prefetch communicator, while `_merge_b12x_dcp_topk` launches the indexer candidate gather on the current stream and the indexer DCP communicator. On a two-rank GB10 pair over dual 200 Gbit/s RoCE, the exact weight-free overlap hangs indefinitely when NCCL 2.31.2 automatically selects 64 channels per communicator.

Both NVIDIA's official NCCL 2.31.2-1 tag and the II image's `local-inference-lab/nccl-canonical` build reproduce the hang. Capping each communicator at four channels completes, as does `NCCL_LAUNCH_ORDER_IMPLICIT=1` in the weight-free reproducer. Native stacks now distinguish this from the proxy-pool exhaustion reported in NVIDIA/nccl#2353: the application had completed enqueueing and was waiting in CUDA synchronization, while both communicator proxy threads were active in progress code.

The result instead matches the documented concurrent-communicator ordering/resource class discussed in NVIDIA/nccl#592 and NVIDIA/nccl#1766. NCCL 2.31.2 documents that `NCCL_LAUNCH_ORDER_IMPLICIT=1` orders operations from different communicators by host program order and prevents deadlock; it is disabled by default. This issue concerns the vLLM-II integration contract: using a separate communicator does not by itself make concurrent collectives unable to collide.

## Source and deployment

- vLLM integration tree: `f0fa1cefc1865d316c2478525f550e7646addc40`
- Spark overlay tree: `f560085b72537567a2d2c5f3032b0bae61422cf4`
- B12X integration tree: `75787c7a7431b3bea414d2ebf5f2b8671b23eb33`
- NCCL: official `v2.31.2-1` at `7b83616df3ae082a1f32bb74c27458bfe8153a13`, and canonical fork commit `fb6f40999a2a9e63104d4ae4a84118bce61528f8`
- Hardware: two NVIDIA GB10 ranks, one DCP2 pair, dual 200 Gbit/s RoCE rails
- Production topology: GLM-5.2 EXL3, TP4/DCP2 across four GB10 nodes

Relevant paths:

- `vllm/v1/attention/backends/mla/b12x_mla_sparse.py::_dcp_gather_ckv`
- `vllm/model_executor/layers/sparse_attn_indexer.py::_merge_b12x_dcp_topk`
- `vllm/distributed/device_communicators/pynccl.py::PyNcclCommunicator`

The CKV source currently says that the side stream uses a dedicated communicator "so it cannot collide with the indexer's DCP merge." Separate communicators prevent accidental operation-order coupling within one communicator, but they can still contend for NCCL proxy operations, channels, transport queues, CUDA resources, and cross-communicator launch ordering.

## Weight-free reproducer

The attached `microbench_glm_dcp2_ckv_prefetch.py` constructs the same two PyNCCL communicators and stream/event dependency as the production path. For each test case it enqueues 75 GLM layers before a host synchronization:

1. Side stream: CKV all-gather on the prefetch communicator, then record an event.
2. Current stream: indexer all-gather on the indexer communicator.
3. Current stream: wait for the CKV event.

The host enqueues 75 layers before synchronizing, so automatic 64-channel execution creates a much larger outstanding communication workload than a four-channel cap. That arithmetic is useful for explaining resource pressure, but it is not a proxy-pool proof: NCCL posts each completed plan through `ncclProxyStart`, and the captured application thread was not blocked in the append path.

The container must use `--ipc host`, and both `VLLM_NCCL_SO_PATH` and `LD_PRELOAD` must name the same tested library. Loading two NCCL builds into one process produces invalid results and is explicitly rejected by the receipt check.

## Results

| NCCL build | Configuration | Result |
|---|---|---|
| official 2.31.2-1 | automatic selection, 64 channels | HANG, timed out at 300 seconds |
| official 2.31.2-1 | automatic 64 + `NCCL_LAUNCH_ORDER_IMPLICIT=1` | PASS, 11 seconds, all outputs validated |
| official 2.31.2-1 | `NCCL_MIN_CTAS=4`, `NCCL_MAX_CTAS=4` | PASS, 8 seconds, all outputs validated |
| official 2.31.2-1 | implicit ordering + exact four CTAs | PASS, both ranks exit 0, all outputs validated |
| official 2.31.2-1 | implicit ordering + deployed exact-four NCHANNELS vars | PASS, both ranks exit 0, all outputs validated |
| canonical fork 2.31.2 | automatic selection | HANG, timed out at 300 seconds |
| canonical fork 2.31.2 | `NCCL_LAUNCH_ORDER_IMPLICIT=1` | PASS |
| canonical fork 2.31.2 | exact four channels | PASS |

The official and fork builds are behaviorally convergent, so the canonical fork's additional policy commit is not the cause of this hang.

## Native stack discrimination

The unmodified official 2.31.2-1 arm was rerun for 74 seconds at automatic 64-channel selection with both ranks ptrace-enabled. GDB loaded symbols from the exact tested `libnccl.so` and captured every thread concurrently.

- Neither rank contains `ncclLocalOpAppend`, `uploadProxyOps`, or `ncclProxyGetPostedOps` in any stack.
- Rank 1's application thread is in `cuCtxSynchronize_v2`; rank 0 is likewise inside `libcuda` during the final synchronization, although its CUDA frames do not fully unwind.
- Each rank has one communicator proxy thread spinning at `ncclProxyProgress` line 1003.
- Rank 0's other communicator proxy thread is inside `recvProxyProgress` -> `progressOps` -> `ncclProxyProgress`; rank 1's corresponding thread is also in proxy progress, not asleep waiting for posted ops.

NVIDIA/nccl#2353 requires the application thread in `ncclLocalOpAppend` and that same rank's proxy thread in `ncclProxyGetPostedOps`. Both required probes are absent here, so its proposed flush-before-block patch is not applicable to this reproducer.

Production-shaped DCP overlap timings with the qualified exact-four mitigation are not slower than the previous GG runtime. For example, r18 measured 2.770 versus 3.304 ms/layer at 8K/3072, 3.223 versus 3.562 ms/layer at 32K/3072, and 1.342 versus 1.604 ms/layer at the 64K cached tail. A separate single-GPU mixed-Trellis benchmark reproduces the remaining prefill regression without NCCL, so the four-channel mitigation is not the observed 7.9-9.8% prefill gap.

## Full-serving collateral is a separate open question

A full TP4/DCP2 serving arm using automatic 64-channel selection plus implicit ordering failed under a model footprint near the GB10 UMA limit. Before engine failure, the hosts logged repeated `NVRM: NV_ERR_NO_MEMORY` and systemd memory-pressure notifications; no Xid, lockup, RCU-stall, or hung-task signature was present. That incident may combine the progress failure with an independent 64-channel-per-communicator allocation/UMA-pressure problem. A proxy-pool fix must not be treated as proving automatic 64-channel serving safe.

## Requested changes

1. Remove or correct the assertion that a dedicated communicator cannot collide with the indexer merge.
2. Define and enforce a deterministic cross-communicator launch order for the CKV-prefetch/indexer overlap on every rank.
3. Configure implicit launch ordering consistently for every NCCL communicator whose operations may overlap in one CUDA context. The immediate qualified candidate uses the process-wide `NCCL_LAUNCH_ORDER_IMPLICIT=1`; enabling it on only the prefetch communicator would violate NCCL's mixed-enabled/disabled overlap contract.
4. Expose per-communicator NCCL configuration through `ncclCommInitRankConfig` for role-specific resource budgets such as `maxCTAs`, after inventorying every communicator that can overlap. This is not a license to mix implicit-ordering modes among overlapping communicators.
5. Add the weight-free two-communicator reproducer as a regression test or qualification tool.

The combined implicit-ordering plus exact-four arm passes the weight-free reproducer and is the source-correct candidate contract. It is not yet serving-qualified. Until that separate staging window is complete, the qualified production contract for this specific GB10 TP4/DCP2 dual-rail topology remains exact four channels with implicit ordering unset.

## Not a duplicate of #202

Issue #202 concerns buffer aliasing, workspace budget semantics, and an idle prefetch communicator at depth zero. This report concerns active concurrent collectives at prefetch depth greater than zero and the progress/resource contract between the CKV-prefetch and indexer communicators.
