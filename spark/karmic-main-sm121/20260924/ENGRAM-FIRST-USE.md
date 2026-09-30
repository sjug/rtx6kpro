# Engram first-use investigation, 2026-09-25

Status: localized, not fixed. Qualification remains incomplete.

The clean ring repair passes the previously failing unrelated-content transition
twice. Its first cold 385-token request still differs from subsequent identical
cold requests at output token zero. All answers are correct, but top logprobs are
not identical. The timing derivative reproduces that residual.

## Frozen evidence

- Clean ring image: `e7b273407022ae74726d6c7b7465aa567b2c5a0594d5e0b46002a742a11f5146`.
- Timing image: `32f93547a3c50a8b72ecfcbd8f0f2f0685bff34cfb4213b77748db80b5a6bebb`.
- Timing lock: `e737a22db6226ab6ca6f6cadfe6393073469d1b8a77e647d8e628cf82f50a10a`.
- First-use receipts: `receipts/engram-timing-20260925T170409Z/`.
- Host events/stacks: `receipts/engram-host-timing-20260925T170403Z/`.
- Additional unseen lengths: `receipts/ring-fix-new-shapes-20260925/` and its
  adjacent `ring-fix-new-shapes-observed-20260925/` telemetry.

Both image arms use TP4/DCP1, K7, 600K maximum context, utilization 0.85,
deterministic MoE and disabled dense split-K turbo. Timing changes no model
arguments or native/B12X bytes. Its vLLM and Inductor caches are new; inherited
B12X/Triton cache existence is recorded per node. Timing and stack dumping can
change scheduling, so this is an instrumented reproduction, not a performance A/B.

## First 385-token prefill, Engram job 384

| Rank host | IDs-ready host wait | Native disk read (two tables) | First lookup launch host call |
| --- | --- | --- | --- |
| dusty | 198 ms | 2.3 / 2.3 ms | 5025.745 ms |
| toby | 230 ms | 10.5 / 10.7 ms | 5017.966 ms |
| rusty | 227 ms | 10.4 / 10.8 ms | 5016.889 ms |
| kirby | 192 ms | 2.9 / 4.5 ms | 5023.971 ms |

On dusty, the next lookup kernel launch takes 0.078 ms. The ready epoch is
published only after the delayed first launch. The request completes in 5.95 s;
later identical requests take 0.32 to 0.38 s. This is not a five-second disk read.

Dusty's stack dump during that job places the lookup thread in Triton's
`backends/nvidia/driver.py:331`, the native `self.launch(...)` call. The main
thread is in B12X `gemm/bf16_gemv/_preparation.py:71`, `torch.mm(x, weight.T,
out=out)`, called by the DS4.1 indexer's `wk(latent)` projection. Python frames
cannot establish which CUDA API or driver lock blocks underneath either call.

The source's `_wait_engram_rows` kernel times out after 5000 ms, writes a failure
flag, then allows the forward to continue. This matches the observed interval,
but the event timestamps alone do not directly read the timeout flag for this
job. Earlier sticky-flag observations cannot substitute for that per-job check.

## Narrowing results

Six 128-token and six 16K requests are bitwise identical within their respective
groups. Only first 385 differs. On the same warmed timing boot, three requests
each at previously unused lengths 386, 387, 513, 1025, 2049 and 8191 are correct
and bitwise identical within each group. Thus an arbitrary new row count does
not inevitably trigger the failure. It remains possible that the 385 request
first selects a kernel/algorithm family subsequently shared by those lengths.

## Next discriminator

Lazy loading or another CUDA synchronization/driver-lock interaction is a
hypothesis, not established attribution. NVIDIA documents that lazy loading can
serialize kernels and deadlock programs relying on cross-kernel concurrency;
preloading or eager loading is an available discriminator:
[CUDA lazy-loading hazards](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#impact-on-concurrent-kernel-execution).

The next controlled arm must retain the same requests and overlap contract,
record effective loading policy and memory, and compare first-use behavior.
Regardless of whether eager loading removes this instance, a timeout must not
silently publish an answer calculated from unavailable Engram rows.

## Eager arm startup observation

The same timing image is booting with only the explicit loading-policy control
`CUDA_MODULE_LOADING=EAGER`. Receipt: `receipts/engram-timing-20260925T171846Z/`.
It has not produced a serving result at this observation. The B12X and vLLM
caches now exist from the previous boot, so this is not a cache-cold comparison.

At about seven minutes after launch, dusty had about 112 GiB MemAvailable and
the API process used about one CPU core. Two brief debugger samples of its main
thread returned PCs `0xe872fa2f2550` and `0xe872fa2f10c0`. Both fall inside the
process's executable mapping `0xe872f9ed0000..0xe872fbce0000` for
`libnvidia-ptxjitcompiler.so.615.65.02`. This establishes active driver compiler
work at those samples, not a CUDA deadlock or a proof of forward progress.
The native backtrace was not usable across the container namespace, even with
its filesystem supplied as sysroot. Both debugger attaches detached normally;
the second paused startup for approximately four seconds. No inference was
running and neither sample can be used as a startup performance measurement.

Claude's follow-up confirms the five-second lookup-call delay on all ranks.
The warmup's separate IDs-ready stalls and the sticky timeout flag mean that a
repair must use per-step failure identity and cover warmup as well as requests.
A proposed fail-closed patch is being developed separately; it is not deployed.

The completed-arm comparison can be replayed without a GPU:

```sh
python -m unittest test_compare_engram_arms -v
python compare_engram_arms.py \
  --arm receipts/engram-host-timing-20260925T170403Z \
  receipts/engram-timing-20260925T170409Z/repeatability/report.json
```

All six parser tests pass. The recorded arm reports 128 and 16K identical,
385 non-identical, all answers correct, and exactly four launch calls above
one second: job 384 at 385 rows, one per node (5016.889 to 5025.745 ms).
An eager arm can be supplied as a second `--arm` only once its driver and log
collection have completed. The tool checks receipt completeness; it does not
infer GPU completion, timeout-flag state or causality from host-call durations.

## Fail-closed proposal review

Claude's initial proposal (lock SHA prefix `76eaf0daedda58cf`) carries one
leading flag row through the existing Engram all-reduce and snapshots the
result into the async output. Its 13 structural and nine CPU-emulated tests
do not prove compilation or graph compatibility.

Independent compile probe on 2026-09-25 reproduced a blocker: invoking the real
patched `_fault_io_rows` in `torch.compile(backend="eager", fullgraph=True)`
with CPU tensors fails with `UserError: DataPtrVariable() has no type` at
the `staged_rows.data_ptr() != io[1:].data_ptr()` comparison. The probe used
the existing `.venv-snapshot-cpu` and the patch test's AST extraction helper.
This is a source-mechanics failure, not a GPU failure. The proposal is not
accepted for build until the invariant is checked at a compile-safe lifecycle
boundary and the compile probe becomes a regression test.

The review also requests explicit tracing of no-output/chunked-prefill and
speculative epoch ordering. Warmup failures being logged by the proposal is
not permission to qualify a final image with Engram timeouts; the progress
repair must remove those failures as well as prevent silent output.

## Eager arm outcome: admission failure, no determinism result

At 17:40:30 UTC all four workers rejected startup before weight loading:
free device memory was 92.15 GiB on dusty, 95.91 on toby, 95.58 on rusty and
95.59 on kirby, below the unchanged utilization requirement of 103.44 GiB.
The eager arm therefore cannot answer whether eager loading removes the
first-request defect. No utilization reduction was made to force admission.

All four containers exited with code 1, `OOMKilled=false`; host MemAvailable
recovered to approximately 116 GiB on every node. They were retained under
the `engram-eager-admission-failed` suffix. Archival receipt:
`receipts/stop-engram-eager-admission-failed-20260925T174210Z/`.
The driver saved complete startup/final logs and disabled the timing flags.
Its final timing-log collection also failed because no Engram runtime had
started and no timing directories existed. That secondary collection error
does not replace the primary memory-admission failure.

The next narrower probe runs only on idle dusty, in isolated CUDA processes:
`probe_cuda_producer_progress.py` compares a bounded consumer/producer wait
with no GEMM, a first-use 385x512 by 512x128 BF16 GEMM, and the same GEMM
prewarmed. `run_cuda_progress_probe.py` records commands, image identity and
outputs. It does not load the model and cannot qualify DS4.1; its purpose is
to seek a cheap reproducer of the same host-launch progress hazard.
