# Bounded GLM decode profiling

The clean K2 repetition runs before this diagnostic. Profiling is not a
benchmark score and cannot establish a performance improvement.

## Questions, in order

1. Does target MoE execution dominate the residual decode cost? If so, the
   trace should show dynamic NVFP4 expert kernels taking the largest share
   of GPU time, rather than draft or communication work.
2. Does the Marlin draft and its proposal head consume the savings from
   additional accepted tokens? K5 already loses throughput; K2 provides
   the shorter-window discriminator. Capture both target and draft kernels.
3. Is communication or waiting between kernels material on the switched
   four-node fabric? Compare all four rank traces rather than summing rank
   times or treating host waits as GPU arithmetic.

## Diagnostic contract

Use the existing image and the no-prefetch K3 profile. Add only the Torch
profiler configuration: frontend ignored, stack/shape/memory/FLOP capture
off, five worker steps, delayed until established decode. Keep graphs,
autotuning, sampling, checkpoint policy and model identity unchanged.
Trace files go under the existing writable cache mount, not tmpfs.

Run C1 and C4 separately with the existing run_bench.sh and a profiling
workload label. Trigger profiling only after metrics show the requested
concurrency running with no waiting requests. Keep the request going until
all ranks finish their bounded capture. Save the trace and capture logs
inside this kit's qualification receipts. Never compare instrumented
throughput with the normal grids.

Before a restart: finish and retain K2 receipts, verify idle, stop workers
before head and retain the containers. Validate the finite diagnostic
profile and all four launch identities, then require a real completion.
After captures: stop profiling, inspect all four ranks and kernel journals,
and use measured hotspots to choose the next single-factor experiment.

Qwen, DS4, the benchmark repository and the retained R38 rollback remain
untouched. No new image is required for this diagnostic.

## Independent review, September 24

Claude reviewed the retained grids and pinned sources, and supports completing
this profile. Read draft head GEMMs, Marlin draft MoE, eager recurrent-state
commit work, RoCEnante kernels and inter-step idle separately. Worker traces
do not directly capture the scheduler in the EngineCore process; a gap is a
lead, not proof of scheduler cost. Compare ranks rather than summing them.

The strongest proposed next arm is `VLLM_MTP_NVFP4_LM_HEAD=1` on the
no-prefetch K3 base. R38's real boot log records online NVFP4 LM-head
quantization with BF16 activations. Its implicit default enabled the MTP
shared-head quantization. Karmic's launcher explicitly disables that path and
uses the separate GLM draft-only FlashInfer copy instead. At pinned vLLM
e77be225, mtp.py constructs the runtime-quantized shared head under this flag;
prepare_draft_lm_head detects it and does not create the separate copy. The
target model's own head construction is unchanged. This distinction was
independently checked after review and is a better-founded arm than further
queue or PDL sweeps. A boot must prove which head actually engaged.

Caveats retained: a single step-rate result within one percent is not proof
of parity; extrapolating K1 from K2/K3/K5 is not a measured exclusion; and the
sampling differences between historical R38 and this profile do not establish
the direction or magnitude of an acceptance change. Do not change sampling to
raise a score. Repeat any finalist and report acceptance alongside throughput.
