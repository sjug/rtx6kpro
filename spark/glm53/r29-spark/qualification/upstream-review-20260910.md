# Upstream master review, 2026-09-10

Read-only source and release review. No composition, build, node operation,
benchmark, commit, push or external post. The research skill was used to
delegate the separate model/community-lead investigation.

## Boundary

Local master and live `git ls-remote upstream refs/heads/master` both resolve to
`59f01d1c3ab25c7a6d39c4e56de75c9df2f63944`. Compared with the previous recorded
check `0df5fbc8ef198ace5fff4a277fc0f4876316558a`, six commits changed 39 files,
all documentation, qualification data or diagnostic scripts in this repository.
The checked-out spark branch is not rebased onto this master: rev-list reports
26 master-only and 34 spark-only commits. No refs were changed.

The GitHub web-rendered history returned a stale August page. The live Git
remote hash, local pinned documents and authenticated GitHub source API were
used as the authorities, not that cached page.

## Latest published release: shared JJ R32

Tag `localinferencelab/vllm:jovian-judgement-community-20260909-r32`.
Published registry receipt digest:
`sha256:c9ad4a6ef4aa55232df9ed1a37e85d94eb8c7d5349561a6cbe828e72b61de83c`.
This review did not inspect registry platforms or establish an ARM image.

| Component | R29 | R32 |
|---|---|---|
| vLLM | 45361846 | 55769270 |
| B12X | 3edbcbce | unchanged |
| LMCache | dcd6ec92 | 35ad809f |
| FlashKDA base/patch | 3b225bf2 / a9537532 | unchanged |
| Upstream FlashInfer | 803c4664 | unchanged |

Source: [R32 report](https://github.com/voipmonitor/rtx6kpro/blob/59f01d1/models/glm-5.3-flash/validation/concurrent-checkpoints-r32.md),
[lock](https://github.com/voipmonitor/rtx6kpro/blob/59f01d1/models/glm-5.3-flash/validation/concurrent-checkpoints-r32.source.lock),
[registry receipt](https://github.com/voipmonitor/rtx6kpro/blob/59f01d1/models/glm-5.3-flash/validation/concurrent-checkpoints-r32.registry.json).

## R30: relevant correctness fixes and revised defaults

1. GLM sparse-pool tail selection, #715, integration commit 57079279. Old
   expansion appended incomplete pools at fixed offset 2048 even when fewer
   history entries were active. The fix appends at min(2048, complete_pools*4),
   placing the last one to three tokens inside the consumed selection prefix.
   The published five-token numerical oracle fails on R29 and passes patched;
   eager/graph tests include short sequences, padding rows and DCP routes.
   This is not a throughput-only fix and has no SM120-only predicate in the diff.
   A Spark GPU test remains necessary; this review did not reproduce a live fault.
2. Mamba cleanup, #718, integration b72ba34a: iterates across null gaps instead
   of stopping retirement prematurely, tracks the already-retired prefix and
   clears its tracking on free. The align-mode manager is relevant to our
   Qwen/GLM hybrid profiles. It is a lifetime/memory correction, not a new kernel.
3. Local queued-checkpoint admission, #721, integration e667409c/60e72555:
   protects checkpoints that already-queued requests need while allowing active
   work to progress. It requires an actual request-boundary checkpoint, no KV
   connector, DCP1 and supported geometry. Our explicit aligned policy disables
   request-boundary checkpoints in config/vllm.py, so the central deferral path
   is not an immediate benefit of our unchanged serving profile.
4. GLM launcher defaults become temperature=1, top_p=0.95, high reasoning and
   clear_thinking=false. This reverses R29's upstream clear_thinking=true preset
   and preserves prior reasoning. Our Spark launcher never adopted that R29
   history-clearing preset. Sampling/default changes must still be separated
   from source performance comparisons.

Primary sources: [R30 report](https://github.com/voipmonitor/rtx6kpro/blob/59f01d1/models/glm-5.3-flash/validation/shared-serving-r30.md),
[exact vLLM source comparison](https://github.com/voipmonitor/vllm/compare/45361846d60622cb5211b902bc893963e5a9eaa6...5576927057cf71b6ec61d120932338b333efa089).

The long-history evidence is bounded: at identical preserved input, no-spec
top_p=1 degenerated in 2/3 samples; top_p=.95 in 0/5, plus three valid final-image
DFlash requests. The report explicitly does not establish a universal numerical
repair. The daily summary's definitive corruption-root-cause wording overstates
that evidence. The report also notes some coding evaluations deliberately use
top_p=1. Keep client override semantics and do not silently change reasoning.

## R31: less warmup allocation, plus LMCache RAM retention

vLLM #724, integration 55769270, allocates one input/output/weights/scratch set
per MoE warmup regime and shares it across the int32/int64 route-ID launches.
Stream ordering and retention through synchronize are preserved. This can lower
startup allocator high-water on constrained systems. The author's roughly
0.9 GiB TP8 recovery is not a measured Spark or TP4 gain. The release's measured
TP4 KV capacity did not change, and verifier speed was essentially unchanged.

LMCache retains filesystem-restored objects in bounded host RAM and fixes
headroom allocation with ownership-safe eviction (#66). This is host-cache
retention, not GPU L2 prefetching. On GB10 the RAM is the same physical pool
as GPU memory, so this is not free capacity and remains disabled in our baseline.

Source: [R31 report](https://github.com/voipmonitor/rtx6kpro/blob/59f01d1/models/glm-5.3-flash/validation/warmup-retention-r31.md).

## R32: LMCache concurrency fix, not a new serving kernel

LMCache #65 retries admission when another generation still owns a shared
immutable page write. Previously the waiting checkpoint could be discarded,
causing later recomputation. Retry happens in the background transfer, not by
blocking the model thread. R32 retains R31 vLLM, B12X and native code unchanged.
Without LMCache, R32 adds no serving behavior over R31.

Important open limitation: [vLLM #726](https://github.com/local-inference-lab/vllm/issues/726)
is still open, zero comments at this review. Concurrent MTP3 strict-JSON requests
can fail with HTTP 500 under LMCache on both R31 and R32. A GPU-only R31 control
passed 32 requests, which is not a proof of universal safety. Do not qualify
constrained output by citing cache-byte or ordinary tool-call tests.

## Spark build implications

The exact GitHub API comparison shows seven vLLM commits changing fifteen
Python files (five implementation files, ten tests), no csrc/CMake/setup/native
changes. The LMCache R29-to-R32 comparison has three commits and nineteen Python
files, with no native-source changes. B12X and native lock identities are stable.
Our source-first R29 native artifacts are therefore plausible reuse inputs,
subject to byte/native-input gates, not a reason to rebuild CUDA blindly.

The next sensible candidate is R32 with the existing Spark architecture,
RoCEnante/NCCL settings, BF16 GLM draft head, aligned checkpoints and LMCache off.
Prioritize the short-pool oracle, recurrent cleanup and warmup contracts, followed
by the established Qwen/GLM semantic, cache, native-context and run_bench battery.
Keep publisher sampling changes a separately identified profile control.
No source/kernel improvement supports promising a substantial throughput gain.

New R32 references in Qwen and DS4 runbooks do not mean new GPU measurements.
Those documents explicitly retain R29/R28.1 evidence; Qwen TP2 on the upstream
shared R32 image remains unqualified. Our own Spark qualification is separate.

Further model/PLE/QAD leads are recorded in upstream-leads-20260910.md by the
background research agent. They must not be described as included in R32 unless
the locked source identities actually contain them.
