# Upstream check: R38 and Qwen QAD

Read-only inspection of local `master` at
`9fdad773ab8ad8b0654d3fa7fe10edfd516aee24` on 2026-09-15. The active
checkout remains `spark`. Sources below were read from Git objects, not from
the working tree or remote hosts. This note does not qualify a Spark deployment.

## Three newest Qwen commits

- `4624ad2` publishes a ten-generation AA-LCR comparison and immutable runtime,
  generation-completeness and judge-summary receipts.
- `ca4441f` renames the candidate-facing `QAD-step1500` label to `QAD` and adds
  author attribution. Immutable checkpoint provenance remains unchanged; this
  is not another trained checkpoint or another evaluation run.
- `9fdad77` publishes a direct-answer arithmetic comparison, runnable test and
  analysis scripts, a frozen 600-task suite, and 9,600 per-attempt receipts.

Sources: the three commit diffs and the pinned
[Qwen model page](https://github.com/voipmonitor/rtx6kpro/blob/9fdad773ab8ad8b0654d3fa7fe10edfd516aee24/models/qwen38-flash-next.md).
These commits add evaluation documentation and artifacts, not a serving-image
recipe or a deployment promotion.

## Checkpoint and hardware boundary

The baseline is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` at
`ada4da32a583a78aa47299f45a70603c950490b8`. QAD is a distinct local export,
`Qwen3.8-Flash-Next-NVFP4-QAD-step1500-v1`, after 2,500 routed-expert trunk
updates and 1,500 joint-refinement updates. Its manifest describes NVFP4 routed
experts with MXFP8 shared experts and attention. The QAD index SHA-256 is
`c528e5628a0f448edc52023933c207a8dbded850afdd960d893c9171a7665dde`.
The report explicitly retains **research-only** deployment status and does not
supply a public QAD download identifier.

Both studies ran one RTX PRO 6000 Blackwell Workstation GPU per TP1 replica on
Frank2, with BF16 activations, FP8 KV, MTP3 and 262,144 maximum context. The
runtime is R36, image digest
`sha256:23ab683d7ce32083f33c163df7dfe554b59aba75bbae8f5b8e4b5a2ef590209b`,
not R37 or R38. Sources:
[QAD runtime receipt](https://github.com/voipmonitor/rtx6kpro/blob/9fdad773ab8ad8b0654d3fa7fe10edfd516aee24/models/qwen38-flash-next/validation/aa-lcr-qad-split-pool-runtime-manifest-20260915.json)
and [AA-LCR report](https://github.com/voipmonitor/rtx6kpro/blob/9fdad773ab8ad8b0654d3fa7fe10edfd516aee24/models/qwen38-flash-next/aa-lcr-nvfp4-vs-qad.md).

## What improved, and what did not

| Test | Published NVFP4 | QAD | Interpretation |
|---|---:|---:|---|
| AA-LCR, 100 questions, ten generations each | 77.5% | 79.4% | +1.9 points; question-cluster 95% interval 0.0 to +3.8 touches zero |
| Source `99 times 17` reproducer, reasoning disabled | 25 errors / 4,200 | 2 errors / 4,200 | Large reduction, not elimination |
| 600-task arithmetic suite, reasoning disabled | 77.17% | 78.69% | +1.53 points; task-cluster 95% interval +0.50 to +2.56 |
| Same arithmetic suite, low reasoning | 99.58% | 99.67% | Low reasoning matters far more; checkpoint difference interval includes zero |

AA-LCR is a local reproduction, not an official leaderboard entry. It uses
`xhigh` candidate reasoning and one Luna-medium verdict per answer. Its public
receipts establish declared completeness and aggregate results, but raw answers
and judge verdicts are retained externally under `/mnt/luke/evals`. Arithmetic
has stronger local auditability: all 9,600 generalized-suite attempts are in
the repository, and scores use algorithmic integer answers rather than an LLM
judge. The attempt journal was independently counted and its aggregate correct
counts agree with the report. The source-reproducer raw receipts remain
external. Sources: [AA-LCR comparison JSON](https://github.com/voipmonitor/rtx6kpro/blob/9fdad773ab8ad8b0654d3fa7fe10edfd516aee24/models/qwen38-flash-next/validation/aa-lcr-nvfp4-vs-qad-ten-generations-20260915.json),
[arithmetic report](https://github.com/voipmonitor/rtx6kpro/blob/9fdad773ab8ad8b0654d3fa7fe10edfd516aee24/models/qwen38-flash-next/direct-arithmetic-stability-nvfp4-vs-qad.md),
[analysis JSON](https://github.com/voipmonitor/rtx6kpro/blob/9fdad773ab8ad8b0654d3fa7fe10edfd516aee24/models/qwen38-flash-next/validation/direct-arithmetic-stability-analysis-20260915.json)
and [attempt journal](https://github.com/voipmonitor/rtx6kpro/blob/9fdad773ab8ad8b0654d3fa7fe10edfd516aee24/models/qwen38-flash-next/validation/direct-arithmetic-stability-attempts-20260915.jsonl.gz).

Neither evaluation isolates checkpoint weights from the complete serving
system. There is no BF16 control, no Spark TP2 arm, no cross-engine arithmetic
arm and no demonstrated throughput gain. Both checkpoints still fail over 90%
of the generalized two-operation attempts with reasoning disabled. These limits
are explicit in the two reports linked above.

## Implication for the Spark profile

Our local [Qwen R37 launcher](launchers/serve-qwen38-flash-next-jj-r37-spark.sh)
defaults to a different repository,
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89`, revision
`c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d`, at TP2. Its model identity and
topology must not be substituted for either upstream evaluation arm. This is a
static launcher comparison, not a fresh check of the live service.

Inference: these commits justify tracking QAD as a future checkpoint candidate
and expose a useful bounded arithmetic regression suite. They do not establish
a drop-in quality or performance improvement for the existing Spark TP2
deployment, and they do not require changing the in-progress R37 build.

## R38 supersedes R37 in local master

The latest documented community release at this master commit is
`jovian-judgement-community-20260914-r38`, published by `99de462`.
Local master, upstream/master and origin/master resolve to `9fdad773`;
this check did not fetch remote branch tips. The R38 registry receipt pins
`sha256:f41ca8bb10bb3a125a50340d70d39ad4b7f5605f3fcb661bc992ed0bc4701a00`.

The [R38 release](https://github.com/voipmonitor/rtx6kpro/blob/9fdad773ab8ad8b0654d3fa7fe10edfd516aee24/models/deepseek-v4.1-flash.md)
reports matched R37/R38 DS4.1 TP4/DCP1, RAM Engram, DSpark K7 results on four
stock-clock RTX PRO 6000 Workstation GPUs. C1 output rises from 226.37 to
259.27 tokens/s; C8 aggregate output from 592.63 to 810.61; uncached 32K
prefill from 20,950.29 to 21,035.18. KV pool tokens rise from 1,546,785 to
4,569,816. These are combined source/default effects, not isolated kernel
causality, and not Spark or Qwen/GLM qualification. C8 verifier rate sums
per-request rates rather than counting physical engine iterations. Acceptance
ratios also change denominator from proposed to actually verified positions.

Notable release changes are compatible padded-graph pricing for adaptive
verification (#748), configurable sliding-window pages with default 256/128
main/SWA geometry (#747/#749), preserved decode-row planning capacity and
optional NVMe Engram controls (#736/B12X #360). Whole-model SSD throughput and
the retained GLM/Qwen/DS4 serving entrypoints were not requalified in this release.

## Implications for the next Spark build

Comparing the published R37/R38 source locks: vLLM changes to `66c29357` and
B12X to `ce419b52`. FlashInfer `803c4664`, LMCache `29bc5a2e`, FlashKDA source
and patch, CUDA 13.3, Torch 2.13 and the three dependency patch bytes remain
unchanged. The dependency patch installer itself changes. The removed R37
FlashInfer component is not recoverable merely because its source pin matches.
The corrected 20-job, one FlashInfer NVCC-thread recipe remains relevant.

The native csrc and cmake tree hashes are unchanged, but CMakeLists and the
published stable extension hashes change. Direct reads of the pinned
[R38 CMakeLists](https://github.com/local-inference-lab/vllm/blob/66c293578412417476f842c1da5805d3a3d959a8/CMakeLists.txt)
and R37 counterpart show a pre-SM90 CUTLASS build option and timestamp-preserving
staging of a patched Torch header. This is build-system work, not evidence of
a new SM121 kernel. Native reuse still needs an explicit input audit before
composition; do not call the entire port a Python-only refresh.

The new daily summaries report an R38 strict-tools/xgrammar assertion failure
on Qwen and DS4. Direct inspection of the frozen
[R38 scheduler](https://github.com/local-inference-lab/vllm/blob/66c293578412417476f842c1da5805d3a3d959a8/vllm/v1/core/sched/scheduler.py#L2709-L2726)
confirms the reported clamp followed by `assert num_accepted <= num_draft_tokens`
is present. This confirms source exposure, not local reproduction or a validated
fix. The September 15 summary mentions a proposed clamp workaround; do not apply
it merely to suppress the assertion without validating the accounting contract.

The same summary announces nightly multiarch JJ images. Treat this as a lead:
registry architecture, SM121 kernels, source provenance and serving gates have
not been inspected. It does not establish an official qualified Spark R38 image.

Recommendation: retarget candidate planning to R38, carrying forward the R37
gate and artifact-retention fixes. First investigate strict-tool accounting and
native inputs, and inspect the multiarch offering before choosing a build path.
No build, host change, source checkout mutation or serving change was performed.

## Follow-up after restoring R32 serving, 2026-09-15

GitHub's live documentation master still resolves to `9fdad773`; R38 remains
the latest documented community release. The following updates refine the
candidate decision, without changing the restored Qwen or GLM services.

### Verified-count ownership fix is now merged

[vLLM #756](https://github.com/local-inference-lab/vllm/pull/756), source commit
`27f0745fc11f2346bacb646c2d4e1864f6dacd89`, merged on September 14 at 23:11 UTC
as `4616abf67e638dfc102d20e9821c7e7b33b83a0d`, after the R38 source cut.
The entire R38 `async_utils.py` to PR-head file diff is exactly the ownership
fix: clone the request-sized verified-count tensor on the model stream,
retain it in the output object and copy that snapshot on the output stream.
This prevents the next model step from overwriting reusable capacity storage
before CPU delivery. The accepted-count assertion is unchanged.

The PR adds two CUDA regression cases, one with deterministically deferred
delivery and one delaying the CUDA copy stream. Its recorded serving checks
cover DS4-0731 K5 and Vision K3, with additional independent fixes present in
those serving images. These are upstream receipts, not tests run in this review.
The report explicitly does not claim a field-identical reproduction of the
original private traffic.

The [R38 call site](https://github.com/local-inference-lab/vllm/blob/66c293578412417476f842c1da5805d3a3d959a8/vllm/v1/worker/gpu/model_runner.py#L2100-L2105)
only supplies this tensor when adaptive verification is active and draft counts
are present. The manager returns a view into `_batch_draft_capacity`. Therefore
the fix proves a real adaptive-verification ownership defect, not a blanket
explanation for all structured-output failures, nor a demonstrated Qwen fix.
Retain strict-tool and mixed-length serving checks in our candidate gates.

### Do not replace the release with mutable JJ head

[Issue #763](https://github.com/local-inference-lab/vllm/issues/763) records a
post-R38 prepared-kernel test image with about 59% lower uncached prefill than
R38. The report attributes that measured case to repeated CPU scratch planning,
not slower GPU MoE kernels. B12X `40bcdf82` implements descriptor retention,
but serving recovery is still marked pending. This is not evidence that R38
has that regression. Select the narrow #756 commit on frozen R38 rather than
implicitly accepting all intervening prepared-kernel changes.

#758 is not needed on R38: its own source report says R38 already returns the
retained DS4 cache-page view. That bug was introduced later in JJ.

### Spark-specific comparison and build boundary

Directly downloading and decompressing the GB10 profile at both frozen R37 and
R38 B12X commits yields identical parsed JSON. The changed compressed blob is
not a new GB10 tuning policy. No Spark-specific throughput improvement is
established by the R38 documentation.

Direct endpoint source comparisons also find `setup.py` and
`tools/build_rust.py` changes that exclude source-addressed wheel tags from
version discovery. Together with the CMake changes above, these need explicit
native-input exceptions or rebuilding, not an unchanged-input assertion.
Published csrc and cmake subtree identities remain equal. Full native reuse
authorization still belongs to the actual Spark composition and ABI gates.

The GitHub compare APIs report divergent source histories for R37 and R38.
Their merge-base-oriented commit/file lists were used only as discovery aids;
the source facts above rely on direct endpoint reads or published tree hashes.

Recommended candidate: frozen R38 vLLM/B12X, pinned #756, our existing SM121
overlay and unchanged qualified serving profiles, carrying all corrected R37
build gates and component-retention receipts. Test the two #756 CUDA cases
before the usual semantic, strict-tool, concurrency, context and benchmark
qualification. Do not remove the assertion or use the forum clamp workaround.
