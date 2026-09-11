# GLM R28.1 and R29 upstream review, 2026-09-09

Scope: read-only review of local Git objects `ece2474` and `de1d5ba`, not a
build, live endpoint inspection or Spark qualification. Upstream report links
below identify files on local `master`, absent from the current `spark`
worktree; read them with `git show master:<repo-relative-path>`.

## Source boundary

- [R28.1 report](https://github.com/voipmonitor/rtx6kpro/blob/0df5fbc/models/glm-5.3-flash/validation/scheduler-serving-r28.1.md),
  [raw JSON](https://github.com/voipmonitor/rtx6kpro/blob/0df5fbc/models/glm-5.3-flash/validation/scheduler-serving-r28.1.json)
  and [lock](https://github.com/voipmonitor/rtx6kpro/blob/0df5fbc/models/glm-5.3-flash/validation/scheduler-serving-r28.1.source.lock)
  were published by `ece2474`. vLLM is `9ff42d83938e74018f9c255e8cfa7ca6df6921b0`;
  image digest is `sha256:52ef7badcc33918f276d778d29bd972a798297584ba776476c7c09b7bdb50e5f`.
- [R29 report](https://github.com/voipmonitor/rtx6kpro/blob/0df5fbc/models/glm-5.3-flash/validation/shared-serving-r29.md)
  and [lock](https://github.com/voipmonitor/rtx6kpro/blob/0df5fbc/models/glm-5.3-flash/validation/shared-serving-r29.source.lock)
  were published by `de1d5ba`. vLLM is `45361846d60622cb5211b902bc893963e5a9eaa6`;
  image digest is `sha256:e44e07e615287605f87bd4db916d683e39066e72a1ba94cf4149089c1ec21b49`.
  This commit adds no corresponding R29 raw JSON beside the report.

## R28.1: scheduler hardening and launcher defaults

The report describes a Python/launcher overlay on immutable R28: all fourteen
vLLM/LMCache shared libraries remain byte-identical. Compute-share timing now
pairs a primitive timestamp with the executor batch and handles an untimed
predecessor boundary without charging its residency to its successor. Untimed
queues avoid clock reads/callbacks. Uncontended decode bypasses extra admission
work. Automatic prefill lanes resolve to `min(4, max_num_seqs)` independently
of cache geometry and token budget; one lane and fixed share 0.4 remain image
defaults. Five scheduler environment controls reach the launcher with native
CLI precedence. Launcher chat reasoning changes to `high`; direct `vllm serve`
does not acquire that launcher default. These are report claims supported by
focused CPU tests, artifact inspection and live tokenization, not a repeated
six-mode numerical study. See the R28.1 report and lock above.

The principal RTX comparison is TP4/DCP4/MTP3 with FP8 KV and LMCache on stock
GPUs 0-3, not our distributed DCP1 GPU-local profile. Three 60-second C1 cells
per arm give +2.90% median output but -0.25% verifier rate, with overlapping
output ranges; the initial short C1 drop of 3.34% is preserved. C64 output is
+0.09%; nominal 32K and exact 204800-token prefill change -0.19% and -0.28%.
This supports bounded near-parity observations, not a universal gain.
DFlash2/DCP1 C8 output falls 2.39% in the short cell; JSON explicitly records
extended controls as not run. The aligned-256 DFlash control proves interior
prefix reuse and literal answers, not MTP3 aligned performance or universal
token parity. See the R28.1 report and raw JSON above.

Four-lane mixed traffic improves late-short-request TTFT while delaying long
prefills; the longest observed decode gap increases from 0.349 to 0.695 s.
Cold-signature JIT also produces a retained 1.080 s gap in a separate collision.
Thus the scheduler feature is a latency trade-off requiring workload-specific
qualification, not a reason to enable fairness or four lanes by default.
See the R28.1 report's cache/fairness gates.

## R29: chat profile, transfer correctness and shared runtime

R29 retains high reasoning and adds `clear_thinking=true`: completed-turn
reasoning is omitted while visible responses, tool exchanges and active
tool-cycle reasoning remain. A captured 1428-message conversation shrinks
from 839815 to 514963 rendered tokens. Three cold replays finish with valid
tool calls and no detected degeneration. The old capture already contained
degenerate reasoning and damaged tool-ID associations. This is bounded chat
profile evidence, not a matched-length kernel repair or general long-context
immunity. Removing historical reasoning changes prefix identity: response
checkpoints cannot be reused beyond the first omitted token. See the R29
report's output-degeneration section.

LMCache changes from `617a1b47...` to `dcd6ec92...`: asynchronous paged gathers
retain immutable host block-ID metadata until consumed, preventing wrong-page
exports when pinned metadata is reused. CPU and native delayed-stream tests
reproduce the defect before the fix and exact copies afterward; four-rank GLM
DFlash2/DCP4 transfer/cancellation checks pass. Migration requires a new empty
external-cache namespace because old corrupt payloads cannot be repaired.
This is separate from the BF16-router synchronization fix, which has passing
synccheck/memcheck and final-image Vision soak evidence but still-failed
racecheck diagnostics without vendor confirmation. See the R29 report's
LMCache and router sections and lock.

The shared GLM/Qwen/DS4 image changes packaging and vLLM/LMCache identities.
B12X's commit/tree and FlashKDA's base/patch/compiled-extension hash remain
unchanged between these two upstream locks. This is not a drop-in Spark native
compatibility proof. The Qwen NVFP4 draft-head default does not qualify or
require changing our GLM BF16 draft head. See both locks and the R29 changelog.

R29 GLM RTX measurements are TP4/DCP1 and therefore closer in topology label,
but still single-host workstation results: C1 MTP3 output +2.78%, verifier
+6.15%, accepted length 2.520 to 2.440, and 32K prefill +0.29%. These cells use
an intermediate integration image, preceding metadata/router corrections;
some host-side activity differs. They do not causally isolate the gain or
repeat final-image performance. C8/C64 and Sieve were not rerun. See the R29
GLM performance section.

## Implications for the recorded Spark production profile

The [Spark rollout record](glm-20260908/RESULTS.md) documents qualified R28
TP4/DCP1, MTP3/BF16 draft head, aligned retention, FP8 KV, 8 sequences and
4096 scheduler tokens, using FlashKDA and RoCEnante/PyNCCL across four SM121
nodes. The [runner](../run-glm53-flash-jj-r28-spark-tp4-node.sh) defaults to
`FAIRNESS_ENGINE=none`, prefill interval 8 and `LMCACHE_ENABLED=0`.
The [build contract](../README.md) preserves Spark-specific architecture,
native artifacts, transport and tuning boundaries. These are repository
records, not a fresh claim about today's live processes.

Assessment and proposed gates, not executed actions:

1. Treat scheduler changes and the chat profile as distinct candidates.
   Preserve aligned/DCP1/MTP3/BF16-head and fairness-off control parity before
   investigating automatic lanes or share. RTX DCP4/LMCache results cannot
   qualify the existing Spark profile.
2. Treat `clear_thinking` and reasoning defaults as explicit prompt-contract
   changes. Check supported low/high/max rendering, overrides, active tool
   cycles, exact prefix invalidation, representative semantic/tool output and
   cached/cold continuations before any production-default change. Match
   request bytes and reasoning settings in performance comparisons.
3. The LMCache fix matters if external caching is enabled later, but the
   recorded production profile disables it. Do not enable LMCache or replace
   cache volumes as part of an unapproved source refresh.
4. Preserve SM121 native build gates and source locks. An unchanged upstream
   FlashKDA hash does not establish availability of a compatible Spark cubin;
   a new vLLM integration requires inspection of its native/API boundaries.
5. Re-run the existing semantic, frozen prefix-pair/triple, admission/HOL,
   native-context and unchanged benchmark-grid controls in an authorized
   Spark window. Neither report establishes a compelling quantified Spark
   throughput gain or qualifies replacing the currently recorded R28 image.

No source composition, build, transfer, restart, deployment or external post
was performed for this review.

## Other master developments checked by the main agent

Local master and `git ls-remote upstream refs/heads/master` both resolve to
`0df5fbc8ef198ace5fff4a277fc0f4876316558a`. Relative to the last checked
`e39bba4`, there are six commits (five content commits and one merge), all
documentation/data in this repository. No branch was changed or rebased.

- [Qwen deployment](https://github.com/voipmonitor/rtx6kpro/blob/0df5fbc8ef198ace5fff4a277fc0f4876316558a/models/qwen38-flash-next.md):
  a real shared-image TP1 text/MTP3 qualification, portable TP1/TP2 Compose,
  mapped-host PLE, BF16 target head and NVFP4 W4A16 draft head. TP2 is explicitly
  unqualified on shared R29, as are vision and LMCache. Checkpoint revision
  b797d2e differs from our c374e7e2 pin. On Spark, mapped host RAM shares the
  physical memory pool, so do not transfer discrete-GPU capacity arithmetic.
  Our R28 launcher already defaults VLLM_LM_HEAD_A16=1; its target-head flag
  defaults VLLM_MXFP8_LM_HEAD=1, unlike the new recipe's zero. Read actual
  dispatch before interpreting that configuration difference as a defect.
- [Qwen engine comparison](https://github.com/voipmonitor/rtx6kpro/blob/0df5fbc8ef198ace5fff4a277fc0f4876316558a/models/qwen38-flash-next/validation/tp1-engine-comparison.md):
  vLLM vs SGLang C1 172.8/152.5 tok/s, C4 485.4/446.8 aggregate, Sieve medians
  239.8/205.3. Different GPUs, recurrent precision, draft heads, budgets and
  cache settings prevent a clean engine-only attribution. Generated code was
  not executed. No justification for reviving our retired SGLang experiment.
- [DS4 R29](https://github.com/voipmonitor/rtx6kpro/blob/0df5fbc8ef198ace5fff4a277fc0f4876316558a/models/ds4-jovian-community-r29.md):
  text0731 K5 and Vision K3 TP2/DCP1 profiles on the shared image, DGLIN dense
  path and attention-aware memory admission. Text verifier +4.28% versus r9
  return control; Vision -0.09%. These pre-final integration measurements
  are bounded and not a proven gain. Router #723 final-image tests and two
  600-second Vision/LMCache runs are useful correctness evidence, not a
  full long-term stability or output-quality pass. Do not copy its generic
  DeepSeek served alias over our exact 0731 model identity.
- [Daily summary](https://github.com/voipmonitor/rtx6kpro/blob/0df5fbc8ef198ace5fff4a277fc0f4876316558a/daily-summaries/2026-09/2026-09-09.md):
  Spark dual-domain NCCL, moving Q quantization before gathering, and
  TrellisMX-MXFP8 are investigation leads only. Linked Discord observations
  were not independently verified here. Its high-context buffer-root-cause
  wording is not established by the more carefully scoped R29 primary report.

The research skill prompted the separate background GLM read and this saved
primary-source note. No daily-summary claim was promoted to a verified fix.
