# September 23 GLM-only qualification

User authorized stopping only sparky, buddy, rocky and lucky after Claude's
review, with the current upstream profile including 32 sequences, capture256
and utilization 0.85. Qwen on dusty/kirby and DS4 on rusty/toby stay serving.

Claude cleared the revised kit at 23:31 UTC. Local tests pass normally and
with Python assertions disabled. Shell syntax and shellcheck pass. The NCCL
foundation path is preserved explicitly; missing or changed paths fail closed.
The current R38 cluster was confirmed idle immediately before cutover.

All four R38 containers were stopped worker-first, head last with a 60-second
grace period. All exited zero without an OOM kill and remain retained.

The existing Docker archive was copied from dusty over switched f0 addresses
10.11.11.1/.2/.4/.3 using rsync without compression or conversion. All four
receiver checksums matched 615f9c36faa6ed6d8af3dec1e73570806b1f33a1c00145448b83f94d26e2d292
and all imported image IDs matched 1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc.
Only the disposable receiver archives were removed after verification. The
source archive, imported images and R38 containers remain.

Receipts: `qualification/window-20260923/`. Four bounded read-only observer
units record clocks, memory, reclaim counters and buddyinfo before startup.
Native gate passed on sparky, including source/native manifest, actual RMSnorm,
one NCCL 2.31.2 object, proxy ABI4 and SM121 FlashKDA target. Draft-head gate
passed (RMSE 0.0949-0.0955, cosine 0.9954-0.9955 at rows 1/4/32).
The device-attached CLI help gate also passed. All four candidate ranks started
worker-first at approximately 23:38 UTC. The correctness driver is running,
beginning with the bounded first-completion gate.

## Startup failure

At 23:40:11 UTC, the head worker failed while loading the MTP draft's MXFP8 MoE
experts into the Marlin format. The traceback surfaces at
`prepare_mxfp8_moe_layer_for_marlin` / `repack_weight` /
`qweight.T.contiguous()` with `cudaErrorUnsupportedPtxVersion`:
"the provided PTX was compiled with an unsupported toolchain."
This is the reporting point, not yet proof of the offending kernel. Native
smoke and NVFP4 draft-head tests do not cover this conversion path.

No first completion, semantic test, native-context test or benchmark ran.
The qualification driver was terminated explicitly; candidate logs, container
state and kernel journals were retained. Claude is reviewing the failure
read-only. R38 restoration completed, workers before head, with a correct `333`
completion and finish_reason=stop at 23:44:26 UTC. No image,
container, cache, driver or model policy was changed to hide the failure.

The readiness loop now also checks every rank's running/OOM state on each
iteration so a failed startup aborts immediately rather than waiting the full
endpoint deadline. This driver-only correction does not change the tested kit
identity or launched command.

All four candidate ranks reported the same unsupported-PTX error. Buddy logged
one NV_ERR_NO_MEMORY warning during model loading; the other candidate kernel
journals were clean. This warning is recorded separately, not attributed as the
cause of the PTX error. Host driver on sparky: 580.173.02. Candidate foundation:
CUDA 13.4; native gate passed but does not establish all PTX-only paths work.

Restoration checks confirm the exact original R38 container and image IDs on
all four nodes, running and not OOM-killed. Temporary observer units were
stopped and collected, with no matching units left. R38 reported 6,236,059 KV
tokens for this boot. Restoration kernel journals recorded 24 allocation
warnings on sparky and 33 on lucky during startup, zero on buddy/rocky, and no
Xid or OOM kill. Successful recovery does not resolve these allocation warnings.

## Completed independent failure review

Claude's review completed at approximately 23:52 UTC. The leading hypothesis is
a launch integration difference, not demonstrated incompatibility of upstream
GLM support. Earlier qualified Spark launchers entered through Bash; this
runner enters Python directly. The NGC foundation ships shell initialization
that manages CUDA compatibility libraries, while the runtime layer replaces
LD_LIBRARY_PATH without the compatibility directory. Passing environment
variables through is not equivalent to executing that initialization.

Source review identifies the Marlin repack kernel as a PTX-JIT-dependent path
not exercised by the passing gates. An earlier expert's unchecked launch could
surface at the next expert's transpose. The exact failed kernel and loaded
libcuda in the failed processes were not measured, so this remains attribution
to test, not proof. Neither the claim that all prior boots used the compatibility
driver nor a need to change the host driver has been established.

Next discriminator: on an idle GPU, use the same candidate and a small Marlin
repack input, synchronizing immediately after the operation, once with direct
Python entry and once through Bash. Record libcuda mappings and driver API
version in both. No whole-model reload or benchmark is needed for this test.
No entrypoint fix or native rebuild has been applied; the fleet is serving.

## Confirmed entrypoint defect and corrected retry

User approved reopening the GLM-only window. R38 was idle and stopped gracefully
again, workers before head. Diagnostic receipts are in
`qualification/entrypoint-20260923/`.

The same image, 16x64 int32 input and real `gptq_marlin_repack` call:

| Entry | libcuda / driver API | Immediate launch status | Result |
| --- | --- | --- | --- |
| Direct Python | host 580.173.02 / 13000 | 222 | failed |
| Bash then Python | image compat 615.65.02 / 13040 | 0 | passed |
| Direct Python, only compat library search path prepended | image compat 615.65.02 / 13040 | 0 | passed |

CUDA's disk cache was disabled in every diagnostic arm. The immediate
cudaGetLastError check places the error at the repack launch, not the next
expert's transpose. The explicit-path arm isolates library selection from other
shell initialization. The Bash probe also passed on buddy, rocky and lucky.

The runner now enters through Bash, restoring NVIDIA shell initialization used
by the existing launchers. launch.py checks the exact compat libcuda mapping and
driver API before exec; verify_boot.py requires its receipt on every rank.
No model options, image, weights or host driver changed. Five local tests pass
normally and with assertions disabled. Claude cleared this correction at
approximately 00:03 UTC September 24. His blanket statements about every prior
boot's driver mapping and lack of prior expandable-segment use are not adopted:
this test measured these diagnostic arms, not all historical deployments.

The failed containers were renamed and retained with the
`-python-entry-20260923` suffix. New observer units and a corrected Bash-entry
native gate preceded the retry. The corrected boot passed all four rank
identity gates and the formerly failing model conversion. The complete
correctness driver then passed: short boundaries 7/7, semantic/vision/tool
15/15, concurrent repeats and multi-turn admission, frozen prefix pairs and
triples, and exact retrieval through 1,048,000 tokens. See QUALIFICATION.md
for cache and allocation-warning caveats.

The unchanged standard benchmark started at 00:26:51 UTC September 24, with
results in this repository and no benchmark-repository changes. Its whole-profile
comparison uses the saved R38 r04 grid, not a newly measured baseline.

The grid completed with 15 valid cells, no flags and no new timed-window
kernel or engine warnings. Claude independently reviewed the arithmetic and
timing after completing its long wait. Decode throughput is 8.9/13.4/10.0%
lower at c1/c2/c4, with engine steps 6.2/9.0/6.0% lower. Prefill remains within
1.8%. This whole profile is not recommended for promotion. A post-grid real
completion returned 333 with stop. All four exact observer units were stopped
and verified absent. Logs and telemetry are retained. The candidate remains
healthy and serving pending the requested R38 restoration decision. Qwen and
DS4 were not changed, and no commit, push or cleanup of retained containers
was performed.

The user then approved Restore R38. The candidate exited cleanly on all four
nodes; the retained R38 workers and head restarted in that order. At 00:57 UTC
a real completion returned exact 333 with stop, all original container IDs
matched and all four were running without OOM kill. One startup allocation
warning occurred on sparky, no Xid or OOM kill. See RESTORATION.md.

At the user's request, waited for Claude and then read its full follow-up.
Its proposed future discrimination is BF16 draft head, then L2 prefetch, then
MoE work source, changing one factor at a time. These remain hypotheses, not
causal findings or authorization for another serving window. No experiment
was started after restoration.

## Subsequent authorized investigation, September 24

The paragraph above describes the earlier restoration, not current serving.
The user subsequently authorized the GLM-only performance investigation and
confirmed the client pause. The completed arms, profiler and health caveats are
recorded in BISECT.md. Qwen and DS4 have not been changed in this investigation.

At 13:05 UTC the separately authorized selective-MXFP8 arm started, after the
shared-head screen/grid completed and all four prior containers stopped with
exit 0 and no OOM kill. Previous containers are retained. Current arm:
`no-prefetch-mxfp8-output-a16`, image `1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc`,
effective profile `3801fdb71cf032381bd994d3d23adb457bcd130317681793aa817d5e4afe2956`.
Exact launch helpers and full boot logs are saved under
`qualification/selective-mxfp8-output-a16-20260924-run1/`.

First completion, four-rank engagement, seven short-pool cases and 15 semantic,
vision and tool cases passed. Full long-context qualification and the first
15-cell grid subsequently passed. The grid improves decode steps 5.4/3.9/2.8%
against no-prefetch K3, but prefill falls 7.2 to 7.9%. There is no promotion.
A same-boot repeat started at 13:40:50 UTC, with receipts in
`qualification/selective-mxfp8-output-a16-20260924-run2/`. Observer units are
`glm-mxfp8-output-20260924.service` on all four GLM hosts, bounded to four hours
and to be stopped after collection. The prior shared-head observer units were
stopped before starting these. See SELECTIVE-MXFP8.md for the reviewed scope.

The user subsequently clarified that new changes absent from R38 are not wanted.
The selective quantization arm was therefore withdrawn. Only our repeat client
was interrupted with SIGINT; three cells were saved and full-grid validation
correctly failed. No repeat-completion or promotion claim is made. Endpoint
metrics then showed zero running and waiting requests. The four original R38
container/image identities were rechecked before starting restoration. No
activation-quantization arm was authorized or run. Qwen and DS4 remain untouched.

Restoration completed at 13:49 UTC. The four MXFP8 containers exited 0 without
OOM kill and remain retained. The original R38 containers and image
`ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5`
are running; a real completion returned exact `333`, finish `stop`. KV admission
is 6,277,606 tokens. Startup NV_ERR_NO_MEMORY occurred 39 times on sparky and
once on lucky; buddy/rocky had none, and no Xid or OOM kill was found. These
allocation-pressure warnings remain unresolved, not a new qualification pass.
All four `glm-mxfp8-output-20260924.service` observers are inactive/dead after
receipt collection. Evidence is in the interrupted run2 receipt, including
original R38 inspections, completion, logs and kernel journals. No R38 benchmark
was rerun and no benchmark-repository files were edited.
