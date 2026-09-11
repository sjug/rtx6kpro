# R29 GLM execution

2026-09-09 11:43 EDT: transfer-only service active on workstation,
`jj-r29-glm-transfer.service`, invocation
`909bbbe0b7a24298aeb0e47c1bdcfa11`.

All four hosts were verified serving R28, with switched fabric ownership and
adequate disk space. The R29 GLM launcher differs from R28 only in release
identity and preflight naming. Runner tests and shell syntax passed.

The existing Docker archive is being copied over switched 200G from dusty,
without compression or format conversion. Each receiver checks archive SHA
and image ID. No serving container is stopped by this service.

The combined cutover/qualification service was rejected by the execution
approval check because stopping the serving production cluster requires
explicit approval. It was not launched. R28 continues serving. The prepared
driver will stop workers before head, retain R28 containers, start R29 workers
before head, then run first-completion, semantics, concurrency, frozen prefix
pairs/triples, native context and the standard grid with matched settings.

Await explicit cutover approval. Do not run the combined driver while the
transfer-only service is active. Qwen and DS4 are untouched.

## Approved cutover, integrity failure and R28 recovery

User explicitly approved the cutover. Transfer completed with identical image
IDs on all four nodes. Qualification service invocation
`137b9a27ff3847c8bbea8ba878ccddc4` began at 12:53:42 EDT. R28 shutdown began at
12:53:48, workers first, head last. No R29 container was created: the first
worker runner refused the GLM launcher identity.

Host and installed R29 launcher SHA both equal
`eb7139772d23e21add2dac63a6683300fdd99b7590a98553f36076ede289efd0`.
The image ENV and label equal R28's launcher SHA
`b5c1cdc827148257d96a0d60a22276dd524fb098cbd52e94aaef252f8a1f3b1e`.
The Dockerfile reused an ARG name already present as an inherited ENV. The
new build argument is named `R29_GLM_LAUNCHER_SHA256`, avoiding that collision.
The immutable current image remains unchanged and unqualified for GLM.

Added a no-GPU gate checking lock, host bytes, installed bytes, ENV and label.
It fails with exit 1 on the current image. Future build publication and the
pre-cutover phase require it. Nine source contracts pass after updating the
Dockerfile digest and derived fingerprint. The original built source lock is
preserved in `qualification/built-ee996ef8.source.lock.json`; the working lock
now describes a future rebuild, not image ee996ef8. No rebuild has run.

Restored the preserved R28 containers, workers then head. At 13:01:32 EDT the
completion probe returned exactly `333`, finish_reason stop, with an R28
system fingerprint. All four restored containers were running. Recovery
receipt: `glm-20260909/r28-restored-completion.json`. About eight minutes from
first shutdown to verified recovery. No R29 GLM qualification or benchmark
was executed. Qwen and DS4 stayed untouched.

## Authorized rebuild and continuation

User approved pausing Qwen for the corrected rebuild. The Qwen worker and head
were stopped in order; their image and containers were preserved. The rejected
R29 tag was removed on dusty only, without deleting its image or archive.

Build service `jj-r29-metadata-rebuild.service`, invocation
`b9ca593c7774439e9803e6063c088a20`, receipt
`build-receipts/20260909T173029Z-765802` is running on dusty. Local and remote
contracts, runner tests and dry run passed before shutdown. Native compilation
is rerunning because the corrected Dockerfile invalidated that build layer.

Workstation continuation `jj-r29-metadata-continuation.service`, invocation
`929e161fa5ea45bea164929a40d7d504`, waits for this exact receipt. Once the build
ends it restores the tested Qwen containers at their original image ID and
requires a correct completion. GLM proceeds only on BUILD-OK and source-lock
identity: save a new image-ID-named Docker archive, transfer on 200G, validate
all receiver IDs, run the no-GPU launcher identity gate on every receiver before
shutdown, then retry aligned MTP3 qualification and the historical R28 comparison.
Receipts are separate under `glm-metadata-rebuild-20260909`.

GLM remains on restored R28 during the rebuild; DS4 is untouched. The corrected
GLM image is not yet built or qualified. No commits or pushes were performed.

## 13:49 EDT: corrected build passed

Build completed at 17:49:36 UTC with exit status zero and BUILD-OK. Corrected
image ID: `0b15723cb87646bb4628c5cd67aa2a7c879ac5a27105109ea5db06c30113dc0b`.
Source-lock SHA: `3fb73885dd8f5b5e40de1342c63aeac37c075e6cee3a86548f9abff889b88754`.
The new GLM launcher identity check passed before GPU gates. The FlashKDA
matrix passed all 12 required cases without skips. Qwen's tested original
containers were restarted worker first, then head; meaningful completion is
pending while model loading proceeds. GLM remains on R28 pending distribution.

## 14:13 EDT: corrected R29 GLM admitted

Qwen recovery passed before distribution. All four GLM nodes loaded corrected
image `0b15723c...` with exact image-ID verification. The no-GPU launcher
identity test passed on every node before cutover at 14:07:37 EDT.
R29 started workers first and sparky last. First completion and all four runtime
identity checks passed; startup reports exactly B12X_ROCENANTE then PYNCCL.
The semantic battery passed all 15 cases over three repetitions. Concurrency,
prefix diagnostics, native context and the standard grid remain in progress.
Boot capacity is 6,196,490 tokens with max model length 1,048,576.

## 14:39 EDT: qualification complete

Continuation and qualification exited zero at 14:39:33 EDT. Corrected R29 stays
serving on all four GLM nodes. All 15 semantic checks, concurrency and head-of-line
probes, frozen prefix pairs/triples, exact retrieval through 1,048,000 tokens,
the standard 15-cell grid and final correct completion passed. Full results and
limitations are in GLM-QUALIFICATION.md. Qwen was restored to its original
qualified R29 image before GLM cutover; DS4 was untouched. R28 rollback remains
available. No commit or push was performed.
