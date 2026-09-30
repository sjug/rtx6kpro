# JJ-main Spark build window

Window authorized September 21, 2026. Build gates passed; model qualification pending.

- Recipe staged and all manifest checks passed at
  `dusty:/home/jugs/git/bld-jj-r38-spark/jj-main-sm121`.
- Reused components and foundation images verified present before shutdown.
- Qwen R32 worker `cae069abb645` on kirby stopped first, then head
  `74e75ce3baff` on dusty, using `podman stop -t 60`. Containers retained.
- Exact pre-stop container inspections saved under remote
  `build-receipts/window/{dusty,kirby}-qwen-before.json`.
- Native build started as `jj-main-vllm-build.service`, invocation
  `01290a70f8a84c0d8a1f2314beeedfbc`. Durable logs and recipe inputs are in the
  remote `build-receipts/vllm-*` directory and user journal.
- Actual compiler invocation verified: CMake `-j=20`, Ninja `-j 20`, SM121a.
- As of September 22 01:21 UTC, compilation had reached 205/407 targets.
  Runtime assembly, GPU gates and model qualification had not run.
- GLM and DS4 services untouched. No prune, reset, image conversion or deletion.

Local validation before this window: ten tests passed normally and with Python
optimization enabled, ShellCheck clean, dry runs side-effect free. Claude reviewed
the initial kit and then verified the fixes; no concrete build blocker remained.

Retention tags protect artifacts from the host's daily prune; they are not serving
qualification tags. The input build lock remains the pre-build contract; exact
outputs are recorded below and in the retained image inspection.

## First attempt

At 01:29 UTC the native wheel had compiled successfully, but normalization failed:
JJ's normalizer does not accept Karmic's `--cutlass-dsl-version` argument. No image
was published. The failed RUN combined compilation and normalization; Podman did
not retain an external build container, so the successful compile cannot be reused
from that layer. The recipe now uses the pinned JJ normalizer interface and commits
the compile result in a separate RUN before normalization. A source-interface test
was added. The retry completed; no serving or kernel policy changed.

## Completed build and gates, September 22 UTC

- Native component: `e60524a185213a45044f6f07d69836b57db778f4b0a6af7dc997e064e199f64e`.
  Retry finished at approximately 01:56 UTC with 20 compile jobs.
- Runtime: `e345783fc7e1a001e639ca42439230d6c3d2adb9ea9a5f0fa457100e700cbc7b`.
- Original serving candidate (subsequently failed TP2 startup): `e7926f763859ba5800cc24198ef297a12322fdaac93b55ac89b88abe02c1b1fb`.
  Its authorized cache-agreement derivative is
  `6633e678fee74f5e1060290a01df812d34d10edcf30e34dd1c9bf7db88260ef3`;
  all build gates passed, including the added 25-case coordinator regression.
  See `gate-evidence/cache-agreement/` and `QUALIFICATION.md` for the live retry.
- Gate service exited zero. Receipt: remote
  `build-receipts/serving-20260922T015757Z-3053878`; small gate logs and image
  inspection copied into `gate-evidence/` locally.
- Native imports, linkage, real RMS normalization, NCCL identity, proxy load,
  launcher argument/preflight checks passed. FlashKDA: 12 cases passed.
  Regression groups: 235 + 7 + 6 + 1 + 7 = 256 passed.
- NVFP4 proposal-head test passed at rows 1, 4 and 32; relative RMSE
  0.094867 / 0.095540 / 0.095373, cosine 0.995507 / 0.995440 / 0.995457.
- FlashKDA stack diagnostic grew from 1,024 to 15,984 bytes. Correctness gates
  passed; retain this observation for admission and model qualification. No
  compensating kernel patch was added.
- Local tests: 11 passed normally and 11 with Python optimization enabled.
- Candidate size: 36,784,563,020 bytes, 89 layers, as recorded by Podman inspect.
- Qwen's retained R32 containers restarted worker-first after GPU gates ended.
  At approximately 02:07 UTC a real chat completion returned `7`, finish `stop`,
  with the R32 system fingerprint; receipt in `gate-evidence/qwen-r32-restored.json`.
  KV capacity logged as 5,318,100 tokens. GLM and DS4 remain untouched.
- Restoration health: dusty logged 12 `NV_ERR_NO_MEMORY` messages from
  `_memdescAllocInternal` in the checked kernel window starting 02:03 UTC.
  Kirby had no NVRM/Xid/OOM messages in that window. Both retained containers
  remained running, not OOM-killed. These warnings are not resolved by this build.

Subsequent Qwen qualification was attempted and blocked at multi-rank preparation;
see `QUALIFICATION.md`. No candidate model request or benchmark ran. The build
gates passed but did not cover the cache-agreement protocol mismatch between the
pinned components. R32 was restored and completion-verified. No promotion,
commit or push has occurred.
