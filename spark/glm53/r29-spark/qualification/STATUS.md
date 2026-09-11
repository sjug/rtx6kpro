# R29 execution record

## 2026-09-09: build started

- Source replay: PASS. Spark vLLM tree
  `ec2025d80cdade8a1083e1ce61f0607bb227f5d1`, package `74b29cc...`.
- B12X unchanged from R28: `c4bfeee9f3c9457400d191c870eb2e44fbcd5c2e`.
- LMCache: `cd8a219da05eec6c7d6c24712f9ee1a9186c3567`, disabled at runtime.
- Eight local build contracts, both runner suites, dry run and shellcheck passed.
  The same contract suites and dry run passed on dusty before shutdown.
- Qwen R28 worker on kirby and head on dusty stopped with `podman stop -t 60`.
  Containers and R28 image retained. No GLM or DS4 changes.
- Build service: `jj-r29-build.service` on dusty, invocation
  `3a84dd70c5dd46ba885114efa02c38d4`.
- Persistent recipe: `/home/jugs/git/bld-jj-r29-spark`; build scratch:
  `/home/jugs/.cache/build-tmp/jj-r29`. Receipt directory created by `build.sh`.
- Source-lock digest at launch:
  `c0de4eb03dfe394b4735b82ccb85939eb710beecda716e3ad8f64684e3bfd35a`.
- Recipe is explicitly uncommitted (`ALLOW_DIRTY_BUILD=1`); its manifest is the
  recipe identity. Base checkout commit is
  `7e5dfef4cb20436e45f60e316606c6981f0b37d4`, not a claim that R29 is committed.
- Native delta: three files change the caller-owned Q output operator. Build
  `_C_stable_libtorch`, assert its new schema and execute the upstream numerical
  regression, while preserving all unrelated vLLM native products.
- Reused FlashKDA SHA:
  `69347e79224ee3a48d1d3604d302dc275a595188b89db53be2a09593dbd07d98`.
- Benchmark source SHA matches R28:
  `2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3`.
- Transfer preflight: `10.11.11.8 dev enp1s0f0np0 src 10.11.11.7`, authenticated
  as kirby using the existing host key. No image transfer yet.

No image is tagged or qualified until the full build gate passes. Qualification
drivers and campaign metadata are prepared under this directory and do not
change the recipe frozen in the running build.

## Native-gate correction and retry

The first attempt compiled and linked all 81 native build steps, then failed
the newly added CPU-side linkage gate because `libcuda.so.1` is injected by CDI
only when a GPU is attached. No candidate image was published.

The same omission reproduced with the known-good R28 object in a GPU-less
container, and resolved to `/usr/local/cuda/compat/lib.real/libcuda.so.1` with
the device attached. The build-only check now permits exactly that missing
driver; runtime checks still reject every missing dependency. A new regression
test rejects missing Torch, CUDA-runtime, and other driver-library names.
The corrected check passed against the R28 native object. Nine local contracts
and both runner suites passed before retrying.

The retry uses source-lock digest
`0c5f7bc829d92a1a47ec9f87a4aa8e332384bab1d015a07b3e54e7b281bad41c`
and service invocation `57497fce27de4d20aab4ebd9a57c0564`. Source trees and
serving settings are unchanged. Compilation and validation are now separate
Docker steps so a later gate failure does not discard the successful compile.

## Post-build execution armed

`jj-r29-qwen-qualification.service` on the workstation is active, invocation
`121850b9fff34874b8bc76008798b709`. It waits on the exact retry receipt
`build-receipts/20260909T125028Z-552785`, requires exit status zero and BUILD-OK,
and verifies the published image against the local source-lock digest.

On success it proceeds through 200G Docker-archive transfer, exact image-ID
verification, Qwen startup, semantic/native-context/acceptance/padding/cache
checks, the standard grid, MTP0 boundary control, and a separate auto-policy
control. It then restores aligned MTP3 and reruns correctness. Every failure
stops the chain with a durable status receipt. It never stops GLM or DS4, clears
caches, deletes R28, commits, or pushes.

The campaign metadata is staged in the private benchmark results repository.
Live progress is in `qualification/execution.log` and each profile's receipts;
completion is not claimed until the driver and result review finish.

## 09:09 EDT: BUILD-OK

The retry passed the full image gate and published
`localhost/voipmonitor/vllm:jj-r29-spark-sm121`, image ID
`ee996ef8e531eb6e9c3208ebe30b141dc41702da1466e6ab2c47daa666d5694f`.
The GPU-attached runtime check, new caller-owned Q-output numerical test,
recurrent-state/cache tests, MXFP8 store-wait regression, draft-head checks,
FlashKDA matrix and launcher negative controls passed. The preflight errors at
the end of the build log are expected negative tests, not serving failures.

The driver verified the source-lock label, preserved the build receipts here,
and immediately began Docker-archive export for kirby. Serving qualification
remains pending; this is a build-gate milestone only.

## 09:15 EDT: distribution verified, Qwen starting

Archive size: 33,883,202,048 bytes. Archive SHA-256:
`fbda51fcf1e0b60fabc8229e10c977feca22a41474fe0e2f392f02f5a6635ec3`.
The switched 200G transfer, remote checksum and post-load image-ID assertions
passed. Both nodes now have image `ee996ef8...` under the normal R29 tag.

The driver started kirby's worker and then dusty's head at 09:15 EDT, using
aligned MTP3 and the unchanged Qwen checkpoint/profile. Model startup and first
meaningful-completion verification are in progress. R28 rollback containers
remain stopped and preserved; GLM and DS4 are untouched.

## 09:20 EDT onward: first completion and initial correctness passed

The initial aligned MTP3 profile passed the exact `333` first-completion gate
at 09:20:06 EDT. Boot logs report 5,189,849 KV tokens. This is a capacity
receipt for this boot, not a throughput result.

All 18 semantic checks passed across three repetitions, including reasoning,
non-thinking, vision and tool round trips. Native-context retrieval returned
the exact needle with `finish_reason: stop` at 2,848, 2,849, 131,072 and
262,000 input tokens. The two long requests took 51.327 and 73.267 seconds;
these are correctness receipts, not cache-normalized prefill comparisons.

The persistent driver continues with fixed-token acceptance, padded transitions,
prefix checks and the standard benchmark grid before the remaining control
boots. Full qualification and the R28 comparison are still pending.
