# R38 Qwen GDN live screen

Authorized 2026-09-15 on dusty/kirby only. The experiment changes no model,
source package, image, production launcher, NCCL setting or memory limit.
It bind-mounts an experiment launcher over the Qwen entrypoint with an
explicit, matching `QWEN_LAUNCHER_SHA256`. The image label retains the
original launcher digest, so record the candidate as image plus launcher.

Arms use the same experiment launcher and differ only in `GDN_SCREEN_ARM`.
`b12x` selects the original R38 GDN combination; `flashinfer` replaces the
decode flag with the prefill flag and clears the decode environment override.
That also changes decode and graph/metadata behavior. This is a screen,
not a recurrence-only causal test or production promotion.

The standard benchmark wrapper and source are unmodified and SHA-pinned.
Campaign metadata is copied into the benchmark results repository before
execution. Each boot has a seed pass and a measured pass. Random prefixes
prevent prefix-cache reuse; they are not identical token streams across
boots. Compare the two arms within this campaign before comparing historical
scout-mode rates. Node clocks and memory are sampled throughout.

## Initial operational failure

The first attempt in `../gdn-screen/` failed before model loading because
the new launcher file was not executable. Both test containers exited with
`catatonit: failed to exec pid1: Permission denied`. R32 was restored and
its completion passed at 19:58:11 EDT. Those receipts remain intact.

The executable bit was corrected on both nodes, an executable-file check
was added to the runner, and container-level DRY_RUNs passed on both nodes
without GPU attachment. Only the two failed experiment containers were
removed after their inspect/log receipts were saved. Images, qualified
containers, Hugging Face and JIT caches were not removed.

This directory contains the corrected retry's immutable driver snapshot.
The preserved qualified R32 containers are restarted after completion or
failure. GLM and DS4 are outside the driver's host list.
