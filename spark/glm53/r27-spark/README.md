# JJ r27 Spark SM121 runtime

This directory defines the DGX Spark derivative of Jovian Judgement r27 for
two workloads:

- Qwen3.8-Flash-Next NVFP4-4p89 at TP2 on dusty and kirby;
- GLM-5.3-Flash NVFP4 at TP4/DCP1 on sparky, buddy, rocky, and lucky.

The image is derived from the exact qualified JJ r26 Spark image
(`bb9cb676…`). It refreshes the vLLM and B12X source trees to the published
r27 identities and pins vLLM PR 674 on top, while inheriting the r26 image's
CUDA 13.3, Torch 2.13, NCCL 2.31.2, SM121 FlashKDA object, SM121 LMCache wheel,
and compiled vLLM extensions unchanged.

## Why r27

Relative to the r26 base, the r27 vLLM tree adds recurrent request-boundary and
leading-instruction checkpoints (`52d9977bc`, #669, #672, #675), the GDN
spec-decode fast-path graph-buffer fix (#667), the single-exchange top-k gather
(#649), FP32 router outputs with stable SM120 graph capture (#666), independent
draft RNG (#653), and the NVFP4 MTP proposal head (#665). The B12X tree adds the
GB10 direct-I/O checkpoint loader sources, shared native NVFP4 scales for A4 and
A16, and the FC1 fragment alias refresh (#317).

PR 674 is pinned explicitly. It is open, mergeable, and absent from the
published r27 lock; it replaces two scalar `.to(tl.int64)` calls in the
endpoint checkpoint restore kernel with `tl.cast`, which upstream hit under
concurrent MTP3 multi-turn serving. Our TP4/DCP1/MTP3 profile is on that path.

## Locked composition

The exact identities and generated patches are recorded in `source.lock.json`.

- Native base image ID: `bb9cb676e464425acb945118bc541fd917fa7eb7f3024418f1733b70de31e8b8`.
- Published r27 vLLM integration tree: `f54cd9ca2b9434727715197d32150b75e82a9ebf`
  (commit `63a82f8d`, base `7d66922a7`).
- Final Spark vLLM tree: `8881b7772e1644d63162ad3b7494888ef7ac54fd`, package
  subtree `f3c3fef548738807f7d1ef8157df63815c498607`.
- Published r27 B12X tree: `f3cd8a9eb00d3226a1acbbed1efedf10cc1c3e71` (commit
  `e8ad299b`, base `a1bbd0278`), package subtree
  `95fdcb1cfea380480b8882fa44055cfef358ddbb`, applied unmodified.
- LMCache integration tree `008ac3e0…`, package subtree `fe5442fb…`: unchanged
  between the r26 and r27 locks, not rebuilt.
- FlashKDA commit `3b225bf2…`: unchanged. The build gates on our SM121 object
  digest `484f88de…`; the published r27 extension digest `16aece5f…` is the
  SM120 workstation object and is recorded for provenance only.

The vLLM Spark overlay is the r26 overlay reapplied (CUDA 13.x architecture
lists admit 12.1; the GLM NVFP4 MTP draft head admits SM120 and SM121) plus PR
674. `patches/vllm/spark-overlay.patch` and
`patches/vllm/pr674-boundary-checkpoint-scalar-cast.patch` are the two
components; `patches/vllm/r26-spark-to-r27-spark.patch` is the single refresh
the Dockerfile applies, and it round-trips from the r26 Spark tree to the
final r27 Spark tree.

Native inventory of the refresh: the vLLM delta is 157 paths, all Python,
shell, and Markdown. The B12X delta is 63 paths and includes
`b12x/loader/*.c`, C99 sources that `b12x.loader._native` compiles with `cc` at
first use. The launchers keep `--load-format instanttensor` and an empty
`VLLM_PLUGINS`, so the loader is inert; the B12X package metadata is not
reinstalled and its new `b12x_loader` plugin entry point is not registered.

## Loader contract change in r27

vLLM `6575b5ac8` removed the InstantTensor `instanttensor_copy` loader option
and switched the upstream launchers to `--load-format fastsafetensors`. The r26
Qwen launcher's `--model-loader-extra-config '{"instanttensor_copy":false}'`
is rejected by the r27 default loader, which is how the first r27 Qwen boot
failed. A second boot with `fastsafetensors` on both ranks exhausted dusty's
host memory during staging (dusty stopped answering ssh, kirby's worker died
on the NCCL watchdog), matching upstream's own GB10 finding for TP1
(`b7e3d0336`). Both r27 launchers therefore keep `--load-format instanttensor`,
which r27 still routes through the default loader with the buffered
`INSTANTTENSOR_*` streaming environment, and pass no extra loader config.
`LOAD_FORMAT` remains overridable.

## Runtime profiles

The GLM profile is TP4/DCP1/MTP3 with the BF16 proposal head, FP8 KV cache,
native 1,048,576-token model length, percentage-sized KV at 0.85 GPU memory
utilization, LMCache disabled, RoCEnante on the switched 200G f0 fabric, and
the launcher environment as corrected on 2026-09-06: the r22 export set only,
no forced L2 prefetch and no `B12X_DYNAMIC_*` overrides.

The Qwen profile is TP2/MTP3 over dusty and kirby's direct 200G f1 links with
the native 262,144-token model length, 0.85 GPU memory utilization, and the
r26 Qwen overlap, compaction, metadata-fastpath, and quantized-LM-head
settings. `VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH=1` stays on; #667 is the fix
for its graph-buffer defect, and qualification must exercise the
uniform-capture to padded-replay transition rather than assume it.

The Qwen profile pins `--recurrent-checkpoint-policy aligned`. r27's default
(`auto`) schedules an exact-repeat prompt as the only work in its step, which
serialized identical concurrent requests and blocked fresh arrivals on
2026-09-06; `aligned` removed both effects under matched settings and passed
the correctness set (see `QUALIFICATION.md`). The launcher defaults to it,
the build gate requires it in the Qwen render and forbids it in the GLM
render, and the runner test requires it in every role. Because image
`ef669fa1` was built before the launcher default, the host runner also passes
the flag; drop that runner line when a rebuilt image carries the launcher
default. GLM's policy is unchanged until its own qualification.

## Build

Run on dusty from a copy of this directory while Qwen is stopped:

```bash
DOCKER_COMMIT=<rtx6kpro commit> ALLOW_DIRTY_BUILD=1 \
IMAGE=localhost/voipmonitor/vllm:glm53-jj-r27-spark-sm121-vllmf3c3fef-b12x95fdcb1-lmcachefe5442f-cu133-torch213-20260906-r1 \
./build-glm53-jj-r27-spark-sm121.sh
```

The build gates cover the r26 set plus: the PR 674 regression on a physical
SM121 GPU, the r27 hybrid-cache regressions (partial prefix hits, Mamba align
chunk split, Mamba prefix state index), the `tl.cast` and
`recurrent_instruction_boundary` source markers, and a launcher dry run that
rejects any reappearance of the removed policy exports.

## Qualification order

1. Qwen on dusty/kirby: semantic battery, boundary and native-context
   retrieval, the standard benchmark against r26, and a c1 to c3 to c1 MTP3
   sequence under full graphs with verification that c3 replays a padded
   batch through a graph captured for a uniform one, watching for persistent
   acceptance collapse.
2. GLM in an approved window, policy unchanged (`tests/qualify-glm53-r27.sh`,
   prepared): all-rank identity gates, semantic x3, the concurrency
   reproducers (identical burst, multi-turn extension, repeat head-of-line)
   under GLM's default policy, the frozen r26/r22 prefix-cache pair matrix and
   the short and long triples (`tests/glm/`), native context with the 262000
   warm before 1048000, then the standard grid and prefill against corrected
   R26 (campaign `2026-09-jj-r27-sm121-qualification`).

   Window sequence (operator-scheduled): ship the image archive dusty to
   rusty, then rusty to sparky, buddy, rocky, and lucky over the 200G mesh,
   `podman load` and verify the image ID on every node; sync this directory
   to each node; stop the r26 GLM containers workers first, head last
   (`ROLE=stop`); start r27 workers first, then the head, with the GLM runner
   as committed (no policy flag); run the driver from the workstation. The
   r26 image stays loaded for rollback.

Status and evidence are in `QUALIFICATION.md`.
