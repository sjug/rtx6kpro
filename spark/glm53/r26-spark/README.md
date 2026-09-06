# JJ r26 Spark SM121 runtime

This directory defines the DGX Spark derivative of Jovian Judgement r26 for
two workloads:

- Qwen3.8-Flash-Next NVFP4-4p89 at TP2 on dusty and kirby;
- GLM-5.3-Flash NVFP4 at TP4/DCP1 on sparky, buddy, rocky, and lucky.

The image is derived from the exact qualified JJ r22 Spark image. It refreshes
the vLLM and B12X source trees to the published r26 identities while preserving
the already-qualified CUDA 13.3, Torch 2.13, NCCL 2.31.2, and SM121 native
artifacts. FlashKDA is unchanged between r22 and r26, so its inherited SM121
object is verified byte-for-byte instead of rebuilt. LMCache changed in r26 and
is rebuilt for SM121, but remains disabled for initial qualification.

This candidate does not carry the empirical packed-MXFP8 allocation touch from
the Qwen r15p image. The r26 B12X tree contains the persistent dense-GEMM store
wait that is the source-level candidate for the underlying corruption fix.
Qwen qualification therefore runs first and must pass the established boundary
and long-context correctness reproducers before GLM qualification is scheduled.

## Locked composition

The exact identities and generated patches are recorded in `source.lock.json`.

- Native base image ID:
  `907c1265f308ea87c77d98e0b787455af228a5dbdfa228b1c7653a48d3433e0a`.
- Published r26 vLLM integration tree:
  `c861b31da5cf527d96b5772a6709450b23c93a6f`.
- Final Spark vLLM tree:
  `d4571e5ba189fcc05534ecae88dee763ffef4e35`, package subtree
  `59c9787400c85d5d4547ef898c5dee41804e7088`.
- Published r26 B12X tree:
  `c20b6aab67ed791cc226ee91de23f7c45509f436`, package subtree
  `00248b09689830e55d8fbce8ee630600b63ef663`.
- LMCache integration tree:
  `008ac3e09ae5917aa0849147480d7bd5b9f8b37a`, package subtree
  `fe5442fbf258accaa7f26d2bbb00d8b7b5c349ca`.
- FlashKDA commit:
  `3b225bf26bb8e218928a1fe14751cb48cf31d11b`, with the inherited SM121
  object pinned by SHA-256.

The vLLM Spark overlay makes two narrow changes:

1. CUDA 13.x supported-architecture lists admit compute capability 12.1.
2. The GLM NVFP4 MTP draft head admits both SM120 and SM121. A unit test covers
   the capability matrix, and a GPU gate compares the real quantized draft head
   against a BF16 reference on GB10.

The generated refresh patches independently round-trip from the exact r22
Spark source trees to the final r26 Spark trees. Changed-path manifests and all
patch digests are fail-closed build inputs.

## Runtime profiles

The GLM first-admission profile is TP4/DCP1/MTP3 with the BF16 proposal head,
FP8 KV cache, native 1,048,576-token model length, percentage-sized KV at 0.85
GPU memory utilization, and LMCache disabled. Distributed traffic is restricted
to the switched 200G f0 fabric. RoCEnante and PyNCCL are expected as the exact
TP backend list. PCIe custom all-reduce and PCIe DMA remain disabled.

The Qwen profile is TP2/MTP3 over dusty and kirby's direct 200G f1 links, with
the native 262,144-token model length, 0.85 GPU memory utilization, and the r26
Qwen overlap, compaction, metadata-fastpath, and quantized-LM-head settings.

Both profiles use `NCCL_PROTO=LL,Simple`. Neither profile pins NCCL channel
counts. Node runners refuse to replace an existing container and refuse public
or incorrect fabric addresses.

## Build

Run on an aarch64 DGX Spark node that holds the locked r22 base image:

```bash
./build-glm53-jj-r26-spark-sm121.sh
```

An uncommitted build requires `ALLOW_DIRTY_BUILD=1` and an explicit image tag.
A dirty build cannot be pushed. Recipe and source-lock hashes retain exact
payload provenance before the release commit exists.

The build gates cover:

- exact source, patch, changed-path, and package-tree identities;
- preservation of inherited native artifacts and the triton-kernels symlink;
- the inherited FlashKDA object's exact SHA-256 and clean dependencies;
- a new LMCache wheel built with `12.1a` and its native modules;
- RoCEnante API 1 and proxy ABI 3 from the immutable in-image cache;
- source-first vLLM/B12X resolution and a real vLLM custom-op execution;
- GLM SM121 NVFP4 draft-head numerical correctness;
- FlashKDA near-collinear-key finiteness;
- B12X persistent dense-GEMM repeated-output correctness;
- launcher and node-runner render contracts.

## Source and evidence separation

This directory tracks the release implementation and its qualification report,
not raw benchmark output. The GLM launcher retains the R22 policy: materialized queue
and the platform default that disables L2 prefetch on SM121. The unqualified
RTX-specific dynamic policy export block is not part of this release recipe.

[QUALIFICATION.md](QUALIFICATION.md) records the built image identity,
qualification outcomes, and post-build launcher changes.

Raw receipts and previous experiments were preserved
under `/home/jugs/git/rtx6kpro-artifacts/20260906-pre-r27/`, retaining their
original repository-relative paths. This is a local archive, not a published
Git artifact. No image rebuild is implied by committing these reviewed source
files and the qualification report.

NVFP4 proposal heads, DCP4, LMCache, and fairness modes remain separate
experiments. Do not inherit upstream RTX tuning as a Spark runtime default.
