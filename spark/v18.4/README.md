# Gilded Gnosis v18.4 Grid48 for DGX Spark

This recipe builds Grid48 from source for the four-node DGX Spark deployment.
It does not apply a Grid48 patch to an older image. The source pins are exact:

| Component | Repository | Branch | Commit |
|---|---|---|---|
| vLLM | `https://github.com/sjug/vllm.git` | `codex/gilded-gnosis-grid48-sm121-20260718` | `df7a0b707f26a2dde80d4285e7b45a5e72f50e6f` |
| B12X | `https://github.com/sjug/b12x.git` | `codex/nf3-grid48-sm121-20260718` | `6b10833ce4880f988ddea9d089ec12058174037c` |
| FlashInfer | `https://github.com/sjug/flashinfer.git` | `codex/sm120-dspark-stack-20260711` | `801d57a08958c13d375ddbb6be3be4808f48a708` |

The B12X pin includes three source commits on top of the v18 B12X baseline:

1. the SM121 Grid48 kernel;
2. the v18.2 mHC compatibility revert;
3. the v18.3 SM121 ultra-tile forced-repin fix.

Consequently `B12X_PATCH_FILE` is deliberately empty. The existing vLLM
SM121 CUDA-13 architecture-list patch and DeepGEMM SM121 MQA-logits patch
remain build inputs because they address separate components.

## Grid48 contract

Grid48 is admitted only on an exact `(12, 1)` CUDA capability with exactly
48 SMs. It launches 48 CTAs, exactly one resident CTA per SM:

| Phase | Tasks | Schedule | Waves |
|---|---:|---|---:|
| FC1 | 128 | 32 CTAs x 3 tasks + 16 CTAs x 2 tasks | 3 |
| FC2 | 768 | 48 CTAs x 16 tasks | 16 |

The production shape is `M=4`, `H=6144`, `I=512`, top-k 8, with 64 NVFP4
experts and 192 NF3 experts. CuTe derives a 45,184-byte dynamic-shared-memory
launch from the allocator; admission also requires 194 workspace words,
1..255 reported registers per thread, no reported local memory, and
whole-grid residency. Any device, geometry, or resource mismatch fails
closed to the serial hybrid path.

The full SM120/SM121 comparison and mapping derivation are in
[`../GRID188-GRID48-SM120-SM121.md`](../GRID188-GRID48-SM120-SM121.md).

## Build and validation

Run on a DGX Spark node from a checkout that also contains the
`spark/blackwell-llm-docker` build worktree:

```bash
./spark/v18.4/build-gilded-gnosis-v18.4-grid48-spark-sm121-cu132.sh
```

The post-build GPU check uses the local GB10 to compile the actual CuTe
Grid48 kernel, queries the generated CUDA function resources, proves exact
one-CTA-per-SM residency, and exercises the NVFP4 MLA cache writer. Set
`SKIP_GPU_CHECK=1` only to build while the GPU is occupied; run the default
check before deployment.

Before a full image build, the same compile/admission check can run against
the existing v18p3 image by mounting only `kernel.py` from the reviewed B12X
branch:

```bash
B12X_KERNEL=/path/to/b12x/b12x/moe/fused/w4a16/kernel.py \
  ./spark/v18.4/smoke-grid48-sm121.sh
```

This is weight-free and single-GPU, but it still compiles the real SM121
CuTe kernel and prints the admitted register, local-memory, dynamic-smem,
workspace, and cubin values. With the source-built v18p4 image, set `IMAGE`
to its tag and omit `B12X_KERNEL`.

Default image tag:

```text
voipmonitor/vllm:gilded-gnosis-v18p4-grid48-spark-sm121-vllmdf7a0b7-b12x6b10833-fi801d57a-cu132-20260718
```

## Four-node deployment

Use `run-glm52-v18.4-grid48-hybrid-tp4-node.sh` on workers first and the
head last. Tensor parallelism remains TP4/DCP1 across one GB10 per node.
NCCL uses the two switched 200 GbE CX-7 rails only:

```text
NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0
NCCL_IB_DISABLE=0
NCCL_PROTO=LL,Simple
```

The back-to-back `f1` interfaces are intentionally excluded because they
are not all-to-all across the four nodes. `VLLM_NF3_MAPPED_GRID_DECODE=1`
is the primary gate; the launcher also exports the old Grid188 variable as
a compatibility alias.
