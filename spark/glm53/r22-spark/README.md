# JJ r22 Spark SM121 GLM and Qwen runtime

This directory builds a JJ r22 candidate for Qwen3.8-Flash-Next on the
dusty/kirby pair and GLM-5.3-Flash on the four-node DGX Spark cluster. It starts
from the qualified stock-r17 Spark image, refreshes tracked source to the exact
JJ r22 trees, rebuilds the R22 FlashKDA extension and LMCache wheel for SM121,
and adds three later B12X commits that are directly relevant to these
deployments.

The candidate deliberately does not contain the empirical packed-MXFP8
allocation touch used by the qualified Qwen r15p image. Qwen is qualified first
so the persistent-store fix can be tested against the exact boundary and 131K
serving reproducers before this image is considered for either workload.

## Composition

- vLLM r22 commit `70b3c1c7`, exact tree `89481110`, with only the CUDA 13.x
  supported-architecture lists extended to admit 12.1. The final Spark tree is
  `997f08f5`; the `vllm/` subtree remains `4fbb1c25`.
- FlashKDA `3b225bf2`, rebuilt for SM121 against the inherited Torch 2.13 and
  CUDA 13.3 ABI. This retains R22's correction for non-finite prefill output.
- B12X r22 `1e59a1fd` plus, in order:
  - `aa90a277`: stripe RoCEnante traffic over both configured rails and bump
    the native proxy ABI to 3;
  - `1a7e3ec2`: size-aware graph collectives and the hot proxy path;
  - `9ae41c5c`: wait for persistent dense-GEMM epilogue stores before shared
    staging is reused.
- LMCache uses public acquisition commit `b13fa35e`, whose repository and
  package trees exactly equal published R22 commit `aefe3ab7`. It is built for
  SM121 but remains disabled for first qualification because a DRAM L1 does not
  add capacity on unified memory.

The canonical source identities, generated refresh patches, and their digests
are in `source.lock.json`. The patches round-trip from the qualified stock-r17
Spark trees to vLLM tree `997f08f5` and B12X tree `8ad308af`.

## GLM first-admission profile

The node runner is intentionally narrow:

- TP4/DCP1/MTP3, `MAX_NUM_SEQS=8`, native 1,048,576-token context;
- percentage-sized KV at `GPU_MEMORY_UTILIZATION=0.85`, no fixed KV byte cap;
- FP8 KV, 4,096 scheduled tokens, prefix caching and chunked prefill;
- FlashKDA prefill and automatic KDA decode selection;
- full-and-piecewise CUDA graphs with the explicit
  `1 2 4 8 12 16 24 32` capture ladder;
- RoCEnante enabled, PCIe custom all-reduce and PCIe DMA disabled, and no
  `--disable-custom-all-reduce` flag;
- only switched 200G f0 rails on `10.11.11.0/24`; public and back-to-back links
  are rejected by the runner;
- LMCache and compute-share fairness disabled for the initial control.

After boot, admission is not established by a log substring alone. The backend
list must be exactly `['B12X_ROCENANTE', 'PYNCCL']`; a disabled RoCEnante
communicator would otherwise fall through to CUDA-IPC and PyNCCL.

## Build and gates

Run on an aarch64 Spark build node that already has the locked stock-r17 base:

```bash
./build-glm53-jj-r22-spark-sm121.sh
```

An uncommitted development build must use `ALLOW_DIRTY_BUILD=1` and an explicit
tag containing `dev`, `test`, or `scratch`. Such an image cannot be pushed.

The build fails closed on:

- source patch, changed-path manifest, and final tree identities;
- preservation of every inherited native artifact except `_flashkda_C`;
- a rebuilt FlashKDA object containing SM121 code and passing R22's 16K
  near-collinear finiteness reproducer;
- B12X proxy ABI 3 loaded from the immutable image cache with `CC=/bin/false`;
- R22's persistent-epilogue repeated-output test on a real GB10;
- the vLLM custom-op execution, native dependency, model registry, LMCache
  native-module, and source-tree-first runtime checks;
- launcher and runner render contracts, including the absence of the global
  custom-all-reduce disable and every non-switched network path.

## Qualification sequence

The initial Qwen and GLM qualification is complete; see
`spark/qwen38-flash-next/JJ-R22-QUALIFICATION.md` and this directory's
`QUALIFICATION.md`. The executed sequence was:

1. distribute the same Docker-v2 identity to kirby over the direct 200G pair;
2. boot Qwen TP2/MTP3 worker-first and run semantic admission, the MTP0/MTP3
   2,784/2,785 and 2,848/2,849 boundary probes, the 131K needle, native-context
   admission, and the standard `llm-inference-bench/run_bench.sh` grid;
3. compare Qwen against qualified r15p using engine steps/s as the primary
   decode metric and preserve acceptance separately;
4. after Qwen passes, distribute the same identity to the four GLM nodes over
   the switched 200G fabric;
5. after approval to stop GLM production, boot workers first and sparky last,
   assert the exact collective backend list, and complete semantic,
   native-context, and standard benchmark admission;
6. compare GLM MTP3 against qualified r17p, then optionally qualify DCP4
   memory-weighted KV and compute-share fairness under mixed load.

The exact image ID qualified for both workloads is promoted under the release
tag recorded in the qualification reports. Optional DCP4, memory-weighted KV,
and compute-share fairness experiments remain separate from this qualified
TP4/DCP1 baseline.
