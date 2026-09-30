# DS4.1 forum determinism review, September 28, 2026

Read-only review of NVIDIA forum topic
[DeepSeek v4.1 Flash](https://forums.developer.nvidia.com/t/deepseek-v4-1-flash/382725)
(175 posts through 2026-09-28 23:57 UTC), the linked
[New Deepseek 4.1 flash recipe](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438)
topic, and every linked repository, gist and model card. Everything was read remotely
(Discourse JSON, `gh api`, raw GitHub, registry API). Nothing was cloned, built or
run, and no host was touched. Earlier snapshots are in the September 18
`ds41-forum-*` notes.

Question: does anyone have a fix for DS4.1 determinism that we lack? Our failure
(`spark/karmic-main-sm121/20260924/`) is the 524K dual needle. Across cold runs
the history is 3/25 correct (R38 2/16, R38 with deterministic MoE 0/6, Karmic 1/3).
Two expanded-mHC selection records alone flip pass to fail. Those kernels differ by
about 4e-7 in FP32 outputs and stay within the upstream reference bounds.

## Answer

No. Nobody in the thread claims a determinism fix or tests cross-boot stability at
500K+. Every relevant mechanism found is one we already have, a broader version of
one, or a reproducibility design that would not change correctness on this input.

| Found | Where | Our equivalent |
| --- | --- | --- |
| Deterministic B12X MoE combine, `RoutingSpec(deterministic_output=True)`, same B12X `a7d7d29b2ef8` | Mia TP4 line `cad252b7` [adapter/moe_b12x_next.py](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/cad252b7d3000cf21d919764e70a6912b6d3b0b0/adapter/moe_b12x_next.py) | `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` (dynamic prefill family; micro decode path deterministic for this checkpoint). Tested: 0/6 on this input. |
| FlashInfer fused MoE finalize off "for numerical accuracy" | SGLang #40105, Mia env | Same class as above. |
| Autotune off (`B12X_AUTOTUNE=0`, FlashInfer autotune off) | christopher_owen [spark3-vllm-ds41f](https://github.com/christopherowen/spark3-vllm-ds41f) `a8014266` | We pin through fixed block count plus seeded selection records, a narrower control. Autotune off changes every component. |
| Persist FlashInfer autotune cache across boots | Mia `adapter/autotune_keep.py` | Same principle as our seeded selections. |
| Token-index top-k tie-break | vLLM #56613 (open), #56749 (draft, index bug) | Our `claude-dsa-topk-tiebreak.patch` for B12X `tiled_topk.py`. |
| Batch-invariant, compensated mHC (fixed K slices, tf32x3; bf16x3 and tf32+residual for large M; no autotuner) | SGLang [#39664](https://github.com/sgl-project/sglang/pull/39664), merged `408d2334c34d` | None. B12X prefill mHC is single-pass TF32 MMA with tunable tiles. |

The last row is the only design we lack. It would make the mHC result identical
across boots and batch sizes, and more precise. It would not settle this input.
Perturbations of about 4e-7 already move the answer by more than 4 nats, so any
fixed order simply lands on one side. The 3/25 history suggests that side is usually
wrong.

## Per source (author-reported unless stated)

- **Mia TP4 production line** (main `cad252b7`, PR #36 merged as `bec90d89`; open
  PR #37 `19b59158`). Runs SGLang dsv4.1 `f80c91a4`, SGLang `dsv4` attention, DeepGEMM
  indexer, SGLang mHC, and b12x only for MoE and some MXFP8 dense. Needles are a
  single phrase, one boot per length, greedy: passes at 99K to 1,011,084. The authors
  state that identical cold ~70K prompts give different greedy continuations, so
  their prefill is not reproducible either. Forum users report 500K needle 4/4 and
  1M 4/4 on this line.
- **knapcio TP4** (`e9ec61d2`): profile on Mia's line, merged upstream as #36; no
  additional determinism work.
- **Rhys TP8** (`3ac75467`, unchanged): SGLang TP8/EP4, FlashInfer CUTLASS MXFP4
  MoE. `enable_deterministic_inference` recorded false. Long-context checks are
  single-run capability smoke (author's words).
- **Mia EXL3 2x** (`6f7d1590`): serving tweaks only.
- **Aiden / 0rand** (`44b0e441`, DS4 0731). The DS4.1 image `production-1.1` uses
  PyPI b12x 1.3.0 (older than our pin). No determinism work.
- **OllieO** (`ac76304a`): the b12x MoE "pelican" corruption at 6 concurrent streams
  was a batch-composition bug in older b12x. A community A/B (topic 381911 post 445)
  cleared it with a 09-21 build. The likely fixes (`720be522`, `ff36d2d1` #362,
  `02407f65` #388, and others) are ancestors of our pin; attribution is inferred.
  His Marlin path enables `VLLM_MARLIN_USE_ATOMIC_ADD=1`, so it is not deterministic.
- **Tony** (`d45538f6`), **Joe** (`9d592115`, unchanged), **Kilork gist**
  (`47c6ec89`, unchanged): no determinism work; Tony uses EXL3 experts.
- **christopher_owen** TP3 (`a8014266`, vLLM Karmic beta `04c30fa9` + 20 patches, B12X
  `e39b437b`): autotune off, DSpark cost curves pinned (patch 0005). His TODO states
  temperature-0 outputs differ within one boot, suspecting the B12X atomic MoE
  combine and split-K, and that `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` fails B12X
  preparation on his build. Context limit 131K; no needles above 64K.
- **LibertAIDAI NVFP4** (`dfce15b9`): expert conversion bit-exact; Engram FP8 to FP4
  lossy. Not relevant.

## Upstream items noted

- vLLM open: #56613 and #56749 (tie-break), #57815 (batch-invariant mode refuses the
  sparse indexer; every indexer top-k backend is batch-dependent), #57092, #57280,
  #58718 (sm121 batch-invariant matmul table), #58623, #57843; NaN fixes #58560
  and #57158 (FlashInfer SM120 sparse path).
- B12X: issue #430 (open) reports `B12xMxfp8LinearKernel` corrupted tokens on DS4.1
  TP4 GB10 with PyPI 1.3.0; probably stale for our pin but cheap to check against a
  BF16 reference. #415 (deterministic prefill reduction) closed, re-split as open
  #436/#437 for the W4A16 path. #417 (GC out of autotune samples) now on master.

## Recommendation

1. Do not expect a kernel or flag to fix this input. The evidence points to an
   input on which this checkpoint is chaotic under legitimate rounding.
2. The most informative experiment is a reference run on an independent stack: the
   exact 524K dual needle on Mia's SGLang TP4 line on the same four nodes, several
   cold boots, verdict rule declared in advance. A similar hit rate would make it a
   model limit, and the qualification bar is then the user's decision. A reliable
   pass would identify a stack difference worth bisecting.
3. If we want reproducibility regardless, the transferable design is SGLang #39664
   ported into B12X mHC (fixed slices, compensated projections, no autotune), plus
   our existing tie-break patch and deterministic MoE. That is reproducibility, not
   correctness.
4. Cheap side check: B12X #430 MXFP8 linear corruption against our pin.

## Addendum: least-invasive reference stack (Aiden vLLM image)

Mia's SGLang line was rejected as a reference run. It needs privileged Docker,
per-node builds from the internet, an NFSv4 exporter with hard worker mounts, and
about 63 GiB per node of packed Engram shards. Stock upstream vLLM nightly
(ARM64 images exist) does not fit four Sparks, because Engram stays in pinned host
memory at about 50 GB per rank.

`aidendle94/sparkrun-vllm-dsv41-gb10:production-1.1` was inspected read-only
through the registry API (config plus the 25 small patch layers, no node use):

- ARM64, created 2026-09-11, about 10 GB. Base is upstream `vllm/vllm-openai`
  nightly `8a728663`, CUDA 13.0.2. FlashInfer `07869c61` is rebuilt with
  `FLASHINFER_CUDA_ARCH_LIST=12.1a`. MoE is `b12x==1.3.0` from PyPI. Entrypoint
  is plain `vllm serve`.
- 22 vLLM patch files: DS4.1 attention, FlashInfer sparse MLA, indexer, compressor,
  KV cache, V2 runner, DFlash, DCP2, b12x MoE glue.
- Engram: `DSV41_ENGRAM_DISK=1` preads rows directly from the original safetensors
  shards (optional `DSV41_ENGRAM_MMAP`, `DSV41_ENGRAM_PREFETCH`). The node-local
  row copy (`DSV41_ENGRAM_DIR`) is optional. No preprocessing is required.
- No hardcoded context cap in the patches. voktolom (topic 382725 post 138) reports
  running this image at TP4, 1M context and original weights, with a 500K needle 4/4.
- Independence from our stack: attention, indexer, compressor and mHC are upstream
  vLLM/FlashInfer/DeepGEMM, not B12X. MoE is still b12x, an older release.

A run would need no build, NFS, preprocessing or host change: one pull, a podman
save/load fan-out over 200G, rootless containers with the read-only HF cache. It
still needs a window on the four DS4.1 nodes.

## Result: reference run on Aiden's vLLM image, 2026-09-29

User-approved window on dusty/toby/rusty/kirby (Qwen and the rusty/toby DS4 Vision
perf job stopped gracefully, logs archived in `~/logs/*-20260929T004057Z.log`).

- Image `docker.io/aidendle94/sparkrun-vllm-dsv41-gb10:production-1.1`, registry digest
  `sha256:ede55d4643b1c9c3a70f524df4e26d361472630588a7518070766de18f66c7fb`, pulled on
  dusty and loaded on the other three over the 10.11.11.x mesh; identical layers on all four.
- Checkpoint `deepseek-ai/DeepSeek-V4.1-Flash` revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d`
  (same as our qualification), local on each node.
- Launch: Aiden's published production-1.1 model arguments and `DSV41_*` switches
  (TP4, DCP2 `ag_rs`, fixed 6 GiB KV per rank, b12x 1.3.0 MoE, FlashInfer sparse MLA,
  disk Engram via safetensors pread/mmap, DSpark K5 probabilistic with block rejection,
  `NCCL_ALGO=Ring`, `NCCL_PROTO=LL,LL128,Simple`). Deviations: `--max-model-len 1048576`,
  served as `DeepSeek-V4.1-Flash` on port 8000, no RoCEnante (not in 1.1), no node-local
  Engram row copy, and podman flags plus fabric NCCL env from our DS4.1 `run_node.py`
  and `launch_contract.py`.
- Boot: weights 233 s, model load 81.55 GiB per rank, KV 3,021,317 tokens, engine init
  655 s, API about 17 min after the head started. Completion probe correct.
- Gate: unchanged `qualify_original_needle.py`, frozen input sha `1b89c7f4...`, 524,288
  prompt tokens, three cold trials (0 cached tokens each).

| Trial | Answer | First token vs `739` | Elapsed |
| --- | --- | --- | --- |
| 0 | `510c94b1bb4f42d2998093bd6b9c3d99, 482617` | `510` by 2.500 nats | 1,147 s |
| 1 | same | `510` by 0.125 nats | 840 s |
| 2 | same | `510` by 0.875 nats | 964 s |

Expected `739184, 482617`. Result **0/3, ORIGINAL-NEEDLE-FAIL**. Trials 1 and 2 were not
identical to trial 0, so this stack is not reproducible within a boot on cold requests.

Caveat added 2026-09-29: vLLM #943 shows every earlier SM12x DS4.1 image left the compressor ring unwritten, including stale windows at prefill chunk boundaries. This Sept 11 image most likely has that defect (our `1989e16d` has our own fix), so this run is weaker reference evidence than stated below.

Interpretation: an independent attention, indexer, compressor and mHC implementation
(upstream vLLM/FlashInfer/DeepGEMM rather than B12X) gives the same wrong answer as our
stack, which was 3/25 correct historically. This supports the finding that the input is
a model-level near-failure for this checkpoint at 524K, not a defect specific to our B12X
composition. Limits: one boot, three trials; MoE was still b12x (older 1.3.0); DCP2 and
DSpark K5 probabilistic differ from our DCP1 K7 greedy profile. The qualification bar for
this input is now a user decision.

Teardown: containers stopped worker-first, logs archived as
`~/logs/ds41-aiden-ref-tp4-<node>-20260929T*.log`, containers and cache volume removed.
The image remains loaded on the four nodes.
