# Upstream check, September 27, 2026

Read-only development and publication review for the next Spark (ARM64/SM121)
composition. Every configured remote in the eleven existing checkouts was fetched
with `git fetch <remote> --no-prune` (22 remotes, all succeeded). No checkout,
branch change, worktree, build, deployment, benchmark or serving change. Previous
check: [September 24](upstream-check-20260924.md).

Baseline is our newest composition, the DS4.1 candidate in
`spark/karmic-main-sm121/20260924/`: publication
`karmic-kraken-20260924-291c78003da67f8d`, recipe `626f7af3205c`, vLLM
`1794dcf18454`, B12X `a7d7d29b2ef8`, LMCache `361a85e88698`, plus our QSA865 and
SM121 overlays. All new tips are fast-forwards of those pins (no rewrites).

## Publication

About 40 main and beta releases have appeared since Sept 24 14:21 UTC. Most main releases
come from recipe and LMCache publisher changes, not engine sources.

| | Newest main | Newest beta |
| --- | --- | --- |
| Image | `karmic-kraken-20260927-7f69112f542883c9` | `karmic-kraken-beta-20260927-c10dcc1b22a84fcf` |
| Published | 2026-09-27 16:32:33 UTC | 2026-09-27 21:37:41 UTC |
| Digest | `sha256:8b30ea7b708e01b90c2ef0293cf0e62af8a6d8a17da608d22cb3341f0fb12750` | `sha256:45ea31f050c6e8b3726280c6f0b4a413f0379e5d87e6e8e57ca9e255dbcad393` |
| Recipe | `323aef21c0f7` | `323aef21c0f7` |
| vLLM | `953a636d3ae1` (dev/karmic-kraken, +1) | `eba11af7a050` (integration/karmic-kraken-beta, +95 since Sept 24 beta) |
| B12X | `d44247b6171f` (master, +2) | `f8069b2c0be1` (integration/karmic-kraken-beta, +33) |
| LMCache | `2915c9d3cd7e` (+38, checkpoint/eviction only) | same |

FlashInfer `2206a14e`, InstantTensor `95d4729b`, NCCL `93fe05d9` (2.31.2) and
FlashKDA are unchanged. Foundation remains linux/amd64, CUDA 13.4.1, PyTorch
2.14; there is still no upstream ARM64/SM121 image. Release scope is still native
smoke plus LMCache contract tests; model serving is explicitly unqualified.
Channel mapping at `323aef2` is unchanged.

## Main delta (small, mostly off our path)

- vLLM #909 [953a636d3a](https://github.com/local-inference-lab/vllm/commit/953a636d3a)
  and B12X #433 [0ff5b395](https://github.com/local-inference-lab/sparkinfer/commit/0ff5b395):
  EXL3 experts through common trellis preparation, aimed at Kimi-K3. Not on the
  NVFP4 GLM path. Side effect for everyone: B12X MoE `query_schema_version` goes
  11 to 14, so all cached MoE decode tunings miss and the first boot is a cold
  autotune (selections may differ). Adds opt-in `B12X_W4A16_PREFILL_FUSED_SUM`
  (changes summation order, not bitwise deterministic).
- B12X #411 [d44247b6](https://github.com/local-inference-lab/sparkinfer/commit/d44247b6):
  PCIe collectives for uneven TP/DCP. Not applicable to Spark (one GPU per node, RoCE).

## Beta items relevant to Spark

### DS4.1 (bears on the 524K blocker)

- vLLM #911 [40371c7bb0](https://github.com/local-inference-lab/vllm/commit/40371c7bb0),
  default on: `VLLM_DS41_ATTENTION_COMPUTE=bf16` (options `auto`, `reference`) removes
  the tuner's FP8 sparse-attention choice, and V4.1 defaults `head_dtype=float32`.
  The gate is `is_device_capability_family(120)`, so it covers SM121 (verified in
  source). Correction after rereading EXECUTION.md: 0.125 is the passing arm's
  first-token margin; the failing arm is wrong by 4.125 nats, bit-identical across
  cold trials, and a two-record mHC selection change alone flips it. FP32 logits
  therefore cannot fix the failure. The BF16 attention default is the only part that
  could plausibly matter (it removes tuner-selected FP8 attention), and that is
  untested. It does NOT change bf16 atomic MoE accumulation or the indexer top-512
  tie-break, so it is not a determinism fix.
  `reference` makes decode single-pass (author-reported up to 17% slower decode).
- vLLM #918 [ce4be0a112](https://github.com/local-inference-lab/vllm/commit/ce4be0a112):
  decode page metadata shared across CUDA graphs (author-reported, RTX: graph memory
  5.34 to 1.90 GiB/GPU). No arithmetic change, but it shifts profiled KV and
  therefore capacity-keyed plans.
- B12X #435 [aef9df3c](https://github.com/local-inference-lab/sparkinfer/commit/aef9df3c):
  V4.1 FP4 KV writer uses round-to-nearest division, bit-identical to DeepSeek's
  reference. Reference parity, not variance reduction; changes about 0.05% of KV
  values (author-reported), which can move a tie either way.
- vLLM #895 startup plan under the pre-profiling fingerprint: opt-in, pins KV bytes
  only, fails open to reprofiling. Equivalent to a fixed-KV contract, which was
  previously rejected; not a reproducibility fix by itself.
- Recipe #84 Engram tables in RAM by default: author-reported about 190 to 230 GiB
  host memory at TP4, roughly 47 GiB pinned per rank (estimate). Not viable on
  128 GB unified GB10; keep `table_memory: disk`. Recipe #85 probabilistic DSpark
  drafts: lossless in distribution; our DS4.1 launch uses greedy.

### All DeepSeek V4 parsers

- DSML optional `string=` fix (upstream #56271, [190268a201](https://github.com/local-inference-lab/vllm/commit/190268a201)):
  previously dropped tool parameters silently. Touches `deepseek_v4` and `deepseek_v41`
  parsers, so it also applies to DS4 Vision (`--tool-call-parser deepseek_v4`).

### Generic engine and kernel

- vLLM [638f90a1fc](https://github.com/local-inference-lab/vllm/commit/638f90a1fc),
  default on: aligned mamba state padding used -1 (addressing memory before the
  state pool) instead of NULL_BLOCK_ID 0 during CUDA-graph padding. Our launchers
  use `--mamba-cache-mode align` widely (Qwen GDN, GLM KDA), so this is an active-path
  out-of-bounds fix. Also fixes DCP LSE for non-power-of-two ranks.
- vLLM #897 [b39ca56600](https://github.com/local-inference-lab/vllm/commit/b39ca56600):
  workers stop when EngineCore fails during startup. Related to, not the same as,
  our post-startup zombie APIServer; completion-probe liveness stays.
- vLLM stall diagnostics [e08582200d](https://github.com/local-inference-lab/vllm/commit/e08582200d),
  [6cf4fc4ebd](https://github.com/local-inference-lab/vllm/commit/6cf4fc4ebd): 60 s
  warnings, useful for NCCL hang forensics.
- B12X [f9e45ccf](https://github.com/local-inference-lab/sparkinfer/commit/f9e45ccf):
  GC kept out of autotune samples, fixes a Qwen `gdn_prefill` startup deadlock.
- B12X [537ff124](https://github.com/local-inference-lab/sparkinfer/commit/537ff124):
  cufile.json comments (NGC base with `--load-format b12x`).
- QSA865 is merged into beta ([9d80bb39b8](https://github.com/local-inference-lab/vllm/commit/9d80bb39b8)),
  still open against main. A beta-based composition drops our carried patch.
- Test only: B12X [48be4e85](https://github.com/local-inference-lab/sparkinfer/commit/48be4e85)
  (16-byte loads in W4A16 gated activation, default on, Qwen path, claimed
  bit-identical), opt-in `B12X_W4A16_FP32_TOPK_WEIGHTS`, `VLLM_B12X_BF16_GEMV`,
  B12X [26c3de2b](https://github.com/local-inference-lab/sparkinfer/commit/26c3de2b)
  block-FP8 small-row GEMV (automatic), `VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS`
  (reachable only with throttled/compute-share prefill flags).

### Not for us

- vLLM #898 `VLLM_SHARE_PYNCCL_COMMS` (opt-in): at TP4/DCP1 only TP and EP collapse;
  DCP subgroups never share with TP. It does not address our DCP2 multi-communicator
  progress failure, and side-stream overlap on a shared communicator would be unordered.
- `glm53-spark-tp2` preset (recipe #87 to #89): `hardware: rtx-pro-6000-pcie`; the
  recipe states "Spark identifies the checkpoint, not DGX Spark hardware". Its
  embed-in-host-RAM saving is accounting only on unified memory; vision MXFP8 is a
  precision change.
- GLM TP3 work, MiMo, PCIe world-size-3, LMCache on-evict/prune: off path.

## Recipe changes affecting a rebase

TORCH_CUDA_ARCH_LIST became a foundation platform default (#98, #100) validated
against the amd64 NGC value `... 12.0+PTX`, and the launcher reapplies it unless a
hardware profile overrides it. There is no GB10 profile. Our SM121 Dockerfiles pin
`12.1a` themselves, so current builds are unaffected; any rebase onto recipe
runtime code needs a GB10 profile with `12.1a` or JIT builds lose arch-specific
features. Capture-size fix #80 matters only when max-num-seqs exceeds a listed
capture set. Runtime file manifest (#103) will flag our overlays as modified.

## Other repositories

- FlashInfer upstream +79: SM100/SM103 Cake kernels and SM120 MiniMax-H3 attention;
  nothing SM121/GB10, and none of it is in the unchanged publisher pin.
- eugr spark-vllm-docker +9 (legacy reference, not a build path): glibc
  `malloc_trim(0)` after startup GC in API servers and workers (3691b4f), a memory
  profiling/capacity mod, and an SWA block-size fallback patch against upstream vLLM.
  The heap trim is a cheap idea worth porting as our own overlay on unified memory.
- llm-inference-bench upstream +9: lil-bench container benchmark with telemetry and
  PCIe ACS reporting; also shipped in the runtime image.
- CUTLASS, dgx-spark-infra: no change. SGLang +162, out of scope. LMCache/LMCache
  dev +12 (includes ARM64 wheel publishing), not our publisher line.
- rtx6kpro upstream/master: two daily summaries, no numbered release.

## Fetched tips

| Checkout | Ref | Tip |
| --- | --- | --- |
| rtx6kpro | upstream/master | c67cb82e046c |
| vllm | lil/dev/karmic-kraken | 953a636d3ae1 |
| vllm | lil/integration/karmic-kraken-beta | eba11af7a050 |
| b12x | lil/master | d44247b6171f |
| b12x | lil/integration/karmic-kraken-beta | f8069b2c0be1 |
| blackwell-llm-docker | origin/main | 323aef21c0f7 |
| flashinfer | upstream/main | 1eb503cd38f5 |
| LMCache | origin/dev | c392138c43f2 |
| cutlass | origin/main | 0b55a2f691d6 |
| sglang | origin/main | 8b2ca8ecc241 |
| spark-vllm-docker | upstream/main | 1da226952173 |
| dgx-spark-infra | origin/master | 3326b9cebb55 |
| llm-inference-bench | upstream/main | 05ec18030f29 |

## Recommendation

The main channel has almost nothing for us; the useful work is beta-only. The next
Spark composition should be a pinned beta-sourced candidate (vLLM `eba11af7a050`,
B12X `f8069b2c0be1`), which also absorbs QSA865, or a main composition carrying
the specific backports below. Nothing here changes production.

1. DS4.1: #911, #918, B12X #435 and the DSML fix. Validate SM121 cold tuning, GB10
   decode A/B (with `reference` as a separate arm), graph memory and KV identity,
   then the unchanged cold 524K trials with the verdict rule declared in advance.
   The two known nondeterminism sources remain open.
2. All models: 638f90a1fc, #897, B12X f9e45ccf, stall diagnostics, cufile.json.
   Backport the DSML fix to DS4 Vision too.
3. Budget for the B12X schema-14 cold autotune and record changed MoE selections.
4. A/B later: W4A16 fused prefill sum and FP32 top-k weights (Qwen prefill and
   accuracy), BF16 GEMV, malloc_trim overlay.
5. Skip: Engram RAM tables, share-comms, the RTX `glm53-spark-tp2` preset, PCIe work.

## Independent review (Codex, same day, read-only)

Live inspection: GLM TP4 serves JJ R38 (leader image `ea031e1d3d05`), Qwen TP2
serves the narrow HC-off plus QSA865 image `a25bedd43581`, DS4 Vision TP2 serves
JJ R38p `ab3ed5285a81`, the DS4.1 candidate is stopped, and nous was unreachable.

Corrections accepted:
- The Qwen HC on/off/on comparison already ran; HC-off plus narrow QSA865 serves
  with prefill near R32 (see `spark/karmic-main-sm121/20260922/qsa865-hcbase/EXECUTION.md`).
  The remaining items are a c4/64K decode deficit and allocation warnings.
- DS4.1: a one-record expanded-mHC selection transplant was sufficient to flip the
  524K answer. #911 does not address that selection sensitivity.
- GLM R26: the control isolated the launcher override set, not L2 prefetch alone.
- 638f90a1fc: align mode alone does not prove the padding path executes in our MTP3
  profiles; include it, but do not attribute past failures to it without tracing.

Correction to the review: B12X beta `f8069b2c` still has MoE `query_schema_version=11`;
schema 14 is master only. A beta build still retunes because the source-derived
cache namespace changes.

Agreed per-model plan (advisory, no build authorized):

| Model | Candidate |
| --- | --- |
| DS4.1 TP4 | Beta vLLM `eba11af7a050` + B12X `f8069b2c0be1`, qualification only |
| GLM TP4 | Same beta pins, qualified separately against R38 with identical launcher, DCP1, prefetch off |
| Qwen TP2 | Narrow derivative of `a25bedd43581`, adding 638f90a1fc, B12X f9e45ccf, #897 and diagnostics; no schema-14 or kernel refresh |
| DS4 Vision TP2 | Keep R38p; next candidate Sept 24 main + DSML fix + #897 + diagnostics, with capacity instrumentation |
