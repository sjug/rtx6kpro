# Qwen Karmic-main prefill investigation

September 23 UTC / September 22 EDT. Read-only source investigation following
the completed dusty/kirby qualification. No restart, inference, kernel patch,
benchmark-repository edit, commit or promotion was performed. Two read-only
queries retrieved dimensions from the pinned model config on dusty.

## Conclusion

The strongest newly identified candidate is upstream vLLM commit
`57ec3c47bb41236c63fdf6fc94f554cd73d57481`, "Shard Qwen Flash Next HyperConnection
projections across TP ranks", dated September 21. It introduces a default-on
Qwen-specific TP path with two all-gathers per HyperConnection mix. The code and
retained configuration establish that our candidate selects it. Timing attribution
is not established until an HC_TP=0 matched arm runs.

This is a more specific lead than generic cold-cache tuning. It is absent from
the tested R32, JJ-main and Karmic-beta sources. Do not call it a proven bug or
claim that all of the R32 prefill difference has this cause.

## Anchors

| Arm | vLLM source | B12X source |
| --- | --- | --- |
| R32 | `5576927057cf71b6ec61d120932338b333efa089`, Spark tree `80a18accc688cdee2974f4c5e03e9416b2896087` | `3edbcbce70f491741b82f5eab9c1b30b39447228` |
| JJ-main | `8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368`, plus cache-agreement overlay tree `976a9ac502a7bcdb25b18cb5597d21caafc97252` | `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` |
| Karmic beta | `57a80980bbf4b40398de7ed851b23e55a3a4c50e` | `e9ce547767ff9ee6509faf294fa1b4e2380dfbf5` |
| Karmic main | `6afb99982576a7a2eb53d667189e859629e22739`, Spark tree `31cb715f3511073f893900248ba8e737646ca50d` | `4f3028b19c1d8290dc72b6f483aba40de23eae5a` |

The Spark overlay changes architecture lists and the draft-head capability gate,
not the HyperConnection implementation. JJ-main's cache-agreement overlay does
not alter that implementation either. No branch tip substituted for these pins.

## Exact activation and added work

At the Karmic-main vLLM pin:

- `vllm/envs.py:1698` defaults `VLLM_QWEN3_8_FLASH_NEXT_HC_TP` to `1`.
- `vllm/models/qwen4_exp/nvidia/hyperconnection.py:208` selects TP sharding when
  that flag is true, CUDA is active, parameters are BF16, and hidden/lowrank
  dimensions divide by TP. Otherwise this HC workspace uses one partition.
- The retained container environment and launcher contain no override of the
  flag. Model revision `c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d` has 48 layers,
  hidden size 2560, HC lowrank 320 and HC count 4. The model explicitly constructs
  BF16 HC configs. At TP2 every condition is satisfied.
- `_mix_normalized()` dispatches to `_mix_sharded()` for TP > 1. That function
  produces the down projection in FP32, copies it to BF16, all-gathers the local
  bottleneck, performs the up projection and local gate mixing, then all-gathers
  the resulting block input. See `hyperconnection.py:633-703`.
- `model.py:294-302,319-365` calls attention and MLP HC mixers in each of 48
  layers, plus the final mixer at `model.py:600`. That is 97 HC mixes and 194
  newly introduced all-gather calls in a complete target forward. Row counts can
  shrink at an output-selection boundary; this is a call count, not a claim all
  calls always carry the full prompt chunk.
- `communication_op.py:17` routes them through the TP group. This profile disables
  custom all-reduce. The CUDA communicator's PyNCCL path allocates output, gathers,
  moves the rank dimension and reshapes; dim-last gathering can also require a
  layout copy (`cuda_communicator.py:492-520`). This is additional communication
  and memory traffic, not simply half-sized GEMMs.

For a full 2848-row chunk, each rank contributes BF16 bottleneck width 160 plus
block-input width 1280 per HC. For 97 full-row mixes this is 795,617,280 bytes
(0.796 GB) of peer payload per rank, plus receiving that amount. It is an upper
bound if final rows are trimmed and excludes the MTP forward. At nominal 200 Gb/s,
the payload alone corresponds to 31.8 ms, excluding collective latency, layout
copies and possible overlap. This is NOT a measured overhead or a net slowdown:
sharding also saves projection compute. It does show why a compute-saving change
can have a different balance across two Sparks than on a faster interconnect.

## Local executable check

A Python AST check extracted the actual `self.tp_size` expression from the pinned
HyperConnectionWorkspace initializer and evaluated it with the verified model
dimensions and CUDA/BF16 predicates. It also counted the actual all-gather call
sites in `_mix_sharded` and checked the four pinned env sources. Output:

```
HC_TP 1 -> actual HC partitions 2
HC_TP 0 -> actual HC partitions 1
all_gathers per HC 2 target forward 194
R32 HC_TP env exists: False
JJ-main HC_TP env exists: False
Karmic-beta HC_TP env exists: False
Karmic-main HC_TP env exists: True
```

This is a source-dispatch check, NOT a GPU regression test or latency reproduction.
The existing full-grid comparison is the recorded performance signal, but a new
single-variable GPU measurement is needed for causal attribution. No extra load
was added to the running pair to manufacture that evidence.

## Other candidates and exclusions

1. **HC TP sharding**, first arm: if this is the added main-vs-JJ cost, disabling
   only HC sharding should improve prefill on the same image. It may affect decode
   and numerical rounding too; do not assume either is unchanged.
2. **GDN backend**, residual R32 comparison: retained boots show FlashInfer
   prefill + B12X decode on R32, versus B12X/B12X on both newer images. Main still
   enforces paired selection in `qwen_gdn_linear_attn.py:279-310`; trying a flag-only
   FlashInfer arm by dropping B12X decode repeats the old confounded experiment.
   The `b12x/sequence/gdn_prefill` subtree is exactly
   `f391cf5fe80421e438caccb3d8ca186735ab62f5` in both JJ-main and Karmic-main;
   their shared delta-prefill and tensor helpers also have no diff. Therefore a
   new GDN kernel-source change is not the added main-vs-JJ delta. Integration,
   chosen tactics and toolchain can still differ.
3. **GDN workspace lifetime**, lower-ranked main-vs-JJ candidate: `50554de256`
   removes persistent activation staging and three per-call copies, but adds a
   caller-owned output allocation. This is not self-evidently slower. If HC does
   not explain the gap, profile allocator/wrapper time before reverting a memory
   improvement.
4. **MoE tuning** changed synthetic route producers and cold selections. This
   needs actual winning configurations and operator timing, not attribution from
   a large source diff. New IQ2 branches do not prove the NVFP4 path slowed.

The checkpoint has only PLE layer 2 and `ple_embedding_dtype=nvfp4`. Although
Karmic-main changes the default BF16 PLE storage to io_uring, its resolver returns
device storage for this NVFP4 checkpoint without an override. Disk PLE is not the
explanation here. QSA DCP additions must likewise not be treated as active TP2/DCP1
collectives. The shared-expert tuning-context omission alone cannot explain the
additional main-vs-JJ difference without proving an actual selection change.

## Smallest decisive next step, not executed

Use the same qualified main image, checkpoint, MTP3, aligned policy and serving
envelope. Add one explicitly recorded runner passthrough:
`VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0`. The present node runner does not forward this
variable, so exporting it only in the SSH shell would be a void experiment.

A worker-first pair restart is required: the flag changes construction and weight
sharding, not merely a request-time branch. Check the effective worker setting,
run semantic/MTP correctness first, then repeated fresh-prefix prefill probes at
the saved prompt lengths, followed by the standard grid if the signal improves.
Do not run another R32 control. Preserve the current main receipts as the on arm;
if boot variability matters, use a main on/off return rather than claiming proof
from one noisy sample. No other model needs interruption.

If prefill improves with this feature disabled, attribute the feature-level
delta, not all individual collective overhead, and record any numerical or
decode tradeoff. If it does not, move to GDN selected plans/wrapper timing. Do not
ship a blanket architecture gate or claim a fixed regression before that result.

## Pre-test Claude review

Claude completed the read-only kit review before any diagnostic inference or
restart. It confirmed the one-variable runner delta, preserved frozen launcher,
image and model identities, and separate receipt paths. The retained HC-on log
also positively records sharding across two ranks with bottleneck 160 and local
features 1280, strengthening the source-dispatch evidence above.

Two requirements were incorporated before the window:

- A pre-inference log gate waits for API readiness, requires completed model
  loading on each rank, and checks the sharding message against the arm. HC-off
  requires its absence as well as the explicit environment value. Absence alone
  without the completed construction log is rejected. Correctness remains a
  separate gate; API readiness does not qualify output.
- The experiment includes a warm HC-on return arm, using the same image and
  caches without clearing anything. The initial on boot compiled from a cold
  cache; the off boot may reuse selections. This is a potential confound, not
  evidence that warm selections necessarily change performance. An improvement
  on the off arm alone will not be called causal.

The driver now supports `QUAL_ARM=hc-off` and `QUAL_ARM=hc-on-return`, recording
separate correctness, window and grid receipts for each. The return arm sets
HC_TP=1 explicitly. A local render test requires the diagnostic command to match
the original command after removing only the HC environment argument.

Existing containers are stopped worker first with `podman stop -t 60`, then
renamed and retained. The runner's destructive `ROLE=stop` is not used. Renaming
avoids the name collision without removing rollback containers, so separate
container names inside every probe are unnecessary.

Claude additionally recommended repeated fresh-prefix prefill probes on the
existing on boot and off boot. These would strengthen the single-scout grid
evidence but are not yet implemented. The standard grid plus the explicit warm
return is the current executable comparison; single-scout limitations remain.

Claude re-reviewed the amendments and cleared the diagnostic window with no
remaining actionable findings. It independently confirmed both ranks' original
loading/sharding messages, the boot gate ordering, arm-specific receipts, and
container retention by rename. Fifteen local contract tests and shellcheck pass.
This clearance is an artifact review, not a live diagnostic result; no HC-off
or HC-on-return inference had run at the time of review.

## Executed on/off/on result

Both new arms completed with exit status zero on September 22 EDT, September 23
UTC. Same image `88867036403cb3c9bc3026eb8c907ad64bc1af9c28d316d22b670dde4f561152`,
checkpoint, MTP3, aligned policy, launch envelope and standard benchmark source.
Only the explicit HC setting changed. Both original and HC-off containers were
gracefully stopped and renamed for retention. The HC-on return is left serving
as the diagnostic candidate, not newly promoted. GLM and DS4 were untouched.

### Conclusion

The feature-level hypothesis is supported: disabling HC TP sharding improves
32K through 128K prefill against both the original on boot and a warm on return.
The warm return loses the improvement, weakening a warm-cache-only explanation.
This attributes a contribution to the HC feature, not a measured cost to each
collective and not the whole difference from R32. One off boot and one scout per
length remain limitations; this is not a confidence interval or a claim of
repeatable gain across multiple off boots. The 16K residual is unexplained.

### Prefill tokens/s

| Context | Saved R32 | Initial HC on | HC off | Warm HC on | Off vs warm on |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 3124 | 2932 | 3076 | 2919 | +5.38% |
| 16K | 3004 | 2870 | 2788 | 2810 | -0.78% |
| 32K | 2918 | 2783 | 2877 | 2762 | +4.16% |
| 64K | 2764 | 2646 | 2734 | 2618 | +4.43% |
| 128K | 2501 | 2412 | 2482 | 2383 | +4.15% |

Compared with saved R32, HC-off is still -1.54/-7.19/-1.41/-1.09/-0.76% across
these lengths. No new R32 run was made. These are fresh randomized benchmark
prompts, not identical token IDs. The token targeting method and full protocol
match; exact realized prompt counts differ slightly. The 8K row is client-only;
the longer rows also retain server-side prefill validation.

### Decode geometric means across five contexts

| Concurrency | HC-off steps/s | Warm-on steps/s | Change | HC-off output tok/s | Warm-on output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 23.605 | 23.153 | +1.95% | 48.245 | 49.731 |
| C2 | 38.366 | 36.681 | +4.59% | 79.180 | 79.643 |
| C4 | 58.025 | 55.049 | +5.41% | 120.572 | 112.947 |

Grid effective acceptance off/on is 2.044/2.148, 2.064/2.171 and 2.078/2.052.
Thus faster engine steps do not guarantee more output tokens: C1 output falls
2.99%, C2 falls 0.58%, and C4 rises 6.75%. Relative to the initial on boot, off
steps rise 1.95/2.50/2.94%, showing some on-boot decode variability. Treat steps
as less acceptance-confounded, not completely independent of token-dependent
routing and speculative work.

### Correctness and health

Both arms passed 18/18 semantic cases, exact retrieval at 2848/2849/131072/262000
with stop finish reasons, fixed-token acceptance, concurrency transitions,
prefix reuse, and queue/identical-prompt checks. Fixed-corpus acceptance was
2.263 off versus 2.276 on return (2.294 initial on). Cross-wave output hashes
were not identical in either arm, as in the initial on receipt; no determinism
claim is made. No acceptance-collapse phase occurred.

HC-off KV admission was 5,153,179 tokens (41.87/43.29 GiB per rank); warm on
was 5,229,340 (42.49/43.82 GiB), versus 5,259,616 on the initial boot. These are
boot-specific values, not performance metrics.

Over each complete benchmark window, dusty mean SM clocks were 2468.31 MHz off
and 2468.20 MHz on; kirby 2487.32 MHz in both. Minimum MemAvailable off/on was
4.43/4.38 GiB on dusty and 6.89/6.64 GiB on kirby. No swap growth was observed
during either grid. No Xid or OOM kill was found. Dusty logged 104 allocation
warnings in the off startup and 87 in the return startup, none during either
grid; kirby logged none. Each rank logged layer_norm and grammar-mask first-use
JIT during correctness, before its grid. Startup warnings remain unresolved.

### Evidence

All files are under the existing task and results trees; no benchmark-repository
source was edited and no commit, push or external filing was made.

- `spark/karmic-main-sm121/20260922/qualification/window-hc-off/` and
  `window-hc-on-return/`: identities, effective HC logs, telemetry, kernel and
  container logs, execution logs and zero exit statuses.
- `spark/karmic-main-sm121/20260922/qualification/qwen-mtp3-hc-off/` and
  `qwen-mtp3-hc-on-return/`: correctness and benchmark receipts.
- Off directory: `vs-initial-on.json`, `vs-warm-on.json`, `vs-r32.json` are
  protocol-gated comparisons made with the existing comparison helper.
- Campaign `2026-09-karmic-main-sm121-qualification`, throughput grid
  `20260922T230925-0400__karmic-main-hc-off-aligned-mtp3__r01.json`, SHA-256
  `2a6248e922cdb30a9049a6d6bc2353642f52cbd5e7a6e2e14043622621ff4cb9`.
- Same campaign, grid
  `20260922T233328-0400__karmic-main-hc-on-return-aligned-mtp3__r01.json`, SHA-256
  `b8730d7b4a57dff063a3287ee99ebd1418399835e2a68b7da71ec62670b670f0`.

The practical candidate change is the Qwen-only HC_TP=0 runtime override. It is
not yet a production default. Before claiming the entire prefill regression
resolved, isolate the 16K scout behavior and any remaining GDN integration cost;
do not remove the B12X decode selector and repeat the previously invalid pairing.
