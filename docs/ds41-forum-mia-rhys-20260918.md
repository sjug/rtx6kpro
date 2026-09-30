# DeepSeek V4.1 Flash: Mia, Rhys and current Aiden recipe evidence

Read-only source review on 2026-09-18. No serving nodes, model files or benchmark repository changed. This is a subset of the NVIDIA forum investigation, covering linked implementation artifacts rather than treating forum claims as qualification.

## Bottom line

There is credible evidence that other implementations serve this model and retrieve synthetic facts at near-million-token context. The strongest published evidence found here is Rhys's **SGLang TP8/EP4** deployment with resident Engram. It is not the B12X-expert vLLM R38 TP4 disk-Engram runtime we tested. Neither his nor Mia's artifacts replay our exact failing 524K dual-needle input. Their success does not identify a fix for that failure or establish universal deterministic output.

The most relevant four-node alternative is Mia's SGLang TP4 disk-Engram recipe, and Aiden now also reports a four-node SGLang deployment. Their expert, attention, scheduler, Engram integration and speculation paths differ materially from ours. A matched request against one of these independent implementations would be an informative reference experiment, not an already-qualified replacement.

## Frozen repositories

| Repository | HEAD inspected |
|---|---|
| [Mia native 3/4-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/tree/fba44707a12026a7bf92c85b234e5aa44923fc14) | `fba44707a12026a7bf92c85b234e5aa44923fc14` |
| [Rhys eight-Spark](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/tree/3ac7546756b86af18728485925c783b54023f803) | `3ac7546756b86af18728485925c783b54023f803` |
| [Mia EXL3 two-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/tree/8404ac7d389c418300d0bee960d52313247930e1) | `8404ac7d389c418300d0bee960d52313247930e1` |

Links below to those repositories' `main` show their current documents; use the fixed HEADs above for the reviewed snapshot.

## Mia native TP4: closest hardware topology, different runtime

Actual [boot.py](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/fba44707a12026a7bf92c85b234e5aa44923fc14/boot.py) invokes **SGLang**, `--attention-backend dsv4`, `--moe-runner-backend flashinfer_mxfp4`, `--fp8-gemm-backend flashinfer_cutlass`, TP4/EP4. The dense adapter selects FlashInfer's B12X MXFP8 kernel; this does not make the expert path our B12X dynamic MoE.

[Current TP4 env](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/fba44707a12026a7bf92c85b234e5aa44923fc14/.env.tp4.example):

- Native checkpoint MXFP4 experts and FP8 dense weights. `boot.py` download revision is `fb2764a5cf321eaa5070ca8f9e892818f477c16d`, but shipped `SKIP_PREPARE=1` and `SKIP_VERIFY=1` accept an existing tree; this is not proof of the measured checkpoint's bytes.
- Per-rank NVMe Engram packed rows, weight and scale adjacent, O_DIRECT C++ row store, 96 I/O threads, zero row cache, resident scales off.
- Context 1,048,576; shared KV pool pinned to 8,000,000 tokens; eight request slots; chunk 1,024; static fraction 0.80. No explicit KV dtype in boot arguments. Do not infer its resolved dtype without a boot receipt.
- DSpark current default K3. README says older TP3 tables were measured K5; default changes must not be silently attached to historical numbers. Optional SPS/STS cost tables change scheduling only when files actually exist.
- `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0` disables BF16 atomic expert finalize; they report nondeterminism in its 32-row autotune bucket before disabling it.
- `expandable_segments:False` and per-long-prefill-chunk empty-cache hook; their TP3 stack reported NaNs with expandable segments. This is their observed stack-specific failure, not proof of the cause of our R38 result.
- NCCL `^LL128`, maximum eight channels, 1 MiB Simple/256 KiB LL128 buffers, NCCL rather than RoCEnante in this recipe.
- Reasoning aliases are remapped to publisher low/high/max = 50/75/100; default 75. Loop detection can finish repeating generations, and omitted/oversized output budgets are capped at 32,768.

The [Dockerfile](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/fba44707a12026a7bf92c85b234e5aa44923fc14/Dockerfile) uses mutable `lmsysorg/sglang:dev-dsv41`, not a pinned ARM64 digest. Its comment names an AMD64 sibling digest, not the measured ARM64 runtime. That limits clean reproduction from today's checkout.

### Reported performance and correctness

[README](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/fba44707a12026a7bf92c85b234e5aa44923fc14/README.md) TP4 prose, 256 output tokens:

| Concurrency | Aggregate tok/s | Reported per-stream tok/s | TTFT |
|---|---:|---:|---:|
| 1 | 45.4 | 45.4 | 212 ms |
| 2 | 72.9 | 37.9 | 232 ms |
| 4 | 103.1 | 26.7 | 270 ms |
| 8 | 114.1 | 23.2 | 2.70 s |
| 16 | 134.2 | 22.0 | 12.45 s |

The per-stream and aggregate columns have different windows and should not be multiplied/divided as if all requests were active throughout. TP4 prefill 4K/16K/32K/64K/128K is 3,350/3,782/3,768/3,531/3,251 tok/s, measured with sparkDash.

README claims a passing 1M needle on the shipped TP4 profile. The tracked tree does not contain that raw 1M response/launch receipt. References to `REPORT.md` and logs are not accompanied by those files in the inspected Git tree.

The public [prefill_distinct.py](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/fba44707a12026a7bf92c85b234e5aa44923fc14/scripts/verify/prefill_distinct.py) uses seeded varied paragraphs, places `PELICAN-<seed>` **near the end**, adds 20 paragraphs, then asks for it with temperature zero/thinking off/20 output tokens. It prints expected and actual text but has **no assertion**. It is not a strong beginning/middle/end retention gate and is not established as the source of the separately claimed 1M result. Boot's default quick smoke checks 19+23=42 and stop; fuller JSON/tool/vision checks exist but are skipped by `SMOKE_QUICK=1`. Batch garbage detection during warmup warns rather than failing startup.

## Rhys: strong capacity receipts, eight nodes and SGLang

Despite its repository name, the current headline implementation is **SGLang**, not vLLM. [Pinned source record](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/blob/main/sglang/docs/build-and-pins.md): SGLang `e087e662ba1ac4ef7747537e2a9141085efd4561`; ARM64 base digest `3475d88ec3124867d9d6f7b3bd49afdcf8ef4d6a9f8454fab331d5adb5204de7`; FlashInfer 0.6.18, CUDA 13.0.3, NCCL 2.30.7; model revision `df42c109f1defefcbfcedbe7d905718a12266e40`.

- TP8/EP4/MoE-TP2, native resident Engram. Each target rank has 96 experts and each draft rank 32, local intermediate width 1,152.
- Native checkpoint MXFP4 expert/FP8 dense precision, BF16 activation dtype, auto KV resolved FP8 E4M3.
- FlashInfer CUTLASS expert modules, **no B12X expert backend**. Small-M dense B12X via FlashInfer is separate.
- Native H8/H16 sparse attention, ratio-2 compression/indexer fixes, DSpark worker context fixes; hashes identify mounted overlays beyond image identity.
- Latest measured SG18 adds RoCEnante, bounded indexer filtering, mHC and WO changes, native TP8 prefill split. Standard Dockerfile/launcher still packages **SG5**, not SG18. The promoted SG18 source snapshot and integration settings are separate; generalized fresh installer is not qualified.

### Near-1M retrieval is real, 8M is aggregate

[Eight-million report](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/blob/main/sglang/docs/eight-million-token-results.md) and [raw summary](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/blob/main/sglang/results/eight-million/summary.json) agree: eight distinct **997,097-token** inputs, 7,976,776 total, one-million per-request limit and eight-million shared pool. TP8/EP4, static K5, chunk 2,048, fraction .80, eight slots. The 8M result is **not an eight-million-token single request**.

Each prompt has distinct early identity, repeated ` the` filler and three generated eight-digit records near beginning/middle/end, temperature zero, thinking off. Eight cold outputs retrieve all records exactly (20 output tokens), followed by cached replays and +50-token extensions. This is materially stronger long-distance retrieval evidence than the available Mia script. It remains a synthetic repeated-filler task, not our exact original dual-needle input or broad long-document reasoning.

Cold all-first-output time 87.84 min, 1,513.50 aggregate input tok/s. Initial peak 5,076.03 is a **single chunk near 4K**, not the entire 128K or 1M rate. First-prompt median chunk rate declines to 823.40 tok/s in the final 768K–1M interval. Cached 8×1,024 outputs: 138.31 tok/s including startup; one eight-active-request server interval 158.12 tok/s. Minimum sampled OS free memory 7.83 GiB. Forced output after correct JSON is explicitly not ordinary prose quality evidence.

[Latest SG18 report](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/blob/main/sglang/docs/sg18-prefill-results.md): repeated capacity test **8×997,100**, 8/8 cold and 8/8 cached retrieval correct; 75.01 min cold, 1,772.28 input tok/s, 60.32 s cached 8,192 outputs. 128 slots, eight-million pool, one-million envelope, K5 and chunk2,048. Minimum sampled memory3.555GiB after reboot. Earlier attempt breached its2GiB guard before retrieval; preserved, not erased. This is sampled rather than continuous health proof.

### Latest SG18 speed and quality limits

| Workload | C1 | C4 | C8 |
|---|---:|---:|---:|
| Coding per-stream decode tok/s | 131.72 | 84.24 | 69.78 |
| Coding full-batch aggregate tok/s | 119.86 | 306.08 | 508.79 |
| Prose aggregate output-window tok/s | 87.30 | 168.47 | 254.65 |

Coding200-output/prose256-output means use different aggregate timing definitions. Cold prefill C1 4K/32K/128K/299K: 4,428/4,196/3,907/3,458 tok/s, one excluded warmup plus three measured trials; cached token counts zero. Capability checks include images/JSON/tools,128/128 concurrent arithmetic requests and three pinned long-context retrievals. These are not directly comparable to our different four-node harness or thinking settings.

Importantly, [prior SG18 quality study](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/blob/main/sglang/docs/sg18-prior-study.md) retains nonidentical repeats and quality failures. Combined IHW code86/96 vs control88/96 before and89/96 after; exact answers51/96 vs52/96 and54/96. Candidate repeat agreement41/48, finalcontrol33/48. Their source qualification explicitly permits different indices for oversubscribed tied-score top-k when the optimal-score oracle passes. Thus their working deployment **does not establish fully deterministic generation or eliminate the family of tie effects we observed**.

### Earlier Rhys vLLM profile

[Retained guide](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/blob/main/docs/vllm-deployment.md) records TP8 resident Engram K5,300K limit,8slots,chunk8192,.80,FlashInferb12x dense for M<=128/CUTLASS otherwise. C1coding95.91tok/s; C6coding305.24 aggregate and C8categorymean237.27. Arithmetic,vision and graph/numeric checks passed, but longest measured prompt93,335 with one output token, explicitly **no full300K retrieval qualification**. Do not transfer SGLang near-1M evidence onto this vLLM build.

## Mia EXL3 TP2: different model precision, not a native reference

[README](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/blob/8404ac7d389c418300d0bee960d52313247930e1/README.md): native vLLM `deepseekv41-flash-0909`, version`0.1.dev20904+g179dd0fa9`, ExLlamaV3v1.4.5`e648f1a1` overlays, **2.9bpw mul1** target checkpoint with4-bitMTP. TP2 diskEngram nativeFP8rows/scales; actual defaultK3; auto DeepSeek fp8_ds_mla KV;2.5GiB fixedKV perrank,600Kcontext,2slots,.88 startuputil. Actual current env has visionon and1536-token chunks, while README memory paragraph still says1024. Historical601K measurements used2048, so distinguish those profiles.

Reported stock prose31.6tok/s C1 and42.5aggregateC2; 128Kprefill961.3,256K872.6tok/s. Long601Kprefill742s, decode22.3tok/s, minhead2.1GiB. Optional cooperativeMoE improves separateC1test31.45→40.23 andC2aggregate45.87→61.06; not shippeddefault and not bit-exact. Vision tests OCR10/10, multi-image9/9, imageafter200Kprefix pass, but imagebidirectional visibility is clamped to128-widewindow and native quality parity is explicitly unestablished. This quantized variant is not a clean oracle for native R38 precision.

## Aiden current SGLang, Sep17

The [current recipe](https://aidenle.com/recipes/deepseek-v4-1-flash-sglang-4x-dgx-spark/) is rendered from [public site asset](https://aidenle.com/assets/index-CwypHSr1.js), inspected as text. It reports replacing his earlier vLLM serving on Sep17. The published launch uses `aidendle94/sparkrun-sglang-dsv41-gb10:production-1.0`, TP4/EP4, diskEngram persistent64-worker pool plus next-chunk prefetch, FlashInferMXFP4experts and FlashInferB12X dense, RoCEnante AR<=1MB/AG<=16MB (NCCLfallback),K5,chunk2048,.80,8slots,524288context,autoKV4,219,904 (4.2–5.8M betweenboots), fusedMoEfinalizeoff and expandableFalse. Reasoningdefault100/thinkingon; quoted speed tests thinkingoff. No matching publicly linked source repository or rawreceipt bundle was established in this review; website says its recipe repository pins the base digest, while published command exposes a tag.

Unique-word needles32K/128K prefill3,068/2,580tok/s; repetitivefiller4,063/3,435. Greedy counting/code/prose94.4/67.8/32.4tok/s; prose C1/2/3/4aggregate30.0/49.6/62.2/72.3. Needle32K/128K,count20,image,toolpass;11–14GBmemoryfloorat128K. **524K is configured, not demonstrated retrieval in the current SGLang evidence.** His earlier490Kneedle belongs the preceding vLLM profile, not this one.

## What this changes for our next investigation

1. It provides independent runtime candidates to run **the identical saved original and revised524K requests**. Do not substitute their needle, shortenedcontext or a new prompt as the reference.
2. Rhys can supply the strongest near1M source/receipt chain, but needs eightnodes/residentEngram and exactSG18sourcecomposition. Mia/Aiden fit fournodes/disk, with weaker public source/longretrievalprovenance.
3. Their FlashInferexpert reduction differs from our B12XdynamicBF16atomic path. Mia also deliberately disables atomic finalize. Our own deterministicMoEarm already failed to remove the524Kanswer issue, so this difference is a testcandidate, not an identifiedfix.
4. SGLang's indexer/attention is an independent implementation relative to our forcedB12Xattention, but at least Rhys's optimized selector also allows tied-indexvariation. Stable greedy outputs cannot be assumed from successful retrieval.
5. No artifacts examined establish that our near-tie original failure is definitely modelbehavior or definitely a runtimebug. Matched-reference requests remain the discriminating next step, subject to user authorization for runtime changes.
