# r18p DGLIN DS4 production qualification

Date: 2026-08-24 EDT

Verdict: REVOKED. The runtime and structural gates below passed, but a
same-prompt comparison after qualification found materially worse response
quality on dusty/r18p-DGLIN than on rusty/r34-B12X. Relaunching the identical
r18p image with the stock B12X backend did not restore r34 response quality,
so DGLIN is not isolated as the primary cause. The broader r18p execution stack
remains unqualified for production pending target-only and semantic-quality
isolation. Production remains on corrected r34. This receipt does not qualify
the image for GLM-5.2.

The earlier PASS was invalid because arithmetic, tool-schema, needle, and
throughput checks did not establish free-form response-quality parity. DGLIN's
non-bit-identical arithmetic was treated as a performance-only difference
without an end-to-end semantic gate.

## Runtime identity

- Hosts: dusty (head) and kirby (worker), one GB10/SM121 each.
- TP transport: the back-to-back 200G fabric, master addresses
  `10.11.1.1` and `10.11.1.2`.
- Image:
  `infernal-invocation-r18p-spark-sm121-vllmf560085-b12x07cdf45-fi1ac6942-cu133-torch213-20260824`.
- Image ID on both nodes:
  `6ff2730764791cf13bca277f22a98b4c31a769bd6d2b4ff03f3414954399ee89`.
- Runner digest on both nodes:
  `8c00b09d705e271b2feb55d66e07767a56c9a69106655bf739b87fc6f159438e`.
- vLLM integration tree: `f0fa1cefc1865d316c2478525f550e7646addc40`.
- B12X integration tree: `07cdf4567b50fa983462f0f0e1bc992de3033adc`.
- Paired-FC2 patch SHA256:
  `e8d399a1c12a3a2ad8a65ab49437b5fb75ae78a00aba551fe2f193c029805ca1`.
- FlashInfer: `1ac6942776b383c6b03c7a5805a22e72a3e3349f`.
- DeepGEMM: `a6b593d2826719dcf4892609af7b84ee23aaf32a`.
- NCCL: 2.31.2 at `fb6f40999a2a9e63104d4ae4a84118bce61528f8`.

The image was transferred directly from dusty to kirby over the 200G pair
interface (`10.11.1.1` to `10.11.1.2` via `enp1s0f1np1`). No model or
Hugging Face cache was cleared.

## Launch contract

- Model and sole served identity: `DeepSeek-V4-Flash-0731`, checkpoint
  revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.
- Backend: `b12x-a8-dglin`; both ranks selected
  `DeepGemmFp8BlockScaledMMKernel` and ran with
  `VLLM_USE_B12X_FP8_GEMM=0`.
- DSpark K5, probabilistic drafting, maximum four sequences.
- CUDA graph ladder: `1,2,4,6,8,12,16,18,24`.
- Maximum model length: 524,288 tokens.
- NCCL: `LL,Simple`, exactly four minimum and maximum channels; implicit
  launch ordering absent.
- Cold process boot: 280 seconds from container start to READY.
- Reported KV capacity: 1,177,876 tokens, or 2.25 concurrent maximum-length
  requests.

The runner now fails closed on the exact image ID, exact-four channel
contract, and absence of `NCCL_LAUNCH_ORDER_IMPLICIT`. Its `DRY_RUN=1` path
renders without stopping or creating a container.

## Correctness and stability

| Gate | Result |
|---|---|
| q-length truncation and 8K prefill-tail reproducers | PASS, engine remained healthy |
| JSON-schema arithmetic | 24/24 correct and schema-valid |
| strict tool calls, concurrency 1 | 24/24 conformant |
| strict tool calls, concurrency 8 | 64/64 conformant |
| 231,641-token strict tool/needle request | PASS, exact sensor/value recall in 124 seconds |
| exact 524,288-token request | PASS, exact `7391` recall in 346.4 seconds |
| 524,289-token request | PASS, immediate HTTP 400 admission rejection |
| mixed-stream corruption soak | 0/200 corruption signatures with three background streams |
| ten-minute inference idle then decode burst | PASS, 42.9/71.2/110.5 tok/s at cc1/2/4 |
| final two-rank fatal/error scan | clean |

The mixed-stream harness requested a six-token templated prompt but the DS4
chat template floor was 85 tokens. Its result is therefore a valid
mixed-stream corruption soak, not an exact-width-six scheduler proof.

## Standard 15-cell sweep

Harness: llm-inference-bench v0.4.29 at
`6838fbeed513433a6403ed56ed15dcf1535f7dd4`; 30 seconds per cell, output
limit 8,192, standard hidden long-context warmup, no capacity skips.

Aggregate decode tok/s:

| context | cc1 | cc2 | cc4 |
|---:|---:|---:|---:|
| 0 | 38.6 | 58.4 | 87.4 |
| 16K | 43.8 | 64.5 | 96.8 |
| 32K | 45.0 | 61.1 | 91.1 |
| 64K | 61.6 | 65.2 | 94.2 |
| 128K | 42.6 | 65.4 | 85.8 |

Acceptance-normalized engine steps/s:

| context | cc1 | cc2 | cc4 |
|---:|---:|---:|---:|
| 0 | 15.1 | 21.6 | 33.1 |
| 16K | 14.8 | 22.0 | 30.8 |
| 32K | 14.8 | 21.4 | 30.3 |
| 64K | 13.7 | 20.9 | 29.9 |
| 128K | 14.3 | 20.4 | 30.1 |

Geomeans were 45.70/62.87/90.98 aggregate tok/s and
14.52/21.28/30.81 engine steps/s at cc1/2/4. Every cell had zero errors,
zero queue fraction, no warmup timeout, and no capacity limit.

Prefill scouts:

| context | tok/s |
|---:|---:|
| 8K | 2,313 |
| 16K | 2,400 |
| 32K | 2,150 |
| 64K | 2,269 |
| 128K | 2,161 |

Against the geometric mean of the three earlier DGLIN boots, the ten matching
cc1/cc4 cells measured +3.16 percent output, -1.80 percent normalized engine
steps, and +5.05 percent effective acceptance. This is inside the established
boot spread and confirms performance equivalence of the rebuilt image. Four of
five single-sample prefill scouts were within one percent of the earlier
DGLIN geomean; the 32K scout was 8.9 percent lower. One scout does not establish
a 32K regression, while the full serving run remains inside the qualified
performance envelope.

## Artifacts

- `standard-sweep.json` and `standard-sweep.log`: complete benchmark result.
- `qlen-and-prefill-smoke.json` and `.log`: scheduler-shape reproducers.
- `strict-tools.log`: constrained-output and long-context tool tests.
- `deep-ctx-needle.log`: exact-cap and cap-plus-one tests.
- `dispatch-race-200.log`: mixed-stream corruption soak.
- `idle-wakeup-burst.log`: post-idle recovery.
- `receipts/`: final container inspections, complete logs, and runtime
  identities from both ranks.

After the response-quality verdict was revoked, dusty/kirby were restarted on
the identical image with `BACKEND=b12x-a8` as the same-image diagnostic
control. The live process selected `B12xFp8BlockScaledMMKernel`, set
`VLLM_USE_B12X_FP8_GEMM=1`, retained exact-four NCCL and all other launch
settings, and reported 1,100,344 KV tokens. The endpoint was READY at
2026-08-24 11:51:19 EDT and advertised only `DeepSeek-V4-Flash-0731`.
Same-prompt retesting still favored corrected r34 materially. This rules out
the DGLIN switch as the sole cause, but does not yet distinguish the newer
DSpark/rejection-sampling path from target-model execution changes in r18p.
