# JJ r22 Qwen3.8 Flash Next qualification

Date: 2026-09-04

## Qualified deployment

- Model: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89`
- Model revision: `c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d`
- Served name: `Qwen3.8-Flash-Next-NVFP4-4p89` (one advertised model)
- Nodes: dusty and kirby, TP2 over the dedicated 200G pair links
- Image ID: `907c1265f308ea87c77d98e0b787455af228a5dbdfa228b1c7653a48d3433e0a`
- Release tag: `localhost/voipmonitor/vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1`
- Qualification build tag: `localhost/voipmonitor/vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1-scratch`
- Runtime: MTP3, `NCCL_PROTO=LL,Simple`, 262,144-token model length

The image contains the upstream B12X persistent-CTA store-wait fix
`9ae41c5c`. It does not contain the local r15p packed-allocation touch.

This is the same image ID qualified under the build tag and promoted without
conversion or rebuild. Its baked recipe SHA-256 is
`359bc15cb8a55f4907ee3fe52f3bcdc88f24ca8b552623f19ce396d7fd65bdd1`;
its source-lock SHA-256 is
`fe3558327415c379496ae1cd2b6e319b97f711d18e37f97bf24739196eace744`.
The embedded source lock, launchers, and runtime verifier match the committed
artifact bytes. Post-build edits are limited to host-runner release defaults
and qualification records and do not change the image payload.

## Correctness

The MTP3 semantic battery passed all 18 cases: low, medium, and xhigh
reasoning; non-thinking; deterministic image reasoning; and strict tool-call
round trips, each repeated three times. The receipt is
`jj-r22-mtp3-semantic-admission-3x-20260904.jsonl`.

MTP3 exact-retrieval probes passed at 2,848, 2,849, 131,072, and 262,000
input tokens. MTP0 exact-retrieval probes passed at 2,784, 2,785, and 131,072
input tokens. No token-zero or exclamation-mark collapse occurred. Receipts:

- `jj-r22-mtp3-boundary-native-context-20260904.jsonl`
- `jj-r22-mtp0-boundary-native-context-20260904.jsonl`

This closes the Qwen corruption reproducer in both speculative and
non-speculative modes and validates the upstream store-wait fix on SM121.

The fixed, pre-rendered token corpus completed 6,144 output tokens with no
errors. R22 used 2,720 engine steps for an acceptance length of 2.2588; r15p
used 2,676 steps for 2.2960. The 1.6% difference is within the observed MTP
variability. Exact generated-token hashes varied across waves on both images,
so hash equality is not used as a quality or acceptance gate. Receipt:
`jj-r22-fixed-token-acceptance-20260904.json`.

Both final containers run the exact image ID and report no recent engine,
traceback, OOM, or killed-process errors.

## Standard decode benchmark

The standard `llm-inference-bench/run_bench.sh` 0.4.29 grid ran all 15 cells:
contexts 0, 16K, 32K, 64K, and 128K at concurrency 1, 2, and 4, with 30-second
measurement windows and an 8,192-token output ceiling. There were zero cell,
warmup, or API errors.

Raw R22 engine steps per second:

| Concurrency | 0 | 16K | 32K | 64K | 128K |
|---:|---:|---:|---:|---:|---:|
| 1 | 16.2418 | 16.4164 | 16.3028 | 16.3338 | 16.0442 |
| 2 | 28.2121 | 27.5706 | 28.0562 | 27.3358 | 27.6911 |
| 4 | 43.4526 | 42.6705 | 42.8049 | 44.0400 | 42.8521 |

Geometric-mean comparison with qualified r15p:

| Concurrency | r15p | r22 | Delta |
|---:|---:|---:|---:|
| 1 | 16.4210 | 16.2673 | -0.94% |
| 2 | 27.8856 | 27.7713 | -0.41% |
| 4 | 43.7626 | 43.1610 | -1.37% |
| All cells | 27.1620 | 26.9154 | -0.91% |

This is decode parity; there is no material regression. Output-token-rate
deltas are acceptance-sensitive and are therefore secondary.

Campaign:
`llm-inference-bench/results/runs/qwen3.8-flash-next/nvfp4-4p89/2026-09-jj-r22-vs-r15p/campaign.yaml`.

Result:
`llm-inference-bench/results/runs/qwen3.8-flash-next/nvfp4-4p89/2026-09-jj-r22-vs-r15p/throughput/20260904T114619-0400__jj-r22-sm121-upstream-store-wait-tp2-mtp3__r01.json`.

## Prefill and capacity

The focused deterministic 16K warm-prefill median was 11.068912 seconds,
versus 11.041812 seconds for r15p, a 0.25% latency difference. The initial
cold scouts were excluded from the comparison because they mixed first-use
work with prefill timing. Receipt:
`jj-r22-prefill-8k-16k-4run-20260904.jsonl`.

KV capacity varied between clean MTP3 boots as CUDA-graph allocation changed:
4,982,992 tokens in the benchmark boot and 5,218,621 tokens in the final
serving boot. The MTP0 qualification boot reported 6,289,728 tokens. Each
figure is tied to its own boot rather than treated as a fixed image property.

## Verdict

JJ r22 passes Qwen correctness, native-context, MTP0/MTP3 boundary,
acceptance, decode, and focused-prefill qualification on the two-node SM121
pair. It is suitable to remain serving on dusty and kirby and to advance to
the separate four-node GLM-5.3-Flash qualification window.
