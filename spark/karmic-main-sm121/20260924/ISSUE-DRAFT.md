# DS4.1 Flash on four GB10s: repeated cold greedy 524K retrieval changes answer

Draft only; not posted. This reports reproducibility and a qualification failure,
not established numerical incorrectness or a proven kernel cause.

## Configuration

- Four DGX Sparks, TP4/DCP1, switched 200G RoCE, Podman, SM121.
- vLLM `1794dcf18454900263e0c66711af8ea4a1283ac1`.
- B12X `a7d7d29b2ef8869086e0ceaa787321f17544e3c9`.
- Model `deepseek-ai/DeepSeek-V4.1-Flash`, revision
  `fb2764a5cf321eaa5070ca8f9e892818f477c16d`.
- Source baseline: the pinned upstream four-Spark launcher, with direct Podman
  multi-node orchestration instead of its external cluster launcher.
- NCCL built from NVIDIA tag `v2.30.7-1` (`73cf1122`), not the patched
  library referenced by the upstream launcher; K7 DSpark;
  disk Engram with resident scales; FP8 KV;
  4 sequences; 8192 batched tokens. Overlap settings remain upstream defaults.
- Explicit approved deviations: utilization 0.85 with automatic KV sizing
  instead of fixed KV bytes; no container memory limits; private IPC with
  64 GiB shm ceiling; max context 600000 after successful 500K admission.
- Local image `eb3adc203432f2e54738c5977b65bc6a618ba4fdb76f23dd510df5910365e08d`.
  ARM64/SM121 build overlays and PR865 are recorded in the accompanying locks;
  this is not the published x86 binary.

## Reproducer and observed results

Same 524288-token input on the same boot, temperature 0, thinking disabled,
64 output-token ceiling, top-5 logprobs, a unique cache salt on each request.
The prompt begins with archive identity `510c94b1bb4f42d2998093bd6b9c3d99`
and retrieval code `739184`, followed by repeated ` filler` text and a second
code `482617` near 70 percent depth. It asks for the initial and late codes.
Expected output is `739184, 482617`.

| Trial | Answer | Code-minus-identity first-token logprob | Seconds |
| --- | --- | ---: | ---: |
| 1 | `739184, 482617` | +1.75 | 682.83 |
| 2 | `510c94b1bb4f42d2998093bd6b9c3d99, 482617` | -3.75 | 470.37 |
| 3 | `510c94b1bb4f42d2998093bd6b9c3d99, 482617` | -6.50 | 522.25 |

All three finish with `stop`, report 524288 prompt tokens, and have identical
input-content SHA256
`b410c6b19d492e83ecb775805266d91eed15ded5e3afd0cf264c25aae41bc7e4`.
The complete series has 1572864 queried tokens, three successful requests,
zero prefix hits, zero preemptions, no concurrent requests and no queued work.
API usage cache details are null, so coldness is established from retained
Prometheus counters rather than inferred from that missing field.

Arithmetic, thinking/non-thinking, vision, tool round trip, concurrent short
requests and fresh-identity dual retrieval through 499000 tokens passed.
The full ladder through 499000 ran on the 500K-context boot. The historical
trials and instruction control ran after a context-only 600K restart on the
same image, with short checks and cold dual retrieval through 262000 repeated.
A final-instruction-only control returned the expected `739184` in 540.32 s
(524274 input tokens). It confirms wording sensitivity, not resolution.

## Limits and questions

An older R38 investigation reproduced the same answer sensitivity, so this is
not established as a new Karmic regression. That investigation found legitimate
BF16 accumulation and indexer tie-selection nondeterminism; neither fact alone
proves incorrect computation. We have no non-B12X reference distribution for
this exact input and cannot attribute the wrong answer to model versus runtime.

Elapsed times vary substantially, but these are diagnostic requests, not a
matched throughput benchmark. An unchecked disk-Engram overlap timeout exists
in the pinned source; we have not observed its failure flag or established
that it fired, caused the latency, or affected an answer. Keep that source-level
concern separate from this measured result.

Questions for maintainers: is this degree of greedy first-token variation an
expected tolerance for the qualified four-Spark profile? Is there a supported
deterministic/reference configuration for this same checkpoint and long input
to distinguish model ambiguity from implementation-induced bias?

## Local evidence

`receipts/historical-524k-600k/` retains complete requests and responses;
`receipts/historical-524k-metrics.jsonl` retains the independent counters.
`review_historical.py` checks trial and aggregate accounting. The replay uses
the unchanged `spark/ds41/r38/probe-needle-diagnostics.py` with `cold-replay:`
pointing at the retained original input. No external report has been submitted.
