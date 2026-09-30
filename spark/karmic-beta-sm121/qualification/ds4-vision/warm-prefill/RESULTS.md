# DS4 Vision prepared-tail diagnostic

Executed September 20, 2026, 23:39 to 23:47 EDT, without restarting either node.
Both container identities and original 22:09 boot times were checked before
and after. Candidate image remains f30dc6d9a2a6, TP2/K3, unchanged serving
configuration. Qwen and GLM were not touched. No commits or benchmark-repository
edits were made.

## Method

`ds4-vision/probe-warm-prefill.py` performs a focused diagnostic, not a replacement
standard benchmark grid. It reads the digest-pinned benchmark's padding text,
generation prompt and message construction helpers without importing its runner
or modifying its repository. Each request has a unique nonce prefix, adjusted
through `/tokenize` to the exact original scout prompt count. Default sampling
and thinking settings, streaming, ignore_eos and one output token match the scout.
The body uses the same benchmark padding; fresh prefixes make these different
token streams, not identical full-input replays. Three passes run ascending,
descending, ascending by length. Full requests and SSE responses are retained.

All 15 samples passed: exact token count, zero cached tokens in usage, one
completed request, one server prefill timing observation, and newly computed KV
tokens equal to the complete input length. Idle counters were checked before
each request. Logs show only the diagnostic client. No first-use preparation or
JIT warnings occurred on either rank during the entire test.

## Results

Client TTFT medians, seconds. Original Karmic values are the single cold-plan
scouts in the completed standard grid, not a simultaneous control.

| Actual prompt tokens | Tail rows | Original Karmic | Warm-plan trials | Warm median | Server median |
|---|---:|---:|---|---:|---:|
| 8198 | 6 | 3.938 | 5.291, 6.622, 3.929 | 5.291 | 5.267 |
| 16147 | 3859 | 8.261 | 7.402, 7.686, 7.462 | 7.462 | 7.415 |
| 32077 | 3405 | 15.286 | 16.734, 15.078, 15.094 | 15.094 | 15.015 |
| 63932 | 2492 | 33.728 | 35.908, 36.246, 33.377 | 35.908 | 35.773 |
| 127645 | 669 | 77.587 | 83.617, 90.556, 90.041 | 90.041 | 89.766 |

Warm median client rates are 1549, 2164, 2125, 1780 and 1418 tok/s respectively.
The historical R38 scout rates were 2104, 2253, 2219, 2142 and 2003 tok/s.
These are historical rate comparisons with fresh prompts, not a matched R38
rerun or attribution proof. The 524K request was not repeated in this diagnostic.

## Conclusion

The long-context deficit survives with plans already prepared and zero prefix
reuse. First-use preparation is not a sufficient explanation. Server timing
tracks client timing closely, so HTTP overhead does not explain the residual.
16K improved by about 0.8 seconds, consistent with a preparation contribution,
but the experiment does not measure that contribution directly. 32K's median
improvement was only about 0.2 seconds, with a slower first repeat. Even 8K has
large spread without a preparation warning. We cannot claim that two independent
causes have now been isolated, or that short-context performance uniformly
returns to R38 parity.

No specific DSA/indexer or autotuner mechanism is established, and no complexity
law is inferred from these five lengths. Profiling a prepared long-context
request is the next discriminating step, not another uninstrumented R38 grid.
Promotion remains held. The candidate is left serving unchanged.

## Health and evidence

No new allocation warnings, Xids or tracebacks occurred. Mean sampled SM clocks
were 2405 MHz on rusty and 2484 MHz on toby; maximum temperatures 87 C and 82 C.
Rusty had four nonzero clock-event samples; toby had none. These are retained
caveats, not a causal explanation for the deficit. See `*-gpu.csv`, full
`*-container.log` and `*-kernel.log`, `identity.json`, `*-final.json`, individual
request/response and Prometheus receipts, `results.json`, and `summary.json`.
