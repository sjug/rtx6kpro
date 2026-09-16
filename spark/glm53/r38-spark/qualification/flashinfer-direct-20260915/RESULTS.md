# R38 FlashInfer screen, 2026-09-15

## Verdict

Rejected at first-completion correctness. No performance benchmark, warmup pass,
semantic battery, or needle suite ran. No new R38 control measurement was taken.
The archived R38/R32 comparisons remain the performance evidence.

The FlashInfer arm completed model loading, target/MTP profiling and graph capture
on both nodes. Three sequential requests with the payload in `request.json`
returned garbled content and exhausted the 32-token budget. Each request was
greedy, with thinking disabled. All three responses begin with `3未知Pe` before
diverging. Expected response: exactly `333` with finish reason `stop`.

Raw responses are `first-completion.json`, `completion-repeat-2.json`, and
`completion-repeat-3.json`. This is a repeatable failure within one boot, not a
multi-boot reproducibility claim. It differs from the earlier B12X-arm startup
illegal-instruction failure: this arm booted and served incorrect output.

## Identity and scope

- Nodes: dusty and kirby only. GLM and DS4 were not touched.
- Image: `ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5`.
- Checkpoint: `c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d`.
- Experimental launcher SHA-256:
  `1caa57eece7fd7470a3ad07190bb72507fc56e89ed7075b825030ba46ed99af0`.
- TP2, MTP3, aligned retention, utilization 0.85, max length 262144,
  max sequences 4, batched-token budget 4096.
- Both rank logs resolve **FlashInfer GDN prefill and CUDA GDN decode**.
  R38 stock resolves B12X for both. R32 resolves FlashInfer prefill and B12X decode.
- The R38 selector therefore changes two backends for this flag-only experiment.
  The failure cannot be attributed to FlashInfer prefill alone, nor does it
  invalidate the earlier performance hypothesis about B12X prefill.
- KV reported on this boot: 5,277,669 tokens, 42.88 GiB on dusty and 44.0 GiB
  on kirby. This is a boot-specific receipt, not a capacity comparison.

The preserved original R38 containers passed the same arithmetic check immediately
before this experiment. See `../stock-r38-replay-20260915/`. That was a startup and
correctness replay, not another benchmark control.

## Receipts and operational outcome

`*-container.json` captures the failed arm's exact configuration. `*-final.log`
captures startup and the first request; `restore-r32/*-flashinfer-stopped.log`
captures the entire run through graceful teardown. `*-kernel.log` and `telemetry/`
retain node observations. Keep experiment and restoration times separate when
reading the continuous telemetry.

The failed containers were stopped worker first, head last, with `podman stop -t 60`,
and preserved. The original qualified R32 containers were then started worker first.
Restoration completion and identity checks are recorded under `restore-r32/`.

R32 restoration passed at **21:08:58 EDT**. Both ranks were verified running image
`74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c`, MTP3 and
aligned retention, with no container OOM flag. The same request returned exactly
`333` with finish reason `stop`. R32 is left serving. The experiment's telemetry
session was stopped after restoration.

The prepared `measure.sh` was deliberately not executed. Its correctness-pass
marker is absent. Both the benchmark source repository and its results repository
were left unchanged; the campaign here records the rejected screen only.

Root cause remains undetermined. A prefill-versus-decode discriminator must retain
B12X decode to reproduce R32's backend combination; that is not expressible with
the tested R38 flag-only selector. No source patch or further restart was made.

## Bounded local review

Claude reviewed the local receipts and reported the fused CUDA speculative-decode
path or its recurrent-state handoff as the leading candidates. These are hypotheses,
not a causal verdict. The logged speculative metrics had acceptance length 1.00
and zero accepted drafts (the first request's interval reports 0 of 93; later
intervals report 0 of 150 and 0 of 36). This is not a per-request acceptance trace
for the repeats.

The three responses share a correct visible first character, but their token IDs
and logprobs were not captured. Do not call this proof of a correct first token or
proof that prefill is sound. Divergence among greedy repeats establishes differing
output in this process, not by itself a particular race or kernel mechanism.

Both repeat responses are preserved as individual JSON receipts. The empty
`readiness-curl.log` is curl stderr on a successful HTTP transaction; HTTP success
does not imply output correctness. The explicit assertion on response content
failed, so no correctness marker or benchmark record was produced.

The next suggested discriminator is correctness-only FlashInfer prefill plus an
explicit Triton decode backend, with token/logprob receipts. Passing would narrow
the problem to the CUDA decode path or its state-layout interaction, not uniquely
prove an internal CUDA-kernel defect. Failure would likewise not by itself exclude
all flag-only alternatives. This additional boot has not been run.
