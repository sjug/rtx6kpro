# Draft upstream report: exact-repeat requests are scheduled exclusively under the auto recurrent checkpoint policy

Status: draft, not filed. Attribution to the scheduler path below is
provisional pending confirmation by source tracing or a maintainer.

## Summary

On Jovian Judgement r27 (vLLM integration `63a82f8d`, tree `f54cd9ca`, plus
PR 674) serving Qwen3.8-Flash-Next NVFP4-4p89 at TP2/MTP3 on two DGX Spark
GB10 nodes with `--recurrent-checkpoint-policy auto` (the default), a request
whose entire prompt matches a previously completed prompt is admitted only
when the engine has no other scheduled work, and while it waits at the head of
the waiting queue no other waiting request is admitted. Two user-visible
effects follow:

1. Concurrent identical prompts serialize. Four identical 200-token requests
   complete one after another (49 tok/s aggregate) while four distinct prompts
   batch at 111 tok/s and four multi-turn extensions batch at 118 tok/s.
2. Fresh requests queued behind a repeat are delayed until running decodes
   finish. With two 600-token decodes running, a repeat submitted at +2 s
   reached first token at 14.7 s, and two fresh requests submitted at +3 s
   reached first token at 9.1 s and 12.1 s instead of under 1 s.

Identical concurrent requests are valid production traffic (shared system
prompts with short user turns, retries, load tests), so this is a serving
regression relative to r26, not only a benchmark artifact. The standard
llm-inference-bench 15-cell grid, which sends the same padded prompt to every
concurrent client, measured c2 and c4 engine steps/s equal to c1 (19.3 vs
31.9 and 49.0 on r26), with the server logging `Running: 1, Waiting: 3` at 1
to 2% KV usage.

## Environment

- Image: our SM121 derivative of r27 (`ef669fa1…`), vLLM Spark tree
  `8881b777…` = r27 integration tree + SM121 overlay + PR 674 head `04813092`.
- Model: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89` @ `c374e7e2`,
  TP2 over a direct 200G link, `--mamba-cache-mode align`,
  `--enable-prefix-caching`, MTP3, FP8 KV, capture sizes 1 2 4 8 16 24 32,
  `--max-num-seqs 4`, native 262,144 context, 0.85 utilization.
- `--recurrent-checkpoint-policy` left at `auto`, which resolves to
  `request_boundaries` for this configuration
  (`vllm/config/vllm.py::use_request_boundary_checkpoints`).

## Reproducers

`qualification/qwen-20260906/probe-concurrency-fresh-extension-repeat.py`
(four concurrent greedy 200-token requests per phase, `ignore_eos`):

```
A: 4 fresh distinct          wall=7.2s  agg_tok/s=110.7 per_req_s=[7.2, 7.0, 6.9, 6.9]
B: 4 extensions of A         wall=6.8s  agg_tok/s=118.4 per_req_s=[6.6, 6.6, 6.8, 6.8]
C: 4 fresh distinct after B  wall=6.8s  agg_tok/s=117.8 per_req_s=[6.2, 6.8, 6.6, 6.3]
D: repeat A verbatim         wall=16.3s agg_tok/s=49.2  per_req_s=[3.9, 12.4, 8.2, 16.3]
E: 4 fresh distinct after D  wall=6.9s  agg_tok/s=116.6 per_req_s=[6.3, 6.7, 6.6, 6.9]
```

`qualification/qwen-20260906/probe-repeat-head-of-line.py` (streaming TTFT):

```
long0    ttft=  0.3s total= 16.2s
long1    ttft=  0.6s total= 16.7s
repeat   ttft= 14.7s total= 16.7s   (submitted at +2 s, verbatim repeat of a completed prompt)
fresh1   ttft=  9.1s total= 10.6s   (submitted at +3 s)
fresh0   ttft= 12.1s total= 13.8s   (submitted at +3 s)
```

Server-side, during the benchmark's c4 cells:
`Running: 1 reqs, Waiting: 3 reqs, GPU KV cache usage: 1.4%`.

## Provisional attribution

`vllm/v1/core/sched/scheduler.py` (r27 tree): when a waiting request's prefix
lookup returns `num_new_local_computed_tokens == request.num_tokens` and the
request carries a `boundary_checkpoint`, `boundary_logits_only` is set. Two
consequences in the waiting-queue loop:

- `if boundary_logits_only and num_scheduled_tokens: break` (around line
  1249): the request is not admitted while anything else is scheduled this
  step, and the `break` also stops admission of every request behind it.
- After admission, `token_budget = 0; break` (around line 1646): "The
  boundary-logits step must be the only scheduled work."

Because running decode requests always schedule tokens, a repeat request is
admitted only once the engine is idle, and any burst of repeats drains the
engine between each one. This reading matches the observed behavior but has
not been confirmed by tracing.

## Expected behavior

A prompt that fully matches a published endpoint checkpoint should be able to
restore and join the running batch without requiring an idle engine, or at
minimum should not block admission of unrelated waiting requests.

## Workaround measured

Relaunching the same image and launcher with
`--recurrent-checkpoint-policy aligned` and nothing else changed removes both
effects:

```
c4 distinct                 wall=7.0s  agg_tok/s=114.3
c4 identical                wall=7.3s  agg_tok/s=110.3   (auto: 16.3s, 49.2)
D: repeat A verbatim        wall=6.7s  agg_tok/s=119.1   (auto: 16.3s, 49.2)

repeat   ttft=0.3s   fresh0 ttft=0.3s   fresh1 ttft=0.3s   (auto: 14.7 / 12.1 / 9.1 s)
```

The unchanged 15-cell benchmark recovers: engine steps/s geometric means c1
19.39, c2 32.82, c4 50.40 (auto 19.31 / 19.32 / 19.29; previous release
19.17 / 31.91 / 48.98). Correctness is unchanged (semantic set 18/18,
retrieval exact through 262,000 tokens, speculative acceptance 2.19 vs
2.25).

## Second model family: GLM-5.3-Flash

The same image reproduces both symptoms with `local-inference-lab/GLM-5.3-Flash-NVFP4` revision `46aaae8a82032f77100f2f03e9cc11b391df3b4d`, TP4/DCP1 on four GB10 nodes over switched 200G RoCE, BF16 MTP3 proposal head, FP8 KV, maximum sequences 8, and native 1048576 context. The matched aligned run changes only the recurrent-checkpoint-policy argument; all four actual command-line receipts otherwise match. This is a second model-family reproduction, not proof of a different mechanism.

| Probe | Auto | Aligned |
| --- | ---: | ---: |
| C4 identical, aggregate output tok/s | 50.4 | 105.5 |
| Repeat four previously completed distinct prompts, tok/s | 56.8 | 113.2 |
| Repeated head-of-line request TTFT, seconds | 13.2 | 0.3 |
| Fresh request 0 behind repeat TTFT, seconds | 10.5 | 0.7 |
| Fresh request 1 behind repeat TTFT, seconds | 7.5 | 0.6 |

Reproducer: `tests/glm/probe-concurrency-glm.py`, same request sequence in both arms. Streaming TTFT counts first nonempty generated text, not an empty role/header delta. Fresh distinct requests batch in both arms. Auto's standard benchmark has all ten C2/C4 cells underfilled at effective concurrency one; aligned admits the requested concurrency in every cell with zero warmup timeouts or request errors. Engine steps/s geometric means at C1/C2/C4 are 19.670/19.729/19.832 under auto (C2/C4 underfilled) and 19.742/31.198/45.427 under aligned.

Both arms pass the semantic reasoning/vision/tool battery and native retrieval through 1048000 tokens. Results are single-boot measurements. Receipts are under `qualification/glm-20260906/` and `qualification/glm-aligned-20260906/`. Cache-reuse tradeoffs are documented separately in `qualification/GLM-ALIGNED-WINDOW.md`; this report's claim remains identical-repeat serialization and head-of-line blocking, with source attribution provisional. This draft has not been filed.
