# Upstream check, September 29, 2026

Refresh of the [September 28 check](upstream-check-20260928.md). Every configured
remote in the eleven existing checkouts was fetched with `git fetch <remote>
--no-prune` (22 remotes, all succeeded); all tips are fast-forwards. No pull, merge,
checkout, build or serving change was made for this check.

## Publication

Seventeen releases since 2026-09-28 16:49 UTC. Newest:

| | Main | Beta |
| --- | --- | --- |
| Image | `karmic-kraken-20260929-6c0c3fd2730e0be3` | `karmic-kraken-beta-20260929-7476365edc3d8ee5` |
| Digest | `sha256:ff2f7b23262cc575...` | `sha256:74c57a9bf4142e03...` |
| Recipe | `f28aea04babb` (+23) | `f28aea04babb` |
| vLLM | `0296a817c885` (+4) | `99cbe782f3b6` (+80) |
| B12X | `a489f972e0dd` (+4) | `1b6cd278626a` (+20) |
| LMCache | `75f2b59d3be8` | `75f2b59d3be8` |

FlashInfer, InstantTensor and NCCL are unchanged; the foundation is still linux/amd64,
CUDA 13.4.1. There is still no ARM64/SM121 publication.

## DS4.1: compressor ring and Engram ordering

- Beta [5eec8ea30a](https://github.com/local-inference-lab/vllm/commit/5eec8ea30a)
  plus [dfeb39654a](https://github.com/local-inference-lab/vllm/commit/dfeb39654a)
  (vllm #943/#926): `CircularBufferSpec.uses_slot_mapping` returned `False`, so every
  compressor-ring row was padding and the ring was never written. Release notes
  `vllm-943` and `vllm-943-long-context`: on every earlier SM12x DS4.1 image, decode
  drifted from prefill (KL 0.038 over the first 256 tokens to 0.156 at 1,536-2,048;
  top-1 differs at 7.6% of positions; 0.018-0.030 with the fix). Prefill was also
  affected at every chunk boundary. An earlier partial port (#927) was reverted
  because it did not engage (`3fbef72918`).
- **Correction to the September 28 report:** it called 706428761f "not applicable"
  because the flag was `False`. The flag being `False` was the defect itself.
- **Our status:** we found the same defect independently on 2026-09-25
  (`spark/karmic-main-sm121/20260924/DETERMINISM-WORK.md`, "Compressor-ring causal
  result") and fixed it in DS4.1 `sparse_mla.py` (ring-fix candidate `e7b273407022`).
  The Sept 27 release `1989e16d` carries that fix; its compressor-ring mapping gate (29
  cases) passed. So the 524K failure on `1989e16d` is **not** explained by this bug.
  Aiden's `production-1.1` image (Sept 11), used for the Sept 29 reference run, most
  likely predates any ring fix. Its 0/3 therefore ran with stale compression windows
  at 64 prefill chunk boundaries and is weaker reference evidence than first reported.
- Main [4211209df5](https://github.com/local-inference-lab/vllm/commit/4211209df5):
  disk Engram gathers are enqueued on the current stream before graph replay, and the
  polling epochs are removed. A late side-stream lookup could be starved by graph
  polling, hit both 5-second timeouts and leave stale rows. This is the defect our
  `ROOT-CAUSE-ANALYSIS.md` flagged (`common/engram.py` 5000 ms spin). `1989e16d`
  carries our own Engram epoch and enqueue fixes; the upstream design differs and is
  simpler.

## Startup and tuning (main)

- vLLM [0296a817c8](https://github.com/local-inference-lab/vllm/commit/0296a817c8)
  with B12X [a489f972](https://github.com/local-inference-lab/sparkinfer/commit/a489f972)
  and [0b9ff5ad](https://github.com/local-inference-lab/sparkinfer/commit/0b9ff5ad):
  ranks fetch the tuning rank's winning CuTe programs instead of recompiling. Candidate
  sharding and winner comparison are unchanged, so tuning still varies per boot; this
  is a startup-time change, not selection pinning.
- [6cf7490681](https://github.com/local-inference-lab/vllm/commit/6cf7490681) and B12X
  [506723a5](https://github.com/local-inference-lab/sparkinfer/commit/506723a5): four
  compiler workers by default on Spark. [7a9de065a2](https://github.com/local-inference-lab/vllm/commit/7a9de065a2)
  and B12X [07932439](https://github.com/local-inference-lab/sparkinfer/commit/07932439):
  batched plan release and MXFP4 program reclamation (memory/startup).

## Other beta fixes

- [296220595b](https://github.com/local-inference-lab/vllm/commit/296220595b):
  `max_token_id` was `vocab_size`, admitting one id past the embedding table on DeepSeek
  V4.x with fastokens.
- [0928737d9f](https://github.com/local-inference-lab/vllm/commit/0928737d9f): structured
  output kept MTP draft placeholders unconstrained after a skipped step.
- [1c4a92571f](https://github.com/local-inference-lab/vllm/commit/1c4a92571f): CUDA graph
  memory estimate read about zero per graph on DS4.1 after #918.
- [eec23823da](https://github.com/local-inference-lab/vllm/commit/eec23823da): GLM parser
  preserves generated whitespace, so the next turn hits the hybrid response checkpoint.

## Recipe and other repositories

- Recipe +23: fastokens tokenizer on by default (opt-in first), restart generated
  deployments after a crash, 60 s stop grace, GLM verbatim-content template, lil-bench
  0.7.4, tuning-exchange verify gate.
- FlashInfer +20 (SM100/SM103/SM110 Cake, PCIe IPC collectives); nothing SM121.
- LMCache dev +10 (SGLang DeepSeek V4.1 support), SGLang +44: out of scope.
- llm-inference-bench +2 (lil-bench 1.3.1), dgx-spark-infra +1 (our upgrade paths),
  rtx6kpro +1 daily summary. CUTLASS and spark-vllm-docker unchanged.

## Fetched tips

| Checkout | Ref | Tip |
| --- | --- | --- |
| rtx6kpro | upstream/master | a8aee42422bd |
| vllm | lil/dev/karmic-kraken | 0296a817c885 |
| vllm | lil/integration/karmic-kraken-beta | 99cbe782f3b6 |
| b12x | lil/master | a489f972e0dd |
| b12x | lil/integration/karmic-kraken-beta | 1b6cd278626a |
| blackwell-llm-docker | origin/main | f28aea04babb |
| flashinfer | upstream/main | 10c80d75ba63 |
| LMCache | origin/dev | 8eb36fa21315 |
| cutlass | origin/main | 0b55a2f691d6 |
| sglang | origin/main | 98fce73d5bd0 |
| spark-vllm-docker | upstream/main | 42f62e349c61 |
| dgx-spark-infra | origin/master | f52bf95a7b3d |
| llm-inference-bench | upstream/main | 72437f9db18b |

## Recommendation

- DS4.1: the upstream ring fix (#943) and Engram ordering fix (4211209df5) confirm two
  defects we had already fixed in `1989e16d`. They should replace our local versions in
  the next DS4.1 composition. Neither explains the 524K gate failure, which stands.
- If an independent reference is still wanted, it must be an image that includes the
  ring fix. Aiden's `production-1.1` does not qualify as a clean reference.
- Other models: no change to the September 27 verdict. The fastokens `max_token_id` fix
  matters only if we adopt fastokens.
