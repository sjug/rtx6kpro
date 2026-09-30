# Upstream check, September 28, 2026

Refresh of the [September 27 check](upstream-check-20260927.md). Every configured
remote in the eleven existing checkouts was fetched with `git fetch <remote>
--no-prune` (22 remotes, all succeeded). No pull, merge, checkout, branch change,
build, deployment or serving change. All tips below are fast-forwards of the
September 27 tips.

## Publication

Twelve releases since 2026-09-27 21:38 UTC. Newest:

| | Main | Beta |
| --- | --- | --- |
| Image | `karmic-kraken-20260928-7c61f87df9a18cad` | `karmic-kraken-beta-20260928-6bd83bf0760b8115` |
| Published | 2026-09-28 16:33:58 UTC | 2026-09-28 16:48:12 UTC |
| Digest | `sha256:d45833596886...` | `sha256:5db12a21e923...` |
| Recipe | `1aa1b59bc29b` | `1aa1b59bc29b` |
| vLLM | `502d6cb5acd2` (+35) | `6ea2fcf67e08` (+18) |
| B12X | `b4b12bcf200a` (+7) | `f8069b2c0be1` (unchanged) |
| LMCache | `ab11b841fcc7` | `ab11b841fcc7` |

FlashInfer, InstantTensor and NCCL pins are unchanged. Foundation remains amd64
CUDA 13.4.1; release scope remains native smoke plus LMCache contracts.

## vLLM main (+35): Kimi work, not on our path

#842, #857, #924 and #925 integrate Kimi-K3 EXL3 precision, padded uneven-TP
loading, MLA memory and DCP transport options, and a B12X dense-MLA backend
(`c27acd0fba`). The new route in `vllm/platforms/cuda.py` applies only when the
B12X backend is selected with `use_mla and not use_sparse`. GLM layers are sparse
when the config sets `index_topk` (our sparse pooled indexer path), and DS4/DS4.1
use sparse MLA, so the dense route should not reach our models. The only GLM file
touched is a plumbing change in `glm5next/nvidia/l2_prefetch.py` (optional
persisting-L2 request), inert with prefetch disabled. None of the beta items from
September 27 (#911, #918, DSML fix, 638f90a1fc, #897, QSA865) were promoted to main.

## vLLM beta (+18)

- Failed-rank handling, worth taking: [b98ece7308](https://github.com/local-inference-lab/vllm/commit/b98ece7308)
  and [34d0206fee](https://github.com/local-inference-lab/vllm/commit/34d0206fee)
  gather replies from every rank without waiting in rank order. The commit's example
  is 4x GB10 at startup, where one rank's CUDA graph capture failed and rank 0 hung in
  the next all-reduce. [64f0d170c7](https://github.com/local-inference-lab/vllm/commit/64f0d170c7)
  surfaces failures during per-step single-rank RPCs, instead of a 300 s wait with the
  server up. [a8ae18928e](https://github.com/local-inference-lab/vllm/commit/a8ae18928e)
  and [1d5a74c62a](https://github.com/local-inference-lab/vllm/commit/1d5a74c62a)
  make `vllm serve` exit 1 when the engine failed, so restart policies notice.
  Together with #897 these target our zombie-APIServer class of failure, though
  completion-probe liveness should stay.
- **Corrected 2026-09-29 (see upstream-check-20260929.md): the analysis below is wrong.** `uses_slot_mapping=False` was the defect itself (ring never written); our release `1989e16d` already carried an independent fix.
- [706428761f](https://github.com/local-inference-lab/vllm/commit/706428761f),
  DS4.1 compressor ring mapping: not applicable to our tree. At our pin
  `1794dcf18454` and at beta, `CircularBufferSpec.uses_slot_mapping` is `False`,
  and the slot-mapping kernel still pads every mapping-disabled group, so the new
  circular branch is a no-op for that spec here. The fix is cherry-picked from a tree
  where rings used the generic mapping. It does not explain the 524K failure.
- Checkpoint-restore admission and scheduler import fixes (#922, #929): LMCache
  external-cache path, off with LMCache disabled.

## B12X master (+7)

- [c616fbf0](https://github.com/local-inference-lab/sparkinfer/commit/c616fbf0) (#417)
  and [332211de](https://github.com/local-inference-lab/sparkinfer/commit/332211de)
  (#431): the Qwen `gdn_prefill` autotune deadlock fix and cufile.json comments,
  previously beta only, are now on master.
- [15c7f6be](https://github.com/local-inference-lab/sparkinfer/commit/15c7f6be):
  preparation shares immutable launch bundles within jobs and reclaims unused
  programs after teardown. Touches DSA indexer, sparse MLA and the other preparation
  packages. A lead for the DS4 Vision JJ-main preparation retention (about 4.4 GiB
  MemAvailable), not a demonstrated recovery; measure per-stage memory before claiming it.
- [163b7951](https://github.com/local-inference-lab/sparkinfer/commit/163b7951):
  cooperative launch candidates rejected with CUDA error 720 are drained and
  released instead of failing preparation. Relevant to tuning robustness on GB10.
- [b4b12bcf](https://github.com/local-inference-lab/sparkinfer/commit/b4b12bcf):
  opt-in A16 token cutoff (default 0). [2e6540dc](https://github.com/local-inference-lab/sparkinfer/commit/2e6540dc):
  W4A16 fallback scale flattening for parameter-backed global scales, a small fix on
  the W4A16 path Qwen uses. [bf7da25a](https://github.com/local-inference-lab/sparkinfer/commit/bf7da25a):
  compile workers keep fake-tensor DSA indexing off CUDA.
- Schema 14 from September 27 still applies to any master-based build.

## Other repositories

- Recipe +3: GLM Spark TP2 preset keeps 0.5 GiB free per GPU (RTX PRO hardware).
- FlashInfer +13: SM100/SM103/SM120 Cake work, CUDA 13.4 CI images, CuTe DSL NVFP4
  W4A16 MegaMoE. Nothing SM121; not in the publisher pin.
- LMCache dev +9 (includes a hybrid DAX false-OOM fix), SGLang +32: out of scope.
- eugr spark-vllm-docker +1 (FlashInfer publishing change), legacy reference.
- dgx-spark-infra +1: our own Podman PPA upgrade path.
- rtx6kpro upstream/master +1 daily summary (`816274a`). CUTLASS and
  llm-inference-bench unchanged.

## Fetched tips

| Checkout | Ref | Tip |
| --- | --- | --- |
| rtx6kpro | upstream/master | 816274ad9fe5 |
| vllm | lil/dev/karmic-kraken | 502d6cb5acd2 |
| vllm | lil/integration/karmic-kraken-beta | 6ea2fcf67e08 |
| b12x | lil/master | b4b12bcf200a |
| b12x | lil/integration/karmic-kraken-beta | f8069b2c0be1 |
| blackwell-llm-docker | origin/main | 1aa1b59bc29b |
| flashinfer | upstream/main | a31c72523ae3 |
| LMCache | origin/dev | e73629e28625 |
| cutlass | origin/main | 0b55a2f691d6 |
| sglang | origin/main | 6fa3fe69e2e5 |
| spark-vllm-docker | upstream/main | 42f62e349c61 |
| dgx-spark-infra | origin/master | 3c3aaaa370c0 |
| llm-inference-bench | upstream/main | 05ec18030f29 |

## Effect on the September 27 recommendation

No change to the verdict. The parser fix remains the only item worth a near-term
image change (DS4 Vision R38p and nous). Add the failed-rank handling commits to the
hardening set that rides along with the next rebuild of any multi-rank image. The B12X
preparation reclamation change is a DS4 Vision capacity lead to test, not a reason to
build. Nothing new addresses the DS4.1 524K failure.
