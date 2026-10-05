# Upstream refresh, October 3, 2026

## Result

All 21 configured remotes in the 11 existing checkouts fetched successfully with
`git fetch <remote> --no-prune`. Local `rtx6kpro/master` fast-forwarded one commit
(`7288ac0`, daily summary publication) from `a968efc0d561424596417246c14e69895b8a0b04`
to `7288ac04678466168c57360392702dd2b0439eee`, matching `upstream/master`. The
active branch remains `spark` and the working tree was not touched. No checkout,
worktree, remote addition, push, build, or serving action was performed.

A new Karmic beta is published (October 3). The October 2 beta (`7b398c9e`) is no
longer listed as a release, and the October 3 changelog is written against our
October 1 kit pin `020df706`. **The B12X autotune selection rule that drives our Qwen
C1 regression is unchanged**, and [b12x#463](https://github.com/local-inference-lab/b12x/issues/463)
is open with no response.

## Publication and source identities

Latest observed publication: [October 3 beta](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-d4499365dba23083d2003d0e77ce487dccdce5d2e15ab00cfe72119d60803e3d), published 2026-10-03 17:45:29 UTC.

Image: `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261003-d4499365dba23083`.

Digest: `ghcr.io/local-inference-lab/vllm@sha256:5d95abf866dbbb2ac7c610f20d4dde08cafbf115cfee2f0ae891d949636ba4fb`.

| Component | October 2 beta | October 3 beta |
| --- | --- | --- |
| b12x | `a77b3f85e5e2a81315ff4a90088912f187708005` | `78ee52c302abcef019d7ce17634c80b76f0975c9` |
| flashinfer | `dbd6238c6655b98195fdf77f04bba6facf5a38a4` | unchanged |
| instanttensor | `95d4729b6d6a991bb8de61877147a9d9d9100b23` | unchanged |
| lmcache | `820af25ff630f4c00f7faefbcc31bc5ccd7bab71` | unchanged |
| nccl | `93fe05d9f9b6963ef841166a69cd0b30e4efe97b` | unchanged |
| vllm | `93dabce32fd4d5355662608296e64d720711cdcc` | `1286ae9c9a3fd9376b82da39f7bc87f6a99037d9` |
| recipe | `6c0e9843bb962f483409b0296b225cca03fe8567` | `353efc679f631206e0b001e67047dea80ee6d76e` |

The fetched vLLM and B12X beta branch tips match these published pins. The recipe
diff `6c0e9843..353efc67` does not change any CUTLASS DSL version.

## What landed since the October 2 check

B12X `a77b3f85..78ee52c3` (29 files, +171/-7054):

- [#468](https://github.com/local-inference-lab/b12x/pull/468), [#467](https://github.com/local-inference-lab/b12x/pull/467): the collective barrier deadline (`B12X_COLLECTIVE_BARRIER_TIMEOUT`, default 120 s) is validated only for sessions that use a barrier and is captured per session.
- [#464](https://github.com/local-inference-lab/b12x/pull/464): `B12X_DYNAMIC_DETERMINISTIC_OUTPUT` is honored before kernel selection.
- [#461](https://github.com/local-inference-lab/b12x/pull/461), [#456](https://github.com/local-inference-lab/b12x/pull/456): CSF release model fixes and the CSF beta port.
- Most of the deletion is load-time CSF scale compression (`_lib/quant/csf_encode.py`) and its validation evidence, removed together with recipe [#121](https://github.com/local-inference-lab/blackwell-llm-docker/pull/121). Stored CSF checkpoints remain supported.

vLLM `93dabce3..1286ae9c` (31 files):

- [#963](https://github.com/local-inference-lab/vllm/pull/963): CSF beta port.
- [#965](https://github.com/local-inference-lab/vllm/pull/965): opt-in GLM Spark online quantization (531 MXFP8 projections, NVFP4 W4A16 MTP experts) via an exact target file. Existing online quantization defaults are unchanged.

Recipe `6c0e9843..353efc67`: a QAD GLM TP2 preset was added in [#120](https://github.com/local-inference-lab/blackwell-llm-docker/pull/120) and then withdrawn "until it ships a stored checkpoint" (`e2e81c4`), plus #121 above.

## Applicability to our fleet

- **Qwen C1 regression.** Since our candidate's B12X `b557d878`, the only change under `preparation/session.py`, `preparation/_measurement.py`, `preparation/_cache.py`, `gemm/`, `_lib/dense_gemm.py`, `norm/` and `sequence/` is the collective barrier in `session.py`. Winner selection still keeps the strict minimum measured latency with no margin. vLLM `b12x_blockscaled.py` and the Qwen3.8 target model are unchanged since `4a379ed4`; only the DFlash and DSpark draft models changed. Nothing here is a fix for the regression; the new image would still need its own matched Spark comparison.
- **GLM.** #965 is opt-in and the TP2 preset was withdrawn, so neither affects our TP4 stored-checkpoint profile.
- **DS4 Vision.** No DS4 Vision specific changes found. The DS4.1 CSF activation-policy change applies to CSF checkpoints only.
- The changelog repeats that model-serving performance remains unqualified; the publication's `qualified` scope is native GPU smoke and LMCache contract tests.

## Fetch audit

| Checkout | Remotes fetched | Principal ref moves |
| --- | --- | --- |
| `rtx6kpro` | `origin`, `upstream` | `upstream/master` `a968efc0` → `7288ac04` |
| `vllm` | `lil`, `origin`, `upstream` | `lil/integration/karmic-kraken-beta` `58d05bc7` → `1286ae9c`; `upstream/main` `6e517b15` → `84bcbc62` |
| `b12x` | `origin`, `upstream`, `voip` | `upstream/integration/karmic-kraken-beta` `a77b3f85` → `78ee52c3` |
| `blackwell-llm-docker` | `origin`, `upstream` | `upstream/main` `6c0e9843` → `353efc67` |
| `flashinfer` | `origin`, `upstream`, `voip` | `upstream/main` `9178cf06` → `c3c33367` |
| `LMCache` | `upstream` | `upstream/dev` `4abc421a` → `1ca53a8a` |
| `cutlass` | `upstream` | none |
| `sglang` | `upstream` | `upstream/main` `cbe070b6` → `b016ca40` |
| `spark-vllm-docker` | `origin`, `upstream` | `upstream/main` `a8f4d693` → `cb51860d` |
| `dgx-spark-infra` | `origin` | none |
| `llm-inference-bench` | `origin`, `upstream` | none |

82 remote refs moved or appeared in total, most of them sglang feature branches. The
vLLM beta tip was already at `58d05bc7` from an intermediate fetch after the October 2
report, which recorded `93dabce3`.
