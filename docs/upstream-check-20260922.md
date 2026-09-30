# Upstream check, 2026-09-22 (evening)

## Scope

Fetched every configured remote with `--no-prune` in the existing checkouts:
rtx6kpro (origin, upstream), vllm (lil, origin, upstream), b12x (lil, origin,
upstream, voip), blackwell-llm-docker (origin, fork), flashinfer (origin,
upstream, voip), LMCache, cutlass, sglang, spark-vllm-docker, dgx-spark-infra.
All fetches succeeded. No checkout, rebase, merge, reset, build, node action,
commit or push. The benchmark repository was not touched.

Comparison anchor: [September 21 check](upstream-check-20260921.md) and its
[publication pins](upstream-publication-20260921.md): JJ main vLLM
`8e1f1e587f` / B12X `0f3a8cbf`, JJ beta branch tips `074707f39a` / `39e0fa7506`,
Kraken main vLLM `af9e4dca10`, recipe `23d674e8`.

Purpose of this check: re-pin the
[DS4 Vision KV recommendation](ds4-vision-jj-main-kv-recommendation-20260922.md)
after the sync. Findings for that plan are in its "Recheck" section.

## Refs after fetch

| Repository | Ref | Before | Now |
| --- | --- | --- | --- |
| rtx6kpro | `upstream/master` | 57ded9e5 | 9097c158 (local `master` already equal; `origin/master` still 57ded9e5) |
| vLLM lil | `dev/jovian-judgement` | 8e1f1e587f | 8e1f1e587f, unchanged |
| vLLM lil | `integration/beta` | 074707f39a | 074707f39a, unchanged |
| vLLM lil | `dev/karmic-kraken` | af9e4dca10 | 6afb999825 (41 commits) |
| vLLM lil | `integration/karmic-kraken-beta` | | 622912b9b2 |
| vLLM upstream | `main` | 6b858751f6 | d110c2f19c; `vllm/utils/mem_utils.py` unchanged |
| B12X lil | `master` | 0f3a8cbf | 4f3028b1 (18 commits) |
| B12X lil | `integration/beta` | 39e0fa7506 | unchanged |
| B12X lil | `integration/karmic-kraken-beta` | | 3113aa0b |
| blackwell-llm-docker | `origin/main` | 23d674e8 | 0ab249f |

rtxk6pro `master` moved by ten documentation commits (Kimi TP9 QSRT, Karmic PR
composition, Qwen TP2 registry, daily summary). They are records, not fleet
state.

## Publication changes

- Recipe `ef8008c` "publish only Karmic canonical and beta containers":
  `tools/jovian_wheel_runtime/community-channel.json` at `origin/main` now
  declares only `karmic-kraken` (vLLM `dev/karmic-kraken`, B12X `master`) and
  `karmic-kraken-beta` (both `integration/karmic-kraken-beta`). The Jovian
  channels and the Qwen NGC runtime workflow were removed. The JJ-main line our
  DS4 Vision candidate was built from is no longer published upstream, and
  `dev/jovian-judgement` has not moved since 2026-09-16.
- Releases on 2026-09-22 (UTC): `karmic-kraken-e58ebf3c…` 17:39 and five
  `karmic-kraken-beta-…` builds (02:35 to 15:55). Digests not inspected here.
- Recipe `89ce009` sets `INSTANTTENSOR_BACKEND: URING,AIO` in the upstream
  `ds4-flash.yaml` profile (direct I/O for the DS4 Flash text checkpoint). Our
  R38 and JJ launchers use `instanttensor_backend=BUFFERED`. Not inherited by
  fetching; relevant to unified-memory accounting (see recheck).
- Recipe `338546b` adds a runtime gate verifying the installed B12X distributed
  tuning protocol (`verify_qwen38_runtime.py`), matching B12X `5a53424d`.

## Source changes relevant to the KV plan

Karmic vLLM `50554de256` "Port caller-owned serving workspaces and DS4.1 API
contracts to Karmic Kraken (#798)":

- `vllm/v1/worker/workspace.py` gains `reserve_by_lane()` and `available_bytes()`;
  `reserve_all()` keeps equal-size semantics but goes through `_reserve_slots`.
- `vllm/v1/worker/b12x_startup.py:306-330`: after each preparation batch the
  coordinator sizes each request in its declared lanes
  (`use_workspace_lane`) and calls `reserve_by_lane()` when lane assignments
  exist, `reserve_all()` otherwise. `b12x_prepare.py:487-489, 581, 679` supply
  `request_workspace_lanes` from unit `workspace_lanes`.
- `vllm/models/deepseek_v4/nvidia/b12x.py`: `_reserve_profile_workspace` is gone;
  the MLA preparation units carry the C128 profile widths
  (`_c128a_profile_widths`, line 972) and decode widths themselves.
- Stage ordering is unchanged: `abstract.py:401` still runs the weights stage
  inside `determine_available_memory`, and `gpu_worker.py:617-655` keeps the
  profile-time `_prepare_b12x_profile_state` and the late-persistent snapshot.
- `race_budget` is still not passed by vLLM; B12X `session.py:328-335` still
  defaults it to `cudaMemGetInfo free / 2`.

B12X `master` since `0f3a8cbf` (18 commits): only `5a53424d` (#414, tuning
result contract in `preparation/types.py`) touches the pinned preparation files;
`f818b3ab` (#407) keys the selection cache on compute capability and SM count
instead of the device name; `1dc77276` changes Spark decode routing, indexer
scheduling and MoE/MHC tuning predicates; `6debca95`, `79a29c1b`, `f20ab3ba`
change MoE/MXFP8 tuning and prepared capacity. None changes `program_cache.py`,
`compile_pool.py`, the compressed-MLA split policy or the scratch layout.

Other Karmic vLLM commits of note: `f040840521` bounds NVFP4 loader scale
memory; `608b70d1ba` reduces GLM speculative state memory; `458766e58f` and
`419112746e` are Spark DSV4.1/GLM and adaptive-verification fixes; `9e5d1793fa`
updates Spark launchers for switched RoCE and DeepSeek TP3.

## Not done

No image digests, runtime manifests or changelog fragments were inspected for
the 2026-09-22 Karmic releases. No Karmic DS4 Vision SM121 candidate exists
locally; no build or node action was taken. `origin/master` of rtx6kpro is
behind `upstream/master`; not pushed.
