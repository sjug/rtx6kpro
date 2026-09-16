# Podman image inventory — DGX Spark fleet (2026-09-15)

Collected via `ssh <node> podman images -a` + `podman system df`. Raw TSVs alongside: `<node>.images.tsv` (per-image), `<node>.systemdf.txt` (storage/df).

## Fleet overview

| node | images | dangling | actual image storage | disk (/) |
|---|---:|---:|---|---|
| sparky | 3 | 0 | 35.87GB (in use: 3) | 2.2T of 3.7T (62%) |
| buddy | 3 | 0 | 35.87GB (in use: 3) | 1.9T of 3.7T (55%) |
| lucky | 3 | 0 | 35.87GB (in use: 3) | 1.3T of 3.7T (38%) |
| rocky | 3 | 0 | 35.87GB (in use: 3) | 1.3T of 3.7T (37%) |
| rusty | 4 | 0 | 59.65GB (in use: 1) | 935G of 3.7T (27%) |
| toby | 3 | 0 | 61.62GB (in use: 1) | 337G of 3.7T (10%) |
| dusty | 3 | 0 | 35.87GB (in use: 2) | 963G of 3.7T (27%) |
| kirby | 3 | 0 | 36.43GB (in use: 3) | 855G of 3.7T (24%) |

Notes: `actual image storage` = `podman system df` SIZE (layer-deduplicated;
naive per-image sums are far higher, e.g. sparky sums to 655GB but stores
164.2GB). `dangling` = `<none>:<none>` entries.

## Presence matrix

Distinct `repo:tag` across the fleet: 7. Sorted by node-presence, then name.

| # | image | nodes present |
|---|---|---|

| 1 | `vllm:jj-r38-spark-sm121` | buddy, dusty, kirby, lucky, rocky, rusty, sparky, toby |
| 2 | `vllm:jj-r32-spark-sm121` | buddy, dusty, kirby, lucky, rocky, sparky, toby |
| 3 | `vllm:jj-r29-spark-sm121` | buddy, dusty, kirby, lucky, rocky, sparky |
| 4 | `localhost/sol-h3-spark/stage2:ce7b6f6a735ad6e2` | rusty |
| 5 | `vm:build-candidates:jj-r38-runtime-20260915T163156Z-64815` | rusty |
| 6 | `vm:build-components:jj-r38-flashinfer-20260915T160431Z-59162` | rusty |
| 7 | `vllm:gilded-gnosis-v20-r34-spark-sm121-vllm17b78ef-b12xcd3ce19-fi1ac6942-cu132-20260818` | toby |

## Distribution
| image present on N nodes | distinct images |
|---:|---:|
| 8 (all) | 1 |
| 7 | 1 |
| 6 | 1 |
| 1 | 4 |
