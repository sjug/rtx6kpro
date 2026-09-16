# Podman image inventory — DGX Spark fleet (2026-09-15)

Collected via `ssh <node> podman images -a` + `podman system df`. Raw TSVs alongside: `<node>.images.tsv` (per-image), `<node>.systemdf.txt` (storage/df).

## Fleet overview

| node | images | dangling | actual image storage | disk (/) |
|---|---:|---:|---|---|
| sparky | 22 | 0 | 164.2GB (in use: 3) | 2.3T of 3.7T (64%) |
| buddy | 22 | 0 | 164.2GB (in use: 3) | 2.0T of 3.7T (57%) |
| lucky | 22 | 0 | 164.2GB (in use: 3) | 1.4T of 3.7T (40%) |
| rocky | 22 | 0 | 164.2GB (in use: 3) | 1.4T of 3.7T (39%) |
| rusty | 683 | 643 | 412.3GB (in use: 63) | 1.4T of 3.7T (40%) |
| toby | 24 | 2 | 304GB (in use: 0) | 532G of 3.7T (15%) |
| dusty | 90 | 86 | 36.47GB (in use: 4) | 963G of 3.7T (28%) |
| kirby | 9 | 0 | 96.34GB (in use: 4) | 911G of 3.7T (26%) |

Notes: `actual image storage` = `podman system df` SIZE (layer-deduplicated;
naive per-image sums are far higher, e.g. sparky sums to 655GB but stores
164.2GB). `dangling` = `<none>:<none>` entries.

## Presence matrix

Distinct `repo:tag` across the fleet: 55. Sorted by node-presence, then name.

| # | image | nodes present |
|---|---|---|

| 1 | `vllm:jj-r32-spark-sm121` | buddy, dusty, kirby, lucky, rocky, rusty, sparky, toby |
| 2 | `vllm:gilded-gnosis-v20-r33-spark-sm121-vllm28e8eaf-b12x06db0f4-fi1ac6942-cu132-20260808` | buddy, kirby, lucky, rocky, rusty, sparky, toby |
| 3 | `vllm:gilded-gnosis-v20-r34-spark-sm121-vllm17b78ef-b12xcd3ce19-fi1ac6942-cu132-20260818` | buddy, kirby, lucky, rocky, rusty, sparky, toby |
| 4 | `vllm:gilded-gnosis-v20-r28-spark-sm121-vllm47d1950-si200c1db-fi7ad08da-cu132-20260804` | buddy, lucky, rocky, rusty, sparky, toby |
| 5 | `vllm:gilded-gnosis-v20-r33-spark-sm121-vllm44700e3-b12x06db0f4-fi1ac6942-cu132-20260808` | buddy, lucky, rocky, rusty, sparky, toby |
| 6 | `vllm:gilded-gnosis-v20-r34-spark-sm121-vllm9eb28db-b12xcd3ce19-fi1ac6942-cu132-20260817` | buddy, lucky, rocky, rusty, sparky, toby |
| 7 | `vllm:gilded-gnosis-v20p2-spark-sm121-vllm6dbe127-si9c34ff8-fi7ad08da-cu132-20260730` | buddy, lucky, rocky, rusty, sparky, toby |
| 8 | `vllm:gilded-gnosis-v20p2-spark-sm121-vllm9e70264-si9c34ff8-fi7ad08da-cu132-20260730` | buddy, lucky, rocky, rusty, sparky, toby |
| 9 | `vllm:gilded-gnosis-v20p3p1-spark-sm121-vllm41ae549-si2b9bf2a-fi7ad08da-cu132-20260803` | buddy, lucky, rocky, rusty, sparky, toby |
| 10 | `vllm:jj-r28-spark-sm121` | buddy, dusty, kirby, lucky, rocky, sparky |
| 11 | `vllm:jj-r29-spark-sm121` | buddy, dusty, kirby, lucky, rocky, sparky |
| 12 | `vllm:glm53-jj-r26-spark-sm121-vllm59c9787-b12x00248b0-lmcachefe5442f-cu133-torch213-20260905-r1` | buddy, kirby, lucky, rocky, sparky |
| 13 | `vllm:glm53-jj-r27-spark-sm121-vllmf3c3fef-b12x95fdcb1-lmcachefe5442f-cu133-torch213-20260906-r1` | buddy, kirby, lucky, rocky, sparky |
| 14 | `vllm:glm53-jj-r12-spark-sm121-vllm12a8897-b12xaace94c-lmcache9086bfb-cu133-torch213-20260901-r2-dev` | buddy, lucky, rocky, sparky |
| 15 | `vllm:glm53-jj-r15-spark-sm121-vllm7e389bd-b12x2a69d69-lmcache9086bfb-cu133-torch213-20260902-r1-dev` | buddy, lucky, rocky, sparky |
| 16 | `vllm:glm53-jj-r17-stock-spark-sm121-vllmbd8e2ab-b12x5ee3a5a-lmcachec867e17-cu133-torch213-20260903-r1-dev` | buddy, lucky, rocky, sparky |
| 17 | `vllm:glm53-jj-r17p1-rocenante-spark-sm121-vllm8c5b282-b12x3c9d909-lmcachec867e17-cu133-torch213-20260903-r1-dev` | buddy, lucky, rocky, sparky |
| 18 | `vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1` | buddy, lucky, rocky, sparky |
| 19 | `vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1-scratch` | buddy, lucky, rocky, sparky |
| 20 | `vllm:infernal-invocation-r17-spark-sm121-vllme879345-b12xc0a44a1-fi1ac6942-cu133-torch213-20260818` | buddy, lucky, rocky, sparky |
| 21 | `vllm:infernal-invocation-r17-spark-sm121-vllme879345-b12xc0a44a1-fi1ac6942-cu133-torch213-20260818b` | buddy, lucky, rocky, sparky |
| 22 | `vllm:infernal-invocation-r18-spark-sm121-vllmf560085-b12x75787c7-fi1ac6942-cu133-torch213-20260818` | buddy, lucky, rocky, sparky |
| 23 | `<none>` | dusty, rusty, toby |
| 24 | `vllm:jj-r38-spark-sm121` | dusty, kirby, rusty |
| 25 | `vllm:fathomless-firmament-v16p1-spark-sm121-vllm8f86f42-b12x1bcc652-fi801d57a-cu132-20260716` | rusty, toby |
| 26 | `vllm:fathomless-firmament-v16p2-spark-sm121-vllm8f86f42-b12x1bcc652-fi3245a18-cu132-20260716` | rusty, toby |
| 27 | `vllm:fathomless-firmament-v16p3-spark-sm121-vllm6a5a106-b12xfe06f49-fic0700ff-cu132-20260716` | rusty, toby |
| 28 | `vllm:gilded-gnosis-v18-spark-sm121-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718` | rusty, toby |
| 29 | `vllm:gilded-gnosis-v18p1-spark-sm121-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718` | rusty, toby |
| 30 | `vllm:gilded-gnosis-v18p2-spark-sm121-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718` | rusty, toby |
| 31 | `vllm:gilded-gnosis-v18p3-spark-sm121-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718` | rusty, toby |
| 32 | `vllm:gilded-gnosis-v18p4-grid48-spark-sm121-vllmdf7a0b7-b12x6b10833-fi801d57a-cu132-20260718` | rusty, toby |
| 33 | `vllm:gilded-gnosis-v18p5-quantfix-spark-sm121-vllm4966149-b12x6b10833-fi7ad08da-cu132-20260718` | rusty, toby |
| 34 | `vllm:gilded-gnosis-v19-dspark-spark-sm121-vllm371085e-b12xc7dc733-fice40d25-cu132-20260719` | rusty, toby |
| 35 | `vllm:gilded-gnosis-v20p0-ds4-spark-sm121-vllm2167295-si6a92bcc-fi801d57a-cu132-20260722` | rusty, toby |
| 36 | `vllm:gilded-gnosis-v20p1-ds4-spark-sm121-vllm2167295-si6a92bcc-fi7ad08da-cu132-20260722` | rusty, toby |
| 37 | `vllm:gilded-gnosis-v20p3-spark-sm121-vllm92b27a4-si2b9bf2a-fi7ad08da-cu132-20260803` | rusty, toby |
| 38 | `cuda:13.0.2-cudnn-devel-ubuntu24.04` | rusty |
| 39 | `cuda:13.2.1-cudnn-devel-ubuntu24.04` | rusty |
| 40 | `docker.io/vllm/vllm-openai:<none>` | kirby |
| 41 | `ghcr.io/astral-sh/uv:0.12.13` | rusty |
| 42 | `localhost/sol-h3-spark/qwen:4a14585f9da54ac2` | rusty |
| 43 | `localhost/sol-h3-spark/stage1:1992711d52ebccab` | rusty |
| 44 | `localhost/sol-h3-spark/stage2:ce7b6f6a735ad6e2` | rusty |
| 45 | `localhost/sol-h3-test/qwen-audio-291:20260911` | rusty |
| 46 | `localhost/sol-h3-test/qwen-audio:20260911` | rusty |
| 47 | `localhost/sol-h3-test/qwen:20260911` | rusty |
| 48 | `localhost/sol-h3-test/stage1:20260911` | rusty |
| 49 | `localhost/sol-h3-test/stage2-audio-candidate:20260911` | rusty |
| 50 | `localhost/sol-h3-test/stage2:20260911` | rusty |
| 51 | `vm:build-candidates:jj-r38-runtime-20260915T163156Z-64815` | rusty |
| 52 | `vm:build-components:jj-r38-flashinfer-20260915T160431Z-59162` | rusty |
| 53 | `vllm:spark-sm121-cu132-build-base-20260708` | rusty |
| 54 | `vllm:spark-sm121-cu132-system-base-20260708` | rusty |
| 55 | `nvcr.io/nvidia/pytorch:<none>` | rusty |

## Distribution
| image present on N nodes | distinct images |
|---:|---:|
| 8 (all) | 1 |
| 7 | 2 |
| 6 | 8 |
| 5 | 2 |
| 4 | 9 |
| 3 | 2 |
| 2 | 13 |
| 1 | 18 |
