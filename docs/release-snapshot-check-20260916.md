# Post-R38 release snapshot check

Checked 2026-09-16 at 18:29 UTC. Public GitHub release assets, Docker Hub
tag metadata and GHCR image metadata were read directly. No repository
fetch, image-layer download, build, container start or node access occurred.

## Answer

There are newer published, source-addressed runtime snapshots after R38.
There is not a numbered community R39 in the checked publishing surfaces.
The new main and beta snapshots are Linux amd64 images, not ready-made
ARM64/SM121 Spark images. They carry a CUDA 13.4/PyTorch 2.14 foundation,
so they are not a source-only refresh of our R38 Spark runtime.

The latest numbered image remains
`localinferencelab/vllm:jovian-judgement-community-20260914-r38` at
`sha256:f41ca8bb10bb3a125a50340d70d39ad4b7f5605f3fcb661bc992ed0bc4701a00`.
The Docker Hub API returned all 15 tags with no next page; R38 was newest,
and each listed tag had only an amd64 image. Source:
[Docker Hub tag API](https://hub.docker.com/v2/repositories/localinferencelab/vllm/tags/?page_size=100&ordering=last_updated).

## New source-addressed snapshots

| Channel | Published UTC | Image tag | vLLM | B12X |
| --- | --- | --- | --- | --- |
| main | 2026-09-16 17:12:48 | `ghcr.io/local-inference-lab/vllm:jovian-judgement-20260916-87a0ed331ac886df` | `59fbf050084aefe2a2dd3b19b271a7eb3527c6ae` | `0add93fbea6b75a3642cb29d7449a81bfec76b8b` |
| beta | 2026-09-16 15:48:07 | `ghcr.io/local-inference-lab/vllm:jovian-judgement-beta-20260916-b58b9d8864783571` | `f1f60826bd091c9c242913272ba55c94ac0a5b60` | `f832e568c0780522280cc27cfebe0980b285d49b` |

The preceding main assembly `5304cd85cf257f78` was published at 16:02:52
with the same vLLM pin and B12X `5cc27234eaf4113dc41cb308753e152b1e71ce8e`.
The later main assembly superseded its moving alias. Sources:
[main release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/jovian-judgement-87a0ed331ac886dfdf73b59cbefa0e23880d61e90babc80adc7fede290fc6bc1),
[beta release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/jovian-judgement-beta-b58b9d8864783571276b01423f3bc93e0f699dfd5c06ca76892e94e95df81f9b),
[preceding main release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/jovian-judgement-5304cd85cf257f78b1e738242f7f877f3d103ffea70e7c4d356064d8a3f3aa96).

Verified with read-only `skopeo inspect --no-tags` against the immutable
main digest and both moving aliases:

- Main digest:
  `sha256:ae4dd5e679ff24a58386b33cf48611cd348c3e6f967814b1c9617fe0433ffea6`.
  Alias `ghcr.io/local-inference-lab/vllm:jovian-judgement` resolved to it.
- Beta digest:
  `sha256:6cd5c6a638e25f57ff85ea1590ed92694869d5139ccab2c7ebfb4d83a2b43f79`.
  Alias `ghcr.io/local-inference-lab/vllm:jovian-judgement-beta` resolved to it.
- Both reported `Architecture=amd64`; main reported `Os=linux`.
- Main image creation was 17:02:27 UTC, beta 15:36:10 UTC.

These are published prereleases with bounded qualification. The release
explicitly limits its evidence to native GPU smoke and LMCache
checkpoint/filesystem contracts; model-serving performance and full GLM
cache end-to-end qualification remain open. Main's receipt records 90
cache-contract tests passing, none skipped. A receipt field saying
`status=qualified` must be read with that scope, not as fleet or model
qualification. The assembly and runtime manifests still say
`research-only`. Sources:
[main container receipt](https://github.com/local-inference-lab/blackwell-llm-docker/releases/download/jovian-judgement-87a0ed331ac886dfdf73b59cbefa0e23880d61e90babc80adc7fede290fc6bc1/container-release.json),
[main assembly](https://github.com/local-inference-lab/blackwell-llm-docker/releases/download/jovian-judgement-87a0ed331ac886dfdf73b59cbefa0e23880d61e90babc80adc7fede290fc6bc1/community-assembly-main.json).

## Main snapshot foundation

The recipe commit is `1712c9d8a740802031a32d797f2432fb19946993`, which
merged shared main/beta runtime-channel publication on September 16.
Main uses the NGC PyTorch base
`nvcr.io/nvidia/pytorch@sha256:33ef5fc15e8937602d64022209cdb2777b32dadf742f41332023d946041b3c14`.
The runtime manifest declares CUDA 13.4.1, Python 3.12 and PyTorch
`2.14.0a0+4fdf77b940.nv26.8.63802676`.

Other source pins shared by the listed main/beta assemblies:

- FlashInfer `2206a14e46387a56c093860a46bbbdd00596b75b`, SM120 wheels.
- InstantTensor `95d4729b6d6a991bb8de61877147a9d9d9100b23`.
- LMCache `b9493dafe09c2d6e3988110069e1785b5f016e05`, SM120 wheels.
- NCCL `93fe05d9f9b6963ef841166a69cd0b30e4efe97b`, the
  `canonical/cu134-nccl2312-amd-turin` source branch.

These source identities are reproducible inputs for a future Spark port,
not permission to reuse x86 native objects on ARM. Sources:
[runtime manifest](https://github.com/local-inference-lab/blackwell-llm-docker/releases/download/jovian-judgement-87a0ed331ac886dfdf73b59cbefa0e23880d61e90babc80adc7fede290fc6bc1/manifest.json),
[recipe commit](https://github.com/local-inference-lab/blackwell-llm-docker/commit/1712c9d8a740802031a32d797f2432fb19946993).

## Composition caution and DS4.1 runner decision

B12X PR #379 remained open at this check, head
`59f0cad9e4aa9eab57b8773ed83c064be608d95a`. The merged vLLM #771 explicitly
requires it for `plan_pooled_selection`. Inspection of the published
B12X main pin's sparse-MLA API did not find that method. Thus the new
main snapshot's native smoke is not sufficient evidence to carry the
whole composition into GLM production; the cross-repository dependency
must be resolved or verified in the actual composed runtime first.
Sources: [B12X #379](https://github.com/local-inference-lab/b12x/pull/379),
[vLLM #771](https://github.com/local-inference-lab/vllm/pull/771),
[pinned B12X API](https://github.com/local-inference-lab/b12x/blob/0add93fbea6b75a3642cb29d7449a81bfec76b8b/b12x/attention/sparse_mla/api.py).

For the requested DS4.1 runner, keep the existing frozen R38 Spark image
as the initial candidate. It already contains the published DS4.1 R38
model-source pins, and native disk Engram is present. A four-node runner
and actual-image disk-Engram gates are independent of undertaking a
CUDA/PyTorch foundation migration. The new snapshots are a separate
porting lead, not a prerequisite for authoring the runner. Prior source
evidence and boundaries are in the
[DS4.1 Spark review](deepseek-v4.1-spark-tp4-source-review-20260916.md).

No claim here qualifies DS4.1 disk serving on our fleet. That still needs
the image/component checks and the approved four-node model window.
