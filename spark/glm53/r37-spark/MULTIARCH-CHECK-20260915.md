# Nightly Spark image check, 2026-09-15

## Conclusion

A real public Spark image exists, but it is **not a prebuilt R38 substitute**.
It targets ARM64/SM121 and has publisher-reported DeepSeek Vision TP2 acceptance.
Its vLLM, B12X, FlashInfer and LMCache pins differ from R38. Treat it as a
separate qualification candidate, not permission to replace the restored R32
GLM/Qwen services or skip our R38 build gates.

## Project and published artifact

The [daily-summary announcement](https://github.com/voipmonitor/rtx6kpro/blob/9fdad773ab8ad8b0654d3fa7fe10edfd516aee24/daily-summaries/2026-09/2026-09-15.md)
names `vllm-multiarch` and links to
[Discord](https://discord.com/channels/1466898002793857221/1517199740444475423/1549039041264222228).
The matching public project is
[randomvariable/vllm-multiarch-oci](https://github.com/randomvariable/vllm-multiarch-oci).
The original Discord message could not be read, so its exact attribution to
this repository remains an inference. The repository itself and its registry
artifacts were inspected directly.

Live registry inspection resolved `ghcr.io/randomvariable/vllm-b12x-multi:latest`
to:

```text
ghcr.io/randomvariable/vllm-b12x-multi@sha256:72431d1f54507c621d7b8c4140c1620867b8a171bdacfb7fa76f4c075db90561
vllmb12x-dev-jovian-judgement-b40673cd006b-5f3804361c05-20260914-n14
```

This agrees with the publisher's [release metadata](https://randomvariable.github.io/vllm-multiarch-oci/latest-image.json),
resolved there at `2026-09-15T10:04:07Z`. The
[registry manifest](https://ghcr.io/v2/randomvariable/vllm-b12x-multi/manifests/sha256:72431d1f54507c621d7b8c4140c1620867b8a171bdacfb7fa76f4c075db90561)
is a single-image Docker schema-2 manifest, not a multi-platform index.
Its [configuration](https://ghcr.io/v2/randomvariable/vllm-b12x-multi/blobs/sha256:1366a06f927158ace5392e2bbc3df5c07a3e15873df0897e7b7fe7a5538fc025)
declares `linux/arm64`, CUDA `13.3.1`, Torch architecture `12.1a` and
FlashInfer architecture `12.1f`. Registry links may require the standard
anonymous bearer-token exchange; `skopeo inspect` performed that exchange.

Despite the project name, the current
[README](https://github.com/randomvariable/vllm-multiarch-oci/blob/36b70acaa5e108657c121c744617dee12538d96d/README.md)
explicitly says x86-64 is only a platform declaration, not a verified second
image. Its warning that a full *local* Spark build is unverified is distinct
from the published CI artifact and the author's deployment acceptance.

## Source and build provenance

The publication tag identifies builder commit
[`5f3804361c05956f8c441b376684406b41d7562e`](https://github.com/randomvariable/vllm-multiarch-oci/commit/5f3804361c05956f8c441b376684406b41d7562e).
We fetched only its 40 KiB
[provenance layer](https://ghcr.io/v2/randomvariable/vllm-b12x-multi/blobs/sha256:2cb63d4d5a5f952fcb1a2db60a050342e3a4a4d326d278108e60ec4fa6e1bbd2),
verified its SHA-256 and read `/opt/vllmb12x/sources.lock.json` plus the
vLLM source identity. They agree with the builder's
[pinned profile](https://github.com/randomvariable/vllm-multiarch-oci/blob/5f3804361c05956f8c441b376684406b41d7562e/profiles/vllmb12x/profile.json).

| Component | Published Spark nightly | Public R38 |
| --- | --- | --- |
| vLLM | `b40673cd006bf3496fdd70361dad2ad29eff54e7` | `66c293578412417476f842c1da5805d3a3d959a8` |
| B12X | `9043b448622764a598969518d413b3fd8b3c0c07` | `ce419b52681b7922bb0972d4b58b590a3fd005b2` |
| FlashInfer | `1ac6942776b383c6b03c7a5805a22e72a3e3349f` | `803c4664f4771ddc418f20a57f752469a237a825` |
| LMCache | `9cebd405d0caf4bebe01d694b5a8bf4e3e354314` | `29bc5a2efde737c436b04499eb62cd1776cebeec` |

R38 values come from its
[source lock](https://github.com/voipmonitor/rtx6kpro/blob/9fdad773ab8ad8b0654d3fa7fe10edfd516aee24/models/deepseek-v4.1-flash/r38/source.lock).
GitHub's [immutable revision comparison](https://github.com/local-inference-lab/vllm/compare/b40673cd006bf3496fdd70361dad2ad29eff54e7...66c293578412417476f842c1da5805d3a3d959a8)
reports R38 36 commits ahead, zero behind. This nightly therefore lacks the
full R38 source delta, including its adaptive-verification graph pricing.

The [build targets](https://github.com/randomvariable/vllm-multiarch-oci/blob/5f3804361c05956f8c441b376684406b41d7562e/components/BUILD.bazel)
compile Torch `cf30153c4c131c8164ee7798e5022d810682e2cb` as `2.13.0`,
vLLM for `121a`, and FlashInfer Python/JIT-cache wheels for `12.1f`.
The embedded vLLM identity records two additional patches: SHM broadcast spin
grace (`e1f50e22...`) and FlashAttention CuTe namespace (`f7af5229...`).
This establishes declared source/build intent, not independent inspection of
compiled GPU code or equivalence to R38's complete dependency patch set.

The public [nightly pipeline](https://github.com/randomvariable/vllm-multiarch-oci/blob/36b70acaa5e108657c121c744617dee12538d96d/.tekton/vllmb12x-nightly.yaml)
builds on ARM64, runs an
[image contract](https://github.com/randomvariable/vllm-multiarch-oci/blob/36b70acaa5e108657c121c744617dee12538d96d/tests/image/vllmb12x.yaml)
and publishes through a serialized tag allocator. The contract checks imports,
libraries, provenance and native helper compilation, not GLM/Qwen model serving.
Private CI run receipts, signing/attestation verification and binary inspection
were not obtained. Embedded locks are provenance evidence, not a cryptographic
attestation that the publisher executed every declared build/test step.

## Runtime evidence and remaining gates

The publisher's
[DeepSeek Vision recipe](https://github.com/randomvariable/vllm-multiarch-oci/blob/36b70acaa5e108657c121c744617dee12538d96d/recipes/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp.yaml)
records functional acceptance on 2026-09-14 in a two-node GB10 kubeadm/Cilium
cluster, TP2 over RoCE, DSpark K3 and one-million-token configured maximum
context. It explicitly does not claim an independent benchmark or Docker
qualification. Its accepted image is the older digest
`sha256:42d5d72bd79a371c3b5a756fe2a1814f3c9393dd9a73d65bcefa01eb8f28d726`.
That acceptance does not prove a one-million-token request was exercised.

Comparing both registry manifests shows all filesystem layer digests match
except the 40 KiB provenance layer; image configuration also changed. Thus n14
is not evidence of a newer native runtime than the accepted image. The
[release page](https://randomvariable.github.io/vllm-multiarch-oci/image-releases/)
correctly separates latest publication from recipe acceptance.

To consider replacing our build path, first choose whether the objective is
exact R38 or this older independent source combination. The latter still needs
our source/dependency compatibility audit, GLM TP4 and Qwen TP2 launchers,
strict-tool/speculation regressions, cold load/restart, NCCL/RoCE, long-prefill,
correctness and matched performance gates. No such local qualification occurred.

Only source files, registry metadata and the single 40 KiB provenance layer were
read. No runtime image layers were pulled, no repositories updated, and no
nodes, builds or serving deployments were touched.
