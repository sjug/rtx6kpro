# October 1 Karmic beta with CUTLASS DSL 4.7.1 for DGX Spark

Status: **prepared and locally checked; not built, GPU-qualified or deployed**.
Target tag: `localhost/voipmonitor/vllm:karmic-beta-20261001-spark-sm121`.
No Spark host was contacted during preparation. An authorized idle build window
on dusty/kirby is required before running the real build.

## Pinned composition

This ports the [October 1 beta publication](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-020df706a373de8da51b361c100c0a2d0a78d56a3a0b326fdcc06620e9e5ef85)
to ARM64 / SM121 using the same content-addressed foundation as our September 29 kit.
The publication's x86_64 / SM120 wheels are source references, not Spark binaries.

| Component | Pin |
| --- | --- |
| Published image | `karmic-kraken-beta-20261001-020df706a373de8d` |
| Published digest | `sha256:b163546994ee4635feb020e5beb644987b852f80bfb96d410e03c430df4cb669` |
| Recipe | `6c0e9843bb962f483409b0296b225cca03fe8567` |
| vLLM | `4a379ed42881ee022aaf5d9ada554f9096e5acf3` |
| B12X | `b557d87850cc836268fd327ccd46eaa6a033cdbf` |
| FlashInfer | `dbd6238c6655b98195fdf77f04bba6facf5a38a4` |
| CUTLASS DSL and all four library distributions | `4.7.1`, ARM64-compatible wheels |
| QuACK | `0.6.5` |
| torch-c-dlpack-ext | `0.1.5`, ARM64 wheel |
| Spark base image | `1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc` |

CUDA 13.4, NVIDIA PyTorch 2.14, FlashKDA, InstantTensor and NCCL 2.31.2
remain from the foundation. LMCache remains at `413ac987` and disabled, as in
our built image; the publication's `820af25f` LMCache is an explicit deviation.
The optional Rust DSML extension also remains at its base bytes. The Python
parser follows the pinned vLLM source.

## Build inputs and native boundaries

`prepare.py` reads the existing `~/git/vllm` and `~/git/b12x` checkouts at exact
commits. It proves the native vLLM inputs unchanged apart from the already
declared deferred Rust files, independently replays the foundation, preserves
the SM121 architecture/draft-head overlay and verifies the merged QSA865 patch.
It freezes 316 vLLM and 342 B12X tracked-file deltas and their resulting Git trees.
The compiler libraries are replaced separately; unchanged vLLM binaries are reused.

`prepare_inputs.py` restores the frozen `inputs.lock.json`; it never resolves
new package versions or rewrites either lock. ARM64 wheels use their locked URLs
and SHA-256 digests. Source transports are checked against pinned Git blob/mode
inventories, then repacked with sorted names and fixed tar/gzip headers. The locked digest covers the
uncompressed tar, so compression-library differences cannot change identity. Changes
to GitHub archive prefixes and timestamps therefore produce identical inputs.
`inputs/` and `refresh.tar` are ignored generated inputs; restore or transfer them
before preflight. No new checkout or remote is needed.

FlashInfer now builds in a separate stage using the original immutable NGC
foundation and `/usr/bin/python3`. Before compilation, the original dependency
precheck verifies setuptools 81.0.0, wheel 0.48.0, apache-tvm-ffi 0.1.13.post3
and the other pinned build dependencies, with CUTLASS upgraded to 4.7.1.
`FLASHINFER_BUILD_NO_PIP=1` disables its dependency-install hook. The compiler
install uses `--break-system-packages` only in this disposable component stage,
so an NGC `EXTERNALLY-MANAGED` marker cannot refuse the initial upgrade.
Assembly uses `--network=none` and `--pull=never`; the NGC and uv images must
already exist locally. Compiler wheels, source refresh and rebuilt FlashInfer
wheels enter the serving stage through temporary mounts. The source/build trees
and input archives are absent from final image layers.

The system NGC compiler remains 4.6.2 as a declared deviation. The serving
venv installs 4.7.1 with import precedence fixed explicitly; a fresh interpreter
and the runtime compiler gate require the imported `cutlass.__file__` under
`/opt/venv` and `cutlass.__version__ == 4.7.1`. All runtime gates explicitly use
that interpreter. Native changes remain restricted to compiler, QuACK/DLPack
and FlashInfer distributions; the foundation independently audits vLLM binaries.

The source-first vLLM/B12X distributions retain their original version metadata.
Only inherited CUTLASS `Requires-Dist` fields are advanced from 4.6.2 to 4.7.1;
their distribution RECORD hashes are updated and before/after digests recorded.
The runtime source lock, image labels and source trees identify the new runtime.
This is a declared packaging overlay, not a claim of a newly compiled vLLM wheel.

The cache fingerprint includes the compiler/source dependency lock, so new
CuTE/B12X/Triton caches cannot reuse the previous compiler's runtime namespace.
Expect cold compilation; performance remains unmeasured on Spark.

## Safeguards and pending GPU gates

`build.sh` verifies all local inputs before building. Real execution requires
dusty, ARM64, a GB10 reported by `nvidia-smi`, the shared dusty/kirby build lock,
the exact base image and successful idle-container/GPU probes on both nodes.
It records the recipe manifest, logs, exit status, image identity and native
replacement audit. The serving tag is created only after every gate succeeds.

The gates include the inherited native/linkage, FlashKDA, launcher, memory and
regression matrices; QSA865; beta's original seven regression files; the new
scheduler/boundary-admission, Qwen GDN and shared PLE table tests; and B12X's compiler-migration
corpus plus the MXFP8 numerical/graph and PLE embedding suites. Test collection must be nonempty,
and every selected case must pass without skips or xfails. The four cuDNN
comparison cases excluded by the upstream compiler corpus remain excluded and
are not acceptance evidence. A new FlashInfer gate compares FP16/BF16 decode,
prefill and RMSNorm with Torch numerical references, disables JIT fallback and
requires every requested nvcc module to have an artifact inside the rebuilt AOT
cache, and blocks its compiler entry point directly. It exercises vLLM's actual
CUDA top-p (0.95), top-k and combined sampler paths, checking allowed tokens and
sampling frequencies against a known Torch probability distribution. It runs at
1,024 tokens (one sampler pass) and at the served vocabularies, 129,280 (DS4
Vision), 154,880 (GLM) and 248,320 (Qwen), with the kept tokens in late blocks
so the multi-pass path must find them. No native
sampler fallback is admitted. The pinned FlashInfer source itself also reads
`FLASHINFER_DISABLE_JIT` in `flashinfer/jit/core.py`; this is checked source
behavior, not a GPU result. The gate also checks that build
inputs are absent from the serving image. No GPU gate has run during preparation.

## Serving profiles and qualification

The candidate Qwen, GLM and DS4 Vision runners preserve the current September 29
kit's head/worker serving arguments. Both roles are checked against the frozen
`runner-baseline.json` fixture, independent of dirty September 29 files.

Only image/container identity and source/compiler provenance change. Each runner
requires `EXPECTED_IMAGE_ID` from the new successful build receipt.

- Qwen: TP2, HC off, MTP3, aligned checkpoints, 262K context, current QAD `7c4f1bc1`.
- GLM: TP4/DCP1, MTP3 by default, aligned checkpoints, current QAD `175ae8ce`.
- DS4 Vision: TP2, DSpark K3, B12X A8, FP8 KV, existing `6821d6ad` checkpoint.

DS4 ships `model-manifest.json` outside ignored receipts; its integrity list
covers the runner, model manifest and runtime helpers. GLM `ROLE=stop` stops and
removes its candidate container, allowing a subsequent start.

`distribute.sh <build-receipt>` verifies the successful build and FlashInfer gate,
then stages the kit and transfers a digest-checked Docker archive over the
switched 200G fabric. `TARGET_GROUP=qwen|ds4-vision|glm|all` selects transfer targets. Serving may continue during copy/load; launch and
build idle checks remain separate. No container is stopped by distribution.
It checks the imported image ID on every node. The current candidate tooling is:

- Qwen: `execute.sh`, then `benchmark.sh` after the existing replay/health review.
- DS4: `ds4-vision/execute.sh`, `benchmark.sh`, structured/prefill probes and comparison.
- GLM: `glm/execute-glm.sh`, `qualify-glm.sh`, `confirm-transport.sh` and acceptance/prefill probes.

All tools use the October 1 tag and staging tree and require the new built image
ID. GLM launch uses the runner's production `NCCL_DEBUG=WARN` default. Both Qwen and
GLM now require a full C1/C2/C4 `BASELINE_GRID` from the reported production image
`500ae05b` and current QAD checkpoint, and write `production-vs-candidate.json`.
Identity, harness, token-budget and protocol drift reject comparison. No historical
R38 result substitutes for this required baseline. Transport confirmation probes
the running candidate and does not stop or replace another deployment.
No model revision, MXFP8 drafter or serving profile change is selected here.
The afternoon publication adds opt-in same-host TP1 replicas and shared PLE
storage. Its recipe identity is recorded, while the qualified Spark launchers
remain inherited: this TP2/TP4 candidate does not select either option or the
new recipe's optional MXFP8 MTP expert configuration.
These scripts are prepared for a separately authorized idle window; none has
been executed on a host. Keep rollback containers separately from candidate stop.
The Qwen failure guidance names the retained September 29 beta containers.
Full model correctness, capacity, acceptance and benchmark qualification remain
required after the native gates.

## Local validation and reproduction

From this directory, reconstruct generated inputs if necessary:

```bash
python3 prepare_inputs.py
python3 prepare.py
python3 preflight.py
python3 -m unittest discover -p 'test_*.py' -v
shellcheck *.sh glm/*.sh ds4-vision/*.sh
DRY_RUN=1 bash build.sh
```

The local suite checks deterministic source-lock/payload preparation, complete
delta replay, remote Git blob equality for all five source archives, rejection
of a wrong source digest, compiler requirement metadata/RECORD handling, and
head/worker command preservation. These are artifact checks, not an image build
or a live runtime result. All 25 local tests, shell checks, input preflight and
the build dry-run passed during preparation. See [the upstream comparison](../../../docs/upstream-check-20261001.md).


Before changing either deployment, capture its production grid in an approved
benchmark window (these commands send inference traffic):

```bash
bash capture-baseline.sh qwen "$PWD/qualification/production-qwen"
bash capture-baseline.sh glm "$PWD/qualification/production-glm"
```

Each captures current container identities, checks the image/checkpoint/profile,
and runs the unchanged standard harness. Supply the printed `BASELINE_GRID`
path when running Qwen's benchmark or GLM's execute/qualify scripts. The September 30 GLM QAD grid and October 1 Qwen QAD qualification grid both
pass `compare-production.py --validate-baseline`; they are the production
baselines for this build. Separate baseline capture runs are unnecessary.
The Qwen record explicitly waives private replay because the trace is unavailable.
Candidate comparisons remain pending.
