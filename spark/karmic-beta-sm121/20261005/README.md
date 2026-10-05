# October 5 Karmic beta with CUTLASS DSL 4.7.1 for DGX Spark

Status: **prepared, not built**. No GPU gate or model qualification has run.
Target tag: `localhost/voipmonitor/vllm:karmic-beta-20261005-spark-sm121`.

This kit is the [October 1 kit](../20261001/README.md) re-pinned to the latest beta publication. The composition method, build inputs, native boundaries, safeguards and GPU gates are unchanged except as listed below; read the October 1 README for them.

## Pinned composition

Ports the [October 5 beta publication `4102660a`](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-4102660abd8a98b16e03386a9972a65dbbff805c1527ae7e94ac9c75156445d7) to ARM64 / SM121. `publication.json` is that release's `container-release.json` verbatim. Its x86_64 / SM120 wheels are source references, not Spark binaries.

| Component | October 1 kit | This kit |
| --- | --- | --- |
| Published image | `karmic-kraken-beta-20261001-020df706a373de8d` | `karmic-kraken-beta-20261005-4102660abd8a98b1` |
| Published digest | `sha256:b1635469...` | `sha256:fd74019fc217a541060093d18427c36330df6ac5700651b55593dafdbb936920` |
| Recipe | `6c0e9843` | `e881183c06a6f64dffb87afb24c7e7703240dca3` |
| vLLM | `4a379ed4` | `f1c2508f1018f47ee8a78478a819d202f359887b` |
| B12X | `b557d878` | `52640cb15d4ad1c7230f747e72c45d45dc2da681` |
| vLLM refreshed tree | `96971e1c` | `743d4acd4f28d1d08c224a3e69f5c1a6f9009c03` |
| B12X tree | `5561aa25` | `4be1c5eb5e11b3322b8f444741b05db721a9a5b5` |

Unchanged: FlashInfer `dbd6238c`, CUTLASS DSL `4.7.1` and its four library distributions, QuACK `0.6.5`, torch-c-dlpack-ext `0.1.5`, Spark base image `1a7a8acf`, CUDA 13.4, NVIDIA PyTorch 2.14, FlashKDA, InstantTensor and NCCL 2.31.2 from the foundation. `inputs.lock.json` and `compiler-arm64.lock` are byte-identical to the October 1 kit. LMCache stays at `413ac987` and disabled (the publication's `820af25f` is a declared deviation, as before).

The publication `b3523c2c` has the same vLLM and B12X; `4102660a` adds only recipe #130 (GLM TP4 CSF preset), which our runners do not use. The recipe's build side changed only the bundled bench lock between the two kit pins.

## Source refresh

`prepare.py` freezes 418 vLLM and 447 B12X tracked-file deltas against the SM121 base. vLLM native inputs are unchanged apart from the already deferred Rust DSML files. B12X has no native source change and the RoCE proxy is unchanged.

New declared deviation: `.dockerignore` and `tools/jovian_wheel_release/Dockerfile` changed upstream (`5ef18472`, wheel-release tooling) and are not admitted shipped suffixes. They stay at the base bytes (`TOOLING_DEFERRED`, recorded as `deferred_tooling` in `runtime.lock.json`); nothing reads them at runtime.

Default-on behavior changes reaching our serving path, beyond opt-in CSF and A4 features:

- B12X #469: W4A16 skips MMAs for empty 16-row route blocks (`B12X_W4A16_SKIP_EMPTY_M_BLOCKS=1` default). Qwen's MTP experts and GLM's MTP experts run W4A16.
- B12X #464: `B12X_DYNAMIC_DETERMINISTIC_OUTPUT` is resolved before split selection.
- B12X #467/#468: the collective barrier deadline is validated per session.
- vLLM #976: NVFP4 MoE input scales are zero-initialized.
- vLLM #841 (`aa86abd0`): InstantTensor staging is bounded, oversized tensors route to CPU and deferred tensors keep their owner. Our runners load with InstantTensor.
- vLLM #858 (`88904938`): the aligned-state index gather for hybrid models is tiled across groups and requests, used by Qwen's GDN layers with aligned checkpoints and MTP; the K3 KV group-size override applies only to `kimi_k3`.
- vLLM #980: weight loading no longer applies `max_split_size_mb:20` when `expandable_segments` is set, which our launchers set. Post-load free memory and KV sizing may move.

## Added gates

Beta regression gate (`gate_beta.py`), in addition to the October 1 files:

- `tests/v1/worker/test_gpu_worker.py::test_weight_loading_split_limit_skips_expandable_segments` (#980)
- `tests/models/qwen4_exp/test_b12x_config.py` (#974)
- `tests/kernels/moe/test_b12x_a4_prefill.py` (#970/#977, opt-in wiring)
- `tests/model_executor/model_loader/instanttensor_loader/test_weight_utils.py` (#841), deselecting `test_instanttensor_model_loader`, which downloads gpt2 unpinned; the gate stays offline
- `tests/v1/worker/test_mamba_utils.py` (#858)

Compiler gate (`gate_compiler.py`), in addition to the October 1 corpus:

- `tests/moe/test_w4a16_e2e.py::test_w4a16_skipped_empty_row_blocks_are_bit_identical` (#469)
- `tests/moe/test_w4a16_a4_prefill.py` (#472/#473/#476)
- `tests/moe/test_nvfp4_split_dispatch_policy.py::test_preparation_resolves_determinism_before_selecting_split` (#464)
- `tests/preparation/test_session.py::test_preparation_import_ignores_unused_barrier_timeout` (#467)

Selected cases must collect and pass without skips, as before. The #976 test needs upstream's `dist_init` fixture from `tests/conftest.py`, which the gate's `--confcutdir` excludes, so it is not run in-image; `test_kit.py` checks the shipped zero-initialization source instead. Local tests add checks that the deferred tooling stays at base bytes and that the #976 and #469 sources ship.

## Serving profiles and qualification

Runners are the October 1 kit's with the new tag, container names and provenance labels; serving arguments are unchanged and checked against the frozen `runner-baseline.json` (production September 29 kit). Checkpoints stay on the originals for a like-for-like comparison with production (user decision, 2026-10-05): Qwen QAD `7c4f1bc1`, GLM QAD `175ae8ce`, DS4 Vision `6821d6ad`. No CSF checkpoint, A4 prefill, GLM expert-activation or router-weight option is selected.

Upstream recipe defaults changed since the October 1 kit and are deliberately not adopted here: CSF checkpoints for Qwen, GLM and DS4 Vision; for GLM TP4, BF16 expert activations (`VLLM_B12X_MOE_FP4_FORCE_A16=1`), FP32 router weights, A4 prefill from 1,536 tokens and `gpu-memory-utilization` 0.96. Each is a separate qualification.

Qwen comparison baseline: the October 4 production grid `runs/qwen3.8-flash-next/nvfp4/2026-10-qad-6909a5be-qualification/throughput/20261004T231819-0400__karmic-beta-20260929-qad-7c4f1bc1-hc-off-mtp3__r01.json` (harness `cc9bb06a`, KV budget 78,619,040). Expect the October 1 C1 decode regression to persist: B12X winner selection is unchanged and [b12x#463](https://github.com/local-inference-lab/b12x/issues/463) is open.

Harness: `~/git/llm-inference-bench` was rebased onto upstream v0.7.7 on 2026-10-05 and no longer hashes to the baseline's `cc9bb06a`. `benchmark.sh` therefore runs a frozen copy that `freeze_harness.sh` rebuilds into the ignored `harness/` directory: `llm_decode_bench.py` from `24fec94b` (`origin/main`) plus `harness-cc9bb06a.patch` (the local metadata, chat-template-kwargs and coding-peak diff), and `harness-run_bench.5c79b976` (verbatim `run_bench.sh`), both verified by SHA-256, with the checkout's `.venv` linked. The GLM (`glm/qualify-glm.sh`) and DS4 Vision (`ds4-vision/benchmark.sh`) grids still pin `2c447f16` against the live checkout and will refuse to run until their own windows settle the harness, the same way.

Results go to campaign `2026-10-karmic-beta-20261005-sm121-qualification`. The build needs an idle dusty/kirby pair, which means stopping Qwen production; that window is the user's decision.

## Local validation

```bash
python3 prepare_inputs.py   # only if inputs/ is absent; inputs were copied from the October 1 kit
python3 prepare.py
python3 preflight.py
python3 -m unittest discover -p 'test_*.py' -v
shellcheck *.sh glm/*.sh ds4-vision/*.sh
DRY_RUN=1 bash build.sh
```

See [the upstream check](../../../docs/upstream-check-20261005.md).
