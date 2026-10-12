# Upstream check 2026-10-09: second fetch pass and the October 9 Spark kit

Follows [the October 8 check](upstream-check-20261008.md). Fetch pass over all AGENTS.md checkouts (now including the `lil` remote of `~/git/flashinfer`, added 2026-10-08 on the user's instruction) with no failures; fork fast-forwards pushed to origin: rtx6kpro master efcd534 to af8a6e6 (daily summary covering October 8), vllm main ab905a885d to 9adba10f3f, LMCache dev 8bf2cf23 to 68a9c6b5, flashinfer main fa7c741e0 to 50829ac26. b12x, blackwell-llm-docker and spark-vllm-docker already equal. llm-inference-bench: the user rebased their two commits onto local-inference-lab main (b545dae = dcdfe05 plus two); nothing to push.

## Beta sources unchanged overnight

| Source | Tip | Note |
| --- | --- | --- |
| blackwell-llm-docker main | `05eaafa` (docker #148) | newest release still `karmic-kraken-beta-20261008.2` |
| vLLM integration/karmic-kraken-beta | `19f2c20e` (#1016) | |
| FlashInfer integration/karmic-kraken-beta | `b4c35e6e` | B12X inside (flashinfer/experimental/b12x) |
| LMCache integration/local-inference-lab | `3171cdb5` | |
| arm64 Spark twins | none after 2026-10-07 | Spark workflow idle since 2026-10-07T23:30Z |

## Serving state (live, all idle)

dusty/kirby Qwen TP2 QAD 7c4f1bc1 on the 20260929 kit (3 days, restored after the October 5 trial), KV 5,174,429; rusty/toby DS4 Vision TP2 (9 days), KV 2,200,602; sparky x4 GLM TP4 QAD 175ae8ce (8 days), KV 6,837,756; nous GLM TP4 CSF dec48abd on 20261008.2, KV 6,895,817 (unreachable at the time of the KV read, last boot value). All three Spark clusters serve original checkpoints; only nous serves CSF. The upstream arm64 presets avoid CSF themselves (GLM-5.3-Flash-NVFP4-Spark, Qwen original), so there is no upstream CSF evidence on GB10.

## The October 9 kit (spark/karmic-beta-sm121/20261009, prepared, not built)

Re-pins the October 5 kit to publication 20261008.2. Not a re-pin in practice: B12X moved into FlashInfer 0.7.1 and the base image, its identity gates and both inherited launchers assert `b12x` at `/opt/jovian-judgement/b12x` with RoCE proxy ABI 4. Findings that shaped the kit:

- vLLM native inputs since the base e77be225: only the deferred Rust DSML files plus `requirements/cuda.txt` and `setup.py` (FlashInfer 0.6.18 to 0.7.1, `b12x` extra to `flashinfer-python[b12x]`). The compiled wheel is reused; installed metadata is overlaid.
- B12X: the FlashInfer subtree differs from the archived tip 236ddff0 in 395 of 536 files, mostly reformatting plus the October 6 to 8 work (per-call A4 prefill, CSF stage scales). Against the base 10a553ef every C source differs: `_roce_proxy.c` is ABI 5 (per-peer rails, up to four HCAs, 1,084 changed lines), the loader sources are reformatted (70 non-whitespace lines in `_batch.c`), `preparation/_memory.cpp` is gone. The GLM TP4 runner enables the B12X RoCE all-reduce, so ABI 5 is on the production path.
- The SM121 overlay's draft-head patch is upstream (vLLM #995); only the CMake architecture list remains.
- FlashInfer 0.7.1 runtime dependencies on arm64 (recipe `linux-arm64/qwen38-runtime.lock`): nccl4py 0.4.1 (base 0.3.1), nccl-extensions 0.1.0 (new, shares the `nccl` namespace), nvidia-cudnn-frontend 1.30.0 (base 1.27.0), apache-tvm-ffi stays 0.1.11. All three have PyPI aarch64 cp312 wheels with the lock's hashes.
- FlashInfer ships three wheels now (python, jit-cache shim, one provider); the provider is built for 12.1a as `flashinfer-jit-cache-sm121a`. The B12X test corpus lives in FlashInfer `tests/experimental/b12x`; the archived corpus file is gone, so the compiler gate pins the twelve entries explicitly.

Kit design: keep the B12X root, refresh it from the frozen FlashInfer archive subtree (byte-identical to the wheel's embedded package, `_compat` excluded), remove the base's b12x 1.3.0 wheel copy, install the three FlashInfer wheels and three runtime wheels, rebuild the RoCE proxy in-image to ABI 5 with a no-compiler load check, adapt the identity gates and the GLM launcher as kit copies differing only in the ABI assertion (pinned by tests), run the compiler gate from the host-unpacked frozen FlashInfer source. Checkpoints stay on the originals (user decision), serving arguments unchanged against `runner-baseline.json`. Local validation: inputs restore, prepare, preflight, 35 unit tests and a dry run all pass.

Codex review (Herdr, read-only) found nine issues, seven of them build or serving blockers, all mechanical consequences of updating the kit by substitution instead of re-reading every consumer of the old composition: the new runtime lock was missing from the `.containerignore` allowlist; the root-versus-wheel identity check counted a README the wheel does not package; the relocated identity gate could not import the foundation modules from `/kit`; the foundation gate's live GLM preflight ran the image's ABI-4 launcher; the inherited warmup matrix is 20 cases at vLLM 19f2c20e, not 8; both serving runners still required the October 5 vLLM tree; the GLM runner's launcher-label check rejected the mounted ABI-5 copy; the embedded `b12x/__init__.py` is an alias stub, so copying it left the root without its public API and allowed duplicate module state across the two import paths; the added parser test loads two unpinned Hub tokenizers. All nine are fixed: the ABI-5 launcher is now installed and labeled in the image, the root carries a kit initializer (`b12x_root_init.py`) with the wheel's bootstrap rooted at `/opt/jovian-judgement/b12x`, and the upgrade step checks wheel-payload identity, the public API and module identity across both import paths. The second Codex pass confirmed all nine closed and found two more, both fixed: the compiler and FlashInfer gates still asserted the root package's `__file__` as `__init__.py` (the kit initializer executes `_api.py` as the package, so it is `_api.py`, now pinned by a test across every consumer), and the inherited `local-inference.roce.proxy-abi="4"` label was not overridden (now `5`). 36 unit tests pass.

Open until the first build: inherited warmup case count against the new `test_b12x.py`, beta gate collection on the eight added files, compiler gate conftest behavior under `--confcutdir`, and whether the stale `test_mamba_utils.py` deselections are still needed.

## Build and Qwen qualification (user go 2026-10-09 15:20Z)

Qwen production on dusty/kirby stopped 15:20Z (containers retained). Build on dusty: eight attempts 15:24Z to 18:27Z (kit README lists each failure: two of my checks, one stale upstream test, container seccomp for io_uring, a designed skip, a `benchmarks` package collision, a two-device test). Attempt eight passed every gate; image `d9ba012f`, tagged `karmic-beta-20261009-spark-sm121`, loaded on kirby. Qwen serving gates passed on a cold and a warm boot. Grid vs the October 4 production grid: about -3% decode tok/s, the same shape as October 1 and 5. Grid vs a same-window fresh-boot production grid on the frozen harness: -2.7/+0.6/+0.4% tok/s, -0.5/+0.7/+1.4% steps/s at C1/C2/C4, prefill +1 to +2%: parity. The October 4 baseline is 3.4-4.3% above tonight's production at C2/C4, confirming the single-baseline pitfall recorded on October 6. Production restored 19:28Z; candidate containers retained; promotion is the user's decision. DS4 Vision (user go, 2026-10-09 evening): image distributed to rusty/toby, candidate passed every serving gate and its grid, vs same-window fresh-boot production +1.0/+3.7/+2.4% tok/s and flat steps/s at C1/C2/C4, prefill -1 to -2%, KV -9.7% (1.98M vs 2.20M tokens; +1.07 GiB consumed memory per rank, +0.33 GiB graph capture). Production restored 20:14 EDT; candidate retained as *-r1. GLM TP4 (RoCE proxy ABI 5 on the four-node fabric) was not touched and needs its own window.

## GLM qualification (2026-10-11, user go)

Image distributed to sparky/buddy/rocky/lucky while production served. Production restarted fresh 14:00 EDT for a same-window baseline grid (frozen harness), then stopped 14:29; candidate booted 14:30, ready 14:37 (RoCEnante ABI 5 live on both rails, backend list exact). Every serving gate passed, including needles to 1,048,000 tokens. Grid vs fresh-boot production: steps/s +0.1/-0.1/+0.7%, tok/s -4.3/+0.3/-2.3% at C1/C2/C4 (acceptance length; a same-window greedy identical-prompt probe reads 2.671 production vs 2.664 candidate, so the dip is sampling noise), grid prefill +0.4 to +0.8%, fresh zero-cache prefill +1 to +3.6%, KV +3.8% (7.19M vs 6.93M tokens). No errors in any container log. Production restored 15:15 EDT (ready 15:18); candidate retained as `*-r1` on all four nodes. All three serving profiles are now qualified on image d9ba012f; promotion of any of them is the user's decision. Record: `runs/glm-5.3-flash/nvfp4/2026-10-karmic-beta-20261009-sm121-qualification/production-fresh-vs-candidate-r01.md`.

## GLM upstream-profile catch-up (2026-10-11, user direction)

The recipe 05eaafa `glm53-flash` profile was adopted on the same image except the Spark choices we keep (`spark/karmic-beta-sm121/20261009/glm/upstream-profile.md`), Codex-reviewed twice. CSF QAD checkpoint dec48abd staged on the four nodes. Second window 17:07 to 17:58 EDT (attempt 1 lost six minutes to the launcher's stale `--fairness-engine` flag; an in-image launch-argument preflight now exists). Every gate passed. Against the same image on the current config: steps/s +27.5/+21.2/+15.4%, tok/s +31.6/+20.1/+16.9% at C1/C2/C4, acceptance unchanged, fresh prefill +4 to +7%, KV -1.1%, weights -4.5 GiB per rank; confirmed by an identical-prompt greedy probe (69.2 vs 54.5 tok/s per user). Production restored; candidate retained as `*-upstream-profile-*-r1`. Record: `runs/glm-5.3-flash/nvfp4/2026-10-karmic-beta-20261009-sm121-qualification/upstream-profile-vs-current-r01.md`.

## GLM promotion (2026-10-11 19:21 EDT, user)

The upstream-profile candidate now serves GLM on sparky x4 (container `glm53-flash-csf-qad-upstream-profile-karmic-beta-20261009-tp4-r1`); 500ae05b containers retained for rollback.

## Not done

Qwen and DS4 Vision promotion (image candidates retained). Per-knob attribution of the GLM gain. The same catch-up review for the Qwen and DS4 Vision profiles. The kit directory, the campaigns and this report are uncommitted.
