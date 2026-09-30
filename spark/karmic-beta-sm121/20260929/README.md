# Full Karmic beta, September 29: Spark SM121 build kit

Location: `spark/karmic-beta-sm121/20260929/`. The SM121 foundation it builds on, meaning the base
image `1a7a8acf`, its source lock, the shared contracts, gates and pair lock, lives in
`spark/karmic-main-sm121/20260922/` and is referenced by one named path (`FOUNDATION`) in
`prepare.py`, `preflight.py`, `test_kit.py` and `build.sh`. This kit is a separate lineage
from the September 20 beta candidate at the top of `spark/karmic-beta-sm121/`.

Status: **built, gated and in production** for Qwen TP2 (dusty/kirby) and DS4 Vision TP2 (rusty/toby) since 2026-09-29, and for GLM TP4 (sparky/buddy/rocky/lucky) since 2026-09-30; see QUALIFICATION.md, ds4-vision/QUALIFICATION.md and glm/EXECUTION.md. Earlier status: Built on dusty on 2026-09-29 in a
user-authorized window (Qwen on dusty/kirby stopped gracefully, logs archived under
`~/logs/qwen38-flash-next-nvfp4-karmic-main-hcbase-qsa865-tp2-*`, containers kept for rollback).

- Image `sha256:500ae05b98da0658c1a5e1820387f96f2121c5659bd7954ad1c5861f20934f05`, tagged
  `localhost/voipmonitor/vllm:karmic-beta-20260929-spark-sm121` only after all gates. Labels
  match: vLLM tree `5a01c2d4`, B12X tree `641d413e`, Qwen launcher `5551db04`.
- Staged on dusty at `~/git/bld-jj-r38-spark/{karmic-main-sm121/20260922,karmic-beta-sm121/20260929}`;
  staged preflight values matched the local ones exactly. Receipt:
  `receipts/build-20260929T143017Z-2072867/` on dusty (exit 0, BUILD-OK).
- Gates: source identity, native reuse, FlashKDA 12/12, CLI arguments, Qwen and GLM launcher
  preflights, inherited regression matrix (5 files), memory gates (7 cases), GLM SM121 draft head,
  PR865 cases 1/2/10/2/2/1, and the beta gate 44/44 across 7 files: compressor ring 4, block
  table 13, NULL_BLOCK padding 1, DCP LSE 8, failed-rank gather 13, engine exit 2,
  structured-output drafts 3.

## Composition

| Input | Pin |
| --- | --- |
| Publication | `karmic-kraken-beta-20260929-7476365edc3d8ee5` (recipe `f28aea04babb0e6b0a5830354f1ef6fd2a1b3d3f`) |
| vLLM | `99cbe782f3b67a7758a85bcc0e7938a1ec27012a` (`lil/integration/karmic-kraken-beta`), result tree `5a01c2d4f2822b2e2b39c0422b555f8aa59c9bb8` |
| B12X | `1b6cd278626aa6b44c519b5d9f4ad576c9af783b` (`lil/integration/karmic-kraken-beta`), tree `641d413e96d96bf673b34d573c0e27d6acb3a9ca` |
| Base image | `1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc` (vLLM `e77be225` + SM121 overlay + PR865, B12X `10a553ef`) |

Both base commits are ancestors of the beta pins. The build is a tracked-source
refresh: `refresh.tar` carries 290 vLLM and 303 B12X changed files, all Python, shell
or text. Native products, CUDA 13.4 / PyTorch 2.14, FlashInfer, FlashKDA, InstantTensor
and NCCL 2.31.2 are reused from the base, and the installer proves native bytes are
unchanged. The SM121 architecture and GLM draft-head overlay (`karmic-main-sm121/20260922/contracts.py`) is
applied unchanged. PR865 is not re-applied: the preparer checks that its patch
reverse-applies exactly to beta (merged as `01f1b874c7`).

Beta supersedes our local DS4.1 fixes (compressor ring, Engram ordering) and the carried
QSA865 patch. It brings #911 (DS4.1 BF16 sparse attention, FP32 logits), #918, B12X #435,
the DSML parser fix, 638f90a1fc (NULL_BLOCK_ID padding, DCP LSE), failed-rank handling,
#897, stall diagnostics and the rest of the beta delta described in
`docs/upstream-check-20260929.md`.

## Provenance in `runtime.lock.json`

The installer writes `runtime.lock.json` as the image's canonical source lock, so the
two provenance levels are kept apart there. Following `docs/karmic-artifact-review-20260929.md`,
the six fields copied from the base lock (`base_build_lock_sha256`, `native_inputs_sha256`,
`native_metadata`, `published_image`, `recipe_commit`, `lmcache_refresh`) live under
`foundation`, alongside the base image ID and base lock digest and path. `publication`
names the beta tag, registry digest `sha256:74c57a9b...`, recipe `f28aea04babb` and
channel, and records that the amd64/SM120 published image is used for source identity
only. `test_kit.py` fails if an inherited field reappears at the top level.

## Declared deviations

1. **Rust DSML parser at base bytes.** The only vLLM native-input change is four files
   under `rust/src/parser/.../deepseek_dsml/`. The compiled Rust extension is not rebuilt,
   and those sources stay at the base bytes. It is used only with `VLLM_USE_RUST_FRONTEND=1`
   (default off). The Python DSML parser carries the fix.
2. **LMCache at base `413ac987`** (beta publishes `75f2b59d`), by user decision. LMCache
   is disabled in every deployment.
3. **No DS4.1 stock-NCCL swap.** The DS4.1 kit's NCCL 2.30.7 component is not included.
   A DS4.1 window would add it as a separate, declared layer.

## Files

- `prepare.py`: replays the base lock independently, enforces the native and allowlist
  contracts, and writes `runtime.lock.json`, `build.lock.json` and `refresh.tar`
  (`refresh.tar` is gitignored and regenerated by `prepare.py`).
- `install.py`: runs inside the build; installs only declared deltas through the base
  image's own installer and fails if any native artifact changes.
- `Dockerfile`, `.containerignore`: offline build over the pinned base.
- `preflight.py`: input digests, frozen trees and reused parent gate assets.
- `build.sh`: dusty only, takes the shared dusty/kirby pair lock
  `karmic-main-sm121/.build.lock`, refuses while dusty or kirby is serving. It runs
  `karmic-main-sm121/20260922/gate.sh`, `karmic-main-sm121/20260922/qsa865/gate.py` (PR865 cases) and `gate_beta.py` (beta's own
  tests for the ring, block table, NULL_BLOCK padding, DCP LSE, failed-rank gather,
  engine exit code and structured-output drafts). It tags
  `localhost/voipmonitor/vllm:karmic-beta-20260929-spark-sm121` only after all gates.
- `run-qwen.sh`: the serving Qwen runner with this image's name and provenance pins.
  HC-off stays the default (`VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0`). It requires
  `EXPECTED_IMAGE_ID` from the build receipt.
- `test_kit.py`: local checks.

Local checks (all pass):

```sh
python3 test_kit.py
python3 -O test_kit.py
shellcheck build.sh run-qwen.sh
python3 preflight.py
DRY_RUN=1 bash build.sh
```

## When the window is given

1. Refresh: rerun `python3 prepare.py` only if the pins are intentionally moved;
   otherwise preflight must pass unchanged.
2. Stop Qwen on dusty/kirby gracefully (worker kirby first, then dusty) and retain the
   containers. Stage `spark/karmic-beta-sm121/20260929/` and `spark/karmic-main-sm121/20260922/` on
   dusty with their relative layout preserved, then run `bash build.sh` on dusty. Receipts
   land in this kit's `receipts/`.
3. Qualify Qwen TP2 first (user decision): distribute to kirby over the 200G mesh,
   `ROLE=worker` on kirby then `ROLE=head` on dusty with `EXPECTED_IMAGE_ID`. Then run
   the same battery as `qsa865-hcbase`: semantic 18, retrieval to 262K, counting,
   concurrency transitions, private replays and two standard grids, compared against
   `a25bedd4`.
4. GLM, DS4 Vision and DS4.1 follow in their own windows. The unchanged 524K gate still
   applies to DS4.1.
