# Infernal Invocation r10 Spark/SM121 qualification record (2026-08-14)

Candidate unified image for DS4, GLM-5.2, and Laguna on the DGX Spark fleet.

Image: `infernal-invocation-r10-spark-sm121-vllmd650d5c-b12x5d648d9-fi1ac6942-cu133-torch213-20260813`
(id `12e1b7c06998`, 30.8 GB). RELEASE_DATE carries the upstream script default
(20260813); the ceremony rebuild stamps the canonical date.
Base image: `ii-spark-sm121-cu133-torch213-nccl2312-20260814-r1` (id
`2a3b0c146aa6`, 25.9 GB).

Status: **Laguna QUALIFIED, DS4 QUALIFIED, GLM NOT QUALIFIED (line
mismatch; see GLM section)**. Fleet posture after this record: DS4 and
Laguna are II cutover candidates on their receipts; GLM remains on the GG
line (r33 production; GG r34 = sanctioned upgrade path). This mirrors
upstream: every II qualification receipt (r2-r10) tests only
DeepSeek-V4-Flash-0731; GLM's serving qualification lives on GG r34. The
original single-unified-image goal is revised to per-model-family qualified
lines. r33-spark retained fleet-wide as rollback.

## Composition

Upstream anchors: build composition `694a55c`, receipt `3dcc751`
(blackwell-llm-docker). Canonical trees reconstructed byte-exact on dusty
before any overlay:

| Component | Base | PRs | Tree |
|---|---|---|---|
| vLLM | `dev/infernal-invocation@ce5f50f6` | 285-296, 298, 300-304 | `a7f04eb1` |
| B12X | `master@954fd017` | 145, 146, 148, 149, 150 | `5d648d94` |
| LMCache | `release/v0.5.2-glm52-dcp-base@9cebd405` | 13 PRs | `5fdf59cf` |
| FlashInfer | `voipmonitor @ 1ac6942` (0.6.18, unchanged pin) | - | source build |
| ExLlamaV3 | `704aefd` (unchanged pin) + aarch64 guards patch | - | - |
| DeepGEMM | `a6b593d` (upstream's own Spark pin; no local patch needed) | - | - |

Platform: NGC `nvcr.io/nvidia/pytorch:26.07-py3` **arm64** digest
(`7531d90bcbe0...`), PyTorch 2.13.0 source-built at `cf30153c` for 12.1a,
NCCL 2.31.2 from `nccl-canonical@fb6f409` (Turin branch) built for sm_121,
CUDA 13.3, cuDNN 9.24, XGrammar 0.2.5, CUTLASS DSL 4.6.2.

Spark vLLM overlay (77 lines, sha `528cf5b6`, overlay tree `d650d5c0`,
applied after canonical-tree verification): CMake CUDA>=13 arch lists accept
12.1; fail-closed online EXL3 Trellis quantization on aarch64 at
`_online_trellis_bits()` + `_load_exl3_online_quantizer()` (literal-"1"
override, envs.py registration). Overlays DROPPED vs r33 (verified, not
assumed): the FlashInfer q_len guard (our #234 predicate present in the II
base verbatim, Laguna reproducers re-prove it e2e below) and the MTP-3D
residual fallback (#288 equivalent upstream). A #258-equivalent
prompt-logprobs bound is present in the II base (KLD capture unblocking
candidate).

## Build findings (Spark vs x86 deltas, all with preserved failing logs)

1. **Per-arch NGC base drift**: the arm64 26.07 base exports
   `NCCL_VERSION=2.30.7` and `PYTORCH_VERSION=2.13.0a0+9186a08` (x86 exports
   compatible values), plus `TORCH_CUDA_ARCH_LIST="8.0 ... 12.0+PTX"` and
   `CUTLASS_DSL_VERSION=4.5.2`.
2. **buildah vs BuildKit ARG/ENV precedence**: buildah resolves `${VAR}`
   against base-image ENV before stage ARGs; BuildKit prefers the ARG. Every
   value the base also exports must be a literal ENV. This produced three
   build failures before the exhaustive ARG-name x base-env audit closed the
   class (one torch wheel was silently built for the base's six-arch list and
   correctly refused by the version gate).
3. **`NCCL_NET_PLUGIN=spcx` inheritance**: the arm64 base selects the
   Spectrum-X NCCL plugin; neutralized in-image (plugin evaluation deferred
   to a deliberate experiment). Same-host NCCL policy env
   (`NCCL_IB_DISABLE=1`, `NCCL_P2P_LEVEL=SYS`, proto pins) stripped from the
   image: NCCL policy is runner-owned on Spark.
4. All 13 x86-era pip kernel deps (tilelang, quack/humming-kernels,
   tokenspeed-*, nccl4py, cupy-cuda13x, ...) ship aarch64 wheels; the
   pip-check exact-match allowlist passed unmodified on aarch64.

## Build gates (all PASS)

r33-style battery adapted to II: canonical + overlay tree verification in
the compose step; version-contract verify; upstream launcher DRY_RUN
contracts (DS4 + GLM52 incl. `serve-ds4-flash-spark.sh` presence); EXL3
extension import; exhaustive codebook decode oracle (bit-pattern +
finiteness); EXL3 GEMM execution parity; online-quant override matrix
(unset/"0"/"true"/"yes"/"2" closed, literal "1" opens); single-GPU + NCCL
collective smokes on the base.

## NCCL 2.31.2 investigation (dusty/kirby pair rails)

Turin-branch delta reviewed in full (140 lines): three hunks x86-AMD/Zen5
guarded (inert on GB10), two generic graph-search hunks. A/B vs r33's
2.30.4 found the "prefer more channels" hunk over-channels the 2-rank RoCE
rail: 2.5x slower at 256KB-1MB allreduce under the production
`NCCL_PROTO=LL,Simple`. 32-cell matrix (2 versions x 4 protos x 4 channel
pins, `~/nccl-matrix` on dusty): winner `LL,Simple + MIN/MAX_NCHANNELS=4`,
2.8-3.2x microbench gain at 128-256KB vs the production env on BOTH
versions; tuned 2.30.4 leads tuned 2.31.2 by ~9% band-geomean.

**End-to-end verdict (A/B/A, three-run means, DS4 TP2)**: no tok/s effect
(base 42.9/61.2/90.0 vs pinned 42.3/60.6/89.2 at cc1/2/4); cc1 boot-to-boot
spread is +/-7%, and DS4's per-step allreduce budget is ~2% of a verify
step. Dispositions: production r33 env unchanged; II runners carry the c4
pin defensively (erases the 2.31.2 regression); the TP4/DCP probe moves to
the GLM window where the collective fraction is several times larger;
Turin-branch finding queued for the upstream report.

## Laguna qualification (PASS)

First Laguna boot on II: KV 1,641,795 tokens (6.26x @262k; r33-era FULL-96
was 1,727,365; -5% consistent with newer sizing accounting). Both q_len
crash-class reproducer shapes complete with the engine alive on
FULL_AND_PIECEWISE (max_tokens=8 truncation; 8,201-token chunk-boundary
prompt answered correctly). FULL vs PIECEWISE outputs token-exact including
full reasoning (`laguna-ii-{FULL,PIECEWISE}.json`). Decode 30.1 vs 22.2
tok/s cc1 (+36%: FULL graphs engaged). Zero error signatures both nodes.
Old GG-era reference outputs are lineage-bound and were not used; the
internal FULL-vs-PIECEWISE comparison is the graph-mode invariant. Bench and
soak deferred by prioritization. Pair re-parked, then reused for DS4.

## DS4 qualification (PASS, dusty/kirby with the production snapshot 9e165c30)

- Boots clean at 262,144 (KV 986,047) and 524,288 (KV 1,029,752, 1.96x).
- Both served names; arithmetic exact; `tool_choice=required` call correct
  (exercises the II tool-grammar stack #294-296/#302).
- **#298 dispatch-race probe**: 0/200 + 0/800-soak corruption-signature hits
  under 3-stream load at width 6. Sustained-load soak clean; zero error
  signatures on both nodes.
- Decode three-run means 40.9/61.3/91.6 tok/s (cc1/2/4) = parity with the
  same-pair r33 reference 42.9/61.2/90.0 (inside the +/-7% boot band).
- Needle at the exact 524,288 cap: identical 519,440-token prompt sizing as
  the r33/rusty run (tokenizer parity across lineages), exact retrieval,
  prefill 349s vs 353s (parity). Cap+1 rejection clean.
- The pair was left serving as a candidate canary (`dusty:8000`).

Exposure measurement on r33 (same pair, production image + env): 0/200
corruption-signature hits at width 6 - reassuring, not proof of absence
(short-prompt trigger shape; upstream's failing control used long-prompt
chunk tails at C4).

## Methodology finding: DSpark-probabilistic profiles are nondeterministic

DS4 with `draft_sample_method=probabilistic` produces different outputs at
temperature 0 **even solo** (verified three ways). Token-exact gates are
therefore impossible on DS4 profiles; corruption-signature or
distribution-based gates only. Laguna's token-exact gates remain valid (its
path is deterministic at temp 0). Recorded so nobody re-derives this.

## Artifacts

`spark/ii-r10/`: candidate DS4 runner (c4 pin, empty-value opt-out),
`probe-298-dispatch-race.py` (signature watchdog), `decode-cells-probe.py`,
`deep-ctx-needle-probe.py`, `laguna-ii-qual.py`, `nccl-ab-node.sh`,
`nccl-matrix-driver.sh`, `nccl-matrix-analyze.py`, this record.
blackwell-llm-docker branch `spark/sm121-arm64-ii-r10`:
`Dockerfile.ii-spark-sm121-cu133-torch213-base`,
`Dockerfile.deepseek-ii-r10-spark-sm121`, `build-ii-spark-sm121-base.sh`,
`build-ii-r10-spark-sm121.sh`,
`patches/releases/infernal-invocation-r10-spark/vllm-spark-overlay.patch`,
EXL3 guards patch + oracle/GEMM gates carried from r33.

## GLM-on-II attempt (2026-08-14/15): NOT QUALIFIED, line mismatch

First-ever GLM-5.2 serving measurement on the II line (upstream's II GLM
runtime is compose+contract-gated only; no serving receipt exists on any
hardware). Seven boot attempts on the cluster produced a decode-capable but
prefill-blocked service:

- PASS: boot (after fixes below), coherence exact, width-4 corruption probe
  0/200 under 3-stream load, decode 3-run means 23.1/39.6/61.3 (cc1/2/4;
  cc1 within ~4% of the r33 records), KV pool 353,152 via 9 GiB pin
  (exceeds r33 production's 309-325k), PYNCCL healthy for TP and DCP groups.
- BLOCKED: uncached 8k prefill 169-299 tok/s vs r33's 767-798 (ladder:
  r33-era EXL3 envs 169; II defaults 253; arena capacity 3072: 299). A 64k
  prompt exceeded the sample_tokens RPC timeout and killed the engine. The
  II runtime routes the 3.42bpw checkpoint through the R7-era mixed-Trellis
  path (per-layer K3/K4 tier plans), whose prefill is unproven on SM121 -
  and unqualified upstream on any hardware.

Candidate-runner findings with permanent value (all fixed in
`run-glm52-ii-r10-tp4-node.sh`):
1. The r33 NCCL pins (`LD_PRELOAD`/`VLLM_NCCL_SO_PATH` at the r33 in-image
   path) silently empty vLLM's allreduce backend list on II - cross-node DCP
   would have shipped broken if the boot had not failed on memory first.
2. II validates `gpu_memory_utilization` against free-at-start in
   `init_device` REGARDLESS of `kv_cache_memory_bytes` (semantics change vs
   r33); these hosts idle at ~107 GiB free (~103 at check time after the
   worker's own context), below 0.86's 104.65 requirement. Working boot
   contract: `GPU_MEM=0.84` + `KV_CACHE_MEMORY_BYTES=9663676416` (pin
   overrides pool sizing; verified in source).
3. r33-era EXL3 tuning envs must not be carried to II (same names, new
   semantics under the mixed-Trellis runtime); the runner now passes them
   only when explicitly set.

Hardware control (2026-08-15, immediately after the restore, same nodes):
r33 prefill 803 tok/s @8k and 886 @64k (records 767-798 / 762-793), decode
cc1 23.7 (record 23.7-24.3), KV 316,928 (band 309-325k). The same hardware
that measured 169-299 tok/s II prefill minutes earlier is on-record under
r33: the deficit is a property of the II r10 GLM runtime path, not silicon,
links, or thermals. Note: upstream qualified GLM-on-II (r11: NVFP4 TP8 +
EXL3 R7 TP4/DCP1, x86) while this window was in progress; our r10 findings
need a re-test against r11+ before upstream delivery.

Disposition: GLM production restored on r33 (bare runner defaults). GLM-II
prefill findings + the runner findings queue for the upstream report. The
TP4 NCCL c4-pin decode A/B on GLM was not reached (open item, GG line).

## Open items

- GLM cluster qualification window (user-scheduled): EXL3 boot
  (ONLINE_QUANT=none), width-4 probe, DCP/#301 coverage, TP4 NCCL grid,
  bench vs r33 records, soak. Optional: first working KLD capture (II
  carries a #258-equivalent).
- Ceremony (both repos) and per-deployment cutover decisions.
- Laguna bench/soak pass on II (deferred).
- Upstream report additions: NCCL Turin over-channeling on 2-rank rails,
  buildah ARG/ENV divergence, arm64 NGC base env drift, spcx plugin
  inheritance, #298 exposure data, nondeterminism note.

## Correction (2026-08-15): encoder-bug finding retracted

The "exhaustive codebook oracle mismatch for the 3INST/MCG codebooks"
rationale cited by this record's overlay description (and embedded in the
II image's guard error message) is RETRACTED: the oracle harness passed
both codebook keys into `quantize_tiles`, whose selection semantics at the
pinned commit are key-PRESENCE, so all three arms encoded with a single
codebook. No encoder defect was ever established on any platform. The
DECODE oracle results (bit-exact, all indices, all codebooks, both
platforms) are unaffected. Consequences: the II image's fail-closed guard
is administratively conservative but its message text is stale (text debt
for the next II respin); the r34-spark program removes the guard entirely
and gates online K6 on a corrected oracle plus a production-path pipeline
gate (see the r34-spark record).
