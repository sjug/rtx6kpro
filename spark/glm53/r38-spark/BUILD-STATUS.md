# R38 Spark build status

The chronological entries below are historical build receipts. The final
serving decisions are in `ROLLOUT.md`: GLM and DSv4 Vision qualified on R38;
Qwen retained on R32. Build receipts are now retained in Git, including the
exact original build-kit archive. Committing the kit does not rewrite the
image's explicitly dirty build provenance or its recipe ancestry label.

## 2026-09-15: source preparation

Scope: build R38 plus pinned vLLM PR #756 on rusty while rusty/toby are
idle. Do not interrupt the restored R32 Qwen or GLM deployments.

Frozen candidate trees, replayed independently from the R32 base:

- vLLM: `077347fdeab296404d0b0cb297d316176acc635a`
- vLLM package: `2d387573ba313b2ac667f6ea8d3f2c5b5cb36125`
- B12X: `6abad73444018f7ce1f0bdc3f0992649a4c21722`
- LMCache: `5a88a1ea9d2627c76288d056e7193f7454669b64`
- Source lock SHA-256: `9a0dc0bb8eac7d3ccf5778765146651d35982810d8fb1ed9e39832f855247be0`

All 36 offline tests pass, both launcher renders preserve the qualified
profiles, the build dry run passes, and shellcheck passes. These are static
checks, not GPU acceptance or serving qualification.

The three reviewed vLLM native-build/versioning transitions are pinned at
both endpoints. Native kernel source and existing binary manifests remain
the reuse authority. FlashInfer must be rebuilt because the R37 component
was automatically pruned. Its effective compiler policy is 20 Ninja jobs,
one NVCC thread per job. Completed component/candidate images receive
non-serving retention tags immediately; the serving tag is gated.

At 11:54 EDT, both build nodes were idle, with about 117 GiB MemAvailable
and the qualified R32 base present. No build had started at that check.
The R37 kit is preserved. R38 work and receipts live on persistent disk.

## 12:04 EDT: build running

The staged kit reproduced all 36 tests and the dry-run recipe digest
`d5d506f52f28c714aa18dad547cdc16fc46b768a0c911db97131b03d29d0f8a2`
on rusty. Idle checks passed again immediately before launch.

- Host directory: `/home/jugs/git/bld-jj-r38-spark`
- Component unit: `jj-r38-flashinfer-build.service`
- Component receipt: `build-receipts/flashinfer-20260915T160431Z-59162`
- Runtime continuation: `jj-r38-runtime-build.service`, bound to that exact
  component receipt. It requires success, rechecks idle state and component
  provenance, then builds and gates the runtime. It does not launch serving.
- Recipe commit: `f7073c09067a9145ac7d09918ade6b32a4c05925`, explicitly dirty.
  This commit is workspace ancestry, not a claim that R38 files are committed.

The compiler output confirms `compute_121a`, `sm_121a` and `--threads=1`.
The AOT cache has 3,406 compile/link steps. At the first compile check,
MemAvailable was 111 GiB with no swap used. No build outcome is claimed yet.

## 12:31 EDT: FlashInfer component passed

Component completed at `2026-09-15T16:31:54Z`, about 27 minutes after launch:

- ID: `45614cd3d6e8821d5ebf11e5417a77e052ae59ad3a677ccacf3923f51e9a6644`
- Retention tag: `localhost/voipmonitor/build-components:jj-r38-flashinfer-20260915T160431Z-59162`
- Exit status: 0. Both wheels, source/submodule pins, recipe copies,
  digests and image inspection are retained in its receipt.

The monitored compiler RSS briefly reached about 92 GiB, with about 23 GiB
MemAvailable at that sample. It recovered without intervention. All observed
samples had zero swap use; the build log had no compiler-failure markers and
the kernel error journal had no entries for the build interval.

Runtime assembly began immediately, using that exact component and the
unchanged staged kit. Receipt: `build-receipts/runtime-20260915T163156Z-64815`.
Runtime gates and image publication are still pending at this update.

## 12:40 EDT: BUILD-OK

The complete runtime gate exited 0 at `2026-09-15T16:40:36Z`.

- Image: `localhost/voipmonitor/vllm:jj-r38-spark-sm121`
- ID: `ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5`
- Candidate retention tag: `localhost/voipmonitor/build-candidates:jj-r38-runtime-20260915T163156Z-64815`
- Size: 35,870,037,422 bytes, 214 layers (Podman inspection, not archive size).
- Lock SHA-256: `9a0dc0bb8eac7d3ccf5778765146651d35982810d8fb1ed9e39832f855247be0`
- Cache fingerprint: `cu133-torch213-jj-r38-sm121-ea978a65fbbc668ea72b`

All 540 selected pytest cases passed, without selected skips. Collection
preflight, runtime identity, native custom-op execution, dependency-path
checks, coherent NCCL, launcher fail-closed checks and the standalone
draft-head numerical gate also passed. The 540 cases comprise:

| Group | Passed |
| --- | ---: |
| GLM pool-tail matrix | 235 |
| Mamba sparse cleanup | 7 |
| MoE warmup regimes | 1 |
| Pooled indexer tail | 1 |
| Mamba retirement | 7 |
| PR #756 asynchronous verified-count snapshot | 2 |
| Boundary scalar restore and RoCE health | 24 |
| Partial prefix hits, Mamba chunk split and state indices | 154 |
| Scheduler boundary-hit admission | 20 |
| MLA BMM storage | 22 |
| Shared-expert stream ownership | 2 |
| GLM draft-head policy/capability | 6 |
| KDA near-collinear keys | 1 |
| Persistent MXFP8 epilogue stores | 1 |
| FlashKDA checkpoint matrix | 12 |
| MoE FC1 fragment rebinding | 10 |
| Small-tile cooperative MoE | 3 |
| Dependency patch regressions | 4 |
| FlashInfer dual-cache sparse-MLA prefill | 24 |
| Rebuilt LMCache filesystem delete behavior | 4 |

Draft-head relative RMSE was 0.094866 / 0.095540 / 0.095373 at rows
1 / 4 / 32, with cosine 0.995507 / 0.995440 / 0.995457. This numerical
gate does not change the serving launcher's qualified BF16-head default.

Both complete receipts are copied into this directory's ignored
`build-receipts/`. Runtime inspection was saved both before and after gates.
The source kit remains explicitly uncommitted and dirty, not attributed to
the recipe ancestry commit as if that commit contained these files.

Final read-only host check: rusty and toby have no running containers.
Dusty/kirby still run the R32 Qwen containers, and sparky/buddy/rocky/lucky
still run the R32 GLM containers, all at image `74e53e710bef` with unchanged
three-hour uptimes. R38 exists on rusty only. No image distribution, serving
cutover, model qualification, benchmark, cleanup, commit or push occurred.

Next phase: an authorized Qwen qualification window on dusty/kirby, followed
by GLM qualification if the Qwen candidate passes. Build acceptance is not
serving qualification or a performance claim.
