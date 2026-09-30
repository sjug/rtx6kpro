# Upstream release discovery and Spark candidate workflow

Last reviewed: **2026-09-20**. This is a discovery procedure and handoff, not a
build lock or deployment authorization. Update this guide when the publishing
process changes; put changing pins and findings in dated reports.

## Current handoff

The absence of a numbered release after JJ R38 does **not** mean publication
stopped. At the last check, upstream published date/hash-addressed images through
Jovian Judgement main/beta and Karmic Kraken main/beta channels in
`ghcr.io/local-inference-lab/vllm`.

Start with these reports for the last verified baseline:

- [September 20 summary and fetch scope](upstream-check-20260920.md).
- [Publication identities, digests and architecture](upstream-publication-20260920.md).
- [Model-specific source changes and evidence](upstream-source-delta-20260920.md).

The recommended next investigation was a pinned Karmic beta ARM64/SM121 build
candidate, not a production cutover. The inspected public artifacts were
x86_64/SM120, CUDA 13.4/PyTorch 2.14; our R38 Spark foundation was CUDA 13.3/
PyTorch 2.13. Native reuse across that change was not qualified.

Qwen's R38 prefill deficit and DS4.1's 524K qualification failure were not closed
by this review. New GDN/QSA changes justify a Qwen comparison, not a claim that
the deficit is fixed. QSA stable selection is not the DS4.1 DSA top-512 path.
Recheck actual serving state separately; these research notes are not a fleet
inventory.

## 1. Establish the comparison baseline

Inspect the working branch, status, remotes, and the previous dated report before
fetching. Spark work belongs on `spark`; `master` is the upstream documentation
line. A fetch does not require checking out, rebasing, merging or resetting either
branch. Preserve unrelated work and the separate benchmark repository.

Record both the prior report's pins and the newly fetched tips. Compare against
those pins, not just commits newer than a wall-clock date: integration branches
can contain older authored commits, rebases and duplicated cherry-picks.

Done when the baseline, current refs, and any unavailable repositories are named.

## 2. Follow the publishing sources, not just the wiki

The repository map at the last review was:

| Role | Source to inspect |
| --- | --- |
| Runbooks and measurement records | `voipmonitor/rtx6kpro`, `master` |
| Image assembly, channels, dependency pins | `local-inference-lab/blackwell-llm-docker` |
| Serving fork | `local-inference-lab/vllm` |
| Kernel library | `local-inference-lab/b12x`, also reached through the configured `sparkinfer` URL |
| Component publishers | Local Inference Lab FlashInfer, LMCache, InstantTensor and nccl-canonical repositories |
| Native dependencies | URLs and exact commits declared by the selected recipe, including FlashKDA and CUTLASS |

Inspect actual remote URLs and redirects rather than assuming a local directory
or remote name identifies the publisher. Where fetching is authorized, fetch
configured remotes without pruning or changing worktrees. Record failed fetches
explicitly. Use the existing checkouts under `~/git/` only: do not clone, mirror
or add repositories, and do not create new directories for logs or receipts (see
`AGENTS.md`). Read sources without a checkout remotely (`git ls-remote`, `gh api`)
and put the refs and findings in the dated report.

Read the build repository's `tools/jovian_wheel_runtime/community-channel.json`
at the selected recipe commit. At this review its source mapping was:

| Channel | vLLM branch | B12X branch |
| --- | --- | --- |
| Jovian main | `dev/jovian-judgement` | `master` |
| Jovian beta | `integration/beta` | `integration/beta` |
| Karmic main | `dev/karmic-kraken` | `master` |
| Karmic beta | `integration/karmic-kraken-beta` | `integration/karmic-kraken-beta` |

Refresh this mapping from source each time. Check the build repository's GitHub
releases and their `container-release.json`, assembly and runtime manifests.
Use the release API when a rendered page is stale or incomplete. Branch tips can
be ahead of the latest successfully built component wheels.

Done when each relevant channel has a publication timestamp, recipe commit,
component source commits, image digest, architecture and stated gate scope.

## 3. Separate changes and evidence

- **Source change:** compare frozen vLLM/B12X/component commits and actual call
  paths. Distinguish ordinary development, integration-only and unmerged work.
- **Recipe change:** compare assembly inputs and launch defaults. A new image can
  have identical engine sources. September 20's beta republish changed GLM MTP
  and Qwen vision defaults without changing its six component source pins.
- **Publication:** a date/hash tag is source-addressed, not guaranteed immutable.
  Pin the registry digest. Main versus beta is source selection, not a readiness
  ranking.
- **Qualification:** native smoke, component tests, model correctness and serving
  performance are separate claims. Read the exact scope even when a receipt says
  `qualified`; inspect model evidence separately from generic release notes.
- **Applicability:** record hardware, topology, checkpoint revision, context,
  speculation, cache policy and sampling. RTX PRO 6000 measurements and a preset
  named "Spark" do not establish GB10 qualification. Separate output tok/s from
  verifier steps/s and preserve restart variability.

Done when each recommendation identifies the code involved, evidence supporting
it, and the local validation still missing. Summaries and commit titles are leads,
not sufficient evidence of an active-path fix.

## 4. Select a candidate without changing the serving contract

Pin one coherent composition and inventory its native ABI/architecture before
authoring a build lock. Reuse prior build safeguards, but revalidate their inputs.
Preserve model-specific launcher settings rather than copying workstation tuning
or newly changed generic defaults wholesale.

Keep transport, loader, checkpoint policy and quantization experiments separate
from the source comparison. In particular, new request-boundary cache behavior
does not automatically apply to our explicitly aligned profiles.

A research/fetch request does not authorize a serving interruption. Obtain the
required window before building on an occupied Spark or restarting production.
For qualification, correctness precedes timing; use the existing standard
`llm-inference-bench/run_bench.sh` workflow without editing that separate repo
unless explicitly requested. Bulk node transfers use the 200G fabric only.

## 5. Close the check

Save a dated report with source links, fetched revisions, failures, architecture
limits, model-specific recommendations and unresolved qualification findings.
Present the useful results directly to the user, not just a file link. A release
check is complete when that report distinguishes what changed, what is usable,
what needs testing, and what was actually done. Commit or publish only within
the user's requested scope.
