# Repository guidance

For upstream-development checks, release selection, or a new Spark image candidate,
read [the upstream release workflow](docs/upstream-release-workflow.md) first.
It explains the publishing channels, source/recipe identities, and qualification
boundaries. Treat its dated reports as historical evidence; refresh mutable refs
and live state before acting.

## Use the existing checkouts; create no repositories or directories

Every associated repository already has a checkout under `~/git/`. Work in those:

| Checkout | Remotes to fetch |
| --- | --- |
| `~/git/rtx6kpro` | `upstream` (voipmonitor), `origin` |
| `~/git/vllm` | `lil` (local-inference-lab/vllm), `upstream`, `origin` |
| `~/git/b12x` | `lil` (local-inference-lab/sparkinfer), `upstream`, `voip`, `origin` |
| `~/git/blackwell-llm-docker` | `origin` (voipmonitor), `fork` |
| `~/git/flashinfer`, `~/git/LMCache`, `~/git/cutlass`, `~/git/sglang`, `~/git/spark-vllm-docker`, `~/git/dgx-spark-infra`, `~/git/llm-inference-bench` | their configured remotes |

Rules for any check or investigation:

- Never clone, mirror, or `git init` a repository, and never add a worktree, for a
  check. If a source has no checkout or configured remote, read it remotely
  (`git ls-remote`, `gh api`, the release API) and say so in the report. Ask
  before adding a remote or a checkout.
- Never create new directories outside the task's own tree: nothing new under
  `~/git/`, no sibling `*-artifacts` or `upstream-check-*` folders, no new log or
  receipt folders. Temporary output goes to the session scratch directory;
  anything worth keeping goes into the dated report under `docs/` (inline the
  refs and findings rather than keeping side files).
- A fetch is `git fetch <remote> --no-prune` in the existing checkout. No
  checkout, rebase, merge, reset, or branch change.
- The directories `~/git/rtx6kpro-artifacts/upstream-mirrors/` and
  `~/git/rtx6kpro-artifacts/upstream-check-*` were created by earlier checks
  against this rule. Do not extend that pattern or add to them.
