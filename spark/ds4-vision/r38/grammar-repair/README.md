# R38 scheduler-assertion repair candidate

Status: qualified and serving on rusty/toby. Correctness and post-grid checks
passed; clean separate C4/16K repeat completed and independently reviewed.
The original contaminated grid remains intact. See QUALIFICATION.md.

Final Docker-v2 image:
`ab3ed5285a81c9fd4df3c9001021c420368e7667177470aa2e055420f89ecdcf`.

The user requested restoration of R38 after rejecting JJ-main's KV-capacity
loss, and a narrow repair of the recurring accepted-draft scheduler assertion.
The user approved the rusty/toby maintenance window. Both R38 containers were
stopped gracefully worker-first and retained. Qwen, GLM and nous are untouched.
Do not run GPU tests or builds concurrently with serving.

## Identity and scope

- Serving R38 image: `ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5`.
- vLLM base: `66c293578412417476f842c1da5805d3a3d959a8`.
- Composed R38 tree declared in `spark/glm53/r38-spark/source.lock.json`:
  `077347fdeab296404d0b0cb297d316176acc635a`.
- Existing PR756 overlay: `27f0745fc11f2346bacb646c2d4e1864f6dacd89`.
  Confirmed in the restored container: AsyncOutput clones verified draft counts
  before copying them to the CPU. Preserve it; do not apply it twice.
- Proposed sole new upstream commit: `8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368`.
- `upstream.patch` SHA-256:
  `e14a19f9441a0eeb305abe0bfe9fccc8d4520354973b2eea98bd365ae05d784c`.

The patch changes five runtime Python files and two test files. It propagates
grammar-invalid suffix counts to the GPU rejection sampler, preserves the
scheduler invariant, and adds counts to the assertion message. No native,
B12X, loader, startup-memory, or launcher change is proposed. A clean apply
against the base commit was checked locally. The declared composed tree is not
available as an object in this checkout; do not confuse a base-only check with
verification of the installed runtime.
The full patch subsequently passed `git apply --check` inside the actual
restored R38 container on rusty. This command was read-only and changed no
serving files.

## Required failing-before/passing-after gate

Use the upstream GPU test
`tests/v1/spec_decode/test_mtp_structured_output.py::test_gpu_sampler_rejects_drafts_after_grammar_termination`.
It has 32 cases spanning valid/verified draft counts and temperature. The old
worker API cannot receive the fifth argument. The baseline-only test adapter
`test_r38_baseline_grammar.py` copies the upstream test module and removes
exactly that argument line. This also removes the access to the missing
GrammarOutput field, so no getattr shim is needed. No sampled-length or
input-token assertion is weakened. Its SHA-256 is
`992a9e3bc23d7f07c19e497ce2a0a77599d18896e880a87ac80e8fb4f263f0b6`.
The baseline GPU run produced exactly 12 semantic failures and 20 passes, with
no errors or skips. All failures were the unchanged expected-input assertion.
The built derivative passed the original upstream 32-case GPU test unchanged,
the two sharding cases, both PR756 tests, and the native RMS-norm smoke.
The final scheduler bookkeeping gate initially failed on a missing offline OPT
fixture. After staging its config/tokenizer, all 37 regression cases passed,
as did the native smoke. Test fixture snapshots contain no model weights and
must not be mistaken for complete model downloads.
Also retain batch-sharding, scheduler-bookkeeping and existing PR756 gates.

This gate needs a CUDA device and the tokenizer used by the upstream fixture.
Read-only metadata verification on restored rusty found xgrammar 0.2.5,
llguidance 1.7.6 and pytest 8.4.1. No gpt2 snapshot was found in the mounted HF
cache; stage and pin its tokenizer assets before the GPU window.
Do not report a skipped collection as a pass. Invalid fixture-setup attempts
are retained separately from the semantic baseline receipt.
The original production request body was not captured, so even a semantic
red/green result establishes the known grammar mechanism, not attribution of
the historical crash. Keep that distinction through serving requalification.

The composed derivative tree is `6b270cd2ef83746bd429d975f8dab942a58f2615`,
package tree `319b5bf8cc5d3c2bb0d66b956764df7751db1834`.
`prepare.py` independently reconstructs R38 and retains the result under
`refs/spark/ds4-r38-grammar/tree` in the existing vLLM checkout without changing
working files or branches. `install.py` verifies the seven before/after file
hashes and that every other source-tree file and build product is unchanged.
The authoritative derivative lock is `/opt/ds4-r38-grammar/source.lock.json`;
the inherited `/opt/glm53-flash/source.lock` describes the base only.

`build.sh` publishes the serving tag only after build/GPU gates, not after model
qualification. A protected build tag keeps failed candidates from nightly prune.
`prepare-runtime.py` preserves the R38 runner with four identity-only changes.
The standard DS4 qualification and benchmark still must pass before promotion.

The first build defaulted to OCI. Docker-archive transport preserved its image
ID but changed the manifest type, so the transfer gate refused it and no model
was launched. `build.sh` now explicitly builds Docker v2 and asserts that format;
the final image passed all gates again. Rejected-format receipts are retained
under `receipts/build-oci/` and `receipts/transfer-oci/`. No conversion is part of
the accepted build/transfer workflow.
