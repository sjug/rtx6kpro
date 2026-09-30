# Build and window status

User authorized pausing Qwen on dusty/kirby, building on dusty after Claude's
review, and qualifying Qwen MTP3 in the same window. GLM and DS4 Vision are not
part of this window.

Claude independently reviewed the initial kit, then cleared the corrected build,
runner, transfer and qualification kit. The host dry run additionally found that
dusty provides `python3`, not `python`; host build calls now use `python3`.

The previous Qwen containers were stopped gracefully, worker first, and retained:
kirby `df1c7b8b19cd`, dusty `1372928bdb6b`, image
`6633e678fee74f5e1060290a01df812d34d10edcf30e34dd1c9bf7db88260ef3`.
They were the current JJ-main evaluation service, not R32. Before-stop inspections
are retained under `receipts/window-open/`. No rollback container was removed.

## First build: gate rejected, no serving tag

Receipt `20260923T012448Z-11803`, candidate `b3a72bee...`.
Assembly and the full tracked-source/package-tree/native-reuse audit passed.
The inherited native verifier then rejected `.claude/skills/ci-fails-buildkite`,
a tracked symlink to a directory. These links exist in both old and new sources;
the old payload manifest silently omitted them, while the new Git-derived
manifest includes them. The verifier incorrectly required a regular file.

The correction checks SHA256 of the link text for symlinks, matching Git blob
semantics and the installer/identity verifier. Regular-file and native checks
are unchanged. Claude reviewed and cleared the focused correction. Twelve local
tests pass, including a dangling-link identity check. No kernel or serving
setting changed because of this gate repair.

## Corrected build: all build gates passed

Receipt `20260923T012848Z-15893`, candidate
`88867036403cb3c9bc3026eb8c907ad64bc1af9c28d316d22b670dde4f561152`.
Durable unit `karmic-main-build-20260922b.service` on dusty. Full logs and native
receipts live under its task directory's `build-receipts/`; the normal image tag
is not assigned until all gates pass. Retention tags protect both candidates.

Source/native checks, 12 FlashKDA checkpoint cases, both live launcher preflights,
256 inherited regression cases, 26 worker cases, 20 new memory cases and the
GLM draft-head numerical check all passed. The 314 pytest cases rejected skips
and count mismatches. Draft-head relative RMSE was 0.094867/0.095540/0.095373 for
rows 1/4/32; cosine was 0.995507/0.995440/0.995457.

The service exited zero and published
`localhost/voipmonitor/vllm:karmic-main-spark-sm121`. Inspect records a size of
37,236,121,028 bytes and 96 layers. Build receipts are mirrored to this local task
tree. The Docker archive transfer over the switched 200G link completed, and
kirby verified both archive SHA256 and image ID. Qwen is running on both nodes.
Its 18 semantic checks, MTP3 boundary/131K/262K retrieval, acceptance,
padded-transition, prefix-reuse and head-of-line checks passed. The standard
15-cell grid finished with no cell flags; the local durable unit
`karmic-main-qwen-qualification-20260922.service` exited zero. Results and
startup/runtime health are in `QUALIFICATION.md`. The performance tradeoff
(higher decode throughput, slower prefill versus R32) is not a promotion.

Read-only inventory confirms GLM's R38 head on sparky and DS4 Vision's R38p head
on rusty remain running. The benchmark source is untouched; its pre-existing
working-tree diff and SHA256s were retained in `receipts/window-open/`.
