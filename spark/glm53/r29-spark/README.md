# R29 Spark candidate

Status, 2026-09-09 14:39 EDT: corrected R29 passed GLM qualification and remains
serving on sparky/buddy/rocky/lucky. See qualification/GLM-QUALIFICATION.md.
Qwen remains on its previously qualified original R29 image on dusty/kirby.
The corrected image fixes GLM launcher metadata; the two image IDs are recorded
separately. R28 GLM containers and images remain available for rollback.
No production DS4 changes. Implementation is preserved in commit 43b34e2;
qualification and build receipts are archived separately.

## Execution order

1. Replay the published R29 lock, retain the qualified Spark architecture and
   draft-head overlay, and verify the R28 derivation base independently.
2. Pass source contracts and runner renders. Stop kirby's Qwen worker, then
   dusty's head with `podman stop -t 60`. Keep the stopped R28 containers and image.
3. Build on idle dusty. Rebuild `_C_stable_libtorch` for SM121 because R29 changes
   the DS4 output-buffer operator schema. Retain FlashKDA from the pinned R28
   base. Build the new LMCache package with its native gates, but keep it disabled.
4. Run the complete image gate before assigning `jj-r29-spark-sm121`. Save a
   Docker archive, copy it to kirby over the switched 200G fabric without
   compression or conversion, load it and assert identical image IDs.
5. Start Qwen worker before head. First prove meaningful output, semantic/tool
   correctness, MTP boundaries, native context, and recurrent-cache behavior.
   Use the same checkpoint revision and qualified R28 launch settings.
6. Run the existing `llm-inference-bench/run_bench.sh` campaign protocol, then
   compare against R28. Record engine steps, output throughput, acceptance,
   prefill, cached tokens and startup KV capacity separately. Any tuning or auto
   checkpoint-policy trial is a separate arm, not a changed baseline.
7. Only after Qwen passes, stage the GLM candidate and request its cutover window.
   Keep BF16 MTP3, aligned, FlashKDA, RoCEnante, InstantTensor and LMCache-off for
   that matched comparison.

Source preparation uses private indices and writes named `refs/spark/jj-r29/*`
refs in the source repositories, but does not change their branches or worktrees.
Generated bundles must be staged before the build and are digest checked.
The inherited Spark FlashInfer is 1ac6942, not R29's RTX-only 803c4664 artifact.
This retained dependency is explicit in the lock and remains subject to model
qualification. No RTX wheel is substituted into the ARM image.

R28 artifacts remain frozen. New work and receipts stay on persistent disk.
No automatic rollback on successful qualification, no cache deletion, and no
Podman system reset are part of this program.
