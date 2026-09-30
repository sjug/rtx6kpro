# GLM Karmic-main upstream-default qualification kit

Executed September 23, 2026 after independent review and an authorized GLM-only
window. The first attempt failed during MTP MXFP8 Marlin conversion because
direct Python entry bypassed NVIDIA's shell initialization and loaded the host
CUDA 13.0 library instead of the image's CUDA 13.4 compatibility library.
An isolated repack test reproduced and resolved it by changing library selection.
The corrected Bash-entry runner is under qualification in a reopened approved
window. Image and model policy are unchanged. See EXECUTION.md.

Use the existing September 23 wide Karmic-main image
`1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc`,
not the earlier HC-off Qwen backport. Its source lock is ../qsa865/runtime.lock.json.
It contains vLLM e77be22511 including GLM boundary fix #860, B12X 10a553ef,
and Qwen PR865, which does not change GLM's attention path. Native gates passed;
GLM serving remains unqualified. B12X master subsequently added cache-artifact
validation 2fca4df8; that change is not in this image. This is an existing-image
qualification, not a claim to build the newest source tips.

prepare.py archives the pinned existing build checkout's runtime at c37f4a0,
resolves its actual current GLM profile in an empty environment, and records
every source digest, resolved argument and environment variable. There is no
hand-copied model-default block. The common/model defaults checked in the prior
reply remain unchanged; newer cache-persistence recipe changes are included in
the resolver. No external cache service is enabled.

## Explicit deployment differences

Only three native option overrides: pinned local checkpoint revision 46aaae8a,
the existing GLM-5.3-Flash client alias, and UMA utilization 0.85 instead of 0.93.
TP4/DCP1 match defaults. Add four-host rank/master/headless addressing, SM121
architecture and the existing switched-fabric RoCEnante/NCCL contract. Replace
the NGC x86 spcx network plugin with the current Spark setting none.
No channel pin, public-NIC transfer, CPU/disk KV cache or fixed KV allocation.

The requested upstream defaults include more than the previously discussed five:
32 sequences and capture256 (versus current 8/capture32), FULL_AND_PIECEWISE
graphs, NVFP4 draft head (rather than BF16), B12X KDA prefill, request_boundaries,
prefill scheduling interval1 and compute share0.4, persistent_grid, L2 prefetch,
split-route settings and MAC224. These are intentionally not silently reverted.
The split/direct-scale/MAC knobs are retained as upstream environment values,
but have no consumers in this pinned B12X tree; do not attribute results to them.
L2 prefetch and persistent_grid are active and previously lost performance on
GB10 in the R26 investigation. They remain enabled for this requested control.
The larger capture envelope
adds admission risk. Qualification must measure them. This is a whole-profile
comparison, not an engine-only performance attribution.

## Review and window gates

1. Run prepare.py, test_profile.py and all four run.py --dry-run renders locally.
2. Independent review before production changes. Verify the default argv is
   accepted by the selected image using GPU-attached help checks on an idle node.
   CPU-only help fails device inference in this image and is not a passed gate.
3. Get a quiet GLM-only window. Qwen and DS4 remain serving. Stop buddy, rocky,
   lucky, then sparky gracefully with podman stop -t 60; retain R38 containers.
4. Transfer the existing Docker-v2 archive over the switched 200G mesh only,
   uncompressed and without conversion. Check disk space first, verify archive
   SHA and image ID on every receiver before launch. Do not load while serving.
5. Sync this kit and compare digests. Run the native/draft-head gates on an idle
   GLM node before boot, entering through Bash as serving does. Require the
   probe_marlin.py pass and exact compatibility-library mapping on every node.
   Start workers before sparky using run.py. Never build
   or run independent GPU tests concurrently with a serving rank.
6. Collect logs, per-rank KV capacity, kernel journals, 1Hz clock/memory/reclaim
   telemetry from before boot. Watch admission closely; no unapproved automatic
   memory/profile changes. Bounded 30-minute first-completion gate, not /v1/models.
7. qualify.sh checks short-pool correctness, unchanged exact-format semantic/tool
   battery x3, identical/distinct bursts, head-of-line and multi-turn behavior,
   frozen prefix pairs/triples and native context through 1M. Old exact-format
   beta failures remain failures if they recur. Review cache/concurrency metrics,
   which are observations and not automatically a PASS from script exit alone.
8. Only after that review, run unchanged run_bench.sh c1/c2/c4 and 8K-128K scouts
   with clear_thinking=false, current profile digest and pinned harness hashes.
   Keep R38 comparison policy difference explicit; the old compare-grids.py
   correctly rejects aligned versus request_boundaries and must not be bypassed
   as if it were a matched-policy engine A/B. No benchmark-repository changes.
9. Review with Claude, show results directly, keep or restore by the approved
   outcome decision. Always clean up one-off observer/systemd units.

No cutover/distribution automation is invoked by preparation or dry-run.
The separate benchmark and deployment commands are intentionally gated behind
the serving window and correctness review, not silently started by this kit.
