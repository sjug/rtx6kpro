# DRAFT — do not file

Repository: `local-inference-lab/nccl-canonical`

Proposed title: `[Cleanup] Scope AMD Zen5 eight-GPU graph policy changes to the target topology`

## Summary

Commit `fb6f40999a2a9e63104d4ae4a84118bce61528f8` adds a narrowly gated `ncclMaybePromoteAmdZen5EightGpuRing()` in `src/init.cc`, but three supporting policy changes occur outside that target gate:

- `ncclTopoCompareGraphs()` globally prefers more channels when aggregate bandwidth is equal.
- `ncclTopoCompute()` globally searches asymmetric channel layouts before reducing bandwidth, removing the previous AMD/SYS-only exception logic.
- `AMD_ZEN5_BW` changes from 32 to 40 GB/s for every topology classified as AMD Zen5.

The GB10/aarch64 two-node reproducer behaves the same with the fork and the bit-coherent official `v2.31.2-1` library, so these changes are not the cause of the GLM DCP multi-communicator hang. This is a provenance and blast-radius cleanup, not a hang fix.

## Requested change

Keep the measured eight-GPU Zen5 promotion, but make every non-upstream graph-selection deviation conditional on the same intended platform contract, or split generally applicable changes into separately justified commits with their own cross-platform tests. At minimum, add negative tests proving graph identity is unchanged for aarch64/GB10, non-Zen5 x86, inter-node topologies, and GPU counts other than eight.

The expected result is byte- or field-identical graph selection to upstream NCCL outside the declared AMD Zen5, single-host, eight-GPU topology. This change should remain separate from the NVIDIA multi-communicator report and from any B12X/vLLM integration fix.
