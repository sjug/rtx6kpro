# DGX Spark kernel 7.0.0-1019 regression: thread review

Reviewed 2026-09-15. All 42 posts returned by the public Discourse post stream were retrieved and read, through post 45 (last posted 2026-09-15 10:06 UTC; post 45 edited 11:59 UTC). Numbers 12, 13 and 36 are absent from the public stream; their contents are unavailable. The initial HTML page exposes only 20 posts, ending at 22. This note summarizes public reports, not locally reproduced results. No Spark hosts were inspected or changed.

## Findings

**There is strong evidence of a kernel-dependent NCCL/RoCE regression, and NVIDIA has published a mitigation.** The affected release is `7.0.0-1019-nvidia`, package `7.0.0-1019.19~24.04.2`. Users repeatedly recover by booting `6.17.0-1032-nvidia` or `6.17.0-1031-nvidia` with their existing workloads. Reports span different containers and models. [Initial A/B](https://forums.developer.nvidia.com/t/383023/1), [rollback confirmation](https://forums.developer.nvidia.com/t/383023/4), [NVIDIA advisory](https://forums.developer.nvidia.com/t/383023/22).

### Mechanism and evidence strength

KHO (kexec handover) is the best-supported explanation for the RDMA registration failure. A contributor reports about 9.3 GiB of scratch pageblocks marked `MIGRATE_CMA` without corresponding CMA-total accounting. Long-term page pinning attempts migration and fails with `ENOMEM`, which propagates through `ib_umem_get()` to `ibv_reg_mr_iova2()`. In a same-image comparison at 108 GiB CUDA pressure, default KHO fails initial registrations while `kho=off` passes 4,096 retained 1 MiB registrations. These are the contributor’s results, not an independently repeated experiment. [Post 28](https://forums.developer.nvidia.com/t/383023/28), [PR590](https://github.com/NVIDIA/NV-Kernels/pull/590).

The early explanation that the machine simply needs a restored 128 MiB CMA region is incomplete. NVIDIA later explains that `cma=128M` effectively disables KHO by preventing its low-memory scratch allocation on Spark. It recommends `kho=off` directly. A separate user reports zero CmaTotal on both good and bad kernels, so that field alone cannot identify the regression. [Post 42](https://forums.developer.nvidia.com/t/383023/42), [post 41](https://forums.developer.nvidia.com/t/383023/41).

Healthy raw RDMA and one successful cold launch do not qualify a system. Repeated launches and NCCL after substantial CUDA allocation expose failures that idle bandwidth tests miss. Reported residency thresholds vary and should not be treated as universal limits. [Post 31](https://forums.developer.nvidia.com/t/383023/31), [post 40](https://forums.developer.nvidia.com/t/383023/40), [post 41](https://forums.developer.nvidia.com/t/383023/41).

### Mitigation and remaining gaps

- NVIDIA first advised deferring the affected update. Its later instructions say to append `kho=off` to the existing `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, preserve other arguments, run `sudo update-grub`, reboot, and check that `/sys/kernel/debug/kho` is absent. These are documented instructions, not actions performed here. [Advisory](https://forums.developer.nvidia.com/t/383023/22), [mitigation](https://forums.developer.nvidia.com/t/383023/43).
- Returning to an installed, known-good 6.17 kernel has the broadest recovery reports. Actual GRUB entries matter: saved entries or mismatched titles can silently select the wrong kernel. Confirm the running kernel after reboot. The attached community script selects the second-highest installed GRUB kernel, which is not necessarily the intended known-good version; it does not install a kernel or hold packages. [Post 15 attachment](https://forums.developer.nvidia.com/t/383023/15), [post 18](https://forums.developer.nvidia.com/t/383023/18), [post 41](https://forums.developer.nvidia.com/t/383023/41).
- TCP fallback has a substantial workload-specific performance cost. The thread does not establish a universal slowdown or explain the reported speculative-acceptance change. [Post 18](https://forums.developer.nvidia.com/t/383023/18).
- A later TP1 report still hangs with `kho=off`, but recovers on 6.17. This leaves a possible additional regression unresolved; the thread does not prove that every driver allocation failure has the KHO cause. [Post 44](https://forums.developer.nvidia.com/t/383023/44).
- The final post concerns Secure Boot rejecting driver modules and mixed packages. Its reported repair is specific to that machine’s driver state. [Post 45](https://forums.developer.nvidia.com/t/383023/45).

### Proposed fix status

The linked [PR590](https://github.com/NVIDIA/NV-Kernels/pull/590) and follow-up [PR591](https://github.com/NVIDIA/NV-Kernels/pull/591) were both open when checked. Maintainer discussion considers a Spark boot parameter versus kernel configuration. The follow-up’s target-branch validation is configuration-only; reported runtime validation belongs to the earlier kernel candidate. No merged or shipped replacement is established by these sources. Recheck their status before relying on this snapshot.

## Every visible post

Each row links directly to its source. Statements below describe what the author reported, not independently established facts.

| Post | Author | Contribution |
| --- | --- | --- |
| [1](https://forums.developer.nvidia.com/t/383023/1) | mlau1 | Initial controlled kernel A/B: DeepSeek Vision TP2 fails RDMA registration on 7.0.0-1019; 6.17.0-1032 passes chat and 47 warmups. Raw RoCE reaches 108 Gb/s. Multiple NCCL versions fail. |
| [2](https://forums.developer.nvidia.com/t/383023/2) | anoop13 | Independent confirmation; kernel rollback resolves the issue. |
| [3](https://forums.developer.nvidia.com/t/383023/3) | hiroshiya | Requests nccl-tests and links NVIDIA’s multi-Spark testing guide. |
| [4](https://forums.developer.nvidia.com/t/383023/4) | chrisb03 | Independent recovery on 6.17.0-1031. |
| [5](https://forums.developer.nvidia.com/t/383023/5) | giles8 | Describes explicit GRUB submenu selection of the installed 6.17 kernel. |
| [6](https://forums.developer.nvidia.com/t/383023/6) | serg.piter | Suggests CMA involvement; reports CmaTotal=0 despite nonzero CmaFree on 7.0. |
| [7](https://forums.developer.nvidia.com/t/383023/7) | OllieJW | Reports NVIDIA driver allocation failure during CUDA graph capture and multi-node NCCL initialization. |
| [8](https://forums.developer.nvidia.com/t/383023/8) | J-Grove | Previously working models fail after updating. |
| [9](https://forums.developer.nvidia.com/t/383023/9) | fidecastro | Another affected user; no independent technical isolation. |
| [10](https://forums.developer.nvidia.com/t/383023/10) | OllieJW | Asks whether saved GRUB entries and changing no-grubmenu.cfg are appropriate. |
| [11](https://forums.developer.nvidia.com/t/383023/11) | kenleo_lucas | Speculation without technical evidence. |
| [14](https://forums.developer.nvidia.com/t/383023/14) | eugr_nv | NVIDIA participant eugr_nv reports internal escalation. |
| [15](https://forums.developer.nvidia.com/t/383023/15) | dbsci | Attaches a community rollback script; no new diagnosis. |
| [16](https://forums.developer.nvidia.com/t/383023/16) | kenleo_lucas | Dashboard update caused failure; explicit GRUB selection and reboot recovered the system. |
| [17](https://forums.developer.nvidia.com/t/383023/17) | hypermac.6502 | Confirms startup recovery after rollback; includes a screenshot. |
| [18](https://forums.developer.nvidia.com/t/383023/18) | gurvann | GX10 TP2 confirmation. Memlock, GID, HCA selection, device exposure, map count and MTU changes did not help. Explicit GRUB path worked; saved entry did not. TCP decode fell from 46-70 to 23 tok/s in this workload. |
| [19](https://forums.developer.nvidia.com/t/383023/19) | OllieJW | Acknowledgement only. |
| [20](https://forums.developer.nvidia.com/t/383023/20) | ForsakenSilver | TP4 ring regression: even 1 MiB AllReduce can fail. Rollback yields approximately 187 Gb/s bus bandwidth and prefill 2,590-2,830 versus 1,430-1,490 tok/s. Same author’s TP2 case showed little performance change. |
| [21](https://forums.developer.nvidia.com/t/383023/21) | james.park4 | Another affected user requests withdrawal of the update. |
| [22](https://forums.developer.nvidia.com/t/383023/22) | NVES | Official NVES advisory: multi-node/RoCE customers should defer 7.0.0-1019 updates, including Dashboard. Investigation ongoing. |
| [23](https://forums.developer.nvidia.com/t/383023/23) | jc2375 | Reports clock limiting helped crashes; no RDMA isolation. |
| [24](https://forums.developer.nvidia.com/t/383023/24) | mashie | Disputes the relevance of the thermal report; offers no additional experiment. |
| [25](https://forums.developer.nvidia.com/t/383023/25) | eugr_nv | NVIDIA requests container identity and reproduction steps. |
| [26](https://forums.developer.nvidia.com/t/383023/26) | pdion | Reproduces with spark-vllm-docker DeepSeek 0731 recipe. |
| [27](https://forums.developer.nvidia.com/t/383023/27) | J-Grove | Reproduces with Vision-Exp recipe and a previously working custom build. |
| [28](https://forums.developer.nvidia.com/t/383023/28) | SloptimistPrime | Contributor traces long-term page-pinning failure to KHO scratch pageblocks and submits PR590. Same-image kho=off A/B and custom default-off kernel pass pressure registration, NCCL and TP2 tests. |
| [29](https://forums.developer.nvidia.com/t/383023/29) | jc2375 | Thermal-report author reiterates timing and ASUS hardware; linkage remains unproven. |
| [30](https://forums.developer.nvidia.com/t/383023/30) | bigun_51 | Custom SGLang DeepSeek V4.1 TP3/EP3 affected. Author explicitly lacks a minimal reproducer and notes custom padding/SM121 changes. |
| [31](https://forums.developer.nvidia.com/t/383023/31) | james.park4 | Detailed Qwen TP2 report: first launch works, subsequent launches fail across both ranks. Substantial free memory remains. Notes CMA config change and proposes cma=128M, not yet tested in this post. |
| [32](https://forums.developer.nvidia.com/t/383023/32) | jordan.mymail | Initial success with cma=128M; benchmarks pending. |
| [33](https://forums.developer.nvidia.com/t/383023/33) | elsaco | Documents CMA default-size change and asks for kernel sources. |
| [34](https://forums.developer.nvidia.com/t/383023/34) | james.park4 | Nontechnical reward speculation and an external image. |
| [35](https://forums.developer.nvidia.com/t/383023/35) | jordan.mymail | cma=128M passes TP2 functional tests and a 380K-token request. Prefill closely matches old kernel; decode 34.2 versus 36.8 tok/s, interpreted by author as within run spread. Does not isolate KHO. |
| [37](https://forums.developer.nvidia.com/t/383023/37) | hiroshiya | Links NVIDIA kernel repository; repeats CMA-size hypothesis. |
| [38](https://forums.developer.nvidia.com/t/383023/38) | hiroshiya | Links documented rationale for disabling default CMA reservation. |
| [39](https://forums.developer.nvidia.com/t/383023/39) | james.park4 | Argues Spark-specific GRUB packaging should have compensated for CMA-default change. Hypothesis predates NVIDIA’s KHO explanation. |
| [40](https://forums.developer.nvidia.com/t/383023/40) | hypermac.6502 | Two consecutive DeepSeek 0731 launch logs: first reaches serving; second fails NCCL during expert-parallel communicator setup. Same container image across nodes. Rollback resolves it. Eight machines owned, but this reproduction uses two. |
| [41](https://forums.developer.nvidia.com/t/383023/41) | ursuciprian | Pressure-controlled 64 MB AllReduce: 7.0 passes at 40/85 GB CUDA residency, varies at 90 GB, fails at 100 GB across three boots. Old kernel passes through 110 GB. Raw RDMA remains healthy. Reports CmaTotal=0 on BOTH kernels. |
| [42](https://forums.developer.nvidia.com/t/383023/42) | eugr_nv | NVIDIA explains cma=128M prevents KHO low-memory scratch reservation on Spark, effectively disabling KHO; recommends kho=off directly. |
| [43](https://forums.developer.nvidia.com/t/383023/43) | eugr_nv | NVIDIA mitigation: append kho=off to existing GRUB_CMDLINE_LINUX_DEFAULT, update-grub, reboot, then check KHO debugfs path is absent. |
| [44](https://forums.developer.nvidia.com/t/383023/44) | Yasuhiko.Ose | Single-node TP1 still hangs with kho=off and driver NV_ERR_NO_MEMORY; rollback to 6.17.0-1032 resolves it. Relationship to the RDMA regression remains unproven. |
| [45](https://forums.developer.nvidia.com/t/383023/45) | Hunlx | Separate Secure Boot/driver-loading problem after update and rollback. Edited report attributes it to unenrolled MOK-signed DKMS modules and mixed driver packages; reports recovery with signed open modules. Not an NCCL reproducer. |
