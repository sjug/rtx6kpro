#!/usr/bin/env bash
# One shared selection list for collection preflight and numerical execution.
set -euo pipefail
mode=${1:?Pass collect or run}
case "$mode" in collect|run) ;; *) echo "Invalid gate mode: $mode" >&2; exit 2 ;; esac
gates=/opt/local-inference/r37-tests
collect=()
if [[ $mode == collect ]]; then collect=(--collect-only-count); fi
required() { python "$gates/run_required.py" "${collect[@]}" "$@"; }
cd /opt/jovian-judgement/vllm
python "$gates/run_r37_regressions.py" "${collect[@]}"
required --confcutdir=tests/v1/worker tests/v1/worker/test_boundary_checkpoint_restore_scalars.py tests/v1/worker/test_b12x_roce_health.py
required --confcutdir=tests/v1 tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py tests/v1/core/test_mamba_align_chunk_split.py tests/v1/worker/test_mamba_prefix_state_index.py
required --count 20 --confcutdir=tests/v1/core tests/v1/core/test_scheduler.py -k full_boundary_hit_is_admitted_while_another_request_decodes
required --confcutdir=tests/kernels/attention tests/kernels/attention/test_mla_bmm_storage.py
required --confcutdir=tests/model_executor/layers tests/model_executor/layers/test_shared_experts_stream.py
required --confcutdir=tests/models tests/models/test_glm5next_model.py -k "mtp_draft_head_mode or mtp_nvfp4_draft_head_capability"
required --confcutdir=tests/models/kimi_k3 tests/models/kimi_k3/test_kda.py -k near_collinear_keys_remain_finite
required --confcutdir=/opt/jovian-judgement/b12x/tests/gemm /opt/jovian-judgement/b12x/tests/gemm/test_mxfp8_linear.py -k persistent_ctas_complete_single_stage_epilogue_stores
python "$gates/run_flashkda_gate.py" "${collect[@]}"
if [[ $mode == run ]]; then python "$gates/verify_glm53_nvfp4_draft_head_sm121.py"; fi
cd /opt/jovian-judgement/b12x
export PYTHONPATH=/build/r37:/opt/jovian-judgement/b12x:/opt/jovian-judgement/vllm
required --count 10 --confcutdir=tests tests/moe/test_dynamic_fc1_fragment_rebind.py
required --count 3 --confcutdir=tests tests/moe/test_w4a8_dynamic_kernel.py -k small_tile_parallel_regime_matches_oracle
required --count 4 --confcutdir=/build/r37/inputs/upstream/tests /build/r37/inputs/upstream/tests/test_dependency_python_patches.py
required --count 24 --confcutdir="$gates" "$gates/test_sparse_mla_sm120.py" -k prefill_dsv4_dual
cd /opt/lmcache/source-r37
# The rebuilt CPU extensions belong to the installed distribution. Default
# prepend mode would put this package root on sys.path and import the
# extension-less source tree instead, so import this selection by spec.
required --count 4 --import-mode=importlib --confcutdir=tests/v1/storage_backend tests/v1/storage_backend/test_fs_native_connector.py -k "test_delete_reconciles_missing_file_from_usage_ledger or test_delete_error_does_not_retire_tracked_bytes"
