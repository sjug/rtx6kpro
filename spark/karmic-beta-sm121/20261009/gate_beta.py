#!/usr/bin/env python3
"""Run beta's own regression tests for the fixes that motivated this build, in-image.

Every pytest session runs in its own process: first a collection pass, then a run that
must pass exactly the collected cases with no skip, xfail or error. Counts are not pinned
in advance (no local vLLM environment); they are printed for the build receipt.
"""
import os
import subprocess
import sys

if not __debug__:
    raise RuntimeError('Regression gates require assertions')
FILES = (
    'tests/models/test_deepseek_v4_1_compressor.py',              # vllm #943 compressor ring writes
    'tests/v1/worker/test_gpu_block_table.py',                    # circular-buffer slot mapping
    'tests/kernels/mamba/test_aligned_state_indices_padding.py',  # NULL_BLOCK_ID padding (638f90a1fc)
    'tests/v1/attention/test_dcp_correct_attn_out.py',            # DCP LSE for non-power-of-two ranks
    'tests/v1/executor/test_multiproc_gather_responses.py',       # failed-rank gather
    'tests/entrypoints/launchers/test_engine_failure_exit.py',    # serve exits 1 on engine failure
    'tests/v1/engine/test_structured_output_draft_handoff.py',    # MTP draft rows stay constrained
    'tests/v1/core/test_prefill_compute_share_scheduler.py',     # stalled prefill lanes (#959/#960)
    'tests/v1/core/test_boundary_admission.py',                  # full-KV preemption (#946)
    'tests/model_executor/layers/test_qwen_gdn_linear_attn.py',          # bound state pool
    'tests/models/qwen4_exp/test_ple_shared_table.py',            # shared PLE table (#961)
    'tests/v1/worker/test_gpu_worker.py::test_weight_loading_split_limit_preserves_runtime_settings',  # #980, d52651e0
    'tests/v1/worker/test_gpu_worker.py::test_weight_loading_split_limit_keeps_non_cuda_defaults',    # #980, d52651e0
    'tests/v1/worker/test_gpu_worker.py::test_startup_plan_revision_rejects_existing_plan',           # plan schema 22 (f4cba696)
    'tests/v1/worker/test_gpu_worker.py::test_serving_thread_count_is_set_before_warmup',             # #988
    'tests/models/qwen4_exp/test_b12x_config.py',                 # PLE storage dtype (#974)
    'tests/model_executor/model_loader/instanttensor_loader/test_weight_utils.py',  # bounded staging (vllm#841)
    'tests/v1/worker/test_mamba_utils.py',                        # tiled aligned-state gather (vllm#858)
    'tests/quantization/test_nvfp4_csf.py',                       # HF-layout CSF reader, direct packed load, scale prefetch (#1002/#1016)
    'tests/models/deepseek_v32/test_mtp_checkpoint_weights.py',   # draft reads only its MTP layers (#1016)
    'tests/model_executor/test_flashinfer_autotune_cache.py',     # tuning revision 22 cache keys
    'tests/model_executor/test_flashinfer_autotune_rank_union.py',  # rank-union marker revision 22
    'tests/v1/executor/test_multiproc_executor_exit.py',          # engine exit on failed rank (vllm#928)
    'tests/v1/worker/test_draft_tokens_handler.py',               # MTP draft rows stay constrained (vllm#931)
    'tests/model_executor/model_loader/test_b12x_loader.py',      # native B12X readers from their defining modules
)
# Cases that need the network (unpinned Hub downloads); deselected so acceptance is offline.
# tests/parser/test_glm47_exact_history.py (vllm#932) is not run: it loads zai-org/GLM-4.7 and
# Qwen/Qwen3-32B tokenizers from the Hub without a revision.
DESELECT = {
    'tests/model_executor/model_loader/instanttensor_loader/test_weight_utils.py': (
        'test_instanttensor_model_loader',),  # downloads openai-community/gpt2
    # vllm#858's merge left these tests stale against beta's own code (they fail on any hardware):
    # the aligned-state cases still expect -1 padding, but beta pads with NULL_BLOCK_ID (638f90a1,
    # gated by test_aligned_state_indices_padding.py); the copy-function cases pass the whole
    # _COPY_FUNCS dict where a per-type tuple (_DEFAULT_COPY_FUNCS) is meant. The single-row
    # aligned-state cases, including 69-group tiling, still run.
    # c67ff04a49 (HF-layout CSF): nvfp4_csf checkpoints no longer have a serving directory with a
    # checkpoint_root, so resolve_ple_embedding_dtype probes the root only for mxfp4_csf; the test's
    # nvfp4_csf parameter still expects the probe and fails on any hardware (2026-10-09 build 3:
    # 42 of 43 passed). The mxfp4_csf case, which our DS4 paths use, still runs.
    'tests/models/qwen4_exp/test_b12x_config.py': (
        'test_omitted_ple_dtype_is_resolved_from_the_csf_checkpoint_root[nvfp4_csf]',),
    # FlashInfer's B12X ships without the standalone b12x.integration.vllm plugin package; the
    # test's register_plugin case importorskips it (build 5: skipped, the gate forbids skips).
    'tests/model_executor/model_loader/test_b12x_loader.py': (
        'test_load_format_selects_native_b12x_with_or_without_plugin[True]',),
    'tests/v1/worker/test_mamba_utils.py': (
        'test_aligned_state_indices_graph_replay_masks_padding_and_refreshes_blocks[4-1-2]',
        'test_aligned_state_indices_graph_replay_masks_padding_and_refreshes_blocks[4-1-69]',
        'test_aligned_state_indices_graph_replay_masks_padding_and_refreshes_blocks[4-8-2]',
        'test_aligned_state_indices_graph_replay_masks_padding_and_refreshes_blocks[4-8-69]',
        'test_aligned_state_indices_graph_replay_masks_padding_and_refreshes_blocks[33-1-2]',
        'test_aligned_state_indices_graph_replay_masks_padding_and_refreshes_blocks[33-1-69]',
        'test_aligned_state_indices_graph_replay_masks_padding_and_refreshes_blocks[33-8-2]',
        'test_aligned_state_indices_graph_replay_masks_padding_and_refreshes_blocks[33-8-69]',
        'test_discovers_heterogeneous_mamba_uniform_group',
        'test_heterogeneous_copy_capacity_is_checked_before_writes',
        'test_populates_heterogeneous_mamba_metadata_with_group_association',
        'test_resolves_legacy_single_type_mamba_copy_api'),
}


def session(path, plugins, *extra):
    import pytest
    os.chdir('/opt/jovian-judgement/vllm')
    deselect = [arg for name in DESELECT.get(path, ()) for arg in ('--deselect', f'{path}::{name}')]
    return pytest.main(['-p', 'no:cacheprovider', '--confcutdir=' + path.rsplit('/', 1)[0], *deselect, *extra, path],
                       plugins=plugins)


def child(mode, index, expected=None):
    path = FILES[index]
    if mode == 'collect':
        class Counter:
            count = 0

            def pytest_collection_finish(self, session):
                Counter.count = len(session.items)  # after --deselect
        code = session(path, [Counter()], '-q', '--collect-only')
        if code != 0 or Counter.count == 0:
            raise RuntimeError(f'Collection failed or empty: {path} exit={code} count={Counter.count}')
        print(f'COLLECTED {Counter.count}', flush=True)
        return
    sys.path.insert(0, '/gate/inherited/tests')
    from run_r38_regressions import RequiredCases
    gate = RequiredCases(expected)
    gate.verify(session(path, [gate, __import__('workspace_fixture')], '-s', '-vv'))
    print(f'KARMIC-BETA-CASE-PASS {path} count={expected}', flush=True)


def main():
    total = 0
    for index, path in enumerate(FILES):
        out = subprocess.run([sys.executable, __file__, 'collect', str(index)],
                             capture_output=True, text=True)
        if out.returncode:
            raise RuntimeError(f'Collection failed for {path}:\n{out.stdout}\n{out.stderr}')
        lines = [line for line in out.stdout.splitlines() if line.startswith('COLLECTED ')]
        if len(lines) != 1:
            raise RuntimeError(f'No collection count for {path}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}')
        count = int(lines[0].split()[1])
        subprocess.run([sys.executable, __file__, 'run', str(index), str(count)], check=True)
        total += count
    print(f'KARMIC-BETA-GATE-PASS files={len(FILES)} cases={total}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        mode, index = sys.argv[1], int(sys.argv[2])
        if mode not in ('collect', 'run'):
            raise RuntimeError(f'Unknown mode: {mode}')
        child(mode, index, int(sys.argv[3]) if mode == 'run' else None)
