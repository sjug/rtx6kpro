"""Only approved adaptations to Luke's frozen four-Spark launch arrays."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NODES = {'dusty': (0, '10.11.11.7'), 'toby': (1, '10.11.11.6'),
         'rusty': (2, '10.11.11.5'), 'kirby': (3, '10.11.11.8')}
NCCL = '/opt/nccl-2.30.7/lib/libnccl.so.2.30.7'
CHUNKING_IMAGE = 'e06df11a8ca18fa514d9f28f67cc691aef296da2eeb22b113a734519853bccd7'
MATCHED_CAPTURE_LOCK = '3eb16ccca36e78ebbf133fb238ed8b0ffc261316c71ada958ab810ef8f07e8b4'
WINDOW_CAPTURE_LOCK = '59100837bbd6fdee5a9c2d689ae44ebd8b1d4b01838f5067dc3cc08706cea63b'
INDEXER_CAPTURE_LOCK = '20ea0e6e920b2f5a7d78cc4ae0c3632b8ab4f140536ba6a58c09427b7bc2520f'
PRECISION_CAPTURE_LOCK = '541ba7997b38b3e9ba2afb286df3f3796beed99def29582200adb3764f198974'
# User-approved NCCL geometry diagnostic, 2026-09-26 (PROPOSED-NCCL-GEOMETRY-20260926.md): the
# pinned upstream batch-invariant NCCL subset only, on the matched window-capture 8192/4096 arms.
# It changes prefill reductions dispatched to PyNCCL, not the small RoCEnante calls.
# Reduction storage remains BF16, so it is a numerical arm:
# never a serving default, never combined with any other capture kind. Within one boot the
# tree is fixed, so cross-grid invariance is the prediction; NCCL builds the tree per boot and
# splits chunks across two complementary trees, so neither cross-boot equality nor complete
# offset independence is promised. NCCL only warns on unsupported values; the driver's
# preflight must fail on such warnings rather than accept a silently different geometry.
NCCL_GEOMETRY = 'tree-simple-1ch'
NCCL_GEOMETRY_ENV = {'NCCL_ALGO': 'allreduce:tree', 'NCCL_PROTO': 'Simple',
                     'NCCL_MIN_NCHANNELS': '1', 'NCCL_MAX_NCHANNELS': '1'}
NCCL_TUNING_UNSET = ['NCCL_PROTO', 'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS']
# User-requested precision diagnostic, 2026-09-26: the same precision-capture image on the matched
# 81389-block 8192/4096 profiles with the ordinary launch NCCL settings (tuning unset list retained,
# no ALGO/PROTO/channel override). Not new NCCL tuning. The explicit selector keeps these boots
# distinct from the tree arm and from earlier non-geometry receipts; precision-capture only.
STANDARD_NCCL_ARM = 'standard-upstream'
# User-authorized ratio-1 compute-mode diagnostic, 2026-09-27 (ds41_ratio1_prepare.py): the precision
# release control/return and the ratio-1 extend BF16 pin variant, each at the release boot's automatic
# 80022-block capacity so attention plan identities match. Diagnostic only, never a serving default.
PRECISION_RELEASE_LOCK = '4b1afffe455d634cbadc82353a64beb3c96bbaf89b595c948c00dbfa3b1f1c32'
RATIO1_LOCK = 'ac3cab6925cf41dedcc596bf26ce8a7071318f1b2d3b57452fb6c5e7e5cafe1d'
RATIO1_BLOCKS = '80022'
# User-authorized selection replay, 2026-09-27: the exact precision release image with the passing
# standard-NCCL 8192 arm's per-rank B12X selections (receipt ...233036Z) at that arm's geometry
# (81389 blocks, threshold 8192), from a separate host cache root. Diagnostic only.
REPLAY_SELECTOR = 'standard8192-233036Z'
REPLAY_CACHE_ROOT = 'git/ds41-r38/karmic-main-20260924/selection-replay-cache-standard8192-233036Z'
# User-authorized selection transplant, 2026-09-27: the same replay geometry and image with the passing
# per-rank selections except exactly the 275 shared records whose config/assignment differ, taken whole
# from the release (seed_selection_transplant.py). Root-cause discrimination only, never a profile.
TRANSPLANT_SELECTOR = 'transplant275-release-into-233036Z'
TRANSPLANT_CACHE_ROOT = 'git/ds41-r38/karmic-main-20260924/selection-transplant-cache-275-233036Z'
# Contingent narrowing, 2026-09-27: only the two mHC 8192-row records (mhc.pre and mhc.pre.expanded)
# from the release, everything else passing, same geometry. Root-cause discrimination only.
MHC8192_SELECTOR = 'transplant2-mhc8192-release-into-233036Z'
MHC8192_CACHE_ROOT = 'git/ds41-r38/karmic-main-20260924/selection-transplant-cache-mhc8192-233036Z'
# One-key split of that pair, 2026-09-27: exactly one of the two mHC 8192-row records from the release,
# everything else passing, same geometry. Root-cause discrimination only, never a profile.
MHC_PRE8192_SELECTOR = 'transplant1-mhc-pre8192-release-into-233036Z'
MHC_PRE8192_CACHE_ROOT = 'git/ds41-r38/karmic-main-20260924/selection-transplant-cache-mhc-pre8192-233036Z'
MHC_EXPANDED8192_SELECTOR = 'transplant1-mhc-expanded8192-release-into-233036Z'
MHC_EXPANDED8192_CACHE_ROOT = 'git/ds41-r38/karmic-main-20260924/selection-transplant-cache-mhc-expanded8192-233036Z'
SELECTION_ROOTS = {REPLAY_SELECTOR: REPLAY_CACHE_ROOT, TRANSPLANT_SELECTOR: TRANSPLANT_CACHE_ROOT,
                   MHC8192_SELECTOR: MHC8192_CACHE_ROOT, MHC_PRE8192_SELECTOR: MHC_PRE8192_CACHE_ROOT,
                   MHC_EXPANDED8192_SELECTOR: MHC_EXPANDED8192_CACHE_ROOT}
# Expanded-mHC capture derivative of the release image (claude-mhc-expanded.lock.json), 2026-09-27:
# copies of the expanded mHC pre inputs and served outputs only, admitted solely on the passing replay
# root or the expanded-only transplant root at the replay geometry. Diagnostic only, never serving.
MHC_EXPANDED_CAPTURE_KIND = 'mhc-expanded-capture'
MHC_EXPANDED_CAPTURE_LOCK = 'aab479ea81fbdf5e96787e6a33493672190df50978d86d655ceb72dadf6ddf5b'
MHC_EXPANDED_CAPTURE_BASE = '1989e16daf38d2966b03a2e2abcb878263fb7d17183c8d6c5b084a51fbed7e8f'
MHC_EXPANDED_CAPTURE_SELECTORS = (REPLAY_SELECTOR, MHC_EXPANDED8192_SELECTOR)


def validate_chunking_image(pin, env):
    diagnostic = pin.get('diagnostic', {})
    arm = env.get('DS41_NCCL_ARM')
    if arm is not None and (arm != STANDARD_NCCL_ARM or diagnostic.get('kind') != 'precision-capture'
                            or 'DS41_NCCL_GEOMETRY' in env):
        raise ValueError('The standard NCCL arm is restricted to the precision-capture image without tree geometry')
    replay = env.get('DS41_SELECTION_REPLAY')
    capture = diagnostic.get('kind') == MHC_EXPANDED_CAPTURE_KIND
    if capture and (replay not in MHC_EXPANDED_CAPTURE_SELECTORS
                    or diagnostic.get('lock_sha256') != MHC_EXPANDED_CAPTURE_LOCK
                    or diagnostic.get('base_image_id') != MHC_EXPANDED_CAPTURE_BASE):
        raise ValueError('The expanded-mHC capture needs its lock on the passing or expanded-only replay root')
    if replay is not None:
        if (replay not in SELECTION_ROOTS
                or (not capture and (diagnostic.get('kind') != 'precision-release-candidate'
                                     or diagnostic.get('lock_sha256') != PRECISION_RELEASE_LOCK))
                or env.get('DS41_DECISION_ROW_BLOCKS') != '81389' or env.get('DS41_PREFILL_THRESHOLD') != '8192'
                or any(key in env for key in ('DS41_CHUNKING_BLOCKS', 'DS41_NCCL_GEOMETRY', 'DS41_NCCL_ARM'))
                or 'diagnostic_overlay' in pin):
            raise ValueError('Selection replay requires the exact release at 81389 blocks and threshold 8192')
        return
    if diagnostic.get('kind') == 'ratio1-bf16-diagnostic' or env.get('DS41_DECISION_ROW_BLOCKS') == RATIO1_BLOCKS:
        allowed = {'precision-release-candidate': PRECISION_RELEASE_LOCK, 'ratio1-bf16-diagnostic': RATIO1_LOCK}
        if (diagnostic.get('kind') not in allowed or diagnostic.get('lock_sha256') != allowed[diagnostic['kind']]
                or env.get('DS41_DECISION_ROW_BLOCKS') != RATIO1_BLOCKS
                or any(key in env for key in ('DS41_PREFILL_THRESHOLD', 'DS41_CHUNKING_BLOCKS',
                                              'DS41_NCCL_GEOMETRY', 'DS41_NCCL_ARM'))
                or 'diagnostic_overlay' in pin):
            raise ValueError('The 80022-block pin is restricted to the ratio-1 compute-mode diagnostic sequence')
        return
    if diagnostic.get('kind') in ('indexer-capture', 'precision-capture'):
        expected_lock = (PRECISION_CAPTURE_LOCK if diagnostic['kind'] == 'precision-capture'
                         else INDEXER_CAPTURE_LOCK)
        if (len(expected_lock) != 64
                or diagnostic.get('lock_sha256') != expected_lock
                or not (env.get('DS41_NCCL_GEOMETRY') == NCCL_GEOMETRY or arm == STANDARD_NCCL_ARM)
                or env.get('DS41_DECISION_ROW_BLOCKS') != '81389'
                or env.get('DS41_PREFILL_THRESHOLD') not in ('8192', '4096')
                or 'diagnostic_overlay' in pin):
            raise ValueError('Indexer capture requires its frozen lock and matched geometry profile')
        return
    if 'DS41_NCCL_GEOMETRY' in env:
        diagnostic = pin.get('diagnostic', {})
        if (env.get('DS41_DECISION_ROW_BLOCKS') != '81389' or diagnostic.get('kind') != 'window-capture'
                or diagnostic.get('lock_sha256') != WINDOW_CAPTURE_LOCK or 'diagnostic_overlay' in pin):
            raise ValueError('NCCL geometry diagnostic requires the reviewed window-capture lock without overlay')
    if env.get('DS41_DECISION_ROW_BLOCKS') == '81389':
        allowed = {'decision-row-capture': MATCHED_CAPTURE_LOCK, 'window-capture': WINDOW_CAPTURE_LOCK}
        diagnostic = pin.get('diagnostic', {})
        if (diagnostic.get('kind') not in allowed
                or diagnostic.get('lock_sha256') != allowed[diagnostic['kind']]
                or 'diagnostic_overlay' in pin):
            raise ValueError('Matched capture requires the reviewed diagnostic lock without overlay')
        return
    if 'DS41_PREFILL_THRESHOLD' in env or 'DS41_CHUNKING_BLOCKS' in env:
        if (pin.get('image_id') != CHUNKING_IMAGE
                or pin.get('diagnostic', {}).get('kind') != 'router-stage-release-candidate'
                or 'diagnostic_overlay' in pin):
            raise ValueError('Chunking diagnostic requires the exact clean e06 router image')


def render(node):
    if node not in NODES:
        raise ValueError(f'Unknown node: {node}')
    upstream = json.loads((ROOT / 'upstream-launch.json').read_text())['reference']
    cluster, model = upstream['cluster'], list(upstream['model'])
    env = dict(cluster[i + 1].split('=', 1) for i, value in enumerate(cluster) if value == '--env')
    # Source/venv and HF paths move inside the image; compilation controls do not.
    env.update(PYTHONPATH='/opt/jovian-judgement/vllm:/opt/jovian-judgement/b12x',
               HF_HOME='/root/.cache/huggingface', LD_PRELOAD=NCCL, VLLM_NCCL_SO_PATH=NCCL)
    # These image-owned paths are checked by the runtime gate, not overwritten
    # with Luke's host paths or a stale cache namespace.
    del env['B12X_COMPILE_CACHE_DIR']
    del env['B12X_ROCE_CACHE_DIR']
    env.update(NCCL_SOCKET_IFNAME='enp1s0f0np0', GLOO_SOCKET_IFNAME='enp1s0f0np0',
               VLLM_HOST_IP=NODES[node][1], NCCL_DEBUG='WARN')
    # User-approved determinism investigation, 2026-09-24. Optional explicit
    # switch only; an ordinary launch retains the frozen upstream default.
    for name in ('B12X_DYNAMIC_DETERMINISTIC_OUTPUT', 'B12X_DENSE_SPLITK_TURBO',
                 'VLLM_DS41_L2_PREFETCH'):
        value = os.environ.get(name)
        if value is not None:
            if value not in ('0', '1'):
                raise ValueError(f'{name} must be 0 or 1')
            env[name] = value
    # Approved root-cause investigation: isolate lazy CUDA loading while keeping
    # Engram overlap and all model settings unchanged. No default override.
    loading = os.environ.get('CUDA_MODULE_LOADING')
    if loading is not None:
        if loading not in ('LAZY', 'EAGER'):
            raise ValueError('CUDA_MODULE_LOADING must be LAZY or EAGER')
        env['CUDA_MODULE_LOADING'] = loading
    model[0] = '/opt/venv/bin/vllm'
    model[model.index('--gpu-memory-utilization') + 1] = '0.85'
    index = model.index('--kv-cache-memory-bytes')
    del model[index:index + 2]
    # Approved historical 524K replay after 500K admission passed.
    model[model.index('--max-model-len') + 1] = '600000'
    # Explicitly approved diagnostic capture only. Preserve the recorded MLA
    # planning geometry; never impose this pin on normal serving.
    decision_blocks = os.environ.get('DS41_DECISION_ROW_BLOCKS')
    if decision_blocks is not None:
        if decision_blocks not in ('80927', '81389', RATIO1_BLOCKS):
            raise ValueError('DS41_DECISION_ROW_BLOCKS must be exactly 80927, 81389 or 80022')
        env['DS41_DECISION_ROW_BLOCKS'] = decision_blocks
        model += ['--num-gpu-blocks-override', decision_blocks]
    # User-approved matched chunking diagnostics, 2026-09-26. Keep preparation
    # capacity at 8192; preserve the measured 81389-block numerical-plan family.
    # Neither control is a normal-serving default or a general passthrough.
    threshold = os.environ.get('DS41_PREFILL_THRESHOLD')
    chunking_blocks = os.environ.get('DS41_CHUNKING_BLOCKS')
    if decision_blocks == '81389':
        # Approved matched-capture derivative, 2026-09-26. Image lock is gated
        # on both host and container paths. No change to normal serving.
        if threshold not in ('8192', '4096') or chunking_blocks is not None:
            raise ValueError('Matched capture requires explicit threshold 8192 or 4096 without chunking blocks')
        if model[model.index('--max-num-batched-tokens') + 1] != '8192' or '--long-prefill-token-threshold' in model:
            raise ValueError('Matched capture preparation contract changed')
        env['DS41_PREFILL_THRESHOLD'] = threshold
        model += ['--long-prefill-token-threshold', threshold]
    elif threshold is not None or chunking_blocks is not None:
        if threshold not in ('7936', '4096') or chunking_blocks != '81389' or decision_blocks is not None:
            raise ValueError('Chunking requires threshold 7936 or 4096 and exactly 81389 blocks, without capture pin')
        if model[model.index('--max-num-batched-tokens') + 1] != '8192':
            raise ValueError('Chunking diagnostic preparation capacity must remain 8192')
        if '--long-prefill-token-threshold' in model:
            raise ValueError('Upstream threshold contract changed; review required')
        env.update(DS41_PREFILL_THRESHOLD=threshold, DS41_CHUNKING_BLOCKS=chunking_blocks)
        model += ['--long-prefill-token-threshold', threshold,
                  '--num-gpu-blocks-override', chunking_blocks]
    replay = os.environ.get('DS41_SELECTION_REPLAY')
    if replay is not None:
        if replay not in SELECTION_ROOTS or decision_blocks != '81389' or threshold != '8192':
            raise ValueError('Selection replay requires the 81389-block, threshold-8192 matched profile')
        env['DS41_SELECTION_REPLAY'] = replay
    # User-approved NCCL geometry diagnostic, 2026-09-26: four explicit overrides, only on the
    # matched 81389-block window-capture arms with an explicit threshold. The tuning variables
    # the normal launch unsets become explicit here; nothing else in the profile changes.
    geometry = os.environ.get('DS41_NCCL_GEOMETRY')
    unset = list(NCCL_TUNING_UNSET)
    if geometry is not None:
        if geometry != NCCL_GEOMETRY or decision_blocks != '81389' or threshold not in ('8192', '4096'):
            raise ValueError('NCCL geometry diagnostic requires tree-simple-1ch on the matched 81389 capture profile')
        env.update(NCCL_GEOMETRY_ENV, DS41_NCCL_GEOMETRY=geometry)
        unset = []
    # Standard NCCL arm: a recorded selector only; the ordinary unset list and cluster NCCL env stay.
    arm = os.environ.get('DS41_NCCL_ARM')
    if arm is not None:
        if (arm != STANDARD_NCCL_ARM or geometry is not None or decision_blocks != '81389'
                or threshold not in ('8192', '4096')):
            raise ValueError('The standard NCCL arm requires the matched 81389 capture profile without tree geometry')
        env['DS41_NCCL_ARM'] = arm
    rank = NODES[node][0]
    model += ['--distributed-executor-backend', 'mp', '--nnodes', '4',
              '--node-rank', str(rank), '--master-addr', NODES['dusty'][1],
              '--master-port', '29656']
    if rank:
        model += ['--headless']
    return {'node': node, 'rank': rank, 'env': env, 'model': model,
            'unset': unset,
            'container_resource_args': ['--ipc=private', '--shm-size=64g', '--pids-limit=-1'],
            'ld_library_path_prefix': '/opt/nccl-2.30.7/lib'}


if __name__ == '__main__':
    import sys
    print(json.dumps(render(sys.argv[1]), indent=2, sort_keys=True))
