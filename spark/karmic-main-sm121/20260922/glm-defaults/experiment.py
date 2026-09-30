"""Finite diagnostic profiles; compare each optimization against its named parent."""
import hashlib
import json

BASE_SHA256 = '78a82d4ec6e674df078e1564e3be73c0cfe5194bb361ea8be11c285863398703'
ARMS = {
    'defaults': {},
    'bf16-head': {'VLLM_GLM53_MTP_DRAFT_HEAD': 'bf16'},
    'no-prefetch': {'VLLM_GLM53_L2_PREFETCH': '0'},
    'queue': {'B12X_DYNAMIC_WORK_SOURCE': 'materialized_queue'},
    'no-replayssm': {},
    'no-prefetch-k5': {'VLLM_GLM53_L2_PREFETCH': '0'},
    'no-prefetch-k2': {'VLLM_GLM53_L2_PREFETCH': '0'},
    'no-prefetch-profile': {'VLLM_GLM53_L2_PREFETCH': '0'},
    'no-prefetch-shared-head': {'VLLM_GLM53_L2_PREFETCH': '0', 'VLLM_MTP_NVFP4_LM_HEAD': '1'},
    'no-prefetch-mxfp8-output-a16': {
        'VLLM_GLM53_L2_PREFETCH': '0',
        'VLLM_B12X_MXFP8_ACTIVATION_MODE': 'a16',
        'VLLM_LOG_MODEL_INSPECTION': '1',
    },
    'no-prefetch-no-mhc-pdl': {'VLLM_GLM53_L2_PREFETCH': '0', 'B12X_MHC_PDL': '0'},
    'no-prefetch-block': {'VLLM_GLM53_L2_PREFETCH': '0'},
    'no-prefetch-no-mhc-pdl-block': {'VLLM_GLM53_L2_PREFETCH': '0', 'B12X_MHC_PDL': '0'},
}
SINGLE_FACTOR_ARMS = ('defaults', 'bf16-head', 'no-prefetch', 'queue', 'no-replayssm')
EXTRA_ARGV = {'no-replayssm': ['--no-use-replayssm']}
# Exact target-model names, not a broad self_attn wildcard. Layer 45 is MTP.
# The KV expansion is absorbed into BF16 attention weights and is left alone.
# Recurrent inputs, routers, experts, vision and vocabulary heads are excluded.
MXFP8_TARGETS = {
    **{f'language_model.model.layers.{i}.self_attn.o_proj': 'mxfp8'
       for i in range(45)},
    **{f'language_model.model.layers.{i}.self_attn.q_b_proj': 'mxfp8'
       for i in range(3, 45, 4)},
}
EXTRA_ARGV['no-prefetch-mxfp8-output-a16'] = [
    '--quantization-config', json.dumps({'targets': MXFP8_TARGETS}, separators=(',', ':')),
]
EXTRA_ARGV['no-prefetch-profile'] = ['--profiler-config', json.dumps({
    'profiler': 'torch',
    'torch_profiler_dir': '/cache/glm-no-prefetch-profile-20260924',
    'torch_profiler_with_stack': False,
    'torch_profiler_record_shapes': False,
    'torch_profiler_with_memory': False,
    'torch_profiler_with_flops': False,
    'ignore_frontend': True,
    'delay_iterations': 3,
    'max_iterations': 5,
}, separators=(',', ':'))]
BASE_NAME = 'glm53-flash-karmic-main-defaults-tp4'


def name(arm):
    if arm not in ARMS:
        raise SystemExit('Unknown diagnostic arm')
    return BASE_NAME if arm == 'defaults' else BASE_NAME + '-' + arm


def resolve(base_bytes, arm):
    if hashlib.sha256(base_bytes).hexdigest() != BASE_SHA256:
        raise SystemExit('Measured baseline profile changed')
    name(arm)
    lock = json.loads(base_bytes)
    lock['environment'].update(ARMS[arm])
    lock['argv'].extend(EXTRA_ARGV.get(arm, []))
    if arm in ('no-prefetch-k5', 'no-prefetch-k2'):
        window = 5 if arm == 'no-prefetch-k5' else 2
        argv = lock['argv']
        spec_index = argv.index('--speculative-config') + 1
        config = json.loads(argv[spec_index])
        config['num_speculative_tokens'] = window
        argv[spec_index] = json.dumps(config, separators=(',', ':'))
        start = argv.index('--cudagraph-capture-sizes') + 1
        end = start
        while end < len(argv) and not argv[end].startswith('--'):
            end += 1
        # Exact target sizes for C1/C2/C3/C4 avoid padding onto K3 graphs.
        sizes = sorted(set(map(int, argv[start:end])) | {(window + 1) * c for c in (1, 2, 3, 4)})
        argv[start:end] = list(map(str, sizes))
    if arm in ('no-prefetch-block', 'no-prefetch-no-mhc-pdl-block'):
        index = lock['argv'].index('--speculative-config') + 1
        config = json.loads(lock['argv'][index])
        config['rejection_sample_method'] = 'block'
        lock['argv'][index] = json.dumps(config, separators=(',', ':'))
    record = {'arm': arm, 'base_profile_sha256': BASE_SHA256,
              'environment_overrides': ARMS[arm],
              'extra_argv': EXTRA_ARGV.get(arm, []),
              'argv': lock['argv'], 'environment': lock['environment']}
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return lock, record, digest


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=ARMS)
    parser.add_argument('--name', action='store_true')
    args = parser.parse_args()
    if args.name:
        print(name(args.arm))
    else:
        _, record, digest = resolve(Path(__file__).with_name('profile.lock.json').read_bytes(), args.arm)
        print(json.dumps({'effective_profile_sha256': digest, **record}, indent=2))
