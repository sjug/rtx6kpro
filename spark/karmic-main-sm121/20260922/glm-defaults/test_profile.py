import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch
import run
import launch
import experiment
import verify_boot

ROOT = Path(__file__).resolve().parent

class ProfileTests(unittest.TestCase):
    def test_independent_single_factor_arms(self):
        baseline = (ROOT / 'profile.lock.json').read_bytes()
        original = json.loads(baseline)
        digests = set()
        for arm in experiment.SINGLE_FACTOR_ARMS:
            overrides = experiment.ARMS[arm]
            lock, record, digest = experiment.resolve(baseline, arm)
            self.assertEqual(lock['argv'], original['argv'] + experiment.EXTRA_ARGV.get(arm, []))
            changed = {k: v for k, v in lock['environment'].items() if v != original['environment'].get(k)}
            self.assertEqual(changed, overrides)
            self.assertEqual(len(changed) + len(experiment.EXTRA_ARGV.get(arm, [])), 0 if arm == 'defaults' else 1)
            self.assertNotIn('NCCL', str(overrides))
            self.assertEqual(record['environment_overrides'], overrides)
            digests.add(digest)
            cmd = run.command('sparky', Path('/home/jugs'), lock, arm)
            self.assertIn('GLM_EFFECTIVE_PROFILE_SHA256=' + digest, cmd)
            self.assertEqual(cmd[cmd.index('--name') + 1], experiment.name(arm))
        self.assertEqual(len(digests), len(experiment.SINGLE_FACTOR_ARMS))
        with self.assertRaises(SystemExit):
            experiment.resolve(baseline + b' ', 'bf16-head')
        with self.assertRaises(SystemExit):
            experiment.resolve(baseline, 'arbitrary')

    def test_k5_only_changes_window_and_matching_graphs_over_no_prefetch(self):
        data = (ROOT / 'profile.lock.json').read_bytes()
        base, _, _ = experiment.resolve(data, 'no-prefetch')
        arm, _, _ = experiment.resolve(data, 'no-prefetch-k5')
        self.assertEqual(base['environment'], arm['environment'])
        def options(argv):
            result = {}
            key = 'command'
            for token in argv:
                if token.startswith('--'):
                    key = token
                    result[key] = []
                else:
                    result.setdefault(key, []).append(token)
            return result
        left, right = options(base['argv']), options(arm['argv'])
        self.assertEqual({k for k in left if left[k] != right[k]},
                         {'--speculative-config', '--cudagraph-capture-sizes'})
        before = json.loads(left['--speculative-config'][0])
        after = json.loads(right['--speculative-config'][0])
        self.assertEqual(after.pop('num_speculative_tokens'), 5)
        before.pop('num_speculative_tokens')
        self.assertEqual(before, after)
        sizes = list(map(int, right['--cudagraph-capture-sizes']))
        self.assertEqual(sizes, sorted(set(sizes)))
        self.assertTrue({6, 12, 18, 24, 192}.issubset(sizes))
        self.assertLessEqual(max(sizes), int(right['--max-cudagraph-capture-size'][0]))
        self.assertGreaterEqual(max(sizes), 6 * int(right['--max-num-seqs'][0]))

    def test_nccl_foundation_preserved(self):
        inherited = {'VLLM_NCCL_SO_PATH': launch.NCCL_PATH, 'VLLM_HOST_IP': '10.11.11.1', 'VLLM_UNKNOWN_POLICY': '1'}
        result = launch.runtime_environment(inherited, {})
        self.assertEqual(result['VLLM_NCCL_SO_PATH'], launch.NCCL_PATH)
        self.assertEqual(result['VLLM_HOST_IP'], '10.11.11.1')
        self.assertNotIn('VLLM_UNKNOWN_POLICY', result)
        with self.assertRaises(SystemExit):
            launch.runtime_environment({}, {})

    def test_profiler_is_bounded_and_changes_no_serving_policy(self):
        data = (ROOT / 'profile.lock.json').read_bytes()
        base, _, _ = experiment.resolve(data, 'no-prefetch')
        arm, _, _ = experiment.resolve(data, 'no-prefetch-profile')
        self.assertEqual(base['environment'], arm['environment'])
        self.assertEqual(arm['argv'][:-2], base['argv'])
        self.assertEqual(arm['argv'][-2], '--profiler-config')
        config = json.loads(arm['argv'][-1])
        self.assertEqual(config['profiler'], 'torch')
        self.assertEqual(config['max_iterations'], 5)
        self.assertEqual(config['delay_iterations'], 3)
        self.assertTrue(config['ignore_frontend'])
        self.assertTrue(config['torch_profiler_dir'].startswith('/cache/'))
        for key in ('stack', 'memory', 'flops'):
            self.assertFalse(config['torch_profiler_with_' + key])
        self.assertFalse(config['torch_profiler_record_shapes'])

    def test_shared_head_is_one_factor_and_requires_distinct_engagement(self):
        data = (ROOT / 'profile.lock.json').read_bytes()
        base, _, _ = experiment.resolve(data, 'no-prefetch')
        arm, _, _ = experiment.resolve(data, 'no-prefetch-shared-head')
        self.assertEqual(base['argv'], arm['argv'])
        self.assertEqual({k: v for k, v in arm['environment'].items()
                          if v != base['environment'][k]}, {'VLLM_MTP_NVFP4_LM_HEAD': '1'})
        shared = 'Quantizing LM head shards to NVFP4 with BF16 activations.'
        copy = 'Using a draft-only NVFP4 GLM MTP vocabulary head copy'
        verify_boot.check_head(shared, arm['environment'])
        verify_boot.check_head(copy, base['environment'])
        for text in ('', copy, shared + copy):
            with self.assertRaises(SystemExit):
                verify_boot.check_head(text, arm['environment'])
        with self.assertRaises(SystemExit):
            verify_boot.check_head(shared, base['environment'])

    def test_selective_mxfp8_preserves_all_other_model_policies(self):
        data = (ROOT / 'profile.lock.json').read_bytes()
        base, _, _ = experiment.resolve(data, 'no-prefetch')
        arm, _, _ = experiment.resolve(data, 'no-prefetch-mxfp8-output-a16')
        self.assertEqual(arm['argv'][:-2], base['argv'])
        self.assertEqual(arm['argv'][-2], '--quantization-config')
        config = json.loads(arm['argv'][-1])
        self.assertEqual(set(config), {'targets'})
        expected = {f'language_model.model.layers.{i}.self_attn.o_proj' for i in range(45)}
        expected.update(f'language_model.model.layers.{i}.self_attn.q_b_proj' for i in range(3, 45, 4))
        self.assertEqual(set(config['targets']), expected)
        self.assertEqual(len(expected), 56)
        self.assertEqual(set(config['targets'].values()), {'mxfp8'})
        changes = {k: v for k, v in arm['environment'].items()
                   if v != base['environment'].get(k)}
        self.assertEqual(changes, {'VLLM_B12X_MXFP8_ACTIVATION_MODE': 'a16',
                                   'VLLM_LOG_MODEL_INSPECTION': '1'})

    def test_selective_mxfp8_engagement_rejects_missing_or_extra_layers(self):
        entries = [f'self_attn.{name.rsplit(".", 1)[1]}: 1 (from targets: {name}, mxfp8)'
                   for name in experiment.MXFP8_TARGETS]
        text = 'Using B12xMxfp8LinearKernel for MXFP8 GEMM\nQuantized 56 layers of types: ' + '; '.join(entries)
        verify_boot.check_selective_mxfp8(text)
        verify_boot.check_selective_mxfp8(text + '\n' + text)
        for wrong in ('', text.replace('56 layers', '55 layers'),
                      text.replace('layers.44.', 'layers.45.'),
                      text.replace('mxfp8)', 'mxfp4)', 1),
                      text.replace('B12xMxfp8LinearKernel', 'OtherKernel'),
                      text + '\n' + text.replace('layers.44.', 'layers.45.')):
            with self.assertRaises(SystemExit):
                verify_boot.check_selective_mxfp8(wrong)

    def test_selective_targets_exist_in_independent_checkpoint_headers(self):
        path = ROOT / 'qualification/profile-no-prefetch-k3-20260924/dense-header-inventory.json'
        inventory = json.loads(path.read_text())
        self.assertEqual(inventory['revision'], self.lock['checkpoint_revision'])
        matrices = {r['name']: r for r in inventory['bf16_matrices']}
        total = 0
        for name in experiment.MXFP8_TARGETS:
            checkpoint_name = name.replace('language_model.model.', 'model.language_model.', 1) + '.weight'
            matrix = matrices[checkpoint_name]
            out_features, in_features = matrix['shape']
            # All selected linears are TP-sharded: row-parallel outputs shard K,
            # column-parallel query projections shard N. Check both partitions.
            if name.endswith('.o_proj'):
                self.assertEqual(in_features % 4, 0)
                in_features //= 4
            else:
                self.assertEqual(out_features % 4, 0)
                out_features //= 4
            self.assertEqual(in_features % 128, 0)
            self.assertEqual(out_features % 8, 0)
            total += matrix['checkpoint_bytes']
        self.assertEqual(total, 4311744512)

    def test_selective_quantization_refuses_screen_only_benchmark(self):
        result = subprocess.run(['bash', str(ROOT / 'benchmark.sh')],
                                env={**os.environ,
                                     'GLM_EXPERIMENT_ARM': 'no-prefetch-mxfp8-output-a16',
                                     'GLM_RECEIPT_DIR': '/nonexistent-qualification-receipt',
                                     'GLM_CORRECTNESS_REVIEWED': '1',
                                     'GLM_QUALIFICATION_SCOPE': 'screen'},
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 78)
        self.assertIn('requires full native-context qualification', result.stdout)

    def test_k2_changes_only_window_and_exact_graph_sizes(self):
        data = (ROOT / 'profile.lock.json').read_bytes()
        base, _, _ = experiment.resolve(data, 'no-prefetch')
        arm, _, _ = experiment.resolve(data, 'no-prefetch-k2')
        self.assertEqual(base['environment'], arm['environment'])
        argv = list(arm['argv'])
        spec = argv.index('--speculative-config') + 1
        value = json.loads(argv[spec])
        self.assertEqual(value['num_speculative_tokens'], 2)
        value['num_speculative_tokens'] = 3
        base_spec = base['argv'].index('--speculative-config') + 1
        self.assertEqual(value, json.loads(base['argv'][base_spec]))
        argv[spec] = base['argv'][base_spec]
        start = argv.index('--cudagraph-capture-sizes') + 1
        end = start
        while not argv[end].startswith('--'):
            end += 1
        sizes = list(map(int, argv[start:end]))
        self.assertTrue({3, 6, 9, 12}.issubset(sizes))
        original_end = start
        while not base['argv'][original_end].startswith('--'):
            original_end += 1
        argv[start:end] = base['argv'][start:original_end]
        self.assertEqual(argv, base['argv'])

    def test_no_mhc_pdl_changes_one_variable_over_no_prefetch(self):
        data = (ROOT / 'profile.lock.json').read_bytes()
        base, _, _ = experiment.resolve(data, 'no-prefetch')
        arm, _, _ = experiment.resolve(data, 'no-prefetch-no-mhc-pdl')
        self.assertEqual(base['argv'], arm['argv'])
        self.assertEqual({k: v for k, v in arm['environment'].items()
                          if v != base['environment'][k]}, {'B12X_MHC_PDL': '0'})

    def test_block_arms_change_only_verification_algorithm(self):
        data = (ROOT / 'profile.lock.json').read_bytes()
        for parent in ('no-prefetch', 'no-prefetch-no-mhc-pdl'):
            base, _, _ = experiment.resolve(data, parent)
            arm, _, _ = experiment.resolve(data, parent + '-block')
            self.assertEqual(base['environment'], arm['environment'])
            argv = list(arm['argv'])
            index = argv.index('--speculative-config') + 1
            config = json.loads(argv[index])
            self.assertEqual(config.pop('rejection_sample_method'), 'block')
            before = json.loads(base['argv'][index])
            self.assertEqual(before.pop('rejection_sample_method'), 'standard')
            self.assertEqual(config, before)
            argv[index] = base['argv'][index]
            self.assertEqual(argv, base['argv'])

    def setUp(self):
        self.lock = json.loads((ROOT / 'profile.lock.json').read_text())

    def test_upstream_defaults(self):
        v = self.lock['options']
        self.assertEqual(v['recurrent-checkpoint-policy'], 'request_boundaries')
        self.assertEqual(v['speculative-config']['num_speculative_tokens'], 3)
        self.assertEqual(v['default-chat-template-kwargs']['clear_thinking'], False)
        self.assertEqual(v['cache-mode'], 'vram')
        self.assertEqual(v['max-num-seqs'], 32)
        self.assertEqual(v['max-cudagraph-capture-size'], 256)
        self.assertEqual(v['additional-config']['kda_prefill_backend'], 'b12x')
        self.assertEqual(self.lock['environment']['VLLM_GLM53_MTP_DRAFT_HEAD'], 'nvfp4')
        self.assertEqual(self.lock['environment']['PYTORCH_CUDA_ALLOC_CONF'], 'expandable_segments:True')

    def test_only_declared_option_changes(self):
        changed = {k for k, v in self.lock['options'].items() if v != self.lock['upstream_options'].get(k)}
        self.assertEqual(changed, set(self.lock['deployment_option_overrides']))

    def test_node_mapping_and_no_side_effects(self):
        with patch('subprocess.run', side_effect=AssertionError('side effect')):
            for node, (rank, address) in run.NODES.items():
                command = run.command(node, Path('/home/jugs'), self.lock)
                self.assertIn(f'GLM_NODE_RANK={rank}', command)
                self.assertIn(f'VLLM_HOST_IP={address}', command)
                self.assertIn('--pull=never', command)
                self.assertNotIn('--privileged', command)
                self.assertNotIn('rm', command)
                self.assertEqual(command[command.index('--entrypoint') + 1], '/bin/bash')
                self.assertEqual(command[-2:], ['-c', 'exec /opt/venv/bin/python /opt/glm-defaults/launch.py'])

    def test_transport(self):
        e = self.lock['environment']
        self.assertEqual(e['NCCL_PROTO'], 'LL,Simple')
        self.assertNotIn('NCCL_MIN_NCHANNELS', e)
        self.assertNotIn('NCCL_MAX_NCHANNELS', e)
        self.assertEqual(e['NCCL_IB_HCA'], 'rocep1s0f0,roceP2p1s0f0')
        self.assertEqual(e['NCCL_NET_PLUGIN'], 'none')

if __name__ == '__main__':
    unittest.main()
