import json
import unittest
from unittest.mock import patch
from pathlib import Path

from launch_contract import NCCL, NODES, render


class ContractTests(unittest.TestCase):
    def test_decision_row_block_pin_is_explicit_and_exact(self):
        self.assertNotIn('--num-gpu-blocks-override', render('dusty')['model'])
        with patch.dict('os.environ', {'DS41_DECISION_ROW_BLOCKS': '80927'}):
            for node in NODES:
                row = render(node)
                self.assertEqual(row['env']['DS41_DECISION_ROW_BLOCKS'], '80927')
                index = row['model'].index('--num-gpu-blocks-override')
                self.assertEqual(row['model'][index + 1], '80927')
        for value in ('', '79000', '81592', '-1', 'automatic'):
            with patch.dict('os.environ', {'DS41_DECISION_ROW_BLOCKS': value}):
                with self.assertRaises(ValueError):
                    render('dusty')

    def test_decision_row_pin_rejects_other_images(self):
        from run_node import command
        with patch.dict('os.environ', {'DS41_DECISION_ROW_BLOCKS': '80927'}):
            for kind in (None, 'router-stage-release-candidate', 'engram-fault-inject'):
                pin = {'image_id': 'a' * 64, 'diagnostic': {'kind': kind}}
                with self.assertRaisesRegex(RuntimeError, 'decision-row-capture'):
                    command('dusty', pin, 'b' * 64)
            pin = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'decision-row-capture'}}
            argv, _, _ = command('dusty', pin, 'b' * 64)
            self.assertIn('DS41_DECISION_ROW_BLOCKS=80927', argv)

    def test_cuda_loading_diagnostic(self):
        self.assertNotIn('CUDA_MODULE_LOADING', render('dusty')['env'])
        for loading in ('LAZY', 'EAGER'):
            with patch.dict('os.environ', {'CUDA_MODULE_LOADING': loading}):
                for node in NODES:
                    self.assertEqual(render(node)['env']['CUDA_MODULE_LOADING'], loading)
        with patch.dict('os.environ', {'CUDA_MODULE_LOADING': 'invalid'}):
            with self.assertRaises(ValueError):
                render('dusty')

    def test_prefetch_diagnostic(self):
        self.assertNotIn('VLLM_DS41_L2_PREFETCH', render('dusty')['env'])
        with patch.dict('os.environ', {'VLLM_DS41_L2_PREFETCH': '0'}):
            self.assertEqual(render('dusty')['env']['VLLM_DS41_L2_PREFETCH'], '0')
        with patch.dict('os.environ', {'VLLM_DS41_L2_PREFETCH': 'invalid'}):
            with self.assertRaises(ValueError):
                render('dusty')

    def test_dense_reduction_diagnostic(self):
        self.assertEqual(render('dusty')['env']['B12X_DENSE_SPLITK_TURBO'], '1')
        with patch.dict('os.environ', {'B12X_DENSE_SPLITK_TURBO': '0'}):
            self.assertEqual(render('dusty')['env']['B12X_DENSE_SPLITK_TURBO'], '0')
        with patch.dict('os.environ', {'B12X_DENSE_SPLITK_TURBO': 'invalid'}):
            with self.assertRaises(ValueError):
                render('dusty')

    def test_determinism_diagnostic(self):
        with patch.dict('os.environ', {'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '1'}):
            self.assertEqual(render('dusty')['env']['B12X_DYNAMIC_DETERMINISTIC_OUTPUT'], '1')
        with patch.dict('os.environ', {'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': 'invalid'}):
            with self.assertRaises(ValueError):
                render('dusty')

    def test_environment_inventory(self):
        cluster = json.loads((Path(__file__).parent / 'upstream-launch.json').read_text())['reference']['cluster']
        keys = {cluster[i + 1].split('=', 1)[0] for i, value in enumerate(cluster) if value == '--env'}
        expected = keys - {'B12X_COMPILE_CACHE_DIR', 'B12X_ROCE_CACHE_DIR'}
        expected |= {'NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME', 'VLLM_HOST_IP', 'NCCL_DEBUG'}
        for node in NODES:
            self.assertEqual(set(render(node)['env']), expected)

    def test_only_approved_model_changes(self):
        reference = json.loads((Path(__file__).parent / 'upstream-launch.json').read_text())['reference']['model']
        for node, (rank, _) in NODES.items():
            result = render(node)['model']
            expected = list(reference)
            expected[0] = '/opt/venv/bin/vllm'
            expected[expected.index('--gpu-memory-utilization') + 1] = '0.85'
            i = expected.index('--kv-cache-memory-bytes')
            del expected[i:i + 2]
            expected[expected.index('--max-model-len') + 1] = '600000'
            expected += ['--distributed-executor-backend', 'mp', '--nnodes', '4',
                         '--node-rank', str(rank), '--master-addr', '10.11.11.7', '--master-port', '29656']
            if rank:
                expected += ['--headless']
            self.assertEqual(result, expected)

    def test_resources_and_library(self):
        for node in NODES:
            row = render(node)
            self.assertEqual(row['container_resource_args'], ['--ipc=private', '--shm-size=64g', '--pids-limit=-1'])
            self.assertEqual(row['env']['LD_PRELOAD'], NCCL)
            self.assertEqual(row['env']['VLLM_NCCL_SO_PATH'], NCCL)
            self.assertEqual(row['env']['NCCL_NET_PLUGIN'], 'none')
            self.assertEqual(row['env']['VLLM_PLUGINS'], 'b12x_loader')
            self.assertEqual(row['env']['NCCL_IB_HCA'], 'rocep1s0f0,roceP2p1s0f0')
            self.assertNotIn('NCCL_PROTO', row['env'])

    def test_unknown_node(self):
        with self.assertRaises(ValueError):
            render('sparky')

    def test_podman_is_direct_retained_and_unlimited(self):
        from run_node import command
        for node in NODES:
            argv, _, _ = command(node, {'image_id': 'a' * 64}, 'b' * 64)
            self.assertNotIn('--rm', argv)
            self.assertNotIn('--privileged', argv)
            self.assertNotIn('--memory', argv)
            self.assertNotIn('--memory-swap', argv)
            self.assertIn('--ipc=private', argv)
            self.assertIn('--shm-size=64g', argv)
            self.assertIn('--pids-limit=-1', argv)
            self.assertIn('--init', argv)
            self.assertIn('unset NCCL_PROTO', argv[-1])
            self.assertIn('export LD_LIBRARY_PATH=/opt/nccl-2.30.7/lib:', argv[-1])


if __name__ == '__main__':
    unittest.main()
