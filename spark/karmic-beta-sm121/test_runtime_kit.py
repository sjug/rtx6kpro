import json
import os
from pathlib import Path
import shlex
import subprocess
import unittest

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'glm53/r38-spark'


def rendered(path, **extra):
    return subprocess.check_output(['bash', str(path)], text=True,
        env={**os.environ, 'DRY_RUN': '1', **extra})


class RuntimeKitTests(unittest.TestCase):
    def test_allocator_control_changes_only_one_env(self):
        settings = dict(ROLE='head', NODE_RANK='0', HOST_IP='10.11.11.1')
        path = ROOT / 'run-glm-tp4-node.sh'
        before = shlex.split(rendered(path, PYTORCH_CUDA_ALLOC_CONF='', **settings))
        after = shlex.split(rendered(path, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', **settings))
        index = after.index('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True')
        self.assertEqual(after[index - 1], '-e')
        del after[index - 1:index + 1]
        self.assertEqual(before, after)

    def test_launch_commands_preserve_r38_contract(self):
        for model in ('qwen38-flash-next', 'glm53-flash'):
            before = rendered(OLD / f'launchers/serve-{model}-jj-r38-spark.sh')
            after = rendered(ROOT / f'launchers/serve-{model}-karmic-spark.sh')
            self.assertEqual(before.replace('JJ r38', 'Karmic beta'), after)

    def test_node_commands_change_only_release_identity(self):
        for role in ('head', 'worker'):
            before = rendered(OLD / 'run-qwen38-flash-next-jj-r38-spark-tp2-node.sh', ROLE=role)
            after = rendered(ROOT / 'run-qwen-tp2-node.sh', ROLE=role)
            self.assertEqual(shlex.split(before.replace('jj-r38', 'karmic')), shlex.split(after))

    def test_launcher_digests_match_generated_files(self):
        import hashlib
        manifest = json.loads((ROOT / 'launchers/manifest.json').read_text())
        for name, record in manifest.items():
            self.assertEqual(hashlib.sha256((ROOT / 'launchers' / name).read_bytes()).hexdigest(), record['sha256'])

    def test_glm_node_contract_except_identity_and_proxy_packaging(self):
        for rank, ip in enumerate(('10.11.11.1', '10.11.11.2', '10.11.11.4', '10.11.11.3')):
            settings = {'ROLE': 'head' if rank == 0 else 'worker', 'NODE_RANK': str(rank),
                        'HOST_IP': ip, 'GLM_LAUNCHER_FILE': ''}
            before = rendered(OLD / 'run-glm53-flash-jj-r38-spark-tp4-node.sh', **settings)
            after = rendered(ROOT / 'run-glm-tp4-node.sh', **settings)
            before = before.replace('jj-r38', 'karmic').replace('/opt/jovian-judgement/b12x-roce', '/opt/b12x-roce-cache')
            before = before.replace(str(OLD), str(ROOT))
            self.assertEqual(shlex.split(before), shlex.split(after))

    def test_dependency_versions_not_upgraded_for_arm(self):
        original = (ROOT / 'runtime-inputs/tools/jovian_wheel_runtime/qwen38-runtime.in').read_text()
        self.assertEqual((ROOT / 'runtime-arm64.in').read_text(), original + '\nquack-kernels==0.6.4\nsetuptools==80.9.0\n')

    def test_ds4_runner_preserves_serving_contract(self):
        for role in ('head', 'worker'):
            before = rendered(ROOT.parent / 'ds4-vision/r38/run-node.sh', ROLE=role)
            after = rendered(ROOT / 'ds4-vision/run-node.sh', ROLE=role)
            before = before.replace('/home/jugs/git/ds4-vision-r38',
                                    '/home/jugs/git/ds4-vision-r38/karmic-beta-sm121')
            before = before.replace('jj-r38', 'karmic')
            self.assertEqual(shlex.split(before), shlex.split(after))


if __name__ == '__main__':
    unittest.main()
