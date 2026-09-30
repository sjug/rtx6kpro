import os
from pathlib import Path
import shlex
import subprocess
import unittest

ROOT = Path(__file__).resolve().parent


class QualificationContract(unittest.TestCase):
    def test_retry_image_is_consistent(self):
        image = '6633e678fee74f5e1060290a01df812d34d10edcf30e34dd1c9bf7db88260ef3'
        for name in ('run-qwen-tp2-node.sh', 'distribute-qwen.sh',
                     'qualify_qwen.py', 'benchmark-qwen.sh'):
            text = (ROOT / name).read_text()
            self.assertIn(image, text, name)
            self.assertNotIn('e7926f763859', text, name)
        self.assertIn('976a9ac502a7bcdb25b18cb5597d21caafc97252',
                      (ROOT / 'run-qwen-tp2-node.sh').read_text())

    def test_node_render_preserves_existing_contract(self):
        for role in ('head', 'worker'):
            env = {**os.environ, 'DRY_RUN': '1', 'ROLE': role}
            old = subprocess.check_output(['bash', str(ROOT.parent / 'karmic-beta-sm121/run-qwen-tp2-node.sh')], env=env, text=True)
            new = subprocess.check_output(['bash', str(ROOT / 'run-qwen-tp2-node.sh')], env=env, text=True)
            self.assertEqual(shlex.split(old.replace('karmic', 'jj-main')), shlex.split(new))
            self.assertIn('NUM_SPECULATIVE_TOKENS=3', new)
            self.assertIn('--recurrent-checkpoint-policy aligned', new)
            self.assertNotIn('NCCL_MIN_NCHANNELS', new)
            self.assertNotIn('NCCL_MAX_NCHANNELS', new)

    def test_stop_retains_container(self):
        text = subprocess.check_output(['bash', str(ROOT / 'run-qwen-tp2-node.sh')], env={**os.environ, 'DRY_RUN': '1', 'ROLE': 'stop'}, text=True)
        self.assertIn('stop -t 60', text)
        self.assertNotIn('podman rm', text)


if __name__ == '__main__':
    unittest.main()
