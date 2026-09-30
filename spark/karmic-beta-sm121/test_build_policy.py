from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent


class BuildPolicyTests(unittest.TestCase):
    def test_checked_overlay_equals_pinned_source_transform(self):
        generated = subprocess.check_output([sys.executable, str(ROOT / "prepare_overlay.py")])
        self.assertEqual(generated, (ROOT / "spark-overlay.patch").read_bytes())

    def test_flashinfer_effective_parallelism_and_interpreter(self):
        text = (ROOT / "Dockerfile.flashinfer").read_text()
        self.assertIn("MAX_JOBS=20 FLASHINFER_NVCC_THREADS=1", text)
        self.assertEqual(text.count("uv build --python /usr/bin/python3 --offline"), 2)
        self.assertIn("FLASHINFER_CUDA_ARCH_LIST=12.1a", text)

    def test_components_do_not_reuse_r38_native_base(self):
        for component in ("flashinfer", "nccl", "vllm", "lmcache", "instanttensor"):
            with self.subTest(component=component):
                text = (ROOT / f"Dockerfile.{component}").read_text()
                self.assertIn("237ecf9ac7373daf91b31bb4f86651ce1ce57b676366ed435aa1aba61dad81d5", text)
                self.assertIn("build-component-not-serving-qualified", text)
                self.assertNotIn("NCCL_MIN_NCHANNELS", text)
                self.assertNotIn("NCCL_MAX_NCHANNELS", text)

    def test_vllm_explicit_cmake_arch(self):
        text = (ROOT / "Dockerfile.vllm").read_text()
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=121a", text)
        self.assertIn("CMAKE_BUILD_PARALLEL_LEVEL=20 MAX_JOBS=20", text)
        self.assertIn("MAX_JOBS=20 NVCC_THREADS=1", text)

    def test_vllm_refreshes_layer_index_before_indexed_apply(self):
        text = (ROOT / "Dockerfile.vllm").read_text()
        self.assertLess(text.index("update-index --refresh"),
                        text.index("apply --check --index"))
        self.assertIn("02a457e2e933d0fb20d5786835110546acf7d8f2", text)

    def test_lmcache_fresh_cuda_build(self):
        text = (ROOT / "Dockerfile.lmcache").read_text()
        self.assertIn("BUILD_WITH_CUDA=1", text)
        self.assertIn("TORCH_CUDA_ARCH_LIST=12.1a", text)
        self.assertNotIn("NO_GPU_EXT=1", text)


if __name__ == "__main__":
    unittest.main()
