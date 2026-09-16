"""Host launch preflight must reject a correct but non-executable mount."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

CHECK = Path(__file__).with_name("check-glm-launcher-identity.sh")


class LauncherPermissions(unittest.TestCase):
    def run_check(self, executable):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / "launcher.sh"
            launcher.write_text("#!/bin/bash\nexit 0\n")
            launcher.chmod(0o755 if executable else 0o644)
            digest = hashlib.sha256(launcher.read_bytes()).hexdigest()
            podman = root / "podman"
            podman.write_text(
                '#!/bin/bash\nif [[ $1 == image ]]; then printf "%s\\n" "$EXPECTED"; fi\n'
            )
            podman.chmod(0o755)
            return subprocess.run(
                ["bash", str(CHECK), "test-image", str(launcher), digest],
                env={**os.environ, "EXPECTED": digest,
                     "PATH": str(root) + os.pathsep + os.environ["PATH"]},
                text=True, capture_output=True, check=False,
            )

    def test_reject_non_executable_launcher(self):
        result = self.run_check(False)
        self.assertEqual(result.returncode, 78, result.stderr)

    def test_accept_executable_launcher(self):
        result = self.run_check(True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
