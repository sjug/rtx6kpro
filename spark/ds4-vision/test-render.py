#!/usr/bin/env python3
"""Render both node roles in disposable CPU-only containers, then parse them."""
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
image = "localhost/voipmonitor/vllm:jj-r32-spark-sm121"
for role in ("head", "worker"):
    text = subprocess.check_output(
        ["bash", str(root / "run-node.sh")], text=True,
        env={**os.environ, "ROLE": role, "DRY_RUN": "1"},
    )
    original = shlex.split(text.removeprefix("DRY-RUN:"))
    cmd = ["podman", "run", "--rm", "--pull", "never", "--network", "none",
           "--memory", "2g", "-e", "DRY_RUN=1"]
    # Keep every candidate environment variable, but no host cache, network,
    # devices, name or serving-related mount in this render-only container.
    i = 3
    while original[i] != "--entrypoint":
        if original[i] == "-e":
            cmd.extend(original[i:i + 2])
        i += 2 if original[i] in (
            "--pull", "--name", "--device", "--security-opt", "--network",
            "--ipc", "--ulimit", "-v", "-e",
        ) else 1
    cmd += ["-v", f"{root}/launch-in-container.sh:/opt/ds4-vision/launch-in-container.sh:ro"]
    cmd += original[i:]
    result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    output = result.stdout + result.stderr
    (root / "receipts" / f"{role}-render.txt").write_text(output)
    command = next(line.removeprefix("Command: ") for line in output.splitlines()
                   if line.startswith("Command: "))
    argv = shlex.split(command)
    serve_args = argv[argv.index("serve") + 1:]
    parsed = subprocess.run(
        ["podman", "run", "--rm", "-i", "--pull", "never", "--network", "none",
         "--memory", "2g", "-v", f"{root}/parse-cli.py:/parse-cli.py:ro",
         "--entrypoint", "/opt/venv/bin/python", image, "/parse-cli.py"],
        input=json.dumps(serve_args), text=True, capture_output=True,
    )
    print(parsed.stdout + parsed.stderr, flush=True)
    parsed.check_returncode()
print("HEAD-WORKER-RENDER-PASS", flush=True)
