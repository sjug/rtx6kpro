#!/usr/bin/env python3
"""Archive read-only node evidence before a qualification profile changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--since", required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)


def capture(host):
    commands = {
        "container.log": ["podman", "logs", "--timestamps", "ds41-flash-jj-r38-tp4"],
        "kernel.log": ["journalctl", "-k", "--since", args.since, "--no-pager", "-o", "short-iso"],
        "observer.log": ["journalctl", "--user", "-u", "ds41-observer", "--since", args.since,
                         "--no-pager", "-o", "cat"],
        "memory.txt": ["cat", "/proc/meminfo", "/proc/pressure/memory"],
        "identity.txt": ["podman", "inspect", "--format",
                         "{{.Id}} {{.Image}} {{.State.Status}} {{.State.OOMKilled}} {{.State.StartedAt}}",
                         "ds41-flash-jj-r38-tp4"],
    }
    import shlex
    for suffix, command in commands.items():
        start = time.time()
        with (args.out / f"{host}-{suffix}").open("xb") as output:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.join(command)],
                stdout=output, stderr=subprocess.STDOUT, timeout=60, check=False)
        receipt = {"host": host, "command": command, "started_unix": start,
                   "elapsed_s": time.time() - start, "exit_code": result.returncode}
        (args.out / f"{host}-{suffix}.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"Capture failed: {host} {suffix}")
    print(f"CAPTURE_PASS {host}", flush=True)


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(capture, ("dusty", "toby", "rusty", "kirby")))
