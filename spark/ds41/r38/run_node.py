"""Render or launch one fixed DS4.1 rank. Never stop or remove any container."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys

from contract import IMAGE, IMAGE_ID, NAME, NODES, ROOT, require, sha256
from contract import runtime_environment, serve_command, settings, verify_kit
from verify_model import verify


def read(command):
    return subprocess.check_output(command, text=True, timeout=30).strip()


def node_for(env, live_host):
    node = env.get("NODE", live_host)
    require(node in NODES, "NODE must be dusty, toby, rusty or kirby")
    if env.get("DRY_RUN", "0") != "1":
        require(node == live_host, f"NODE={node} does not match host {live_host}")
    rank, ip = NODES[node]
    expected_role = "head" if rank == 0 else "worker"
    require(env.get("ROLE", expected_role) == expected_role, f"{node} requires ROLE={expected_role}")
    require(env.get("NODE_RANK", str(rank)) == str(rank), "NODE_RANK disagrees with fixed node map")
    require(env.get("HOST_IP", ip) == ip, "HOST_IP disagrees with switched fabric map")
    return node


def command(env, node, cfg, digest):
    home_dir = Path.home()
    hf = Path(env.get("HF_CACHE", home_dir / ".cache/huggingface")).resolve()
    cache = Path(env.get("CACHE", home_dir / ".cache/vllm-jj-ds41-tp4")).resolve()
    require(cache != hf and not cache.is_relative_to(hf) and not hf.is_relative_to(cache),
            "HF cache and writable runtime cache must not overlap")
    for path in (hf, cache, ROOT):
        require(":" not in str(path) and "," not in str(path), "Unsupported mount path punctuation")
    values = runtime_environment(cfg, node)
    values["DS41_KIT_SHA256"] = digest
    cmd = ["podman", "run", "-d", "--pull", "never", "--name", NAME,
           "--device", "nvidia.com/gpu=all", "--device", "/dev/infiniband",
           "--security-opt", "label=disable", "--network", "host", "--ipc", "host", "--init",
           "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", "--ulimit", "nofile=500000:500000",
           "-v", f"{hf}:/root/.cache/huggingface:ro", "-v", f"{cache}:/cache:rw",
           "-v", f"{cache}/tmp:/container-tmp:rw", "-v", f"{ROOT}:/opt/ds41:ro"]
    seccomp = env.get("IO_URING_SECCOMP_PROFILE")
    if seccomp:
        path = Path(seccomp)
        require(path.is_absolute() and path.is_file(), "io_uring seccomp profile must be a local absolute file")
        require(":" not in str(path) and "," not in str(path), "Unsupported seccomp path punctuation")
        policy = json.loads(path.read_text())
        require(policy.get("defaultAction") in ("SCMP_ACT_ERRNO", "SCMP_ACT_KILL",
                                               "SCMP_ACT_KILL_PROCESS", "SCMP_ACT_TRAP"),
                "Seccomp profile must default to deny")
        require(isinstance(policy.get("syscalls"), list), "Malformed seccomp policy")
        values["DS41_SECCOMP_SHA256"] = sha256(path)
        cmd += ["--security-opt", f"seccomp={path}"]
    for key, value in values.items():
        cmd += ["-e", f"{key}={value}"]
    cmd += ["--label", f"local-inference.ds41.kit.sha256={digest}",
            "--entrypoint", "/opt/venv/bin/python", IMAGE_ID, "/opt/ds41/launch.py"]
    return cmd, hf, cache


def verify_memory(info):
    values = {line.split(":", 1)[0]: int(line.split()[1]) * 1024 for line in info.splitlines()}
    require(values["MemAvailable"] >= 100 * (1 << 30), "Admission requires 100 GiB MemAvailable while idle")
    require(values["SwapTotal"] - values["SwapFree"] <= 1 << 30,
            "Admission requires at most 1 GiB swap used while idle")


def verify_fabric(node, sysfs=Path("/sys/class/infiniband")):
    for hca, interface, subnet in (("rocep1s0f0", "enp1s0f0np0", "10.11.11"),
                                   ("roceP2p1s0f0", "enP2p1s0f0np0", "10.11.12")):
        address = f"{subnet}.{NODES[node][1].rsplit('.', 1)[1]}"
        addresses = json.loads(read(["ip", "-j", "-4", "address", "show", "dev", interface]))
        require(any(a["local"] == address for iface in addresses for a in iface["addr_info"]),
                f"Switched address {address} is not assigned to {interface}")
        for peer, (_, primary) in NODES.items():
            if peer == node:
                continue
            peer_ip = f"{subnet}.{primary.rsplit('.', 1)[1]}"
            route = json.loads(read(["ip", "-j", "route", "get", peer_ip]))[0]
            require(route.get("dev") == interface and route.get("prefsrc") == address,
                    f"Peer route to {peer} does not use switched rail {interface}")
        port = sysfs / hca / "ports/1"
        require("ACTIVE" in (port / "state").read_text(), f"Inactive HCA {hca}")
        require("RoCE v2" in (port / "gid_attrs/types/3").read_text(), f"Wrong GID type on {hca}")
        require((port / "gid_attrs/ndevs/3").read_text().strip() == interface,
                f"GID 3 maps to the wrong netdev on {hca}")
        gid_ip = ipaddress.IPv6Address((port / "gids/3").read_text().strip()).ipv4_mapped
        require(str(gid_ip) == address, f"GID 3 does not carry {address} on {hca}")


def preflight(node, hf):
    require(platform.machine() == "aarch64", "This runner requires aarch64")
    info = json.loads(read(["podman", "image", "inspect", IMAGE]))[0]
    require(info["Id"].removeprefix("sha256:") == IMAGE_ID, "R38 image ID mismatch")
    require(info["Architecture"] == "arm64", "Not an ARM64 image")
    labels = info.get("Labels") or info.get("Config", {}).get("Labels", {})
    require(labels.get("local-inference.release.name") == "jj-r38-spark-sm121", "Wrong release label")
    require(labels.get("local-inference.lmcache.default") == "disabled", "LMCache must remain disabled")
    # Explicit status capture: command failures never mean 'idle'.
    require(not read(["podman", "ps", "--format", "{{.Names}}"]),
            "Existing containers are running; arrange an approved serving window first")
    require(not read(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]),
            "GPU processes exist; refuse to overlap this launch with other GPU work")
    exists = subprocess.run(["podman", "container", "exists", NAME], check=False, timeout=30)
    require(exists.returncode == 1, "Candidate exists or container-state probe failed; refusing replacement")
    verify_memory(Path("/proc/meminfo").read_text())
    verify_fabric(node)
    return verify(hf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true", help="Host checks only, no container or GPU")
    args = parser.parse_args()
    require(os.environ.get("DRY_RUN", "0") in ("0", "1"), "DRY_RUN must be 0 or 1")
    node = node_for(os.environ, socket.gethostname().split(".")[0])
    cfg = settings(os.environ)
    digest = verify_kit()
    cmd, hf, cache = command(os.environ, node, cfg, digest)
    if os.environ.get("DRY_RUN", "0") == "1":
        print(json.dumps({"status": "DRY_RUN", "node": node, "image_id": IMAGE_ID,
                          "kit_sha256": digest, "podman_argv": cmd,
                          "serve_argv": serve_command(cfg, node)}, indent=2))
        return
    model = preflight(node, hf)
    if args.preflight_only:
        print(json.dumps({"status": "HOST_PREFLIGHT_PASS", "model": model, "node": node}, indent=2))
        return
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True, timeout=90)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"DS41 runner refused: {exc}", file=sys.stderr)
        sys.exit(78)
