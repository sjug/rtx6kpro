#!/usr/bin/env python3
"""User-authorized rusty/toby-only cutover after staging, with durable receipts."""
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

root = Path(__file__).resolve().parent
out = root / "receipts"
out.mkdir(exist_ok=True)
remote = "/home/jugs/git/ds4-vision"
image = "74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c"
name = "ds4-vision-jj-r32-tp2"
model = "DeepSeek-V4-Flash-Vision-Exp"


def ssh(node, command, timeout=90):
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, command],
        text=True, timeout=timeout,
    )


deadline = time.monotonic() + 7200
while True:
    status = ssh("rusty", f"systemctl --user show ds4-vision-peer-stage -p ActiveState -p Result; "
                 f"tail -c 1000 {remote}/receipts/peer-transfer.log")
    if "MODEL-TRANSFER-PASS" in status:
        break
    if "ActiveState=failed" in status or time.monotonic() > deadline:
        raise RuntimeError(f"Staging failed or timed out: {status}")
    print(f"WAITING-FOR-MODEL-STAGING {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
    time.sleep(30)

for node in ("rusty", "toby"):
    actual = ssh(node, "podman image inspect localhost/voipmonitor/vllm:jj-r32-spark-sm121 --format '{{.Id}}'").strip()
    if actual != image:
        raise RuntimeError(f"Wrong candidate on {node}: {actual}")
    print(ssh(node, f"cd {remote} && sha256sum --check runtime-files.sha256 && "
              f"python3 verify-model.py --cache /home/jugs/.cache/huggingface "
              f"--manifest receipts/model-manifest.json"), flush=True)
    # The existing container remains the rollback, unchanged and not removed.
    old = ssh(node, "podman inspect ds4-0731-tp2 --format '{{json .State}} {{.Image}}'")
    (out / f"{node}-0731-before.txt").write_text(old)

print("CUTOVER-START " + time.strftime("%Y-%m-%dT%H:%M:%S%z"), flush=True)
for node in ("toby", "rusty"):
    print(ssh(node, "podman stop -t 60 ds4-0731-tp2", timeout=100), flush=True)
for node, role in (("toby", "worker"), ("rusty", "head")):
    launch = ssh(node, f"ROLE={role} bash {remote}/run-node.sh", timeout=100)
    (out / f"{node}-launch.txt").write_text(launch)
    print(launch, flush=True)

deadline = time.monotonic() + 1200
while time.monotonic() < deadline:
    for node in ("rusty", "toby"):
        state = json.loads(ssh(node, f"podman inspect {name} --format '{{{{json .State}}}}'"))
        if not state["Running"] or state["OOMKilled"]:
            raise RuntimeError(f"Candidate stopped on {node}: {state}")
        log = ssh(node, f"podman logs --tail 15 {name} 2>&1")
        print(f"BOOT {node}\n{log}", flush=True)
    try:
        with urllib.request.urlopen("http://rusty:8000/v1/models", timeout=5) as response:
            listing = json.load(response)
        if [m["id"] for m in listing["data"]] != [model]:
            raise RuntimeError(f"Unexpected advertised identity: {listing}")
        request = urllib.request.Request("http://rusty:8000/v1/chat/completions", json.dumps({
            "model": model, "messages": [{"role": "user", "content": "What is 17 times 23 minus 58? Reply with only the integer."}],
            "chat_template_kwargs": {"thinking": False}, "temperature": 0, "max_tokens": 64,
        }).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=90) as response:
            completion = json.load(response)
        (out / "first-completion.json").write_text(json.dumps(completion, indent=2) + "\n")
        choice = completion["choices"][0]
        if choice["finish_reason"] != "stop" or choice["message"]["content"].strip() != "333":
            raise RuntimeError(f"First completion incorrect: {completion}")
        print("FIRST-COMPLETION-PASS " + time.strftime("%Y-%m-%dT%H:%M:%S%z"), flush=True)
        break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(20)
else:
    raise RuntimeError("Readiness timed out after 20 minutes")

subprocess.run([sys.executable, "-u", str(root / "qualify.py"), "--out", str(out / "semantic"), "--long"], check=True)
for node in ("rusty", "toby"):
    (out / f"{node}-serve.log").write_text(ssh(node, f"podman logs {name} 2>&1"))
    final = ssh(node, f"podman inspect {name} --format '{{{{json .State}}}} {{{{.Image}}}}'")
    (out / f"{node}-final.txt").write_text(final)
    print(f"FINAL {node} {final}", flush=True)
(out / "QUALIFICATION-PASS").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
print("CUTOVER-AND-QUALIFICATION-PASS", flush=True)
