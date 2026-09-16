#!/usr/bin/env python3
"""Podman image inventory over a fleet of SSH-reachable nodes.

Collects `podman images -a` + `podman system df` + `df -h /` from each node
(in parallel, read-only), stores raw TSV/txt per node, and generates a
README.md report: fleet overview table, image->nodes presence matrix,
distribution histogram.

Usage:
    scripts/podman-image-inventory.py [--out DIR] [--nodes n1,n2,...] [--force]

Default --out: logs/podman-image-inventory-$(YYYYMMDD) under the repo root.
Existing per-node files are reused unless --force is given.
"""

import argparse
import collections
import concurrent.futures
import datetime
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NAMES = ["sparky", "buddy", "lucky", "rocky", "rusty", "toby", "dusty", "kirby"]

IMAGES_FMT = (
    "podman images -a --format "
    '"{{.Repository}}\\t{{.Tag}}\\t{{.ID}}\\t{{.Size}}\\t{{.Created}}"'
)
SYSDF_CMD = "df -h / | tail -1; podman system df"

SSH_ARGS = ["-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]


def collect(nodes, outdir, force):
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in nodes if force or not (outdir / f"{n}.images.tsv").exists()]

    def one(n):
        with open(outdir / f"{n}.images.tsv", "w") as f:
            r = subprocess.run(
                ["ssh", *SSH_ARGS, "-n", n, IMAGES_FMT],
                capture_output=True, text=True, timeout=120,
            )
            f.write(r.stdout + r.stderr)
        with open(outdir / f"{n}.systemdf.txt", "w") as f:
            r = subprocess.run(
                ["ssh", *SSH_ARGS, "-n", n, SYSDF_CMD],
                capture_output=True, text=True, timeout=120,
            )
            f.write(r.stdout + r.stderr)
        return n, r.returncode

    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(todo)) as ex:
            for n, rc in ex.map(one, todo):
                if rc != 0:
                    print(f"WARNING: {n}: ssh/podman failed (rc={rc})")
    return nodes


def load(nodes, outdir):
    """Return (per-node info, ...) where per-node image data is:
    imgs: distinct repo:tag keys (for the presence matrix),
    count: total entries, dangling: number of `<none>` entries.
    Each TSV line is one image; all dangling lines collapse to a single
    `<none>` key in the matrix, so counts come from the raw lines."""
    nodeimg, nodeinfo = collections.defaultdict(list), {}
    for n in nodes:
        seen, count, dangling = set(), 0, 0
        for line in open(outdir / f"{n}.images.tsv"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            count += 1
            repo, tag = p[0], p[1]
            if repo == "<none>":
                dangling += 1
                t = "<none>"
            else:
                t = f"{repo}:{tag}"
            if t not in seen:
                seen.add(t)
                nodeimg[n].append((t, parse_size(p[3])))
        txt = open(outdir / f"{n}.systemdf.txt").read()
        nodeinfo[n] = {
            "count": count,
            "dangling": dangling,
            "df": disk_line(txt),
            "images": re.search(
                r"^\s*Images\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)", txt, re.M),
        }
    return nodeimg, nodeinfo


def parse_size(s):
    s = s.strip()
    for suf, mult in (("TB", 1e3), ("GB", 1), ("MB", 1e-3), ("kB", 1e-6), ("B", 0)):
        if s.endswith(suf):
            return float(s[: -len(suf)]) * mult
    return 0


def disk_line(txt):
    """First line of df output whose mountpoint column ends in '/'. df
    prints 'Filesystem Size ... /' header first, data row second."""
    for line in txt.splitlines():
        f = line.split()
        if len(f) >= 6 and f[-1] == "/" and f[0] != "Filesystem":
            return f"{f[2]} of {f[1]} ({f[4]})"
    return "?"


def shortname(t):
    t = t.replace("localhost/voipmonitor/vllm:", "vllm:")
    t = t.replace("docker.io/nvidia/cuda:", "cuda:")
    t = t.replace("localhost/voipmonitor/", "vm:")
    return t


def report(nodes, nodeimg, nodeinfo, outdir):
    img2nodes = collections.defaultdict(set)
    for n in nodes:
        for t, _ in nodeimg[n]:
            img2nodes[t].add(n)
    allt = sorted(img2nodes)

    o = [f"# Podman image inventory — DGX Spark fleet "
         f"({datetime.date.today():%Y-%m-%d})\n",
         "Collected via `ssh <node> podman images -a` + `podman system df`. "
         "Raw TSVs alongside: `<node>.images.tsv` (per-image), "
         "`<node>.systemdf.txt` (storage/df).\n",
         "## Fleet overview\n",
         "| node | images | dangling | actual image storage | disk (/) |",
         "|---|---:|---:|---|---|"]
    for n in nodes:
        m = nodeinfo[n]["images"]
        dang = nodeinfo[n]["dangling"]
        o.append(f"| {n} | {nodeinfo[n]['count']} | {dang} | "
                 f"{m.group(3)} (in use: {m.group(2)}) | {nodeinfo[n]['df']} |")
    o.append("""
Notes: `actual image storage` = `podman system df` SIZE (layer-deduplicated;
naive per-image sums are far higher, e.g. sparky sums to 655GB but stores
164.2GB). `dangling` = `<none>:<none>` entries.

## Presence matrix

Distinct `repo:tag` across the fleet: {n}. Sorted by node-presence, then name.

| # | image | nodes present |
|---|---|---|
""".format(n=len(allt)))
    for i, t in enumerate(sorted(allt, key=lambda t: (-len(img2nodes[t]), t)), 1):
        here = ", ".join(sorted(img2nodes[t]))
        o.append(f"| {i} | `{shortname(t)}` | {here} |")
    o.append("\n## Distribution\n"
             "| image present on N nodes | distinct images |\n|---:|---:|")
    cnt = collections.Counter(len(img2nodes[t]) for t in allt)
    for k in sorted(cnt, reverse=True):
        o.append(f"| {k if k < 8 else '8 (all)'} | {cnt[k]} |")
    path = outdir / "README.md"
    path.write_text("\n".join(o) + "\n")
    return path, allt, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default=",".join(NAMES))
    ap.add_argument("--out", default=str(
        REPO / f"logs/podman-image-inventory-{datetime.date.today():%Y%m%d}"))
    ap.add_argument("--force", action="store_true",
                    help="re-collect even if per-node files exist")
    args = ap.parse_args()
    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    outdir = Path(args.out)
    collect(nodes, outdir, args.force)
    nodeimg, nodeinfo = load(nodes, outdir)
    path, allt, cnt = report(nodes, nodeimg, nodeinfo, outdir)
    print(f"report: {path}")
    print(f"fleet: {len(nodes)} nodes, {len(allt)} distinct repo:tag")
    for n in nodes:
        m = nodeinfo[n]["images"]
        print(f"  {n:8} {nodeinfo[n]['count']:4} images "
              f"({nodeinfo[n]['dangling']} dangling, {m.group(3)} actual)")


if __name__ == "__main__":
    main()
