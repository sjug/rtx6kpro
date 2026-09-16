#!/usr/bin/env python3
"""Podman image keep-latest-3 prune plan over the Spark fleet (plan only, no removal).

Per node: keep the currently-serving image (image of any running container)
plus the 2 most recent *named* images, all other IDs go on the removal list.
References a fresh collection (`podman images -a` + `podman ps -a`) into
outdir; existing files are reused unless --force.

Outputs per node: <node>.keep.txt, <node>.remove.tsv (ID\\ttag\\tcreated\\tnote),
<node>.rmi-command.txt (ready-to-run ssh command). Plus README.md summary.

Usage:
    scripts/podman-image-prune-plan.py [--out DIR] [--nodes n1,...] [--force]
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
KEEP = 3
SSH_ARGS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
IMAGES_FMT = ('podman images -a --format '
              '"{{.ID}}\\t{{.Repository}}\\t{{.Tag}}\\t{{.CreatedAt}}\\t{{.Size}}"')
PS_FMT = 'podman ps -a --format "{{.Image}}\\t{{.State}}\\t{{.Names}}"'


def parse_ts(s):
    m = re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S* ([-+]\d{4})", s)
    return datetime.datetime.strptime(f"{m.group(1)} {m.group(2)}",
                                      "%Y-%m-%d %H:%M:%S %z")


def norm(name):
    return name.replace("docker.io/", "").replace("localhost/", "") \
                .replace("library/", "")


def collect(nodes, outdir, force):
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in nodes if force or not (outdir / f"{n}.images-id.tsv").exists()]

    def one(node):
        for suffix, fmt in ((".images-id.tsv", IMAGES_FMT), (".ps.tsv", PS_FMT)):
            r = subprocess.run(["ssh", *SSH_ARGS, "-n", node, fmt],
                               capture_output=True, text=True, timeout=180)
            (outdir / f"{node}{suffix}").write_text(r.stdout + r.stderr)
            if r.returncode != 0:
                return node, r.returncode
        return node, 0

    for node, rc in (concurrent.futures.ThreadPoolExecutor(max_workers=len(todo))
                     .map(one, todo) if todo else []):
        if rc:
            print(f"WARNING: {node}: collection failed (rc={rc})")
    return nodes


def nid(s):
    if re.fullmatch(r"(sha256:)?[0-9a-f]{12}", s):
        return s.replace("sha256:", "")[:12]
    if re.fullmatch(r"sha256:[0-9a-f]{64}", s):
        return s[7:19]
    return None


def plan(node, outdir):
    images = collections.defaultdict(list)   # id -> [(kind, name, size, created)]
    for line in open(outdir / f"{node}.images-id.tsv"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 5 or not nid(p[0]):
            continue
        repo, name = p[1], p[2]
        images[nid(p[0])].append(
            {"name": f"{repo}:{name}" if repo != "<none>" else "<none>",
             "size": parse_size(p[4]), "created": parse_ts(p[3])})

    running, blockers = collections.defaultdict(list), {}
    for line in open(outdir / f"{node}.ps.tsv"):
        p = line.rstrip('\n').rstrip('"').split('\t')
        if len(p) < 3:
            continue
        img, state, cname = p[0], p[1], p[2]
        iid = nid(img)
        if iid is None:
            iid = next((i for i, ts in images.items()
                        if any(norm(t["name"]) == norm(img) for t in ts)), None)
            if iid is None:
                continue
        if state == "running":
            running[iid].append(cname)
        blockers[iid] = cname

    def recent(ids):   # most recent first; callers exclude <none> for recency
        return sorted(ids, key=lambda i: max(t["created"] for t in images[i]),
                      reverse=True)

    all_ids = list(images)
    serving = [i for i in recent(all_ids) if i in running]
    keep = list(serving)
    keep += [i for i in recent(all_ids)
             if i not in running and images[i][0]["name"] != "<none>"][: max(0, KEEP - len(keep))]
    keep = list(dict.fromkeys(keep))
    remove = [i for i in all_ids if i not in keep]
    return images, keep, remove, running, blockers


def parse_size(s):
    s = s.strip()
    for suf, mult in (("TB", 1e3), ("GB", 1), ("MB", 1e-3), ("kB", 1e-6), ("B", 0)):
        if s.endswith(suf):
            return float(s[:-len(suf)]) * mult
    return 0


def human(nimages, gb_total):
    gb = gb_total / 1000
    if gb >= 1:
        return f"{nimages} images, ~{gb:.1f}TB nominal"
    return f"{nimages} images, ~{gb_total:.1f}GB nominal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default=",".join(NAMES))
    ap.add_argument("--out", default=str(
        REPO / f"logs/podman-prune-plan-{datetime.date.today():%Y%m%d}"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    outdir = Path(args.out)
    collect(nodes, outdir, args.force)

    rows = []
    for n in nodes:
        images, keep, remove, running, blockers = plan(n, outdir)
        keepf = outdir / f"{n}.keep.txt"
        remf = outdir / f"{n}.remove.tsv"
        with keepf.open("w") as f, remf.open("w") as g:
            f.write("# ID      role      created                     tags\n")
            for i in keep:
                role = "serving" if i in running else "recent"
                newest = max(images[i], key=lambda t: t["created"])
                f.write(f"{i}  {role:8}  {newest['created']:%Y-%m-%d %H:%M}  "
                        f"{', '.join(t['name'] for t in images[i])}\n")
            for i in remove:
                note = "dangling" if any(t["name"] == "<none>" for t in images[i]) else ""
                if i in blockers:
                    note = f"USED BY container {blockers[i]}"
                for t in images[i]:
                    g.write(f"{i}\t{t['name']}\t{t['created']:%Y-%m-%d %H:%M}\t{note}\n")
        cmd = f'ssh -n {n} "podman rmi {" ".join(remove)}"'
        (outdir / f"{n}.rmi-command.txt").write_text(cmd + "\n")
        nom = sum(t["size"] for i in remove for t in images[i])
        rows.append((n, len(images), len(keep), len(remove),
                     sum(1 for i in remove if i not in blockers),
                     sum(t["size"] for i in remove for t in images[i]),
                     nom))

    o = [f"# Podman keep-latest-{KEEP} prune plan ({datetime.date.today():%Y-%m-%d})\n",
         f"Plan only — removals NOT executed. Rule: keep the serving image "
         f"(running container) + {KEEP - 1} most recent named images per node.\n",
         "| node | total | keep | remove | removable now | blocked (container) | nominal sum to free |",
         "|---|---:|---:|---:|---:|---:|---|"]
    for n, total, k, r, unblocked, sz, nom in rows:
        o.append(f"| {n} | {total} | {k} | {r} | {unblocked} | {r - unblocked} | {human(r, nom)} |")
    o.append("\nPer node: `<node>.keep.txt`, `<node>.remove.tsv`, `<node>.rmi-command.txt`. "
             "Images referenced by any container (even exited) fail plain `rmi`; those are "
             "flagged `USED BY container ...` in the remove TSV — remove the container first "
             "(`podman rm`) or use `rmi -f` deliberately.\n")
    (outdir / "README.md").write_text("\n".join(o) + "\n")
    print(f"report: {outdir/'README.md'}")
    for n, total, k, r, unblocked, sz, nom in rows:
        print(f"  {n:8} total {total:4} keep {k} remove {r:4} "
              f"(now-removable {unblocked}, blocked {r-unblocked})")


if __name__ == "__main__":
    main()
