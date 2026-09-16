#!/usr/bin/env python3
"""Execute (or dry-run) the keep-latest-3 image prune plan over the Spark fleet.

Phases (per node):
  0. fresh snapshot (images + containers), pre-checks:
       - rebuild keep/remove with plan() against the FRESH data
       - abort node if fresh keep-set differs from the approved plan,
         if any serving (running-container) image is on the remove list,
         or if a podman/buildah build process is active
  1. (execute) rm approved exited containers, then `podman rmi <ids...>`;
     (dry-run) write the commands to <node>.dry-run.txt instead
  2/3. (execute-only, not yet implemented here) leftovers + post-inventory

Plan lists come from --plan-dir (podman-image-prune-plan.py output).
Only image/container removal by explicit list; never `podman prune`.

Usage:
    scripts/podman-image-prune-exec.py [--dry-run] [--plan-dir DIR]
        [--out DIR] [--nodes ...]
"""

import argparse
import datetime
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SSH_ARGS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]

_spec = importlib.util.spec_from_file_location(
    "prune_plan", REPO / "scripts/podman-image-prune-plan.py")
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)


def ssh(node, cmd, timeout=120):
    r = subprocess.run(["ssh", *SSH_ARGS, "-n", node, cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout, r.stderr, r.returncode


def load_plan(plan_dir, node):
    keep = [l.split()[0] for l in open(plan_dir / f"{node}.keep.txt")
            if not l.startswith("#")]
    remove = list(dict.fromkeys(
        l.split("\t")[0] for l in open(plan_dir / f"{node}.remove.tsv")
        if len(l.split("\t")) >= 4))
    return keep, remove


def resolve_leftovers(node, plan_dir, execdir):
    """Phase 2: rm exit/external containers pinning removal-list images
    (buildah working containers via buildah rm), then rmi -f the remaining
    leftover image IDs (justified: every tag of those IDs is on the removal
    list and no live container references them). Returns leftover text."""
    pp.collect([node], execdir, force=True)
    out, err, rc = ssh(node, 'podman ps -a --external --format '
                                 '"{{.ImageID}}\\t{{.Image}}\\t{{.State}}\\t{{.Names}}"',
                       timeout=60)
    (execdir / f"{node}.psx.tsv").write_text(out + err)

    present = {l.split("\t")[0] for l in open(execdir / f"{node}.images-id.tsv")
               if len(l.split("\t")) >= 5 and pp.nid(l.split("\t")[0])}
    _, remove_p = load_plan(plan_dir, node)
    leftovers = [i for i in remove_p if i in present]

    pins = []
    for l in open(execdir / f"{node}.psx.tsv"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 4 and p[0] in leftovers and p[2] != "running":
            pins.append(p)
    notes = []
    for iid, img, state, name in pins:
        rmc = "podman rm"
        if state == "storage" or name.endswith("-working-container"):
            _, _, brc = ssh(node, f"command -v buildah >/dev/null && buildah rm {name}",
                            timeout=120)
            if brc == 0:
                notes.append(f"buildah rm {name} (storage artifact pinning {iid})")
            else:
                rmc = "podman rm --storage"
                _, err2, rc2 = ssh(node, f"{rmc} {name}", timeout=120)
                notes.append(f"{rmc} {name} (state={state})" if rc2 == 0
                             else f"FAILED rm {name}: {err2.strip()[:120]}")
            continue
        _, err2, rc2 = ssh(node, f"{rmc} {name}", timeout=120)
        notes.append(f"{rmc} {name} (state={state}, pinned {iid})" if rc2 == 0
                     else f"FAILED rm {name}: {err2.strip()[:120]}")

    # re-read what still remains and force-rmi it
    pp.collect([node], execdir, force=True)
    present2 = {l.split("\t")[0] for l in open(execdir / f"{node}.images-id.tsv")
                if len(l.split("\t")) >= 5 and pp.nid(l.split("\t")[0])}
    still = [i for i in leftovers if i in present2]
    if still:
        previous_leftovers = set(leftovers)
        still_forced = [i for i in still if i in previous_leftovers]
        if still_forced:
            _, err3, rc3 = ssh(node, f"podman rmi -f {' '.join(still_forced)}",
                               timeout=900)
            notes.append(f"podman rmi -f {len(still_forced)} "
                         f"leftovers rc={rc3}" + ("\n" + err3[:500] if rc3 else ""))
        pp.collect([node], execdir, force=True)
        present3 = {l.split("\t")[0] for l in
                    open(execdir / f"{node}.images-id.tsv")
                    if len(l.split("\t")) >= 5 and pp.nid(l.split("\t")[0])}
        remain = [i for i in still if i in present3]
        if remain:
            notes.append(f"STILL LEFTOVER: {remain}")
    return notes


def precheck(node, plan_dir, execdir):
    """Return (status, reason, keep, remove, rm_containers)."""
    pp.collect([node], execdir, force=True)
    images, keep_f, remove_f, running, blockers = pp.plan(node, execdir)
    keep_p, remove_p = load_plan(plan_dir, node)

    if keep_f != keep_p:
        return "ABORT", (f"keep-set changed since plan: fresh {keep_f} vs planned {keep_p}"
                         f" (new image? re-run plan)"), keep_f, remove_f, [], None
    out, _, rc = ssh(node, "ps ax -o args | grep -E '[p]odman (build|pull)|[b]uildah'")
    if rc == 0:
        return "ABORT", f"build/pull in flight: {out.strip()}", keep_f, remove_f, [], None

    gone = [i for i in remove_p if i not in images]
    running_blocked = [i for i in remove_p if i in running]
    if running_blocked:
        return "ABORT", f"removal list now serving: {running_blocked}", keep_f, remove_f, [], None

    # exited containers referencing removal images (rm-able with approval)
    rm_containers = []
    for l in open(execdir / f"{node}.ps.tsv"):
        p = l.rstrip('\n').split('\t')
        if len(p) < 3 or p[1] != "exited":
            continue
        iid = pp.nid(p[0]) or next(
            (i for i, ts in images.items()
             if any(pp.norm(t["name"]) == pp.norm(p[0]) for t in ts)), None)
        if iid and iid in remove_p:
            rm_containers.append((iid, p[2]))

    return "OK", f"{len(gone)} already absent", keep_f, remove_f, rm_containers, gone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-dir", default=str(
        REPO / "logs/podman-prune-plan-20260915"))
    ap.add_argument("--out", default=str(
        REPO / f"logs/podman-prune-exec-{datetime.date.today():%Y%m%d}"))
    ap.add_argument("--nodes", default=",".join(pp.NAMES))
    ap.add_argument("--rm-blockers", action="store_true",
                    help="rm exited containers referencing removal-list images "
                         "(approved for the jj-r28 case on dusty/kirby)")
    args = ap.parse_args()
    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    plan_dir, execdir = Path(args.plan_dir), Path(args.out)
    execdir.mkdir(parents=True, exist_ok=True)

    report = [f"Podman image prune exec — {datetime.datetime.now():%Y-%m-%d %H:%M} "
              f"({'DRY RUN' if args.dry_run else 'EXECUTE'})", ""]
    print(f"mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    for n in nodes:
        status, reason, keep, remove, rm_containers, gone = precheck(n, plan_dir, execdir)
        line = (f"{status} {n}: {reason}; exited-blockers: "
                + (", ".join(nm for _, nm in rm_containers) or "-"))
        report.append(line)
        print(line)
        if status == "ABORT":
            continue
        exec_ids = [i for i in remove if i not in gone]
        pending_rm = [nm for _, nm in rm_containers] if args.rm_blockers else []
        if not args.rm_blockers:
            blocked = {iid for iid, _ in rm_containers}
            exec_ids = [i for i in exec_ids if i not in blocked]
        would = []
        if pending_rm:
            would.append(f"podman rm {' '.join(pending_rm)}")
        would.append(f"podman rmi {' '.join(exec_ids)}")
        (execdir / f"{n}.dry-run.txt").write_text(
            f"# status: {status} ({reason})\n# would-remove: {len(exec_ids)} images"
            f" (+{len(gone)} already absent); rm-containers: {len(pending_rm)}\n"
            + "\n".join(f"ssh -n {n} \"{c}\"" for c in would) + "\n")
        if not args.dry_run:
            for c in would:
                out, err, rc = ssh(n, c, timeout=900)
                log = f"cmd: {c}\nrc={rc}\n{out}{err}\n"
                (execdir / f"{n}.rmi.log").open("a").write(log)
                report.append(log.strip())
            print(f"  {n}: executed, log at {execdir/f'{n}.rmi.log'}")

    if not args.dry_run:
        report.append("")
        for n in nodes:
            try:
                notes = resolve_leftovers(n, plan_dir, execdir)
            except Exception as e:
                notes = [f"resolve error: {e}"]
            report.append(f"{n}: " + ("; ".join(notes) if notes else "no leftover pins"))
            print(f"  {n}: " + ("; ".join(notes) or "clean"))
    (execdir / "README.md").write_text("\n".join(report) + "\n")
    print(f"report: {execdir/'README.md'}")


if __name__ == "__main__":
    main()
