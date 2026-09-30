"""Compare completed diagnostic arms; host launch durations are not GPU times."""
import argparse
import json
from pathlib import Path

NODES = {"dusty", "toby", "rusty", "kirby"}
LENGTHS = {128, 385, 16384}


def launches(lines):
    pending = {}
    rows = []
    for line in lines:
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        meta = dict(item.split("=", 1) for item in fields[1:] if "=" in item)
        if not {"node", "pid", "tid", "job"} <= meta.keys():
            continue
        events = [item for item in fields[1:] if "=" not in item]
        if len(events) != 1:
            raise ValueError("Ambiguous timing event")
        event = events[0]
        if event not in {"kernel-launch-begin", "kernel-launch-end"}:
            continue
        key = tuple(meta[name] for name in ("node", "pid", "tid", "job"))
        now = int(fields[0])
        if event.endswith("begin"):
            if key in pending:
                raise ValueError("Overlapping launch pair")
            pending[key] = now, int(meta["prepared"])
        else:
            if key not in pending:
                raise ValueError("Launch end without begin")
            start, prepared = pending.pop(key)
            if now < start:
                raise ValueError("Nonmonotonic launch pair")
            rows.append({"node": key[0], "pid": key[1], "tid": key[2],
                         "job": key[3], "prepared": prepared,
                         "start_monotonic_ns": start,
                         "host_ms": (now - start) / 1_000_000})
    if pending:
        raise ValueError("Incomplete launch pair: arm may still be running")
    return rows


def arm(host_directory, report_path):
    if not (host_directory / "exit-code").is_file():
        raise ValueError("Arm has no terminal driver receipt")
    code = int((host_directory / "exit-code").read_text())
    if code not in (0, 1):
        raise ValueError("Unexpected diagnostic driver status")
    report = json.loads(report_path.read_text())
    if len(report) != 3 or {r["length"] for r in report} != LENGTHS:
        raise ValueError("Wrong repeatability matrix")
    if any(r["repeats"] != 6 or type(r["correct"]) is not bool
           or type(r["identical"]) is not bool for r in report):
        raise ValueError("Incomplete repeatability verdict")
    rows = []
    for node in sorted(NODES):
        paths = sorted((host_directory / node).glob("*.log"))
        if not paths:
            raise ValueError("Missing host timing logs: " + node)
        node_rows = [row for path in paths for row in launches(path.read_text().splitlines())]
        if not node_rows or any(row["node"] != node for row in node_rows):
            raise ValueError("Wrong or empty node timing receipt: " + node)
        if not any(row["prepared"] == 385 for row in node_rows):
            raise ValueError("Missing 385-row lookup evidence: " + node)
        rows.extend(node_rows)
    return {"host_directory": str(host_directory), "report": str(report_path),
            "driver_exit": code, "repeatability": report,
            "all_correct_and_identical": all(r["correct"] and r["identical"] for r in report),
            "long_launch_calls": [r for r in rows if r["host_ms"] >= 1000],
            "first_385_call_per_node": {
                node: min((r for r in rows if r["node"] == node and r["prepared"] == 385),
                          key=lambda r: r["start_monotonic_ns"]) for node in sorted(NODES)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", nargs=2, required=True,
                        metavar=("HOST_DIRECTORY", "REPEATABILITY_REPORT"))
    args = parser.parse_args()
    result = {"scope": "Diagnostic comparison only; host timings, not GPU duration or causality",
              "arms": [arm(Path(directory), Path(report)) for directory, report in args.arm]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
