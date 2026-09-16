#!/usr/bin/env python3
"""One pytest session per process; zero cases and skips are not acceptance."""

import argparse
import sys

if not __debug__:
    raise RuntimeError("R37 gates require assertions enabled")

from flashkda_counts import RequiredMatrix


def main():
    import pytest

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int)
    parser.add_argument("--collect-only-count", action="store_true")
    args, pytest_args = parser.parse_known_args()
    if pytest_args and pytest_args[0] == "--":
        pytest_args.pop(0)
    gate = RequiredMatrix()
    if args.collect_only_count:
        status = pytest.main(["--collect-only", "-q", *pytest_args], plugins=[gate])
        gate.verify_collection(status, args.count)
        print(f"R37 COLLECTION PASS: {gate.collected} cases", flush=True)
        return
    status = pytest.main(["-s", "-vv", *pytest_args], plugins=[gate])
    if (status or gate.invalid or gate.collected == 0 or gate.passed != gate.collected
            or (args.count is not None and gate.collected != args.count)):
        raise RuntimeError(f"Incomplete gate: expected={args.count}, collected={gate.collected}, "
                           f"passed={gate.passed}, invalid={gate.invalid}, exit={status}")
    print(f"R37 REQUIRED PASS: {gate.passed} cases", flush=True)


if __name__ == "__main__":
    main()
