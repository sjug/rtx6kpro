#!/usr/bin/env python3
"""Execute the frozen R32 regressions, rejecting skips or incomplete matrices."""

if not __debug__:
    raise RuntimeError("R32 verification requires Python assertions enabled")

import argparse
import os
import subprocess
import sys
from pathlib import Path

from flashkda_counts import RequiredMatrix

CASES = {
    "pool_tail": ("tests/kernels/test_glm_pool_tail_selection.py", 235, None),
    "sparse_cleanup": ("tests/v1/core/test_mamba_sparse_cleanup.py", 7, None),
    "warmup": (
        "tests/kernels/moe/test_b12x.py::test_b12x_moe_warmup_runs_each_planner_regime_once",
        1, None,
    ),
    "pooled_indexer": (
        "tests/models/test_glm5next_pooled_indexer.py::test_glm53_pool_expansion_appends_only_the_incomplete_tail",
        1, None,
    ),
    "mamba_retirement": (
        "tests/v1/core/test_single_type_kv_cache_manager.py", 7,
        "test_mamba_retirement_crosses_null_gaps or test_mamba_retirement_bounds_prefill_states",
    ),
}


class RequiredCases(RequiredMatrix):
    def __init__(self, count):
        super().__init__()
        self.expected = count

    def verify(self, exit_code):
        if (exit_code != 0 or self.collected != self.expected
                or self.passed != self.expected or self.invalid):
            raise RuntimeError(
                f"R32 matrix incomplete: expected={self.expected}, collected={self.collected}, "
                f"passed={self.passed}, exit={exit_code}, invalid={self.invalid}"
            )


def run_case(name):
    # Imported only in the child: each process owns exactly one pytest session.
    import pytest

    os.chdir("/opt/jovian-judgement/vllm")
    os.environ["B12X_GLM53_GPU_TEST"] = "1"
    path, count, selector = CASES[name]
    gate = RequiredCases(count)
    confcutdir = path.rsplit("/", 1)[0]
    args = ["-s", "-vv", f"--confcutdir={confcutdir}", path]
    if selector:
        args.extend(["-k", selector])
    status = pytest.main(args, plugins=[gate])
    gate.verify(status)
    print(f"R32-REGRESSION-PASS {path}: {count} passed", flush=True)


def run_all():
    for name in CASES:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--case", name],
            check=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "B12X_GLM53_GPU_TEST": "1"},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.case:
        run_case(args.case)
    else:
        run_all()


if __name__ == "__main__":
    main()
