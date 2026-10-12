#!/usr/bin/env python3
"""Retain inherited strict matrices; the warmup matrix is 20 cases at vLLM 19f2c20e (5 dtype pairs x 2 capacities x 2 cutoffs)."""
import sys

WARMUP = 'tests/kernels/moe/test_b12x.py::test_b12x_moe_prefill_capacity_and_exact_decode_reuse'


def adapted_cases(cases):
    if cases.get('warmup') != (WARMUP, 6, None):
        raise RuntimeError('Inherited warmup contract changed')
    return {**cases, 'warmup': (WARMUP, 20, None)}


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('Inherited gates require assertions')
    sys.path.insert(0, '/gate/inherited/tests')
    import run_r38_regressions as inherited
    inherited.CASES = adapted_cases(inherited.CASES)
    # Its child processes must repeat this adapter, preserving one pytest session each.
    inherited.__file__ = __file__
    inherited.main()
