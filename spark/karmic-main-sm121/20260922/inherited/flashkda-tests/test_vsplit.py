"""Exactness matrix for K2 dispatch and packed recurrent checkpoint export.

Short fixed and varlen cases cover both sequence-offset dtypes, empty sequences,
all state input/output modes and both state dtypes against ``torch_ref``.
Packed-checkpoint cases additionally require unchanged native outputs and exact
independent prefix states, including 256-token checkpoints through long queries.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from torch_ref import torch_ref

import flash_kda

D = 128
H = 2
LOWER_BOUND = -5.0


@dataclasses.dataclass(frozen=True)
class StateCase:
    name: str
    has_input: bool
    has_output: bool
    dtype: torch.dtype | None


# ``no_state`` has no meaningful dtype.  Every state-bearing input/output
# mode is exercised independently with BF16 and FP32 state storage.
STATE_CASES = (
    StateCase("no_state", False, False, None),
    StateCase("in_only_bf16", True, False, torch.bfloat16),
    StateCase("in_only_fp32", True, False, torch.float32),
    StateCase("out_only_bf16", False, True, torch.bfloat16),
    StateCase("out_only_fp32", False, True, torch.float32),
    StateCase("in_out_bf16", True, True, torch.bfloat16),
    StateCase("in_out_fp32", True, True, torch.float32),
)


@dataclasses.dataclass(frozen=True)
class Problem:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    cu_seqlens: torch.Tensor | None
    num_sequences: int

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(D)


def _make_problem(
    *,
    batch: int,
    seq_len: int,
    cu_seqlens: torch.Tensor | None,
    num_sequences: int,
    seed: int,
) -> Problem:
    torch.manual_seed(seed)
    shape = (batch, seq_len, H, D)
    q = F.normalize(
        torch.randn(shape, dtype=torch.float32, device="cuda"), p=2, dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(shape, dtype=torch.float32, device="cuda"), p=2, dim=-1
    ).to(torch.bfloat16)
    return Problem(
        q=q,
        k=k,
        v=torch.randn(shape, dtype=torch.bfloat16, device="cuda"),
        g=torch.randn(shape, dtype=torch.bfloat16, device="cuda"),
        beta=torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda"),
        A_log=torch.rand(H, dtype=torch.float32, device="cuda"),
        dt_bias=torch.rand(H, D, dtype=torch.float32, device="cuda"),
        cu_seqlens=cu_seqlens,
        num_sequences=num_sequences,
    )


def _make_fixed_problem() -> Problem:
    # B>1 also checks fixed-length sequence-to-state indexing.  T=17 crosses a
    # chunk boundary and exercises the manual tail output store.
    return _make_problem(
        batch=2,
        seq_len=17,
        cu_seqlens=None,
        num_sequences=2,
        seed=101,
    )


def _make_stage_reuse_problem() -> Problem:
    # InputStages=3, so four full chunks are the smallest case that forces a
    # circular input stage to be reused.
    return _make_problem(
        batch=1,
        seq_len=64,
        cu_seqlens=None,
        num_sequences=1,
        seed=404,
    )


def _make_varlen_problem(cu_dtype: torch.dtype) -> Problem:
    # Repeated offsets put empty sequences at the beginning, middle, and end.
    # The two non-empty sequences cover both a one-token tail and 17 tokens
    # across two chunks without making the torch reference expensive.
    lengths = (0, 1, 0, 17, 0)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=cu_dtype, device="cuda")
    return _make_problem(
        batch=1,
        seq_len=offsets[-1],
        cu_seqlens=cu_seqlens,
        num_sequences=len(lengths),
        seed=202,
    )


def _state_template(problem: Problem, state_case: StateCase) -> torch.Tensor | None:
    if not state_case.has_input:
        return None
    assert state_case.dtype is not None
    # Generate through FP32 for deterministic BF16/FP32 cases with identical
    # logical values.  Each execution receives its own clone.
    torch.manual_seed(303)
    return torch.randn(
        (problem.num_sequences, H, D, D),
        dtype=torch.float32,
        device="cuda",
    ).to(state_case.dtype)


def _fresh_final_state(
    problem: Problem, state_case: StateCase
) -> torch.Tensor | None:
    if not state_case.has_output:
        return None
    assert state_case.dtype is not None
    # NaN makes a missing state store fail exact comparison, including for an
    # empty sequence whose expected state is the input state or all zeros.
    return torch.full(
        (problem.num_sequences, H, D, D),
        float("nan"),
        dtype=state_case.dtype,
        device="cuda",
    )


def _run_reference(
    problem: Problem,
    state_case: StateCase,
    initial_template: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    out = torch.full_like(problem.q, float("nan"))
    final_state = _fresh_final_state(problem, state_case)
    torch_ref(
        problem.q,
        problem.k,
        problem.v,
        problem.g,
        problem.beta,
        problem.scale,
        out,
        A_log=problem.A_log,
        dt_bias=problem.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=(
            initial_template.clone() if initial_template is not None else None
        ),
        final_state=final_state,
        cu_seqlens=problem.cu_seqlens,
    )
    torch.cuda.synchronize()
    return out, final_state


def _run_kernel(
    problem: Problem,
    state_case: StateCase,
    initial_template: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    out = torch.full_like(problem.q, float("nan"))
    final_state = _fresh_final_state(problem, state_case)
    workspace = torch.empty(
        flash_kda.get_workspace_size(
            problem.q.shape[0] * problem.q.shape[1],
            H,
            problem.num_sequences,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    flash_kda.fwd(
        problem.q,
        problem.k,
        problem.v,
        problem.g,
        problem.beta,
        problem.scale,
        out,
        A_log=problem.A_log,
        dt_bias=problem.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=(
            initial_template.clone() if initial_template is not None else None
        ),
        final_state=final_state,
        cu_seqlens=problem.cu_seqlens,
        workspace=workspace,
    )
    torch.cuda.synchronize()
    return out, final_state


def _assert_vsplit_exact(problem: Problem, state_case: StateCase) -> None:
    initial_template = _state_template(problem, state_case)
    expected_out, expected_state = _run_reference(
        problem, state_case, initial_template
    )

    actual_out, actual_state = _run_kernel(
        problem, state_case, initial_template
    )
    assert torch.equal(actual_out, expected_out), (
        f"output differs from torch_ref for {state_case.name}"
    )
    if expected_state is not None:
        assert actual_state is not None
        assert torch.equal(actual_state, expected_state), (
            f"final_state differs from torch_ref for {state_case.name}"
        )


@pytest.mark.parametrize("state_case", STATE_CASES, ids=lambda case: case.name)
def test_vsplit_fixed_exact(state_case: StateCase) -> None:
    _assert_vsplit_exact(_make_fixed_problem(), state_case)


def test_vsplit_stage_reuse_exact() -> None:
    _assert_vsplit_exact(
        _make_stage_reuse_problem(),
        StateCase("stage_reuse_fp32", True, True, torch.float32),
    )


@pytest.mark.parametrize(
    "cu_dtype",
    (torch.int32, torch.int64),
    ids=("cu_int32", "cu_int64"),
)
@pytest.mark.parametrize("state_case", STATE_CASES, ids=lambda case: case.name)
def test_vsplit_varlen_exact(
    state_case: StateCase, cu_dtype: torch.dtype
) -> None:
    _assert_vsplit_exact(_make_varlen_problem(cu_dtype), state_case)


@pytest.mark.parametrize("num_sequences", (4, 64))
@pytest.mark.parametrize("cu_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float32))
def test_packed_checkpoints_match_prefix_states_without_changing_outputs(
    num_sequences: int, cu_dtype: torch.dtype, state_dtype: torch.dtype
) -> None:
    """Export several states per sequence, including empty checkpoint ranges.

    Four sequences select the V-split kernel; 64 sequences select the complete
    value tile on a 188-SM GPU. Every exported state is checked against the
    independent recurrence evaluated on that exact sequence prefix.
    """
    lengths = [0, 17, 64, 65] * (num_sequences // 4)
    starts = [0]
    indptr = [0]
    checkpoints = []
    owners = []
    for seq, length in enumerate(lengths):
        starts.append(starts[-1] + length)
        for offset in (16, 32, 64):
            if offset <= length:
                checkpoints.append(offset)
                owners.append(seq)
        indptr.append(len(checkpoints))
    problem = _make_problem(
        batch=1, seq_len=starts[-1], num_sequences=num_sequences, seed=607,
        cu_seqlens=torch.tensor(starts, dtype=cu_dtype, device="cuda"),
    )
    state_case = StateCase("packed_checkpoints", True, True, state_dtype)
    initial = _state_template(problem, state_case)
    reference_out, reference_final = _run_reference(problem, state_case, initial)
    expected_out, expected_final = _run_kernel(problem, state_case, initial)
    # Native matrix accumulation can differ from torch_ref by one BF16 ULP
    # on these packed inputs, with or without checkpoint export. Preserve the
    # non-checkpoint CUDA output bit-for-bit and bound reference rounding.
    reference_ulp = (
        expected_out.view(torch.int16).int() - reference_out.view(torch.int16).int()
    ).abs()
    assert int(reference_ulp.max()) <= 1
    assert torch.equal(expected_final, reference_final)
    output = torch.empty_like(problem.q)
    final = _fresh_final_state(problem, state_case)
    saved = torch.full(
        (len(checkpoints), H, D, D), float("nan"), dtype=torch.float32, device="cuda"
    )
    offsets = torch.tensor(checkpoints, dtype=cu_dtype, device="cuda")
    pointers = torch.tensor(indptr, dtype=cu_dtype, device="cuda")
    workspace = torch.empty(
        flash_kda.get_workspace_size(starts[-1], H, num_sequences),
        dtype=torch.uint8, device="cuda",
    )

    def run() -> None:
        flash_kda.fwd(
            problem.q, problem.k, problem.v, problem.g, problem.beta,
            problem.scale, output, problem.A_log, problem.dt_bias, LOWER_BOUND,
            initial_state=initial, final_state=final,
            cu_seqlens=problem.cu_seqlens, workspace=workspace,
            checkpoint_state=saved, checkpoint_offsets=offsets,
            checkpoint_indptr=pointers,
        )

    run()
    torch.cuda.synchronize()
    assert torch.equal(output, expected_out)
    assert torch.equal(final, expected_final)
    expected_checkpoints = torch.empty_like(saved)
    for cutoff in (16, 32, 64):
        clipped_lengths = [min(length, cutoff) for length in lengths]
        clipped_starts = [0]
        for length in clipped_lengths:
            clipped_starts.append(clipped_starts[-1] + length)
        fields = {}
        for name in ("q", "k", "v", "g", "beta"):
            tensor = getattr(problem, name)
            fields[name] = torch.cat(
                [tensor[:, start : start + length]
                 for start, length in zip(starts, clipped_lengths)], dim=1,
            )
        clipped = dataclasses.replace(
            problem, **fields,
            cu_seqlens=torch.tensor(clipped_starts, dtype=cu_dtype, device="cuda"),
        )
        _, states = _run_kernel(clipped, state_case, initial)
        for index, (owner, offset) in enumerate(zip(owners, checkpoints)):
            if offset == cutoff:
                expected_checkpoints[index].copy_(states[owner])
    assert torch.equal(saved, expected_checkpoints)

    # Capture and replay use the caller's output buffers, including all packed
    # checkpoint rows. Poison them between replays to catch missing stores.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        saved.fill_(float("nan"))
        output.fill_(float("nan"))
        final.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected_out)
        assert torch.equal(final, expected_final)
        assert torch.equal(saved, expected_checkpoints)


@pytest.mark.parametrize("num_sequences", (4, 64))
@pytest.mark.parametrize("cu_dtype", (torch.int32, torch.int64))
def test_dense_checkpoints_across_long_prefill_match_independent_prefixes(
    num_sequences: int, cu_dtype: torch.dtype
) -> None:
    """Check 256-token state exports through a 4096-token packed query.

    Empty and short sequences coexist with a long query. Both K2 dispatches
    must preserve outputs and every checkpoint across repeated pipeline-stage
    reuse, not just the first few chunks. The reference uses the identical
    native recurrence without checkpoint export on each truncated prefix.
    """
    lengths = [0, 257, 1025, 4096] + [16] * (num_sequences - 4)
    starts = [0]
    indptr = [0]
    checkpoints = []
    owners = []
    for seq, length in enumerate(lengths):
        starts.append(starts[-1] + length)
        for offset in range(256, length + 1, 256):
            checkpoints.append(offset)
            owners.append(seq)
        indptr.append(len(checkpoints))
    problem = _make_problem(
        batch=1, seq_len=starts[-1], num_sequences=num_sequences, seed=809,
        cu_seqlens=torch.tensor(starts, dtype=cu_dtype, device="cuda"),
    )
    state_case = StateCase("dense_checkpoints", True, True, torch.float32)
    initial = _state_template(problem, state_case)
    expected_out, expected_final = _run_kernel(problem, state_case, initial)
    output = torch.empty_like(problem.q)
    final = _fresh_final_state(problem, state_case)
    saved = torch.full(
        (len(checkpoints), H, D, D), float("nan"), dtype=torch.float32, device="cuda"
    )
    offsets = torch.tensor(checkpoints, dtype=cu_dtype, device="cuda")
    pointers = torch.tensor(indptr, dtype=cu_dtype, device="cuda")
    workspace = torch.empty(
        flash_kda.get_workspace_size(starts[-1], H, num_sequences),
        dtype=torch.uint8, device="cuda",
    )

    def run() -> None:
        flash_kda.fwd(
            problem.q, problem.k, problem.v, problem.g, problem.beta,
            problem.scale, output, problem.A_log, problem.dt_bias, LOWER_BOUND,
            initial_state=initial, final_state=final,
            cu_seqlens=problem.cu_seqlens, workspace=workspace,
            checkpoint_state=saved, checkpoint_offsets=offsets,
            checkpoint_indptr=pointers,
        )

    run()
    torch.cuda.synchronize()
    assert torch.equal(output, expected_out)
    assert torch.equal(final, expected_final)
    expected_checkpoints = torch.empty_like(saved)
    for cutoff in sorted(set(checkpoints)):
        clipped_lengths = [min(length, cutoff) for length in lengths]
        clipped_starts = [0]
        for length in clipped_lengths:
            clipped_starts.append(clipped_starts[-1] + length)
        fields = {}
        for name in ("q", "k", "v", "g", "beta"):
            tensor = getattr(problem, name)
            fields[name] = torch.cat(
                [tensor[:, start : start + length]
                 for start, length in zip(starts, clipped_lengths)], dim=1,
            )
        clipped = dataclasses.replace(
            problem, **fields,
            cu_seqlens=torch.tensor(clipped_starts, dtype=cu_dtype, device="cuda"),
        )
        _, states = _run_kernel(clipped, state_case, initial)
        for index, (owner, offset) in enumerate(zip(owners, checkpoints)):
            if offset == cutoff:
                expected_checkpoints[index].copy_(states[owner])
    assert torch.equal(saved, expected_checkpoints)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        saved.fill_(float("nan"))
        output.fill_(float("nan"))
        final.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected_out)
        assert torch.equal(final, expected_final)
        assert torch.equal(saved, expected_checkpoints)
