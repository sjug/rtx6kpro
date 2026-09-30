"""GPU regression for disabled generic mapping on DS4.1 circular state groups.

Run in an idle pinned image, never alongside the serving process. These tests
exercise the production metadata and ring store/load kernels, not a replica.
"""
import pytest
import torch

from vllm.models.deepseek_v4_1.sparse_mla import _tokens
from vllm.models.deepseek_v4_1.compressor import (
    _load_partial_state, _save_partial_states,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec


def metadata(lengths, seqs, blocks, *, circular=True, ratio=1, page=16,
             capacity=None, masked=(), padded_nt=None):
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    nt = padded_nt or starts[-1]
    capacity = capacity or ((nt + 127) // 128 * 128)
    assert capacity >= nt
    tensor = lambda data, dtype=torch.int64: torch.tensor(data, dtype=dtype, device='cuda')
    starts_gpu = tensor(starts, torch.int32)
    seqs_gpu = tensor(seqs, torch.int32)
    table = tensor([[block] for block in blocks], torch.int32)
    # The real runner writes PAD_ID for every CircularBufferSpec token.
    inputs = torch.full((capacity,), -1 if circular else 1,
                        dtype=torch.int64, device='cuda')
    for index in masked:
        inputs[index] = -1
    outputs = [torch.full((capacity,), -99, dtype=dtype, device='cuda')
               for dtype in (torch.int64, torch.int32, torch.int64, torch.int32)]
    def launch():
        _tokens[((capacity + 127) // 128,)](
            starts_gpu, seqs_gpu, table, inputs, *outputs,
            len(lengths), nt, table.stride(0), table.shape[1], capacity,
            page, ratio, 128, circular,
        )
    expected = [[-1] * capacity, [-1] * capacity, [-1] * capacity, [0] * capacity]
    for req, (length, seq, block) in enumerate(zip(lengths, seqs, blocks)):
        for t in range(starts[req], starts[req + 1]):
            pos = seq - length + t - starts[req]
            logical = pos // ratio
            valid = pos >= 0 and block > 0
            if not circular:
                valid &= t not in masked and logical // page < 1
            if valid:
                expected[0][t] = pos
                expected[1][t] = req
                expected[3][t] = (pos + 1) // ratio
                if (pos + 1) % ratio == 0 and (not circular or t >= starts[req + 1] - page):
                    expected[2][t] = block * page + logical % page
    return launch, outputs, [tensor(e, o.dtype) for e, o in zip(expected, outputs)]


@pytest.mark.parametrize('length', [1, 8, 15, 16, 17, 128, 256, 385, 8192])
@pytest.mark.parametrize('graph', [False, True])
def test_circular_disabled_generic_slots(length, graph):
    spec = CircularBufferSpec(block_size=16, num_kv_heads=1, head_size=1024,
                              head_size_v=0, dtype=torch.float32)
    assert spec.uses_slot_mapping is False
    launch, outputs, expected = metadata([length], [length], [2])
    launch()
    if graph:
        torch.cuda.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            launch()
        for output in outputs:
            output.fill_(-99)
        capture.replay()
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_circular_ragged_null_and_padded_requests():
    launch, outputs, expected = metadata([385, 8, 3, 1], [385, 776, 19, 0],
                                        [2, 4, 0, 3], capacity=512)
    launch()
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_circular_full_graph_zero_length_padding():
    launch, outputs, expected = metadata([17, 8, 0, 0], [385, 776, 0, 0],
                                        [2, 4, 0, 0], capacity=128, padded_nt=32)
    launch()
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize('ratio', [1, 2])
def test_noncircular_input_mask_preserved(ratio):
    launch, outputs, expected = metadata([16, 3], [16, 3], [2, 0],
        circular=False, ratio=ratio, masked=(0, 7))
    launch()
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize('length', [1, 17, 385])
@pytest.mark.parametrize('graph', [False, True])
def test_odd_continuation_loads_own_saved_projection(length, graph):
    launch, outputs, _ = metadata([length], [length], [2])
    page = 16
    pool = torch.full((4, page * 1024), float('nan'), device='cuda')
    values = torch.arange(length * 512, device='cuda', dtype=torch.float32).view(length, 512)
    gates = values.neg().add(0.25)
    counts = torch.tensor([length, 1], dtype=torch.int32, device='cuda')
    starts = torch.tensor([0, 8], dtype=torch.int32, device='cuda')
    positions = torch.tensor([length], dtype=torch.int64, device='cuda')
    table = torch.tensor([[2]], dtype=torch.int32, device='cuda')
    next_counts = torch.tensor([8, 1], dtype=torch.int32, device='cuda')
    pending = [torch.empty((1, 512), device='cuda') for _ in range(2)]
    tags = torch.empty(1, dtype=torch.int64, device='cuda')
    ids = torch.empty(1, dtype=torch.int32, device='cuda')
    def run():
        launch()
        _save_partial_states[(length,)](pool, outputs[2], values, gates,
                                       counts, pool.stride(0), page)
        _load_partial_state[(1,)](pool, table, starts, positions, *pending,
            tags, ids, next_counts, pool.stride(0), table.stride(0), page)
    run()
    if graph:
        torch.cuda.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            run()
        pool.fill_(float('nan'))
        for output in pending:
            output.fill_(float('nan'))
        tags.fill_(-99)
        ids.fill_(-99)
        capture.replay()
    torch.testing.assert_close(pending[0][0], values[-1], rtol=0, atol=0)
    torch.testing.assert_close(pending[1][0], gates[-1], rtol=0, atol=0)
    assert tags.item() == length - 1 and ids.item() == 0
    assert torch.isnan(pool[0]).all() and torch.isnan(pool[1]).all() and torch.isnan(pool[3]).all()


def test_k7_lookahead_preserves_every_accepted_predecessor():
    page, drafts, length = 16, 7, 385
    assert page >= 2 * (1 + drafts)
    pool = torch.full((4, page * 1024), float('nan'), device='cuda')
    def save(start, size):
        launch, outputs, _ = metadata([size], [start + size], [2])
        launch()
        values = torch.arange(start, start + size, device='cuda', dtype=torch.float32)[:, None].expand(size, 512).contiguous()
        gates = values.neg().add(0.25)
        counts = torch.tensor([size, 1], dtype=torch.int32, device='cuda')
        _save_partial_states[(size,)](pool, outputs[2], values, gates, counts, pool.stride(0), page)
    save(0, length)
    save(length, 8)
    save(length + 8, 8)
    starts = torch.tensor([0, 8], dtype=torch.int32, device='cuda')
    table = torch.tensor([[2]], dtype=torch.int32, device='cuda')
    counts = torch.tensor([8, 1], dtype=torch.int32, device='cuda')
    for accepted in range(1, 9):
        pos = length + accepted
        positions = torch.tensor([pos], dtype=torch.int64, device='cuda')
        pending = [torch.full((1, 512), float('nan'), device='cuda') for _ in range(2)]
        tags = torch.full((1,), -99, dtype=torch.int64, device='cuda')
        ids = torch.full((1,), -99, dtype=torch.int32, device='cuda')
        _load_partial_state[(1,)](pool, table, starts, positions, *pending,
            tags, ids, counts, pool.stride(0), table.stride(0), page)
        if pos % 2:
            assert tags.item() == pos - 1
            torch.testing.assert_close(pending[0], torch.full_like(pending[0], pos - 1), rtol=0, atol=0)
            torch.testing.assert_close(pending[1], torch.full_like(pending[1], -(pos - 1) + 0.25), rtol=0, atol=0)
        else:
            assert tags.item() == -1
            assert (pending[0] == 0).all() and (pending[1] == 0).all()
