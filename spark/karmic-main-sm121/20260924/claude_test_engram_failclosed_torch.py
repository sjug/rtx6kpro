"""CPU-torch tests of the proposed Engram fail-closed protocol.

Run with the task-local .venv-snapshot-cpu (torch CPU, no Triton, no GPU).
The REAL patched Python (Engram.forward, Engram.invalidate_disk_output and
its alias check, DeepseekV4Model.snapshot_engram_fault, _check_model_fault)
runs unchanged. Only the Triton wait kernel and the TP all-reduce are
emulated. The emulated wait mirrors the patched kernel's contract: it
always writes timed_out (1 or 0) to element 0 of its last argument.

Also compiles the real forward with torch.compile(fullgraph=True,
backend="eager"), and shows the previous data_ptr() placement fails that
compile (negative control for the review finding).

Not verified here: Triton compilation, GPU timing, CUDA graph capture,
RoCEnante/NCCL behaviour.
"""
import ast
import importlib.util
import threading
import types
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prep', HERE / 'claude_prepare_engram_failclosed.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
SOURCE = {path: new for path, _old, new in prep.compose()}
TP, TOKENS, WIDTH = 4, 6, 6144


def extract(path, name, owner=None, namespace=None):
    tree = ast.parse(SOURCE[path])
    scope = tree.body
    if owner:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(namespace or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), f'<{name}>', 'exec'), namespace)
    return namespace[name]


class Kernel:
    """Emulates `kernel[grid](*args)` for a single-program kernel."""

    def __init__(self, body):
        self.body = body

    def __getitem__(self, grid):
        return self.body


def wait_rows(ready, expected, failed, timeout_ms, flag):
    """Tensor-only emulation of the patched _wait_engram_rows contract.

    The lookup either published before the timeout (ready >= expected) or
    did not; `failed` keeps the legacy sticky 1; `flag` always gets 0 or 1.
    """
    timed_out = ready < expected
    failed.copy_(torch.where(timed_out, torch.ones_like(failed), failed))
    flag.view(-1)[:1].copy_(timed_out.to(flag.dtype))


class AllReduce:
    """Sum across rank threads (bf16), optionally in place like NCCL."""

    def __init__(self, in_place=False):
        self.in_place = in_place
        self.slots = [None] * TP
        self.barrier = threading.Barrier(TP)
        self.local = threading.local()

    def __call__(self, tensor):
        rank = self.local.rank
        self.slots[rank] = tensor.clone()
        self.barrier.wait()
        total = torch.zeros_like(tensor)
        for r in range(TP):
            total = total + self.slots[r]
        self.barrier.wait()
        if self.in_place:
            tensor.copy_(total)
            return tensor
        return total


class Layer:
    pass


def make_layer(epochs, fault, reducer, seen):
    ns = {'torch': torch, '_wait_engram_rows': Kernel(wait_rows),
          'tensor_model_parallel_all_reduce': reducer,
          'hyperconnection': types.SimpleNamespace(
              run_engram_mix=lambda state, kv, w, *, eps, plan, out, token_mask: out.copy_(state))}
    layer = Layer()
    layer.engram_io_rows = torch.empty(TOKENS + 1, WIDTH, dtype=torch.bfloat16)
    layer.engram_io_rows[0].zero_()
    layer.staged_rows = layer.engram_io_rows[1:]
    layer.overlap_epochs = tuple(epochs[i:i + 1] for i in range(3))
    layer.overlap_fault_epoch = fault
    layer.embed_tokens = types.SimpleNamespace(table_memory='ram')
    layer.wkv = lambda rows: seen.append(rows.clone()) or rows
    layer.norm_weights, layer.eps, layer.mix_plans = None, 1e-6, {False: 'plan'}
    for name in ('forward', 'invalidate_disk_output', '_check_fault_io_alias'):
        setattr(layer, name, extract(prep.ENGRAM, name, 'Engram', ns).__get__(layer))
    return layer


class Rank:
    """One TP rank: two Engram layers sharing the model's fault epoch."""

    def __init__(self, reducer):
        self.epochs = torch.zeros(3, dtype=torch.int64)
        self.fault = torch.zeros(1, dtype=torch.int64)
        self.seen = []
        self.layers = [make_layer(self.epochs, self.fault, reducer, self.seen) for _ in range(2)]
        snapshot = extract(prep.MODEL, 'snapshot_engram_fault', 'DeepseekV4Model', {'torch': torch})
        model = types.SimpleNamespace(_engram_fault_epoch=self.fault, _engram_epochs=self.epochs)
        self.snapshot = snapshot.__get__(model)

    def prepare(self, epoch, published, rows):
        for layer in self.layers:
            layer.invalidate_disk_output()               # eager seam, runs the alias check
        self.epochs[1] = epoch                           # expected, main stream
        if published:
            self.epochs[0] = epoch                       # side stream published in time
        for layer, local in zip(self.layers, rows):
            layer.staged_rows[:TOKENS].copy_(local)

    def forward(self, reducer, rank):
        reducer.local.rank = rank
        hidden = torch.zeros(TOKENS, 2, 8, dtype=torch.bfloat16)
        hashes = torch.zeros(TOKENS, 24, dtype=torch.int64)
        for layer in self.layers:
            layer.forward(hidden, hashes)


def shard_rows(seed):
    """Per-rank staged rows where each element has exactly one owning rank."""
    g = torch.Generator().manual_seed(seed)
    full = torch.randn(2, TOKENS, WIDTH, generator=g).to(torch.bfloat16)
    owner = torch.randint(0, TP, (2, TOKENS, 24), generator=g).repeat_interleave(256, dim=2)
    return full, [torch.where(owner == r, full, torch.zeros_like(full)) for r in range(TP)]


def run_step(ranks, reducer, epoch, published, seed=None):
    full, rows = shard_rows(epoch if seed is None else seed)
    for rank, ok, local in zip(ranks, published, rows):
        rank.prepare(epoch, ok, local)
    threads = [threading.Thread(target=r.forward, args=(reducer, i)) for i, r in enumerate(ranks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return full, [r.snapshot().tolist() for r in ranks]


CHECK = extract(prep.ASYNC, '_check_model_fault')


class Protocol(unittest.TestCase):
    def make(self, **kw):
        reducer = AllReduce(**kw)
        return reducer, [Rank(reducer) for _ in range(TP)]

    def test_healthy_steps_pass_and_rows_unchanged(self):
        reducer, ranks = self.make()
        for epoch in (1, 2, 3):
            full, snaps = run_step(ranks, reducer, epoch, [True] * TP)
            self.assertEqual(snaps, [[0, epoch]] * TP)
            CHECK(snaps[0])
            for rank in ranks:
                for rows, expected in zip(rank.seen[-2:], full):
                    self.assertTrue(torch.equal(rows, expected))

    def test_one_rank_timeout_fails_output_rank(self):
        reducer, ranks = self.make()
        run_step(ranks, reducer, 1, [True] * TP)
        _, snaps = run_step(ranks, reducer, 2, [True, True, False, True])
        self.assertEqual(snaps, [[2, 2]] * TP)            # rank 2 alone timed out
        with self.assertRaisesRegex(RuntimeError, 'in this step .fault epoch 2, step epoch 2'):
            CHECK(snaps[0])                                # rank 0 is the output rank

    def test_fault_in_unchecked_step_is_not_lost(self):
        # Warmup, dummy or any step whose output is never materialized: the
        # next checked output (even many epochs later) still fails.
        for in_place in (False, True):
            reducer, ranks = self.make(in_place=in_place)
            run_step(ranks, reducer, 1, [False] * TP)      # e.g. warmup stall, no get_output
            for epoch in (2, 3, 4):
                _, snaps = run_step(ranks, reducer, epoch, [True] * TP)
                self.assertEqual(snaps, [[1, epoch]] * TP)
                with self.assertRaisesRegex(RuntimeError, 'in an earlier step'):
                    CHECK(snaps[0])

    def test_stale_flag_row_and_sticky_failed_word_are_not_faults(self):
        reducer, ranks = self.make(in_place=True)
        for rank in ranks:
            for layer in rank.layers:
                layer.engram_io_rows[0, 0] = 4.0           # stale sum left by an in-place reduce
            rank.epochs[2] = 1                             # legacy sticky `failed`
        _, snaps = run_step(ranks, reducer, 5, [True] * TP)
        self.assertEqual(snaps, [[0, 5]] * TP)
        CHECK(snaps[0])

    def test_late_lookup_cannot_touch_flag(self):
        reducer, ranks = self.make()
        run_step(ranks, reducer, 1, [True, False, True, True])
        layer = ranks[1].layers[0]
        flag_before = layer.engram_io_rows[0].clone()
        layer.staged_rows.fill_(7.0)                       # a late lookup writes every token row
        self.assertTrue(torch.equal(layer.engram_io_rows[0], flag_before))

    def test_alias_break_fails_at_eager_seam(self):
        reducer, ranks = self.make()
        layer = ranks[0].layers[0]
        layer.staged_rows = layer.staged_rows.clone()      # e.g. a module move copying buffers
        with self.assertRaisesRegex(RuntimeError, 'no longer alias'):
            layer.invalidate_disk_output()
        layer.staged_rows = layer.engram_io_rows[1:]
        layer.overlap_fault_epoch = None
        with self.assertRaisesRegex(RuntimeError, 'fault epoch buffer'):
            layer.invalidate_disk_output()
        layer.overlap_epochs = None                        # non-overlap path: no check
        layer.invalidate_disk_output()

    def test_snapshot_is_a_copy(self):
        reducer, ranks = self.make()
        _, snaps = run_step(ranks, reducer, 1, [True] * TP)
        snap = ranks[0].snapshot()
        ranks[0].fault.fill_(9)                            # later steps cannot rewrite it
        self.assertEqual(snap.tolist(), [0, 1])


class Compile(unittest.TestCase):
    """torch.compile(fullgraph=True, backend="eager") around the REAL forward."""

    def setUp(self):
        torch._dynamo.reset()
        self.epochs = torch.zeros(3, dtype=torch.int64)
        self.fault = torch.zeros(1, dtype=torch.int64)
        self.seen = []
        self.layer = make_layer(self.epochs, self.fault, lambda t: t * 1, self.seen)
        self.layer.wkv = lambda rows: rows * 1
        self.hidden = torch.zeros(TOKENS, 2, 8, dtype=torch.bfloat16)
        self.hashes = torch.zeros(TOKENS, 24, dtype=torch.int64)

    def test_forward_compiles_fullgraph_and_records_fault(self):
        compiled = torch.compile(self.layer.forward, fullgraph=True, backend='eager')
        self.epochs[:2] = torch.tensor([3, 3])             # published
        compiled(self.hidden, self.hashes)
        self.assertEqual(self.fault.tolist(), [0])
        self.assertEqual(float(self.layer.engram_io_rows[0, 0]), 0.0)
        self.epochs[1] = 4                                 # not published: timeout
        compiled(self.hidden, self.hashes)
        self.assertEqual(self.fault.tolist(), [4])
        self.assertEqual(self.epochs[2].item(), 1)

    def test_previous_data_ptr_placement_fails_fullgraph(self):
        # Negative control: the prior revision validated the alias inside
        # forward with data_ptr(); Dynamo cannot trace that.
        def old_io_rows(layer):
            if layer.staged_rows.data_ptr() != layer.engram_io_rows[1:].data_ptr():
                raise RuntimeError('alias')
            return layer.engram_io_rows
        with self.assertRaises(Exception) as caught:
            torch.compile(old_io_rows, fullgraph=True, backend='eager')(self.layer)
        self.assertIn('DataPtrVariable', str(caught.exception))


class ReductionExactness(unittest.TestCase):
    def test_single_owner_sums_are_order_independent(self):
        full, rows = shard_rows(9)
        for order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
            total = torch.zeros_like(full)
            for r in order:
                total = total + rows[r]
            self.assertTrue(torch.equal(total, full))

    def test_counterexample_when_elements_have_two_contributors(self):
        a = torch.tensor([1.0], dtype=torch.bfloat16)
        b = torch.tensor([2 ** -8], dtype=torch.bfloat16)
        self.assertFalse(torch.equal((a + b) + b, a + (b + b)))


if __name__ == '__main__':
    unittest.main()
