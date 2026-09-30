"""CPU tests for diagnostic snapshots, not regression tests for the model bug."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class Tensor:
    def __init__(self, value):
        self.value = value
    def detach(self):
        return self
    def clone(self):
        return Tensor(self.value)
    def view(self, *shape):
        return self
    def __getitem__(self, key):
        return self
    def zero_(self):
        self.value = 0
        return self
    def data_ptr(self):
        return id(self)


class TraceTests(unittest.TestCase):
    def setUp(self):
        torch = SimpleNamespace(Tensor=Tensor, uint8='uint8', empty_like=lambda x: Tensor(-99),
                                cuda=SimpleNamespace(is_current_stream_capturing=lambda: False, synchronize=lambda: None))
        spec = importlib.util.spec_from_file_location('ops_test_module', Path(__file__).with_name('trace_attention_ops.py'))
        self.m = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch': torch}):
            spec.loader.exec_module(self.m)
        self.m._active = {'layers': [{}, {}, {}], 'operators': []}

    def test_input_before_output_after(self):
        source, out = Tensor(1), Tensor(99)
        def original(x, *, out):
            out.value = 2
            x.value = 3
            return out
        result = self.m._wrap('run_post', original)(source, out=out)
        row = self.m._active['operators'][0]
        self.assertIs(result, out)
        self.assertEqual(row['inputs']['arg0'].value, 1)
        self.assertNotIn('out', row['kwargs'])
        self.assertEqual(row['outputs']['kw_out'].value, 2)

    def test_moe_capture_after_join_and_unchanged_result(self):
        source, shared, routed = Tensor(1), Tensor(2), Tensor(3)
        result = (shared, routed)
        def original(self, x):
            shared.value = 4
            return result
        actual = self.m._wrap_moe(original)(SimpleNamespace(layer_name='layer2'), source)
        row = self.m._active['operators'][0]
        self.assertIs(actual, result)
        self.assertEqual(row['outputs']['shared'].value, 4)
        self.assertEqual(row['outputs']['routed'].value, 3)
        self.assertEqual(row['after_attention_layer'], 2)

    def test_disabled_does_not_snapshot(self):
        self.m._active = None
        value = Tensor(1)
        self.assertIs(self.m._wrap('run_post', lambda x: x)(value), value)

    def test_engram_projection_captures_source_before_and_output_after(self):
        self.m._active['layers'] = [{}]
        source, output = Tensor(1), Tensor(99)
        source.shape = (160, 6144)
        output.shape = (160, 25600, 1)
        binding = SimpleNamespace(source=source, output=output,
                                  plan=SimpleNamespace(query='engram'),
                                  x_q=SimpleNamespace(values=Tensor(3), scale_rows=Tensor(4), scale_mma=Tensor(5)))
        def original(*, binding):
            binding.output.value = 2
            return output
        actual = self.m._wrap_engram_projection(original)(binding=binding)
        row = self.m._active['operators'][0]
        self.assertIs(actual, output)
        self.assertEqual(row['inputs']['source'].value, 1)
        self.assertEqual(row['outputs']['projected_kv'].value, 2)
        self.assertEqual(row['outputs']['quantized_scale_mma'].value, 5)

    def test_dense_replay_keeps_original_result_and_reuses_captured_operands(self):
        source = Tensor(1)
        source.shape = (129, 6144)
        output = Tensor(2)
        values, scales = Tensor(3), Tensor(4)
        weights = SimpleNamespace(values=Tensor(5), scale_mma=Tensor(6))
        binding = SimpleNamespace(source=source, output=output, plan=object(), workspace=None, expected_m=256,
                                  x_q=SimpleNamespace(values=values, scale_mma=scales),
                                  packed_weight=SimpleNamespace(weight=weights))
        calls = []
        def run(lhs, rhs, *, out, stream, split_k_workspace):
            calls.append((lhs, rhs, stream, split_k_workspace))
            self.assertIsNot(out, output)
            out.value = 7
        state = SimpleNamespace(config='tactic', dense=SimpleNamespace(run=run, gemm=object(),
            lowering=SimpleNamespace(m=256, policy=SimpleNamespace(split_k_slices=1))))
        module = SimpleNamespace(require_prepared=lambda plan, family: state)
        row = {'outputs': {}}
        with patch.dict(sys.modules, {'b12x.preparation.types': module,
                                     'b12x._lib.compile_plan': SimpleNamespace(program_keys=lambda x: ())}):
            self.m._replay_engram_dense(binding, row)
        self.assertEqual(output.value, 2)
        self.assertEqual(row['outputs']['dense_replay'].value, 7)
        self.assertEqual(row['outputs']['dense_replay_synchronized'].value, 7)
        self.assertEqual(len(calls), 2)
        self.assertEqual(row['dense_rows'], {'live_m': 129, 'expected_m': 256, 'lowering_m': 256})
        self.assertIs(calls[0][0][0], values)
        self.assertIs(calls[0][0][1], scales)

    def test_dense_replay_rejects_split_workspace(self):
        state = SimpleNamespace(dense=SimpleNamespace(lowering=SimpleNamespace(
            policy=SimpleNamespace(split_k_slices=2))))
        with patch.dict(sys.modules, {'b12x.preparation.types': SimpleNamespace(require_prepared=lambda *a: state),
                                     'b12x._lib.compile_plan': SimpleNamespace(program_keys=lambda x: ())}):
            with self.assertRaisesRegex(RuntimeError, 'unsplit'):
                self.m._replay_engram_dense(SimpleNamespace(plan=object()), {'outputs': {}})

    def test_engram_allreduce_preserves_in_place_input(self):
        self.m._active['layers'] = [{}]
        source = Tensor(1)
        source.ndim, source.shape = 2, (160, 6144)
        def original(self, value):
            value.value = 7
            return value
        result = self.m._wrap_engram_allreduce(original)(None, source)
        row = self.m._active['operators'][0]
        self.assertIs(result, source)
        self.assertEqual(row['inputs']['local_rows'].value, 1)
        self.assertEqual(row['outputs']['rows'].value, 7)

    def test_engram_epochs_captured_after_original(self):
        self.m._active['layers'] = [{}]
        epoch = Tensor(0)
        owner = SimpleNamespace(overlap_epochs=(epoch,))
        hidden, hashes = Tensor(1), Tensor(2)
        def original(self, hidden_states, hash_ids, token_mask):
            epoch.value = 7
            return hidden_states
        result = self.m._wrap_engram_forward(original)(owner, hidden, hashes)
        row = self.m._active['operators'][0]
        self.assertIs(result, hidden)
        self.assertEqual(row['outputs']['epoch0'].value, 7)

    def test_replay_epochs_reset_only_failure_before_wait(self):
        self.m._active.update(layers=[{}], dense_replay=True)
        ready, expected, failed = Tensor(8), Tensor(8), Tensor(1)
        owner = SimpleNamespace(overlap_epochs=(ready, expected, failed))
        def original(self, hidden, hashes, mask):
            self_test.assertEqual(failed.value, 0)
            self_test.assertEqual((ready.value, expected.value), (8, 8))
            return hidden
        self_test = self
        self.m._wrap_engram_forward(original)(owner, Tensor(0), Tensor(0))
        row = self.m._active['operators'][0]
        self.assertEqual(row['outputs']['prior_failed'].value, 1)
        self.assertEqual(row['outputs']['epoch2'].value, 0)


if __name__ == '__main__':
    unittest.main()
