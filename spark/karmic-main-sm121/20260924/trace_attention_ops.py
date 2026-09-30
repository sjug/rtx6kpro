"""Diagnostic-only extension: snapshot mHC boundaries around the first divergence.

Mounted over the trace helper with a pinned read-only overlay. Arithmetic and
arguments are unchanged. Clones perturb memory/timing; retain the external repro.
"""
import functools
import json
import os
from pathlib import Path
import re
import torch

_active = None
_seen = set()
_installed = False
CONTROL = Path('/cache/ds41-trace-control.json')
OUTPUT_KEYS = {'out', 'residual_out', 'y_out', 'post_out', 'comb_out', 'pre_out'}


def _clone(value):
    return value.detach().clone()


def _wrap(name, original):
    @functools.wraps(original)
    def call(*args, **kwargs):
        enabled = (_active is not None and 0 <= len(_active['layers']) - 1 <= 3
                   and not torch.cuda.is_current_stream_capturing())
        row = None
        if enabled:
            row = {'name': name, 'after_attention_layer': len(_active['layers']) - 1,
                   'inputs': {f'arg{i}': _clone(x) for i, x in enumerate(args) if isinstance(x, torch.Tensor)},
                   'kwargs': {k: _clone(x) for k, x in kwargs.items()
                              if isinstance(x, torch.Tensor) and k not in OUTPUT_KEYS},
                   'outputs': {}}
            _active['operators'].append(row)
        result = original(*args, **kwargs)
        if row is not None:
            binding = kwargs.get('binding')
            if binding is not None:
                row['query'] = repr(binding.plan.query)
                for field in ('out', 'y', 'post_buffer', 'comb_buffer', 'pre_out'):
                    value = getattr(binding, field)
                    if isinstance(value, torch.Tensor):
                        row['outputs'][field] = _clone(value)
            if isinstance(result, torch.Tensor):
                row['outputs']['return'] = _clone(result)
            for key in OUTPUT_KEYS:
                value = kwargs.get(key)
                if isinstance(value, torch.Tensor):
                    row['outputs']['kw_' + key] = _clone(value)
        return result
    return call


def _install():
    global _installed
    if not _installed:
        from b12x.norm import mhc
        for name in ('run_pre', 'run_post_pre', 'run_post'):
            setattr(mhc, name, _wrap(name, getattr(mhc, name)))
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        MoERunner._forward_impl = _wrap_moe(MoERunner._forward_impl)
        from b12x.gemm import block_fp8_linear
        block_fp8_linear.run = _wrap_engram_projection(block_fp8_linear.run)
        from b12x.norm import hyperconnection
        hyperconnection.run_engram_mix = _wrap('engram_mix', hyperconnection.run_engram_mix)
        from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
        CudaCommunicator.all_reduce = _wrap_engram_allreduce(CudaCommunicator.all_reduce)
        from vllm.models.deepseek_v4_1.common.engram import Engram
        Engram.forward = _wrap_engram_forward(Engram.forward)
        _installed = True


def _engram_active():
    return (_active is not None and len(_active['layers']) == 1
            and not torch.cuda.is_current_stream_capturing())


def _wrap_engram_allreduce(original):
    @functools.wraps(original)
    def call(self, input_):
        row = None
        if _engram_active() and input_.ndim == 2 and input_.shape[1] == 6144:
            # This hook is after Engram's stream-ordered lookup epoch wait.
            row = {'name': 'engram_allreduce', 'after_attention_layer': 0,
                   'inputs': {'local_rows': _clone(input_)}, 'kwargs': {}, 'outputs': {}}
            _active['operators'].append(row)
        result = original(self, input_)
        if row is not None:
            row['outputs']['rows'] = _clone(result)
        return result
    return call


def _wrap_engram_forward(original):
    @functools.wraps(original)
    def call(self, hidden_states, hash_ids, token_mask=None):
        enabled = _engram_active()
        prior_failed = None
        if enabled and _active.get('dense_replay', False) and self.overlap_epochs is not None:
            prior_failed = _clone(self.overlap_epochs[2])
            # Diagnostic only: the main-stream wait is the flag's sole writer.
            # Reset before that wait so this request's timeout can be observed.
            self.overlap_epochs[2].zero_()
        result = original(self, hidden_states, hash_ids, token_mask)
        if enabled:
            # Do not read staged rows or epochs before the original lookup wait.
            row = {'name': 'engram_epochs', 'after_attention_layer': 0,
                   'inputs': {'hash_ids': _clone(hash_ids)}, 'kwargs': {}, 'outputs': {}}
            if self.overlap_epochs is not None:
                for i, value in enumerate(self.overlap_epochs):
                    row['outputs'][f'epoch{i}'] = _clone(value)
            if prior_failed is not None:
                row['outputs']['prior_failed'] = prior_failed
            _active['operators'].append(row)
        return result
    return call


def _wrap_engram_projection(original):
    @functools.wraps(original)
    def call(*args, **kwargs):
        binding = kwargs.get('binding')
        enabled = (_active is not None and len(_active['layers']) == 1
                   and binding is not None and binding.source.shape[-1] == 6144
                   and binding.output.shape[1] == 25600
                   and not torch.cuda.is_current_stream_capturing())
        row = None
        if enabled:
            row = {'name': 'engram_projection', 'after_attention_layer': 0,
                   'inputs': {'source': _clone(binding.source)}, 'kwargs': {},
                   'outputs': {}, 'query': repr(binding.plan.query)}
            _active['operators'].append(row)
        result = original(*args, **kwargs)
        if row is not None:
            row['outputs']['projected_kv'] = _clone(binding.output)
            for name in ('values', 'scale_rows', 'scale_mma'):
                row['outputs']['quantized_' + name] = _clone(getattr(binding.x_q, name).view(torch.uint8))
            row['addresses'] = {name: getattr(binding.x_q, name).data_ptr()
                                for name in ('values', 'scale_rows', 'scale_mma')}
            row['addresses'].update(source=binding.source.data_ptr(), output=binding.output.data_ptr())
            if _active.get('dense_replay', False):
                _replay_engram_dense(binding, row)
        return result
    return call


def _replay_engram_dense(binding, row):
    """Diagnostic second execution only; never substitute the serving output."""
    from b12x.preparation.types import require_prepared
    from b12x._lib.compile_plan import program_keys
    state = require_prepared(binding.plan, 'gemm.block_fp8_linear')
    if state.dense.lowering.policy.split_k_slices != 1:
        raise RuntimeError('Diagnostic replay only admits the observed unsplit projection')
    m, k = binding.source.shape
    weight = binding.packed_weight.weight
    replay = torch.empty_like(binding.output)
    # Capture the weight rows covering both observed faulty column windows.
    row['outputs']['weight_prefix'] = _clone(weight.values[:1024].view(torch.uint8))
    row['outputs']['weight_scale_mma'] = _clone(weight.scale_mma.view(torch.uint8))
    state.dense.run(
        (binding.x_q.values.view(m, k, 1), binding.x_q.scale_mma),
        (weight.values.view(25600, k, 1), weight.scale_mma),
        out=replay, stream=None, split_k_workspace=binding.workspace,
    )
    row['outputs']['dense_replay'] = replay
    # Second diagnostic sample after all previously submitted device work drains.
    # It retains the same operands and does not overwrite the model's output.
    torch.cuda.synchronize()
    synchronized = torch.empty_like(binding.output)
    state.dense.run(
        (binding.x_q.values.view(m, k, 1), binding.x_q.scale_mma),
        (weight.values.view(25600, k, 1), weight.scale_mma),
        out=synchronized, stream=None, split_k_workspace=binding.workspace,
    )
    row['outputs']['dense_replay_synchronized'] = synchronized
    row['dense_config'] = repr(state.config)
    row['dense_policy'] = repr(state.dense.lowering.policy)
    row['dense_rows'] = {'live_m': m, 'expected_m': binding.expected_m,
                         'lowering_m': state.dense.lowering.m}
    row['dense_programs'] = [{'dialect': p.dialect, 'key': p.key, 'name': p.name}
                             for p in program_keys(state.dense.gemm)]


def _wrap_moe(original):
    @functools.wraps(original)
    def call(self, *args, **kwargs):
        enabled = (_active is not None and 0 <= len(_active['layers']) - 1 <= 3
                   and not torch.cuda.is_current_stream_capturing())
        result = original(self, *args, **kwargs)
        if enabled:
            # Clone only after the runner has joined its shared-expert stream.
            row = {'name': 'moe_forward_impl', 'after_attention_layer': len(_active['layers']) - 1,
                   'input_snapshot_timing': 'after_return_not_original_input_contract',
                   'inputs': {f'arg{i}': _clone(x) for i, x in enumerate(args) if isinstance(x, torch.Tensor)},
                   'kwargs': {k: _clone(x) for k, x in kwargs.items() if isinstance(x, torch.Tensor)},
                   'outputs': {}, 'query': self.layer_name}
            values = result if isinstance(result, tuple) else (None, result)
            if len(values) != 2:
                raise RuntimeError('Unexpected MoE output structure')
            for name, value in zip(('shared', 'routed'), values, strict=True):
                if value is not None:
                    if not isinstance(value, torch.Tensor):
                        raise RuntimeError('Deferred MoE output is not admitted by this trace')
                    row['outputs'][name] = _clone(value)
            _active['operators'].append(row)
        return result
    return call


def before(hidden, positions, prefix):
    global _active
    if torch.cuda.is_current_stream_capturing():
        return None
    match = re.search(r'layers\.(\d+)', prefix)
    if match is None:
        raise RuntimeError('DS41 trace cannot identify layer')
    layer = int(match.group(1))
    if layer == 0 and _active is None and CONTROL.is_file():
        control = json.loads(CONTROL.read_text())
        run = control['run']
        if not isinstance(control.get('dense_replay', False), bool):
            raise RuntimeError('dense_replay must be a boolean')
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', run):
            raise RuntimeError('Unsafe trace run name')
        if hidden.shape[0] == control['rows'] and run not in _seen:
            free, _ = torch.cuda.mem_get_info(hidden.device)
            # Five attention boundaries plus four intervening mHC/MoE windows.
            # Bound includes copied weights and duplicate alias snapshots.
            footprint = 192 * hidden.numel() * hidden.element_size()
            if control.get('dense_replay', False):
                footprint += 1024 * 6144 + 25600 * 192 + hidden.shape[0] * 25600 * 4 + 8
            if free < 2 * footprint:
                print(f'[DS41-DIAG-TRACE-REFUSED] free={free} need={2 * footprint}', flush=True)
                return None
            _install()
            _seen.add(run)
            _active = {'run': run, 'node': os.environ['DS41_NODE'], 'layers': [], 'operators': [],
                       'dense_replay': control.get('dense_replay', False),
                       'free_bytes_before': free, 'snapshot_bound_bytes': footprint}
    if _active is None:
        return None
    if layer != len(_active['layers']):
        raise RuntimeError('DS41 trace layer order changed')
    row = {'layer': layer, 'prefix': prefix, 'input': _clone(hidden), 'positions': _clone(positions)}
    _active['layers'].append(row)
    return row


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    return value


def _nbytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_nbytes(v) for v in value.values())
    if isinstance(value, list):
        return sum(_nbytes(v) for v in value)
    return 0


def after(row, output):
    global _active
    if row is None:
        return
    row['output'] = _clone(output)
    if row['layer'] != 4:
        return
    moe_layers = [r['after_attention_layer'] for r in _active['operators'] if r['name'] == 'moe_forward_impl']
    mhc_layers = {r['after_attention_layer'] for r in _active['operators'] if r['name'] != 'moe_forward_impl'}
    if moe_layers != [0, 1, 2, 3] or mhc_layers != {0, 1, 2, 3}:
        raise RuntimeError(f'Diagnostic hooks did not engage: MoE={moe_layers}, mHC={mhc_layers}')
    for name in ('engram_allreduce', 'engram_projection', 'engram_mix', 'engram_epochs'):
        if sum(r['name'] == name for r in _active['operators']) != 1:
            raise RuntimeError('Diagnostic Engram hook did not engage exactly once: ' + name)
    _active['snapshot_actual_bytes'] = _nbytes(_active)
    if _active['snapshot_actual_bytes'] > _active['snapshot_bound_bytes']:
        raise RuntimeError('Diagnostic snapshot exceeded its declared memory bound')
    host = _cpu(_active)
    directory = Path('/cache/ds41-attention-trace')
    directory.mkdir(exist_ok=True)
    path = directory / f"{_active['run']}-{_active['node']}.pt"
    if path.exists():
        raise RuntimeError(f'Trace already exists: {path}')
    torch.save(host, path)
    print('[DS41-DIAG-TRACE-OPS] ' + str(path), flush=True)
    _active = None
