"""DIAGNOSTIC ONLY: real inputs and served outputs of the expanded mHC pre on the decision forward.

Installed as vllm/models/deepseek_v4_1/claude_mhc_expanded.py in a labelled derivative of the precision
release image (claude-mhc-expanded-install.py); DeepseekV4Model.__init__ calls install(self) once.
Nothing is allocated or copied until the trigger exists.

Why: the two-mHC transplant reproduces the failing first token exactly, and at layer 0 the two native
mhc.pre configurations are bit-identical on real inputs. The remaining 8192-row record, mhc.pre.expanded
(tf32, k-splits 8 vs 1), runs only on engram layers (the only callers of the unfused pre), on all 8192
rows of every chunk (both engram layers precede the CED decoder start; sequence parallelism is off in
the release model, so every rank holds all rows). Its synthetic comparison stayed within every B12X
bound; only real inputs can separate autotuner rounding from a defect.

Source calls at vLLM 1794dcf1: the engram decoder layer calls
  self._b12x_mhc.pre(residual, fn, scale, base, norm, pre_mix, previous_output=None, ...)
and B12xMHC.post_pre calls self.pre(..., previous_output=x, ...) for every fused call. The instance
wrapper therefore sees both; fused calls (previous_output is not None) pass through untouched, before
any decision test or other work.

One layer per capture (trigger "layer"), on the final chunk of the frozen prompt only. The mHC state is
replicated on every rank, so the full real inputs are staged ROW-SHARDED: rank k keeps rows
[k*rows/W, (k+1)*rows/W) of every row tensor, and the four files together hold every row once.
  every rank  its row shard of residual [rows, 4, H] bf16 (after the Engram update, the tensor handed to
              pre), pre_mix, positions, post, comb and pre_out, in ONE exact-size host buffer registered
              with cudaHostRegister (no caching-allocator rounding), released right after the save
  rank 0      also fn, scale, base and the attn_norm and ffn_norm weights
  last rank   also the 128-row tails of y, residual_out and the attention output
  every rank  bounded device digests (digest(): two int64 lanes, chunked) of every captured tensor at
              FULL size, including y, residual_out and the attention output: equal digests on every rank
              prove the shards come from identical tensors, the reassembled rows must reproduce them,
              and the served y is checkable without staging it
  every rank  the served plan's installed Selection (component, encoded query, config, source)

Memory: memory_plan() gives the exact added bytes per rank (DS4.1: about 80 MiB of registered host memory
plus a 15 MiB device digest bound). Before any allocation every rank requires MemAvailable - need >=
reserve_mib (trigger); otherwise it records a refusal and allocates nothing. The save streams the buffer
to disk in 16 MiB zero-copy slices while hashing it, then releases it; nothing re-reads or copies it.

Trigger (same content on every node, after readiness):
  /cache/claude-mhc-expanded.json
  {"token": "<8-64 of [a-z0-9-]>", "prompt_tokens": 524288, "chunk_rows": 8192, "layer": 1, "reserve_mib": 1024}
Output: /cache/claude-mhc-expanded/<node>-rank<k>-<token>.json and .bin on every rank, and
<token>-rank<k>.mhc-consumed | .mhc-aborted. Copies and digests change timing, never arithmetic. Any
error disarms, releases staging only after the stream has drained, and serving continues.
"""
import ctypes
import hashlib
import json
import os
import re
import threading
import time

import torch

TRIGGER = '/cache/claude-mhc-expanded.json'
OUT_DIR = '/cache/claude-mhc-expanded'
MEMINFO = '/proc/meminfo'
SCHEMA = 'claude-mhc-expanded-v2'
PROMPT_TOKENS, CHUNK_ROWS, TAIL, POLL_SECONDS = 524288, 8192, 128, 1.0
ALIGN, PAGE, WRITE_SLICE = 256, 4096, 16 << 20
DIGEST_BLOCK_BYTES = 2 << 20
# Device temporaries of one digest block: the contiguous block (1x), the int64 widening (2x), the int64
# index weights (2x) and their product (2x), plus 1 MiB for scalars and allocator slack.
DIGEST_TRANSIENT_BYTES = DIGEST_BLOCK_BYTES * 7 + (1 << 20)
RESERVE_MIB = (256, 16384)
TOKEN = re.compile(r'^[a-z0-9-]{8,64}$')
DIGESTED = ('residual', 'pre_mix', 'positions', 'post', 'comb', 'pre_out', 'fn', 'scale', 'base', 'norm',
            'ffn_norm', 'y', 'residual_out', 'attn_out')
ROW_SHARDED = ('residual', 'pre_mix', 'positions', 'post', 'comb', 'pre_out')
WEIGHTS = ('fn', 'scale', 'base', 'norm', 'ffn_norm')
TAILS = ('y_tail', 'residual_out_tail', 'attn_out_tail')
_IMPORTED_AT = time.time()
_state = {'armed': None, 'checked': 0.0, 'done': set(), 'layers': {}}


def _log(message):
    print(f'CLAUDE-MHC-EXPANDED {message}', flush=True)


def _rank():
    from vllm.distributed import get_tensor_model_parallel_rank
    return get_tensor_model_parallel_rank()


def _world_size():
    from vllm.distributed import get_tensor_model_parallel_world_size
    return get_tensor_model_parallel_world_size()


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# ---------------------------------------------------------------- pure pieces (locally tested)

def shard_rows(rows, rank, world):
    if rows % world:
        raise ValueError(f'{rows} rows do not shard evenly over {world} ranks')
    per = rows // world
    return rank * per, (rank + 1) * per


def staging_layout(hidden, rows=CHUNK_ROWS, rank=0, world=1):
    """This rank's staging: [(name, shape, dtype)]: its row shard, the weights on rank 0, tails on the last."""
    lo, hi = shard_rows(rows, rank, world)
    n = hi - lo
    layout = [('residual', (n, 4, hidden), torch.bfloat16), ('pre_mix', (n, 4), torch.float32),
              ('positions', (n,), torch.int64), ('post', (n, 4), torch.float32),
              ('comb', (n, 4, 4), torch.float32), ('pre_out', (n, 4), torch.float32)]
    if rank == 0:
        layout += [('fn', (24, 4 * hidden), torch.float32), ('scale', (3,), torch.float32),
                   ('base', (24,), torch.float32), ('norm', (hidden,), torch.bfloat16),
                   ('ffn_norm', (hidden,), torch.bfloat16)]
    if rank == world - 1:
        layout += [('y_tail', (TAIL, hidden), torch.bfloat16), ('residual_out_tail', (TAIL, 4, hidden), torch.bfloat16),
                   ('attn_out_tail', (TAIL, hidden), torch.bfloat16)]
    return layout


def layout_offsets(layout):
    """{name: (offset, nbytes, shape, dtype)} packed at ALIGN, and the total byte count."""
    offsets, cursor = {}, 0
    for name, shape, dtype in layout:
        nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        offsets[name] = (cursor, nbytes, tuple(shape), dtype)
        cursor = (cursor + nbytes + ALIGN - 1) // ALIGN * ALIGN
    return offsets, cursor


def memory_plan(hidden, rows=CHUNK_ROWS, world=1):
    """Exact added bytes per rank while armed: the registered staging region (plus one page for
    alignment), two small digest tables, and the per-block device digest bound."""
    table = len(DIGESTED) * 2 * 8
    ranks = []
    for rank in range(world):
        _, total = layout_offsets(staging_layout(hidden, rows, rank, world))
        ranks.append({'rank': rank, 'host_registered_bytes': total, 'host_allocation_bytes': total + PAGE,
                      'peak_bytes': total + PAGE + 2 * table + DIGEST_TRANSIENT_BYTES})
    return {'world': world, 'digest_table_bytes': table, 'device_digest_transient_bound_bytes': DIGEST_TRANSIENT_BYTES,
            'ranks': ranks, 'max_peak_bytes': max(r['peak_bytes'] for r in ranks)}


def trigger_spec(spec, layers=None):
    """The validated trigger, or None when this helper must not arm."""
    if (not isinstance(spec, dict) or set(spec) != {'token', 'prompt_tokens', 'chunk_rows', 'layer', 'reserve_mib'}
            or not isinstance(spec['token'], str) or not TOKEN.match(spec['token'])
            or any(type(spec[k]) is not int for k in ('prompt_tokens', 'chunk_rows', 'layer', 'reserve_mib'))
            or spec['prompt_tokens'] != PROMPT_TOKENS or spec['chunk_rows'] != CHUNK_ROWS
            or not RESERVE_MIB[0] <= spec['reserve_mib'] <= RESERVE_MIB[1]
            or (layers is not None and spec['layer'] not in layers)):
        return None
    return spec


def is_decision(metadata, rows, spec):
    """Host-only decision-forward test on one layer's attention metadata (the window helper's rule)."""
    return (not metadata.is_decode and metadata.num_reqs == 1 and metadata.max_seq_len == spec['prompt_tokens']
            and metadata.num_actual_tokens == spec['chunk_rows'] and rows == spec['chunk_rows'])


def mem_available(path=None):
    with open(path or MEMINFO) as stream:
        for line in stream:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing from ' + (path or MEMINFO))


def memory_problem(available, need, reserve_mib):
    reserve = reserve_mib << 20
    if available - need < reserve:
        return f'MemAvailable {available} minus need {need} is below the {reserve}-byte reserve'
    return None


def digest(tensor, out=None):
    """Two int64 lanes over the tensor's raw 32-bit words in C order: lane 0 = sum(word), lane 1 =
    sum(word * ((index mod 65521) + 1)), both wrapping. Integer addition is associative, so the result
    is independent of reduction order and identical on CPU and GPU. Leading-dimension blocks of at most
    DIGEST_BLOCK_BYTES bound the temporaries whatever the strides. `out` (int64 [2], same device)
    accumulates in place and is returned."""
    device = tensor.device
    out = torch.zeros(2, dtype=torch.int64, device=device) if out is None else out
    rows = tensor if tensor.dim() >= 2 else tensor.reshape(-1)
    row_bytes = (rows[0].numel() if rows.dim() >= 2 else 1) * rows.element_size()
    per_block = max(1, DIGEST_BLOCK_BYTES // row_bytes)
    start = 0
    for first in range(0, rows.shape[0], per_block):
        raw = rows[first:first + per_block].contiguous().reshape(-1).view(torch.uint8)
        if raw.numel() % 4:
            raise ValueError('digest needs a whole number of 32-bit words')
        words = raw.view(torch.int32)
        out[0] += words.sum(dtype=torch.int64)
        weights = torch.arange(start, start + words.numel(), dtype=torch.int64, device=device).remainder_(65521).add_(1)
        out[1] += (words.to(torch.int64) * weights).sum()
        start += words.numel()
    return out


def write_region(path, region):
    """Stream a contiguous uint8 host tensor to path in WRITE_SLICE zero-copy views, hashing as it goes."""
    total = region.numel()
    view = memoryview((ctypes.c_char * total).from_address(region.data_ptr())).cast('B')
    hasher = hashlib.sha256()
    with open(path + '.tmp', 'wb') as stream:
        for first in range(0, total, WRITE_SLICE):
            piece = view[first:first + WRITE_SLICE]
            hasher.update(piece)
            stream.write(piece)
    os.replace(path + '.tmp', path)
    return hasher.hexdigest()


def _plain(value):
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value if isinstance(value, (str, int, float, bool)) or value is None else repr(value)


# ---------------------------------------------------------------- install and hooks

def engram_layers(model):
    """{layer_id: decoder layer} for target decoder layers with an Engram module (unfused pre callers)."""
    found = {}
    for layer in model.layers:
        attn = getattr(layer, 'attn', None)
        if getattr(layer, 'engram', None) is None or attn is None or getattr(attn, 'is_draft', False):
            continue
        found[int(attn.layer_id)] = layer
    return found


def install(model):
    """Wrap each engram layer's B12xMHC.pre on this instance and hook its attention output; idempotent."""
    for layer_id, layer in engram_layers(model).items():
        if layer_id in _state['layers']:
            continue
        mhc = layer._b12x_mhc
        original = mhc.pre

        def pre(*args, _original=original, _layer=layer, _id=layer_id, **kwargs):
            outputs = _original(*args, **kwargs)
            # Fused post_pre arrives here through self.pre(previous_output=x): never touched.
            if kwargs.get('previous_output') is None and _state['armed'] is not None:
                _after_pre(_id, _layer, args, outputs)
            return outputs
        object.__setattr__(mhc, 'pre', pre)
        layer.register_forward_pre_hook(lambda module, args, _id=layer_id: _positions(_id, args))
        layer.attn.register_forward_hook(lambda module, args, output, _id=layer_id: _after_attention(_id, output))
        _state['layers'][layer_id] = layer
    _log(f'installed layers={sorted(_state["layers"])}')


def _read_trigger(rank):
    try:
        if os.stat(TRIGGER).st_mtime <= _IMPORTED_AT:
            return None
        with open(TRIGGER) as stream:
            spec = trigger_spec(json.load(stream), set(_state['layers']))
    except (OSError, ValueError):
        return None
    if spec is None or spec['token'] in _state['done']:
        return None
    if os.path.exists(os.path.join(OUT_DIR, f'{spec["token"]}-rank{rank}.mhc-consumed')):
        _state['done'].add(spec['token'])
        return None
    return spec


def _write_abort(token, rank, error, extra=None):
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, f'{token}-rank{rank}.mhc-aborted'), 'a') as stream:
            stream.write(json.dumps({'error': f'{type(error).__name__}: {error}', 'unix': time.time(),
                                     **(extra or {})}, default=str) + '\n')
    except OSError as failure:
        _log(f'abort receipt not written: {failure}')


def _abort(error):
    session = _state['armed']
    _state['armed'] = None
    token = session.token if session is not None else None
    _log(f'aborted token={token} {type(error).__name__}: {error}')
    if session is not None:
        session.fired = True
        _write_abort(token, session.rank, error, {'memory': session.memory})
        session.release_after_stream()


def _guarded(function):
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:                      # a diagnostic must not kill the serving forward
            _abort(error)
            return None
    wrapper.__name__, wrapper.__doc__ = function.__name__, function.__doc__
    return wrapper


@_guarded
def _positions(layer_id, args):
    if _capturing():
        return
    state = _state
    if state['armed'] is None:
        now = time.monotonic()
        if now - state['checked'] < POLL_SECONDS:
            return
        state['checked'] = now
        rank, world = _rank(), _world_size()
        spec = _read_trigger(rank)
        if spec is None:
            return
        state['done'].add(spec['token'])               # one attempt per token, whatever happens next
        layer = state['layers'][spec['layer']]
        plan = memory_plan(int(layer._b12x_mhc.hidden_size), spec['chunk_rows'], world)
        need = plan['ranks'][rank]['peak_bytes']
        try:
            available = mem_available()
            problem = memory_problem(available, need, spec['reserve_mib'])
            if problem:
                raise MemoryError(problem)
            state['armed'] = _Session(spec, rank, world, layer, plan, available)
        except Exception as error:
            _write_abort(spec['token'], rank, error, {'memory': {'plan': plan, 'need': need}})
            _log(f'refused token={spec["token"]} rank={rank}: {type(error).__name__}: {error}')
            return
        _log(f'armed token={spec["token"]} rank={rank} layer={spec["layer"]} need={need}')
    session = state['armed']
    if session is not None and layer_id == session.layer_id and not session.fired:
        session.positions = args[1]


@_guarded
def _after_pre(layer_id, layer, args, outputs):
    session = _state['armed']
    if session is None or session.fired or _capturing() or layer_id != session.layer_id:
        return
    session.pre(layer, args, outputs)


@_guarded
def _after_attention(layer_id, output):
    session = _state['armed']
    if session is None or session.fired or _capturing() or layer_id != session.layer_id:
        return
    session.attention(output)


def _cudart():
    return torch.cuda.cudart()


def _event():
    event = torch.cuda.Event()
    event.record()
    return event


class _Session:
    """One armed token on one rank: rank-0 registered staging, device digest slot, event waiter, save."""

    def __init__(self, spec, rank, world, layer, plan, available):
        self.token, self.rank, self.world, self.spec, self.layer_id = spec['token'], rank, world, spec, spec['layer']
        self.shard = shard_rows(spec['chunk_rows'], rank, world)
        self.fired, self.captured, self.positions, self.plan_record = False, False, None, None
        self.started, self.pending_copies = time.time(), False
        self.memory = {'plan': plan, 'available_at_arm': available}
        self.hidden = int(layer._b12x_mhc.hidden_size)
        cuda = torch.cuda.is_available()
        self.slot = torch.zeros((len(DIGESTED), 2), dtype=torch.int64, device='cuda' if cuda else 'cpu')
        self.slot_host = torch.zeros((len(DIGESTED), 2), dtype=torch.int64, pin_memory=cuda)
        self.registered = None
        self.views = {}
        self.offsets, self.total = layout_offsets(staging_layout(self.hidden, spec['chunk_rows'], rank, world))
        self.buffer = torch.empty(self.total + PAGE, dtype=torch.uint8)
        start = (-self.buffer.data_ptr()) % PAGE
        if cuda:
            code = int(_cudart().cudaHostRegister(self.buffer.data_ptr() + start, self.total, 0))
            if code != 0:
                self.buffer = None
                raise RuntimeError(f'cudaHostRegister failed with {code}')
            self.registered = self.buffer.data_ptr() + start
        self.region = self.buffer[start:start + self.total]
        for name, (offset, nbytes, shape, dtype) in self.offsets.items():
            self.views[name] = self.region[offset:offset + nbytes].view(dtype).view(shape)
        self.memory['available_after_allocation'] = mem_available()

    def _stage(self, name, value):
        """Copy into this rank's staging when it holds `name`; row-sharded tensors are sliced first."""
        staged = self.views.get(name)
        if staged is None:
            return
        if name in ROW_SHARDED:
            value = value[self.shard[0]:self.shard[1]]
        if tuple(value.shape) != tuple(staged.shape) or value.dtype != staged.dtype:
            raise RuntimeError(f'{name} is {tuple(value.shape)} {value.dtype}, staging expects '
                               f'{tuple(staged.shape)} {staged.dtype}')
        self.pending_copies = True
        staged.copy_(value, non_blocking=True)

    def pre(self, layer, args, outputs):
        if len(args) != 6:
            raise RuntimeError('unfused pre was not called with six positional operands')
        residual, fn, scale, base, norm, pre = args
        from vllm.forward_context import get_forward_context
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            raise RuntimeError('attention metadata is not a per-layer mapping')
        if not is_decision(metadata[layer.attn.swa_cache_layer.prefix], int(residual.shape[0]), self.spec):
            return
        if self.captured:
            raise RuntimeError('decision forward entered twice')
        if self.positions is None or int(self.positions.shape[0]) != int(residual.shape[0]):
            raise RuntimeError('decoder-layer positions were not observed for this call')
        residual_out, post, comb, y, pre_out = outputs
        tensors = {'residual': residual, 'pre_mix': pre, 'positions': self.positions.to(torch.int64), 'post': post,
                   'comb': comb, 'pre_out': pre_out, 'fn': fn, 'scale': scale, 'base': base, 'norm': norm,
                   'ffn_norm': layer.ffn_norm.weight, 'y': y, 'residual_out': residual_out}
        selection = layer._b12x_mhc._plan_for('pre', int(residual.shape[0])).selection
        if selection is None:
            raise RuntimeError('served pre plan has no installed selection')
        self.plan_record = {'component_id': selection.component_id, 'source': selection.source,
                            'query': _plain(selection.query), 'config': _plain(selection.config)}
        expected = {name: (shape, dtype) for name, shape, dtype in staging_layout(self.hidden, self.spec['chunk_rows'], 0, 1)}
        for name in ROW_SHARDED + WEIGHTS:              # validate every full shape before any work is enqueued
            shape, dtype = expected[name]
            if tuple(tensors[name].shape) != tuple(shape) or tensors[name].dtype != dtype:
                raise RuntimeError(f'{name} is {tuple(tensors[name].shape)} {tensors[name].dtype}, '
                                   f'expected {tuple(shape)} {dtype}')
        for name, value in tensors.items():
            digest(value, self.slot[DIGESTED.index(name)])
        for name in ROW_SHARDED + WEIGHTS:
            self._stage(name, tensors[name])
        self._stage('y_tail', y[-TAIL:])
        self._stage('residual_out_tail', residual_out[-TAIL:])
        self.captured = True

    def attention(self, output):
        if not self.captured:
            return
        digest(output, self.slot[DIGESTED.index('attn_out')])
        self._stage('attn_out_tail', output[-TAIL:])
        self.fired = True
        self.slot_host.copy_(self.slot, non_blocking=True)
        event = _event()
        threading.Thread(target=self.complete, args=(event,), name='claude-mhc-expanded-waiter', daemon=True).start()

    def release_after_stream(self):
        """Free staging only after every copy already enqueued into it has completed."""
        if self.buffer is None:
            return
        if not self.pending_copies:
            self._release()
            return
        event = _event()
        threading.Thread(target=lambda: (event.synchronize(), self._release()), daemon=True).start()

    def _release(self):
        if self.registered is not None:
            code = int(_cudart().cudaHostUnregister(self.registered))
            self.registered = None
            if code != 0:
                _log(f'cudaHostUnregister returned {code}')
        self.views, self.region, self.buffer = {}, None, None

    def complete(self, event):
        try:
            event.synchronize()
            self.save()
        except BaseException as error:
            _log(f'error token={self.token} rank={self.rank} {type(error).__name__}: {error}')
            _write_abort(self.token, self.rank, error, {'memory': self.memory})
        finally:
            self._release()
            _state['armed'] = None

    def save(self, out_dir=None, node=None):
        out_dir, node = out_dir or OUT_DIR, node or os.environ.get('DS41_NODE')
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, f'{node}-rank{self.rank}-{self.token}')
        record = {'schema': SCHEMA, 'token': self.token, 'rank': self.rank, 'node': node, 'layer': self.layer_id,
                  'prompt_tokens': self.spec['prompt_tokens'], 'chunk_rows': self.spec['chunk_rows'],
                  'world': self.world, 'shard': list(self.shard),
                  'hidden': self.hidden, 'kit_sha256': os.environ.get('DS41_KIT_SHA256'),
                  'armed_unix': self.started, 'plan': self.plan_record,
                  'digests': {name: [int(v) for v in self.slot_host[i]] for i, name in enumerate(DIGESTED)}}
        receipt = {'json': stem + '.json'}
        self.memory['available_at_save'] = mem_available()
        record['staging'] = {'total': self.total, 'tensors': {
            n: {'offset': o, 'nbytes': b, 'shape': list(s), 'dtype': str(d).removeprefix('torch.')}
            for n, (o, b, s, d) in self.offsets.items()}}
        receipt['bin'], receipt['bin_sha256'] = stem + '.bin', write_region(stem + '.bin', self.region)
        self._release()
        self.memory['available_after_release'] = mem_available()
        record['memory'], record['saved_unix'] = self.memory, time.time()
        with open(stem + '.json.tmp', 'w') as stream:
            stream.write(json.dumps(record, indent=1, sort_keys=True) + '\n')
        os.replace(stem + '.json.tmp', stem + '.json')
        with open(os.path.join(out_dir, f'{self.token}-rank{self.rank}.mhc-consumed'), 'w') as stream:
            stream.write(json.dumps(receipt, sort_keys=True) + '\n')
        _log(f'saved token={self.token} rank={self.rank} layer={self.layer_id}')
