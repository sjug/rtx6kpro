"""DIAGNOSTIC ONLY: final-window capture through layers 0 and 1 (schema claude-window-v1).

Installed as vllm/models/deepseek_v4_1/claude_window.py in a labelled
window-capture derivative of the router fence candidate, beside the unchanged
decision-row helper (claude_decision_row.py, schema claude-decision-row-v3).
Both helpers arm from the SAME trigger file; this one adds nothing to the
decision-row file and writes its own sibling file plus its own receipts:

  trigger   /cache/claude-decision-row.json  {"token": ..., "prompt_tokens": 524288, "chunk_rows": 8192|4096,
                                              "window": true | {"rows": 128, "layers": [0, 1]} (optional)}
            "window": false disarms this helper only; absent means armed. The other keys are the
            decision-row helper's and are validated identically here.
  output    /cache/claude-decision-row/<node>-rank<k>-<token>-window.pt
  receipts  /cache/claude-decision-row/<token>-rank<k>.window-consumed | .window-aborted

Target: the same decision forward the decision-row helper targets (one prefill
request, num_actual_tokens == chunk_rows, max_seq_len == prompt_tokens), on
target layers 0 and 1 only (compress ratio 0, full-row encoder layers, no CED
view). Captured rows are the last WINDOW = 128 rows of that chunk, positions
prompt_tokens - 128 .. prompt_tokens - 1. Every boundary is an existing eager
segment of the pinned attention source: the body of the vllm::dsv41_b12x_attention
custom op (_forward), insert_context_kv inside the eager_break _cache_context_kv,
forward_mqa (eager_break) and _o_proj (custom-op body). No hook adds a new outer
compile boundary; the copies still add kernels and copies to those segments and
can perturb execution, which the matched unarmed controls and the same-boot
decision-row comparison bound but do not exclude.

Per layer, in producer order:
  inputs      hidden_in [128, hidden] (custom-op input rows), kv_norm [128, 512] (after kv_norm, before
              rotation), positions [128]
  rotated     kv_rot [128, 512] (the tensor handed to mla.write_cache) and write_slots [128] (its slot_mapping)
  attention   q [128, heads, 512], out [128, heads, 512], attn_sink [heads], swa_indices [128, width],
              swa_lengths [128], record_slots [256] and records [256, 528]: the window of the first captured
              row followed by the window of the last captured row, i.e. the 255 distinct positions
              prompt_tokens - 255 .. prompt_tokens - 1 (one position appears twice), gathered from the SWA
              cache after the attention kernel read it
  wo_partial  [128, hidden] the WO projection result before the TP all-reduce
  wo_reduced  [128, hidden] after the TP all-reduce (equal to wo_partial when TP is 1)

Copies are non_blocking D2H into pinned staging preallocated at arm time; the
record gather is one device index followed by a stream-ordered D2H, as the
decision-row helper gathers its records. After layer 1's
wo_reduced an event is recorded and a waiter thread saves from host memory.

Stated limit: hooks add kernels and copies after producers, so timing changes.
Equal full responses and equal decision-row tensors against the uninstrumented
captures show no observed output perturbation; they do not prove hidden
numerics were untouched.
"""
import hashlib
import json
import os
import re
import threading
import time

import torch

TRIGGER = '/cache/claude-decision-row.json'
OUT_DIR = '/cache/claude-decision-row'
SOURCE_LOCK = '/opt/ds41-window/claude-window.lock.json'
SCHEMA = 'claude-window-v1'
DEFAULT_CHUNK, WINDOW, RECORDS, KV_DIM, HEAD_DIM, SWA_RECORD, POLL_SECONDS = 8192, 128, 256, 512, 512, 528, 1.0
LAYERS = (0, 1)
BOUNDARIES = ('inputs', 'rotated', 'attention', 'wo_partial', 'wo_reduced')
TOKEN = re.compile(r'^[a-z0-9-]{8,64}$')
_IMPORTED_AT = time.time()
_state = {'armed': None, 'checked': 0.0, 'done': set()}


def _log(message):
    print(f'CLAUDE-WINDOW {message}', flush=True)


def _rank():
    from vllm.distributed import get_tensor_model_parallel_rank
    return get_tensor_model_parallel_rank()


def _world_size():
    from vllm.distributed import get_tensor_model_parallel_world_size
    return get_tensor_model_parallel_world_size()


def _source_trees():
    try:
        with open(SOURCE_LOCK) as stream:
            return json.load(stream)['trees']
    except (OSError, ValueError, KeyError):
        return None


def _capturing():
    return torch.cuda.is_current_stream_capturing()


def window_spec(spec):
    """The validated window request of a trigger, or None when this helper must not arm."""
    if not isinstance(spec, dict):
        return None
    token = spec.get('token')
    chunk_rows = spec.get('chunk_rows', DEFAULT_CHUNK)
    if (not isinstance(token, str) or not TOKEN.match(token) or spec.get('prompt_tokens') != 524288
            or type(chunk_rows) is not int or chunk_rows not in (8192, 4096)):
        return None
    window = spec.get('window', True)
    if window is False:
        return None
    if window is True:
        window = {'rows': WINDOW, 'layers': list(LAYERS)}
    if (not isinstance(window, dict) or window.get('rows') != WINDOW
            or window.get('layers') != list(LAYERS)):
        return None
    return {'token': token, 'prompt_tokens': 524288, 'chunk_rows': chunk_rows, 'window': window}


def _read_trigger(rank):
    try:
        if os.stat(TRIGGER).st_mtime <= _IMPORTED_AT:
            return None
        with open(TRIGGER) as stream:
            spec = window_spec(json.load(stream))
    except (OSError, ValueError):
        return None
    if spec is None or spec['token'] in _state['done']:
        return None
    if os.path.exists(os.path.join(OUT_DIR, f'{spec["token"]}-rank{rank}.window-consumed')):
        _state['done'].add(spec['token'])
        return None
    return spec


def _target_layers():
    from vllm.forward_context import get_forward_context
    found = {}
    for module in get_forward_context().no_compile_layers.values():
        layer_id = getattr(module, 'layer_id', None)
        if (layer_id in LAYERS and hasattr(module, 'swa_cache_layer') and hasattr(module, 'forward_mqa')
                and not getattr(module, 'is_draft', False)):
            found[layer_id] = module
    if sorted(found) != list(LAYERS):
        raise RuntimeError(f'expected target attention layers {list(LAYERS)}, found {sorted(found)}')
    for layer in found.values():
        if getattr(layer, 'compress_ratio', None) or getattr(layer, 'is_ced_decoder', False):
            raise RuntimeError(f'layer {layer.layer_id} is not a full-row SWA encoder layer')
    return found


def _abort(error, token=None):
    """Fail open for serving: log, disarm, never save; production kernels already ran."""
    session = _state['armed']
    token = token or (session.token if session is not None else None)
    if session is not None:
        session.fired = True
    _state['armed'] = None
    if token:
        _state['done'].add(token)
    _log(f'aborted token={token} {type(error).__name__}: {error}')
    if token and session is not None:
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(os.path.join(OUT_DIR, f'{token}-rank{session.rank}.window-aborted'), 'a') as stream:
                stream.write(json.dumps({'error': f'{type(error).__name__}: {error}', 'unix': time.time()}) + '\n')
        except OSError as failure:
            _log(f'abort receipt not written: {failure}')


def _guarded(method):
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception as error:                      # a diagnostic must not kill the serving forward
            _abort(error)
            return None
    wrapper.__name__, wrapper.__doc__ = method.__name__, method.__doc__
    return wrapper


def begin(layer, metadata):
    """In _forward once the SWA metadata is resolved: the open layer capture on the decision chunk, else None."""
    try:
        return _begin(layer, metadata)
    except Exception as error:
        _abort(error)
        return None


def _begin(layer, metadata):
    if _capturing():                                   # never arm, allocate or copy under graph capture
        return None
    state = _state
    if state['armed'] is None:
        now = time.monotonic()
        if now - state['checked'] < POLL_SECONDS:
            return None
        state['checked'] = now
        rank = _rank()
        spec = _read_trigger(rank)
        if spec is None:
            return None
        state['armed'] = _Session(spec, rank, _target_layers())
        _log(f'armed token={spec["token"]} rank={rank} layers={list(LAYERS)} rows={WINDOW}')
    session = state['armed']
    if session.fired or getattr(layer, 'is_draft', False) or layer.layer_id not in session.layers:
        return None
    original = metadata[layer.swa_cache_layer.prefix]
    if (original.is_decode or original.num_reqs != 1
            or original.max_seq_len != session.prompt_tokens):
        return None
    if original.num_actual_tokens != session.chunk_rows:
        raise RuntimeError(f'final chunk rows {original.num_actual_tokens} differ from armed {session.chunk_rows}')
    return session.open(layer, original)


def _active(layer):
    """The open capture of `layer` on the armed decision forward, else None (never raises, no device work)."""
    session = _state['armed']
    if session is None or session.fired or _capturing():
        return None
    entry = session.layers.get(getattr(layer, 'layer_id', None))
    return entry if entry is not None and entry.open else None


def rotated(layer, *, rotated, slot_mapping, positions):
    entry = _active(layer)
    if entry is not None:
        entry.rotated(rotated=rotated, slot_mapping=slot_mapping, positions=positions)


def attention(layer, *, q, output, swa_indices, swa_lengths):
    entry = _active(layer)
    if entry is not None:
        entry.attention(layer, q=q, output=output, swa_indices=swa_indices, swa_lengths=swa_lengths)


def wo_partial(layer, local):
    entry = _active(layer)
    if entry is not None:
        entry.wo_partial(local)


def wo_reduced(layer, local):
    entry = _active(layer)
    if entry is not None:
        entry.wo_reduced(local)


def _pinned(shape, dtype):
    return torch.empty(shape, dtype=dtype, pin_memory=True)


def _gather(cache, slots, page, record):
    """Records of a block-major cache [blocks, page*record] at physical slots (invalid slots read slot 0)."""
    safe = slots.clamp_min(0)
    columns = (safe % page)[:, None] * record + torch.arange(record, device=slots.device)
    return cache[(safe // page)[:, None], columns]


class _Session:
    """One armed token on one rank: all staging preallocated, in-step copies, event waiter, save."""

    def __init__(self, spec, rank, layers):
        self.token, self.rank, self.prompt_tokens = spec['token'], rank, spec['prompt_tokens']
        self.chunk_rows = spec['chunk_rows']
        self.fired, self.event, self.waiter = False, None, None
        self.started = time.time()
        self.world_size = _world_size()
        self.layers = {lid: _LayerCapture(layer, self) for lid, layer in sorted(layers.items())}

    def open(self, layer, original):
        entry = self.layers[layer.layer_id]
        entry.begin(original)
        return entry

    def finish(self):
        """After layer 1's reduced WO output; completion no longer depends on any later forward."""
        if self.fired or _state['armed'] is not self:
            return
        self.fired = True
        incomplete = {lid: sorted(set(BOUNDARIES) - entry.seen) for lid, entry in self.layers.items()
                      if entry.seen != set(BOUNDARIES)}
        if incomplete:
            _log(f'incomplete decision forward token={self.token} rank={self.rank} missing={incomplete}')
            _state['armed'] = None
            return
        self.event = torch.cuda.Event()
        self.event.record()
        self.waiter = threading.Thread(target=self.complete, name='claude-window-waiter', daemon=True)
        self.waiter.start()

    def complete(self):
        try:
            self.event.synchronize()
            self.save()
        except BaseException as error:
            _log(f'error token={self.token} rank={self.rank} {type(error).__name__}: {error}')
        finally:
            _state['done'].add(self.token)
            _state['armed'] = None

    def save(self):
        problems = []
        layers = {lid: entry.host(problems) for lid, entry in sorted(self.layers.items())}
        last = self.prompt_tokens - 1
        expected = torch.arange(last - WINDOW + 1, last + 1, dtype=torch.int64)
        for lid, entry in layers.items():
            if not torch.equal(entry['positions'], expected):
                problems.append(f'layer {lid}: captured positions are not the final {WINDOW} of the request')
        node = os.environ.get('DS41_NODE')
        meta = {'rank': self.rank, 'node': node, 'generation': self.token, 'kit_sha256': os.environ.get('DS41_KIT_SHA256'),
                'source_trees': _source_trees(), 'prompt_tokens': self.prompt_tokens, 'chunk_rows': self.chunk_rows,
                'window_rows': WINDOW, 'record_rows': RECORDS, 'layers': list(LAYERS),
                'tp_world_size': self.world_size, 'batch_requests': 1, 'row_position': last,
                'decision_row_file': f'{node}-rank{self.rank}-{self.token}.pt',
                'problems': problems, 'pid': os.getpid(), 'armed_unix': self.started, 'saved_unix': time.time()}
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f'{node}-rank{self.rank}-{self.token}-window.pt')
        torch.save({'schema': SCHEMA, 'meta': meta, 'layers': layers}, path + '.tmp')
        os.replace(path + '.tmp', path)
        with open(path, 'rb') as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        with open(os.path.join(OUT_DIR, f'{self.token}-rank{self.rank}.window-consumed'), 'x') as stream:
            stream.write(json.dumps({'file': path, 'sha256': digest, 'problems': problems}) + '\n')
        _log(f'saved token={self.token} rank={self.rank} file={path} sha256={digest} problems={len(problems)}')


class _LayerCapture:
    """Pinned staging for one target layer, allocated at arm time; open only on the decision forward."""

    def __init__(self, layer, session):
        self.session, self.layer_id = session, layer.layer_id
        self.rows, self.open, self.seen = session.chunk_rows, False, set()
        hidden, heads, width = int(layer.hidden_size), int(layer.n_local_heads), int(layer.swa_width)
        self.hidden, self.heads, self.width = hidden, heads, width
        self.page = layer.swa_cache_layer.block_size
        self.staged = {
            'positions': _pinned((WINDOW,), torch.int64),
            'hidden_in': _pinned((WINDOW, hidden), torch.bfloat16),
            'kv_norm': _pinned((WINDOW, KV_DIM), torch.bfloat16),
            'kv_rot': _pinned((WINDOW, KV_DIM), torch.bfloat16),
            'write_slots': _pinned((WINDOW,), torch.int64),
            'q': _pinned((WINDOW, heads, HEAD_DIM), torch.bfloat16),
            'out': _pinned((WINDOW, heads, HEAD_DIM), torch.bfloat16),
            'attn_sink': _pinned((heads,), torch.float32),
            'swa_indices': _pinned((WINDOW, width), torch.int64),
            'swa_lengths': _pinned((WINDOW,), torch.int32),
            'record_slots': _pinned((RECORDS,), torch.int64),
            'records': _pinned((RECORDS, SWA_RECORD), torch.uint8),
            'wo_partial': _pinned((WINDOW, hidden), torch.bfloat16),
            'wo_reduced': _pinned((WINDOW, hidden), torch.bfloat16),
        }

    def _d2h(self, name, value):
        staged = self.staged[name]
        if tuple(value.shape) != tuple(staged.shape) or value.dtype != staged.dtype:
            raise RuntimeError(f'layer {self.layer_id}: {name} has shape {tuple(value.shape)} {value.dtype}, '
                               f'staging expects {tuple(staged.shape)} {staged.dtype}')
        staged.copy_(value, non_blocking=True)

    def _tail(self, tensor, name):
        if tensor.shape[0] != self.rows:
            raise RuntimeError(f'layer {self.layer_id}: {name} has {tensor.shape[0]} rows, expected {self.rows}')
        return tensor[self.rows - WINDOW:self.rows]

    def begin(self, original):
        if self.open or 'inputs' in self.seen:
            raise RuntimeError(f'layer {self.layer_id}: decision forward entered twice')
        if int(original.num_actual_tokens) != self.rows:
            raise RuntimeError(f'layer {self.layer_id}: query rows {original.num_actual_tokens}, expected {self.rows}')
        self.open = True

    @_guarded
    def inputs(self, layer, *, hidden_states, kv, positions):
        """Custom-op input rows and the normalized, unrotated KV latent (before _cache_context_kv)."""
        self._d2h('positions', self._tail(positions, 'positions').to(torch.int64))
        self._d2h('hidden_in', self._tail(hidden_states, 'hidden_states'))
        self._d2h('kv_norm', self._tail(kv, 'kv'))
        self.seen.add('inputs')

    @_guarded
    def rotated(self, *, rotated, slot_mapping, positions):
        """insert_context_kv: the rotated KV and the slots handed to mla.write_cache."""
        if 'inputs' not in self.seen:
            raise RuntimeError(f'layer {self.layer_id}: rotated KV before inputs')
        self._d2h('kv_rot', self._tail(rotated, 'rotated kv'))
        self._d2h('write_slots', self._tail(slot_mapping, 'slot_mapping').to(torch.int64))
        self.seen.add('rotated')

    @_guarded
    def attention(self, layer, *, q, output, swa_indices, swa_lengths):
        """forward_mqa after mla.run: the kernel's query and output rows and the records their windows read."""
        if 'rotated' not in self.seen:
            raise RuntimeError(f'layer {self.layer_id}: attention before the cache write')
        self._d2h('q', self._tail(q, 'attention query'))
        self._d2h('out', self._tail(output, 'attention output'))
        self._d2h('attn_sink', layer.attn_sink)
        indices = self._tail(swa_indices, 'swa_indices')
        self._d2h('swa_indices', indices.to(torch.int64))
        self._d2h('swa_lengths', self._tail(swa_lengths, 'swa_lengths').to(torch.int32))
        slots = torch.cat((indices[0], indices[WINDOW - 1])).to(torch.int64)
        self._d2h('record_slots', slots)
        # Stream-ordered: the temporary device gather outlives the non_blocking D2H on the same stream.
        self._d2h('records', _gather(layer.swa_cache_layer.kv_cache, slots, self.page, SWA_RECORD))
        self.seen.add('attention')

    @_guarded
    def wo_partial(self, local):
        """_o_proj: the WO projection result before the TP all-reduce."""
        if 'attention' not in self.seen:
            raise RuntimeError(f'layer {self.layer_id}: WO output before attention')
        self._d2h('wo_partial', self._tail(local, 'wo local').reshape(WINDOW, self.hidden))
        self.seen.add('wo_partial')

    @_guarded
    def wo_reduced(self, local):
        """_o_proj: the layer's returned attention block output after the TP all-reduce."""
        if 'wo_partial' not in self.seen:
            raise RuntimeError(f'layer {self.layer_id}: reduced WO output before the partial')
        self._d2h('wo_reduced', self._tail(local, 'wo reduced').reshape(WINDOW, self.hidden))
        self.seen.add('wo_reduced')
        self.open = False
        if self.layer_id == LAYERS[-1]:
            self.session.finish()

    def host(self, problems):
        """Waiter thread, after the event: schema entry from pinned staging only (no device access)."""
        s = {k: v.clone() for k, v in self.staged.items()}
        if self.seen != set(BOUNDARIES):
            problems.append(f'layer {self.layer_id}: boundaries missing {sorted(set(BOUNDARIES) - self.seen)}')
        s.update(layer_id=self.layer_id, chunk_rows=self.rows, chunk_row_start=self.rows - WINDOW,
                 hidden=self.hidden, heads=self.heads, swa_width=self.width, page_size=self.page)
        return s
