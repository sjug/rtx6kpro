"""DIAGNOSTIC ONLY: decision-row capture for the DS4.1 524288-token audit (schema claude-decision-row-v3).

Installed as vllm/models/deepseek_v4_1/claude_decision_row.py in a labelled
diagnostic derivative of the router fence candidate, never in the candidate.
DeepseekV41Attention.forward_mqa calls begin() once per layer call; everything
else runs only on the returned capture, so an unarmed process adds no device
work (a graph-capture check, a boolean, at most one trigger stat per second).

Arming (on every rank host, before sending the request):
  /cache/claude-decision-row.json  {"token": "<8-64 of [a-z0-9-]>", "prompt_tokens": 524288, "chunk_rows": 8192}
chunk_rows must be the integer 8192 or 4096; omitted keeps the historical 8192.
written after the server is ready (mtime after import). An abort writes a
durable <token>-rank<k>.aborted receipt beside the capture output. Arming happens on the
next eager forward that is not being graph-captured; it allocates all pinned
host staging and one device key-staging buffer then (cudaHostAlloc may
synchronize), never on the decision chunk. A token is used once: a durable
<token>-rank<k>.consumed receipt blocks reuse across restarts.

Target: the forward whose original SWA metadata is prefill, one request,
num_actual_tokens == chunk_rows and max_seq_len == prompt_tokens (host fields; no
device read). Target layers 0..39 (DSpark draft layers skipped). The row
geometry differs per layer, exactly as production resolves it through
layer._query_metadata: encoder layers (before ced_decoder_start, layer 20)
run all chunk_rows rows and the decision row is row chunk_rows - 1; CED decoder layers
(20..39) run only the gathered last CED_WINDOW = 128 rows of the request
(ced.py _decoder_indices: original_end - compact_end + row) and the decision
row is row 127. Every hook indexes row query_rows - 1 of the tensors it is
handed, checks that their leading dimension equals the query metadata's
num_actual_tokens, and stages that row's position; a position that is not
prompt_tokens - 1 is recorded as a problem at save time. Any other geometry
aborts the capture (fail open) instead of reading a wrong row. Every byte
the audit needs is copied inside that forward, on the current stream,
immediately after its producer: small tensors by non_blocking D2H into pinned
staging; index-key pages of each key owner (layers whose indexer owns its k
cache) by index_select into the preallocated device buffer followed by
non_blocking D2H into pinned storage. Stream order makes the device buffer
safe to reuse for the next owner.

Completion needs no later forward: after layer 39 an event is recorded and a
dedicated waiter thread blocks on that event alone, then builds and saves the
file from host memory. Nothing reads GPU caches after the decision forward,
so KV block reuse after the request ends cannot reach the capture.

Stated limit: hooks add kernels and copies after producers, so timing
changes. Equal full responses against the uninstrumented run show no observed
output perturbation; they do not prove hidden numerics were untouched.
"""
import ctypes
import dataclasses
import hashlib
import json
import os
import re
import threading
import time

import torch

TRIGGER = '/cache/claude-decision-row.json'
OUT_DIR = '/cache/claude-decision-row'
SOURCE_LOCK = '/opt/ds41-decision-row/claude-decision-row.lock.json'
SCHEMA = 'claude-decision-row-v3'
DEFAULT_CHUNK, CED_WINDOW, TARGET_LAYERS, POLL_SECONDS = 8192, 128, 40, 1.0
TOKEN = re.compile(r'^[a-z0-9-]{8,64}$')
_IMPORTED_AT = time.time()
_state = {'armed': None, 'checked': 0.0, 'done': set()}


def _log(message):
    print(f'CLAUDE-DECISION-ROW {message}', flush=True)


def _rank():
    from vllm.distributed import get_tensor_model_parallel_rank
    return get_tensor_model_parallel_rank()


def _source_trees():
    try:
        with open(SOURCE_LOCK) as stream:
            return json.load(stream)['trees']
    except (OSError, ValueError, KeyError):
        return None


def _capturing():
    return torch.cuda.is_current_stream_capturing()


def sha(tensor):
    t = tensor.detach().contiguous().view(torch.uint8)
    return hashlib.sha256(ctypes.string_at(t.data_ptr(), t.numel()) if t.numel() else b'').hexdigest()


def _read_trigger(rank):
    try:
        if os.stat(TRIGGER).st_mtime <= _IMPORTED_AT:
            return None
        with open(TRIGGER) as stream:
            spec = json.load(stream)
    except (OSError, ValueError):
        return None
    if not isinstance(spec, dict):
        return None
    token = spec.get('token')
    chunk_rows = spec.get('chunk_rows', DEFAULT_CHUNK)
    if (not isinstance(token, str) or not TOKEN.match(token) or token in _state['done']
            or spec.get('prompt_tokens') != 524288
            or type(chunk_rows) is not int or chunk_rows not in (8192, 4096)):
        return None
    if os.path.exists(os.path.join(OUT_DIR, f'{token}-rank{rank}.consumed')):
        _state['done'].add(token)
        return None
    return spec


def _target_layers():
    from vllm.forward_context import get_forward_context
    found = {}
    for module in get_forward_context().no_compile_layers.values():
        layer_id = getattr(module, 'layer_id', None)
        if (layer_id is not None and hasattr(module, 'swa_cache_layer') and hasattr(module, 'forward_mqa')
                and not getattr(module, 'is_draft', False) and layer_id < TARGET_LAYERS):
            found[layer_id] = module
    if sorted(found) != list(range(TARGET_LAYERS)):
        raise RuntimeError(f'expected {TARGET_LAYERS} target attention layers, found {sorted(found)}')
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
        try:                                            # durable failure receipt for the external driver
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(os.path.join(OUT_DIR, f'{token}-rank{session.rank}.aborted'), 'a') as stream:
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
    """Return the layer capture when armed and on the decision chunk, else None (never raises)."""
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
        _log(f'armed token={spec["token"]} rank={rank} layers={len(state["armed"].layers)}')
    session = state['armed']
    if session.fired or getattr(layer, 'is_draft', False) or layer.layer_id >= TARGET_LAYERS:
        return None
    original = metadata[layer.swa_cache_layer.prefix]
    if (original.is_decode or original.num_reqs != 1
            or original.max_seq_len != session.prompt_tokens):
        return None
    if original.num_actual_tokens != session.chunk_rows:
        raise RuntimeError(f'final chunk rows {original.num_actual_tokens} differ from armed {session.chunk_rows}')
    # The rows this layer actually computes: the CED decoder view for layers >= ced_decoder_start.
    query = layer._query_metadata(original)
    return session.layer(layer, query)


def _pinned(shape, dtype):
    return torch.empty(shape, dtype=dtype, pin_memory=True)


def _device_empty(shape, dtype, device):
    return torch.empty(shape, dtype=dtype, device=device)


class _Session:
    """One armed token on one rank: all staging preallocated, in-step copies, event waiter, save."""

    def __init__(self, spec, rank, layers):
        self.token, self.rank, self.prompt_tokens = spec['token'], rank, spec['prompt_tokens']
        self.chunk_rows = spec.get('chunk_rows', DEFAULT_CHUNK)
        self.fired, self.event, self.waiter = False, None, None
        self.started = time.time()
        self.layers = {lid: _LayerCapture(layer, self) for lid, layer in sorted(layers.items())}
        owners = {lid: layer for lid, layer in layers.items()
                  if layer.indexer is not None and getattr(layer.indexer, 'owns_k', False)}
        self.key_host, device, largest = {}, None, 0
        for lid, layer in owners.items():
            cache = layer.indexer.k_cache.kv_cache
            self.key_host[lid] = _pinned((layer._index_width, cache.shape[1]), torch.uint8)
            largest, device = max(largest, layer._index_width * cache.shape[1]), cache.device
        self.key_device = _device_empty((largest,), torch.uint8, device) if owners else None
        self.seen = set()

    def layer(self, layer, query):
        entry = self.layers[layer.layer_id]
        entry.expect(query)                                    # raises on any unexpected row geometry
        self.seen.add(layer.layer_id)
        return entry

    def stage_keys(self, layer, index_pages_row):
        """Owner layers only: the whole page-table row's pages, device gather then D2H (stream ordered)."""
        host = self.key_host[layer.layer_id]
        view = self.key_device[:host.numel()].view(host.shape)
        torch.index_select(layer.indexer.k_cache.kv_cache, 0, index_pages_row.clamp_min(0), out=view)
        host.copy_(view, non_blocking=True)

    def finish(self):
        """After layer 39's attention; completion no longer depends on any later forward."""
        if self.fired or _state['armed'] is not self:           # aborted or superseded
            return
        self.fired = True
        if self.seen != set(range(TARGET_LAYERS)):
            _log(f'incomplete decision forward token={self.token} rank={self.rank} layers={len(self.seen)}')
            _state['armed'] = None
            return
        self.event = torch.cuda.Event()
        self.event.record()
        self.waiter = threading.Thread(target=self.complete, name='claude-decision-row-waiter', daemon=True)
        self.waiter.start()

    def complete(self):
        """Waiter thread: wait for this event only, then build and save from host memory."""
        try:
            self.event.synchronize()
            self.save()
        except BaseException as error:                         # reported, never silent
            _log(f'error token={self.token} rank={self.rank} {type(error).__name__}: {error}')
        finally:
            _state['done'].add(self.token)
            _state['armed'] = None

    def save(self):
        problems, shared = [], {}
        layers = {lid: entry.host(self.rank, self.key_host, shared, problems)
                  for lid, entry in sorted(self.layers.items())}
        for lid, entry in sorted(layers.items()):
            if entry['position'] != self.prompt_tokens - 1:
                problems.append(f'layer {lid}: captured row position {entry["position"]} is not the decision row')
        meta = {'rank': self.rank, 'node': os.environ.get('DS41_NODE'), 'generation': self.token,
                'kit_sha256': os.environ.get('DS41_KIT_SHA256'), 'source_trees': _source_trees(),
                'prompt_tokens': self.prompt_tokens, 'chunk_rows': self.chunk_rows, 'row_index': self.chunk_rows - 1,
                'decoder_rows': CED_WINDOW, 'batch_requests': 1, 'row_position': layers[0]['position'],
                'problems': problems, 'pid': os.getpid(), 'armed_unix': self.started, 'saved_unix': time.time()}
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f'{meta["node"]}-rank{self.rank}-{self.token}.pt')
        torch.save({'schema': SCHEMA, 'meta': meta, 'layers': layers}, path + '.tmp')
        os.replace(path + '.tmp', path)
        with open(path, 'rb') as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        with open(os.path.join(OUT_DIR, f'{self.token}-rank{self.rank}.consumed'), 'x') as stream:
            stream.write(json.dumps({'file': path, 'sha256': digest, 'problems': problems}) + '\n')
        _log(f'saved token={self.token} rank={self.rank} file={path} sha256={digest} problems={len(problems)}')


class _LayerCapture:
    """Pinned staging for one target layer, allocated at arm time."""

    def __init__(self, layer, session):
        self.session = session
        self.layer_id, self.ratio = layer.layer_id, layer.compress_ratio
        self.decoder = bool(getattr(layer, 'is_ced_decoder', False))
        self.rows = CED_WINDOW if self.decoder else session.chunk_rows  # confirmed against query metadata per forward
        self.main_page, self.index_page = layer._main_page, layer._index_page
        self.is_index_source = layer.indexer is not None
        self.owns_keys = self.is_index_source and getattr(layer.indexer, 'owns_k', False)
        self.key_owner = getattr(layer, 'kv_source_layer_id', None)
        self.candidate_role = None
        if self.is_index_source:
            source = layer.candidate_source_layer
            self.candidate_role = ('produce' if layer.layer_id == source else
                                   'consume' if layer.layer_id > source else 'full')
        width = layer.swa_width
        s = {'position': _pinned((), torch.int64), 'q': _pinned((16, 512), torch.bfloat16),
             'out': _pinned((16, 512), torch.bfloat16), 'attn_sink': _pinned((16,), torch.float32),
             'swa_len': _pinned((), torch.int32), 'swa_slots': _pinned((width,), torch.int64),
             'swa_records': _pinned((width, 528), torch.uint8)}
        if self.ratio:
            s.update(indexed_len=_pinned((), torch.int32), indexed_logical=_pinned((512,), torch.int32),
                     indexed_page_table_row=_pinned((layer._main_width,), torch.int32),
                     indexed_slots=_pinned((512,), torch.int64), indexed_records=_pinned((512, 288), torch.uint8))
        if self.is_index_source:
            s.update(iq_data=_pinned((32, 64), torch.uint8), iq_scales=_pinned((32, 4), torch.uint8),
                     iw=_pinned((32,), torch.bfloat16), cache_length=_pinned((), torch.int32),
                     index_pages=_pinned((layer._index_width,), torch.int32), topk=_pinned((512,), torch.int32))
            if self.candidate_role == 'produce':
                s.update(candidates=_pinned((16384,), torch.int32), candidate_len=_pinned((), torch.int32))
        self.staged, self.plan = s, None

    def _d2h(self, name, value):
        self.staged[name].copy_(value, non_blocking=True)

    def expect(self, query):
        """The query metadata this layer computes on; the decision row is its last row."""
        rows = int(query.num_actual_tokens)
        if query.is_decode or int(query.num_reqs) != 1 or rows != self.rows:
            raise RuntimeError(f'layer {self.layer_id}: query rows {rows} (reqs {query.num_reqs}), '
                               f'expected {self.rows} for a {"CED decoder" if self.decoder else "full-row"} layer')
        self._d2h('position', query.positions[rows - 1])

    def _row(self, tensor, name):
        if tensor.shape[0] != self.rows:
            raise RuntimeError(f'layer {self.layer_id}: {name} has {tensor.shape[0]} rows, expected {self.rows}')
        return self.rows - 1

    @_guarded
    def indexer(self, layer, *, row_in_chunk, iq_data, iq_scale, iw, cache_lengths, index_pages):
        """After dsa_indexer.select for the index chunk holding the decision row (row indices of this layer)."""
        r = self._row(iq_data, 'index query')
        if not 0 <= row_in_chunk <= r or row_in_chunk >= index_pages.shape[0]:
            raise RuntimeError(f'layer {self.layer_id}: index chunk row {row_in_chunk} outside the page table')
        self._d2h('iq_data', iq_data[r])
        self._d2h('iq_scales', iq_scale[r])
        self._d2h('iw', iw[r])
        self._d2h('cache_length', cache_lengths[r])
        self._d2h('index_pages', index_pages[row_in_chunk])
        self._d2h('topk', layer.topk_indices_buffer[r])
        if self.candidate_role == 'produce':
            self._d2h('candidates', layer._candidates[r])
            self._d2h('candidate_len', layer._candidate_lens[r])
        if self.owns_keys:
            self.session.stage_keys(layer, index_pages[row_in_chunk])

    @_guarded
    def attention(self, layer, *, state, q, output, swa_indices, swa_lengths, top_lengths, page_table,
                  binding, owner):
        """After mla.run: inputs as the kernel consumed them, its output, and the records it read."""
        r = self._row(q, 'attention query')
        self._row(output, 'attention output')
        self.plan = {'mode': state.query.mode, 'config': dataclasses.asdict(state.config)}
        self._d2h('q', q[r])
        self._d2h('out', output[r])
        self._d2h('attn_sink', layer.attn_sink)
        self._d2h('swa_len', swa_lengths[r])
        slots = swa_indices[r].to(torch.int64)
        self._d2h('swa_slots', slots)
        self._d2h('swa_records', _gather(layer.swa_cache_layer.kv_cache, slots, layer.swa_cache_layer.block_size, 528))
        if self.ratio:
            mapped = binding.scratch.mapped_indices[r].to(torch.int64)
            self._d2h('indexed_len', top_lengths[r])
            self._d2h('indexed_logical', owner.topk_indices_buffer[r])
            self._d2h('indexed_page_table_row', page_table[r])
            self._d2h('indexed_slots', mapped)
            self._d2h('indexed_records', _gather(layer._owner().kv_cache, mapped, self.main_page, 288))
        if self.layer_id == TARGET_LAYERS - 1:
            self.session.finish()

    def host(self, rank, key_host, shared, problems):
        """Waiter thread, after the event: schema entry from pinned staging only (no device access)."""
        s = {k: v.clone() for k, v in self.staged.items()}
        entry = {'q': s['q'], 'out': s['out'], 'attn_sink': s['attn_sink'], 'swa_len': int(s['swa_len']),
                 'swa_slots': s['swa_slots'], 'swa_records': s['swa_records'], 'plan': self.plan,
                 'indexed_len': None, 'ced_decoder': self.decoder, 'query_rows': self.rows,
                 'row': self.rows - 1, 'position': int(s['position'])}
        if self.ratio:
            entry.update(indexed_len=int(s['indexed_len']), indexed_logical=s['indexed_logical'],
                         indexed_page_table_row=s['indexed_page_table_row'], indexed_slots=s['indexed_slots'],
                         indexed_records=s['indexed_records'])
        if not self.is_index_source:
            return entry
        count = int(s['cache_length'])
        used = (count + self.index_page - 1) // self.index_page
        row = s['index_pages'][:used]
        ix = {'q_data': s['iq_data'], 'q_scales': s['iq_scales'], 'weights': s['iw'], 'cache_length': count,
              'page_size': self.index_page, 'index_pages': s['index_pages'], 'topk': s['topk']}
        if self.candidate_role == 'produce':
            ix.update(candidates=s['candidates'], candidate_len=int(s['candidate_len']))
        owner = self.session.layers.get(self.key_owner)
        if owner is None or self.key_owner not in key_host:
            problems.append(f'layer {self.layer_id}: no staged key owner')
        elif bool((row <= 0).any()):
            problems.append(f'layer {self.layer_id}: index page table has holes inside cache_length')
        else:
            owner_count = int(owner.staged['cache_length'])
            owner_row = owner.staged['index_pages'][:used]
            if owner_count != count or not torch.equal(owner_row, row):
                # dedupe only on the complete used page table and length; otherwise the bytes are not staged
                problems.append(f'layer {self.layer_id}: page table differs from key owner {self.key_owner}')
            else:
                key = (self.key_owner, count, tuple(row.tolist()))
                if key not in shared:
                    pages = key_host[self.key_owner][:used].clone()
                    shared[key] = (pages, sha(pages))
                pages, digest = shared[key]
                ix['key_pages_sha256'] = digest
                if rank == 0:
                    ix['key_pages'] = pages
        entry['indexer'] = ix
        return entry


def _gather(cache, slots, page, record):
    """Records of a block-major cache [blocks, page*record] at physical slots (invalid slots read slot 0)."""
    safe = slots.clamp_min(0)
    columns = (safe % page)[:, None] * record + torch.arange(record, device=slots.device)
    return cache[(safe // page)[:, None], columns]
