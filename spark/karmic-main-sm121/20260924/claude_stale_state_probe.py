"""Diagnostic-only capture of request-to-request state for the DS4.1 385 residual.

Not installed anywhere. To use it, a diagnostic image copies this file next to
`vllm/models/deepseek_v4_1/nvidia/model_state.py` and applies
`claude-stale-state-model_state.patch` (two lines at the end of
`DeepseekV41ModelState.__init__`). Without the control file it does nothing.

Where it runs: on the host, inside `prepare_inputs`, after the model runner has
built this step's attention metadata (`prepare_attn`) and model inputs, and
before the forward is launched. Nothing is added to any CUDA graph; the FULL
graph for the 8-token verify step is captured and replayed unchanged. Observe
mode adds a device synchronisation and device-to-host copies before the
matched step; transplant and poison add device writes on the current stream
before it. The external reproducer must still show the divergence in observe
mode before any transplant result is trusted.

Fail-closed rules
- Every layer of every KV-cache group is classified. A layer that cannot be
  captured (not in the forward context, no or empty cache, unsupported spec,
  layout mismatch) aborts the matched step unless the control lists it in
  `allow_omitted` (a regex); omitted layers and reasons are always recorded.
- The number of KV-cache groups must equal the number of block tables.
- Required coverage (control `coverage`, DS4.1 defaults in DEFAULT_COVERAGE)
  must be met: sliding-window caches for target layers 0..39 and draft layers
  40..42, compressor rings for ratio-2 layers 2..19, main and indexer caches
  for the KV source layers 2, 8, 14, 20.
- Each write select item names one domain: `positions` [lo, hi) of the
  request, or `tail` [lo, hi) offsets into the unwritten tail slots after
  seq_len in the last touched block. Transplant requires the source layer's
  layout to equal the current one (spec, page, ratio, circular, record bytes,
  which positions are backed, tail length). Each physical slot is written
  once; a slot selected twice with different source bytes is an error. A
  write action that writes no slot is an error.

Control file: /cache/ds41-stale-control.json (re-read when its mtime changes)

  {"prompt_token_sha256": "<sha256 of json list of prompt token ids>",
   "scheduled_tokens": 8,
   "allow_omitted": "",                       # optional regex, default none
   "coverage": {...},                         # optional, see DEFAULT_COVERAGE
   "actions": [
     {"run": "orig-first",  "mode": "observe"},
     {"run": "orig-repeat", "mode": "observe"},
     {"run": "tx-1", "mode": "transplant", "source": "orig-repeat",
      "select": [{"layer_regex": "layers\\\\.2\\\\d\\\\.", "positions": [0, 257]},
                 {"layer_regex": "layers\\\\.20\\\\.", "tail": [0, 16]}]},
     {"run": "px-1", "mode": "poison", "byte": 255,
      "select": [{"layer_regex": "layers\\\\.20\\\\.", "positions": [0, 128]}]}]}

Snapshots: /cache/ds41-stale-state/<run>-<node>.pt with, per captured layer,
its group, spec class, page, ratio, record bytes, block-table row,
per-position records for [0, seq_len) (None where no block backs it) and the
tail records; plus the step's input ids and positions (the drafts), the
metadata tensors, CED state, lookback ids, compressor pending buffers, the
omitted-layer list and coverage counts. `python3 claude_stale_state_probe.py
diff A.pt B.pt` (inside an image with torch) prints where two snapshots
differ.

Byte layout: V4.1 caches store contiguous token records (b12x `page_nbytes`:
SWA 528 bytes, indexed 288 bytes; index keys 68 bytes; compressor ring 1024
fp32). A block is `page * record` payload bytes, possibly followed by
allocator padding (`bind_kv_cache` keeps the allocator stride). The record
size is `spec.page_size_bytes // spec.num_states`; the helper views exactly
the payload as (blocks, page, record) and refuses anything that would copy.
Position-to-slot mapping replicates `_tokens` in
vllm/models/deepseek_v4_1/sparse_mla.py at vLLM 1794dcf1.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

CONTROL = Path('/cache/ds41-stale-control.json')
OUTPUT = Path('/cache/ds41-stale-state')
MODES = ('observe', 'transplant', 'poison')
RUN_NAME = re.compile(r'[A-Za-z0-9_-]{1,64}')
LAYER = re.compile(r'layers\.(\d+)\.')
# DS4.1 at config fb2764a5: 40 target layers, 3 draft layers (layers.40..42),
# ratio-2 layers 2..19 carry compressor rings, KV sources 2, 8, 14, 20.
DEFAULT_COVERAGE = {
    'swa': {'suffix': '.swa_cache', 'layers': [[0, 40], [40, 43]]},
    'ring': {'suffix': '.compressor.state_cache', 'layers': [[2, 20]]},
    'main': {'suffix': '.attn', 'layers': [[2, 3], [8, 9], [14, 15], [20, 21]]},
    'indexer': {'suffix': '.indexer.k_cache', 'layers': [[2, 3], [8, 9], [14, 15], [20, 21]]},
}
KNOWN_SPECS = ('SlidingWindowMLASpec', 'MLAAttentionSpec', 'CircularBufferSpec')


# --------------------------------------------------------------------------
# Pure logic (stdlib only; unit tested)
# --------------------------------------------------------------------------

def token_sha256(token_ids) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in token_ids]).encode()).hexdigest()


def layer_index(name: str):
    match = LAYER.search(name)
    return int(match.group(1)) if match else None


def position_slot(block_row, pos: int, *, page: int, ratio: int, circular: bool):
    """Physical slot for `pos` in one cache layer, or None when no block backs it."""
    if pos < 0 or page <= 0 or ratio <= 0:
        return None
    logical = pos // ratio
    column = 0 if circular else logical // page
    if column >= len(block_row):
        return None
    block = int(block_row[column])
    if block <= 0:
        return None
    return block * page + logical % page


def request_slots(block_row, seq_len: int, *, page: int, ratio: int, circular: bool):
    """(slot per position in [0, seq_len), unwritten tail slots of the last touched block)."""
    per_position = [position_slot(block_row, p, page=page, ratio=ratio, circular=circular)
                    for p in range(seq_len)]
    tail = []
    last = next((s for s in reversed(per_position) if s is not None), None)
    if last is not None and not circular:
        block, offset = divmod(last, page)
        tail = [block * page + o for o in range(offset + 1, page)]
    return per_position, tail


def record_layout(*, spec_name: str, page: int, ratio: int, page_bytes: int, block_bytes: int) -> int:
    """Validate one layer's byte layout; returns the record size or raises ValueError."""
    if spec_name not in KNOWN_SPECS:
        raise ValueError(f'unsupported spec {spec_name}')
    if page <= 0 or ratio <= 0:
        raise ValueError(f'non-positive page {page} or ratio {ratio}')
    if page_bytes <= 0 or page_bytes % page:
        raise ValueError(f'page bytes {page_bytes} not a multiple of page {page}')
    if block_bytes < page_bytes:
        raise ValueError(f'block row has {block_bytes} bytes, payload needs {page_bytes}')
    return page_bytes // page


def classify(groups, num_block_tables: int, allow_omitted: str = ''):
    """Split layer descriptors into captured and omitted; fail closed.

    `groups`: list of {index, spec_name, layers: [{name, status}]} where status
    is 'ok' or an omission reason. Returns (captured, omitted) with entries
    (group, name) and (group, name, reason).
    """
    if len(groups) != num_block_tables:
        raise RuntimeError(f'{len(groups)} KV-cache groups but {num_block_tables} block tables')
    allow = re.compile(allow_omitted) if allow_omitted else None
    captured, omitted, fatal = [], [], []
    for group in groups:
        if not group['layers']:
            omitted.append((group['index'], None, 'group has no layers'))
            fatal.append(f"group {group['index']} ({group['spec_name']}) has no layers")
        for layer in group['layers']:
            if layer['status'] == 'ok':
                captured.append((group['index'], layer['name']))
                continue
            omitted.append((group['index'], layer['name'], layer['status']))
            if not (allow and allow.search(layer['name'])):
                fatal.append(f"{layer['name']} ({layer['status']})")
    if fatal:
        raise RuntimeError('uncaptured KV-cache layers: ' + '; '.join(fatal[:8])
                           + (f' and {len(fatal) - 8} more' if len(fatal) > 8 else ''))
    if not captured:
        raise RuntimeError('no KV-cache layer captured')
    return captured, omitted


def check_coverage(captured_names, coverage):
    """Every required (kind suffix, layer index) pair must be captured; returns counts per kind."""
    if not coverage:
        raise ValueError('coverage must not be empty')
    have = {kind: set() for kind in coverage}
    for name in captured_names:
        index = layer_index(name)
        if index is None:
            continue
        for kind, rule in coverage.items():
            if name.endswith(rule['suffix']):
                have[kind].add(index)
    missing, counts = [], {}
    for kind, rule in coverage.items():
        need = {i for lo, hi in rule['layers'] for i in range(int(lo), int(hi))}
        if not need:
            raise ValueError(f'coverage {kind} names no layers')
        counts[kind] = len(have[kind] & need)
        if need - have[kind]:
            missing.append(f'{kind}: layers {sorted(need - have[kind])[:10]}')
    if missing:
        raise RuntimeError('required coverage missing: ' + '; '.join(missing))
    return counts


def parse_control(raw: dict) -> dict:
    """Validate a control document; raises ValueError on anything unexpected."""
    if not isinstance(raw, dict):
        raise ValueError('control must be an object')
    allowed = {'prompt_token_sha256', 'scheduled_tokens', 'actions', 'allow_omitted', 'coverage'}
    if set(raw) - allowed or not {'prompt_token_sha256', 'actions'} <= set(raw):
        raise ValueError(f'control keys must be within {sorted(allowed)}')
    if not re.fullmatch(r'[0-9a-f]{64}', str(raw['prompt_token_sha256'])):
        raise ValueError('prompt_token_sha256 must be a sha256 hex digest')
    scheduled = int(raw.get('scheduled_tokens', 8))
    if scheduled <= 0:
        raise ValueError('scheduled_tokens must be positive')
    allow_omitted = str(raw.get('allow_omitted', ''))
    if allow_omitted:
        re.compile(allow_omitted)
    coverage = raw.get('coverage', DEFAULT_COVERAGE)
    if not isinstance(coverage, dict) or not coverage:
        raise ValueError('coverage must be a non-empty object')
    for kind, rule in coverage.items():
        if not isinstance(rule, dict) or set(rule) != {'suffix', 'layers'} or not rule['layers']:
            raise ValueError(f'coverage {kind} needs suffix and non-empty layers')
        for lo, hi in rule['layers']:
            if not 0 <= int(lo) < int(hi):
                raise ValueError(f'coverage {kind} has an empty layer range')
    actions = raw['actions']
    if not isinstance(actions, list) or not actions:
        raise ValueError('actions must be a non-empty list')
    runs = set()
    for action in actions:
        mode, run = action.get('mode'), action.get('run', '')
        if mode not in MODES or not RUN_NAME.fullmatch(run) or run in runs:
            raise ValueError(f'invalid or duplicate action {action!r}')
        runs.add(run)
        if mode == 'observe':
            if set(action) != {'run', 'mode'}:
                raise ValueError(f'{run}: observe takes no other keys')
            continue
        expected = {'run', 'mode', 'select'} | ({'source'} if mode == 'transplant' else {'byte'})
        if set(action) != expected:
            raise ValueError(f'{run}: {mode} takes exactly {sorted(expected)}')
        select = action['select']
        if not isinstance(select, list) or not select:
            raise ValueError(f'{run}: {mode} needs a non-empty select list')
        for item in select:
            domains = [d for d in ('positions', 'tail') if d in item]
            if len(domains) != 1 or set(item) != {'layer_regex', domains[0]}:
                raise ValueError(f'{run}: each select item needs layer_regex and exactly one of positions/tail')
            re.compile(item['layer_regex'])
            lo, hi = item[domains[0]]
            if not 0 <= int(lo) < int(hi):
                raise ValueError(f'{run}: bad {domains[0]} range {item[domains[0]]}')
        if mode == 'transplant' and (not RUN_NAME.fullmatch(str(action['source'])) or action['source'] in (run,)):
            raise ValueError(f'{run}: transplant needs a different, valid source run name')
        if mode == 'poison' and not 0 <= int(action['byte']) <= 255:
            raise ValueError(f'{run}: poison needs a byte value 0..255')
    return {'prompt_token_sha256': raw['prompt_token_sha256'], 'scheduled_tokens': scheduled,
            'allow_omitted': allow_omitted, 'coverage': coverage, 'actions': actions}


def matches(control: dict, *, num_reqs: int, scheduled: int, prompt_sha: str) -> bool:
    return (num_reqs == 1 and scheduled == control['scheduled_tokens']
            and prompt_sha == control['prompt_token_sha256'])


def plan_writes(select, layer_name: str, per_position, tail):
    """Unique physical writes for one layer: [(slot, [(domain, index), ...])] in first-seen order.

    Unbacked positions are skipped. Several sources for one slot (shared slots
    at ratio > 1, or overlapping select items) are grouped under that slot so
    the caller can require identical source bytes before writing it once.
    """
    order, sources = [], {}
    for item in select:
        if not re.search(item['layer_regex'], layer_name):
            continue
        domain = 'positions' if 'positions' in item else 'tail'
        slots = per_position if domain == 'positions' else tail
        lo, hi = map(int, item[domain])
        for index in range(lo, min(hi, len(slots))):
            slot = slots[index]
            if slot is None:
                continue
            if slot not in sources:
                sources[slot] = []
                order.append(slot)
            if (domain, index) not in sources[slot]:
                sources[slot].append((domain, index))
    return [(slot, sources[slot]) for slot in order]


def layout_key(layer: dict):
    """What must agree between a transplant source and target for one layer."""
    return (layer['spec'], layer['page'], layer['ratio'], layer['circular'], layer['record_bytes'],
            tuple(s is not None for s in layer['slots']), len(layer['tail_slots']))


def source_row(layer: dict, domain: str, index: int):
    """('values' | 'tail', row) of a snapshot layer holding (domain, index)."""
    if domain == 'tail':
        if not 0 <= index < len(layer['tail_slots']):
            raise IndexError(f'tail offset {index} outside the captured tail')
        return 'tail', index
    if not 0 <= index < len(layer['slots']) or layer['slots'][index] is None:
        raise IndexError(f'position {index} is not backed in the source')
    return 'values', sum(1 for s in layer['slots'][:index] if s is not None)


def ranges(indices):
    """Collapse sorted integers into [lo, hi) pairs."""
    out = []
    for i in indices:
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


def bisect_selects(select):
    """Split every range of a select list in half, for the next transplant round."""
    halves = ([], [])
    for item in select:
        domain = 'positions' if 'positions' in item else 'tail'
        lo, hi = map(int, item[domain])
        mid = (lo + hi) // 2
        if mid == lo:
            halves[0].append(dict(item))
            continue
        halves[0].append(dict(item, **{domain: [lo, mid]}))
        halves[1].append(dict(item, **{domain: [mid, hi]}))
    return [h for h in halves if h]


# --------------------------------------------------------------------------
# Torch side (runs only where torch exists; torch tests in the image)
# --------------------------------------------------------------------------

def payload_view(cache, page: int, record: int):
    """(blocks, page, record) uint8 VIEW of a bound cache's payload; never copies."""
    import torch
    flat = cache if cache.dtype == torch.uint8 else cache.view(torch.uint8)
    flat = flat.view(flat.shape[0], -1)
    if flat.shape[1] < page * record:
        raise ValueError(f'block row has {flat.shape[1]} bytes, payload needs {page * record}')
    view = flat[:, :page * record].view(flat.shape[0], page, record)
    if view.data_ptr() != flat.data_ptr():
        raise RuntimeError('payload view does not alias the cache')
    return view


def _slot_index(view, slots, page):
    import torch
    if page <= 0 or page != view.shape[1]:
        raise ValueError('invalid kernel page for slot indexing')
    pairs = [divmod(int(s), page) for s in slots]
    if any(not 0 < b < view.shape[0] or not 0 <= o < page for b, o in pairs):
        raise ValueError(f'slot outside cache with {view.shape[0]} blocks and page {page}')
    blocks = torch.tensor([b for b, _ in pairs], dtype=torch.int64, device=view.device)
    offsets = torch.tensor([o for _, o in pairs], dtype=torch.int64, device=view.device)
    return blocks, offsets


def gather_records(view, slots, page):
    present = [s for s in slots if s is not None]
    return view[_slot_index(view, present, page)].cpu() if present else None


def apply_write(action, layer_name, layer, seq_len, source_layers, *, plan_only=False):
    """Plan and perform one layer's transplant or poison; returns slots written."""
    import torch
    per_position, tail = request_slots(layer['block_row'], seq_len, page=layer['page'],
                                       ratio=layer['ratio'], circular=layer['circular'])
    plan = plan_writes(action['select'], layer_name, per_position, tail)
    if not plan:
        return (0, None) if plan_only else 0
    view = payload_view(layer['cache'], layer['page'], layer['record_bytes'])
    index = _slot_index(view, [slot for slot, _ in plan], layer['page'])
    if action['mode'] == 'poison':
        if plan_only:
            return len(plan), lambda: view.__setitem__(index, int(action['byte']))
        view[index] = int(action['byte'])
        return len(plan)
    src = source_layers.get(layer_name)
    if src is None:
        raise RuntimeError(f'transplant source lacks layer {layer_name}')
    current = dict(layer, slots=per_position, tail_slots=tail)
    if layout_key(src) != layout_key(current):
        raise RuntimeError(f'transplant layout differs for {layer_name}')
    rows = []
    for slot, origins in plan:
        picked = [src[kind][row] for kind, row in (source_row(src, d, i) for d, i in origins)]
        if any(not torch.equal(picked[0], other) for other in picked[1:]):
            raise RuntimeError(f'{layer_name}: slot {slot} selected with different source bytes')
        rows.append(picked[0])
    replacement = torch.stack(rows).to(view.device)
    if plan_only:
        return len(plan), lambda: view.__setitem__(index, replacement)
    view[index] = replacement
    return len(plan)


class _Probe:
    def __init__(self, model_state, vllm_config):
        self.state = model_state
        self.context = vllm_config.compilation_config.static_forward_context
        self.node = os.environ.get('DS41_NODE', 'unknown')
        self.control_mtime = None
        self.control = None
        self.next_action = 0
        self.step = None  # (block_tables, kv_cache_config, metadata) from prepare_attn

    def _load_control(self):
        try:
            mtime = CONTROL.stat().st_mtime_ns
            if mtime != self.control_mtime:
                self.control_mtime = mtime
                self.control = None
                self.control = parse_control(json.loads(CONTROL.read_text()))
                self.next_action = 0
                print(f'[DS41-STALE-PROBE] control loaded, {len(self.control["actions"])} actions', flush=True)
        except FileNotFoundError:
            self.control = None
            return
        except (OSError, ValueError, TypeError, KeyError, re.error) as error:
            self.control = None
            print(f'[DS41-STALE-PROBE] control-rejected reason={error!r}', flush=True)

    def layers(self, kv_cache_config, block_tables, metadata):
        """Fail-closed classification of every KV-cache layer; returns (layers, omitted, counts)."""
        from vllm.v1.kv_cache_interface import CircularBufferSpec, UniformTypeKVCacheSpecs
        groups, entries = [], {}
        for index, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            descriptor = {'index': index, 'spec_name': type(group_spec).__name__, 'layers': []}
            groups.append(descriptor)
            for name in group.layer_names:
                geometry = {}
                # Group specs sum per-layer storage and deliberately do not
                # implement individual state geometry. Resolve by layer name,
                # exactly as vLLM does when constructing attention metadata.
                spec = (group_spec.kv_cache_specs.get(name)
                        if isinstance(group_spec, UniformTypeKVCacheSpecs) else group_spec)
                spec_name = type(spec).__name__
                try:
                    ratio = int(spec.tokens_per_state)
                    record = int(spec.state_content_size_bytes)
                    heads = int(spec.num_heads)
                    spec_error = None
                except (NotImplementedError, TypeError, ValueError, AttributeError) as error:
                    ratio = record = heads = 0
                    spec_error = f'unsupported spec {spec_name}: {error!r}'
                module = self.context.get(name)
                cache = getattr(module, 'kv_cache', None) if module is not None else None
                if isinstance(cache, (list, tuple)):
                    cache = cache[0] if len(cache) == 1 else None
                if module is None:
                    status = 'not_in_forward_context'
                elif cache is None or not hasattr(cache, 'numel'):
                    status = 'no_cache'
                elif cache.numel() == 0:
                    status = 'empty_cache'
                elif spec_error:
                    status = spec_error
                elif index >= len(block_tables):
                    status = 'no_block_table'
                elif name not in metadata:
                    status = 'no_layer_metadata'
                else:
                    try:
                        layer_metadata = metadata[name]
                        page = int(layer_metadata.block_size)
                        block_bytes = cache.view(cache.shape[0], -1).shape[1] * cache.element_size()
                        geometry = {'page': page, 'record_bytes': record, 'row_bytes': block_bytes,
                                    'heads': heads, 'ratio': ratio,
                                    'circular': isinstance(spec, CircularBufferSpec)}
                        if heads != 1 or block_bytes != page * record:
                            raise ValueError(f'heads={heads}, row={block_bytes}, kernel page={page}, record={record}')
                        record = record_layout(spec_name=spec_name, page=page, ratio=ratio,
                                               page_bytes=page * record, block_bytes=block_bytes)
                        payload_view(cache, page, record)
                        entries[name] = {'group': index, 'spec': spec_name, 'page': page, 'ratio': ratio,
                                         'circular': isinstance(spec, CircularBufferSpec),
                                         'record_bytes': record, 'cache': cache,
                                         'block_row': [int(b) for b in layer_metadata.block_table[0].tolist()],
                                         'group_block_row': [int(b) for b in block_tables[index][0].tolist()]}
                        status = 'ok'
                    except (ValueError, RuntimeError, AttributeError, TypeError, IndexError) as error:
                        import torch
                        if isinstance(error, torch.AcceleratorError) or 'CUDA error' in str(error):
                            raise
                        status = f'layout: {error}'
                descriptor['layers'].append({'name': name, 'status': status, 'geometry': geometry})
        self.last_inventory = {'groups': groups, 'table_count': len(block_tables)}
        captured, omitted = classify(groups, len(block_tables), self.control['allow_omitted'])
        counts = check_coverage([name for _, name in captured], self.control['coverage'])
        return {name: entries[name] for _, name in captured}, omitted, counts

    def _extra_state(self):
        extra = {}
        ced = getattr(self.state, 'ced_state', None)
        if ced is not None:
            for key in ('indices', 'starts', 'request_positions', 'replay_start', 'counts', 'keep_full', 'prefix_start'):
                extra[f'ced.{key}'] = getattr(ced, key).detach().cpu().clone()
        lookback = getattr(self.state, 'lookback_token_ids', None)
        if lookback is not None:
            extra['lookback_token_ids'] = lookback.detach().cpu().clone()
        for name, module in self.state.model.named_modules():
            for key in ('_pending_values', '_pending_gates', '_pending_tags', '_local_ids'):
                value = getattr(module, key, None)
                if value is not None and hasattr(value, 'detach'):
                    extra[f'{name}.{key}'] = value.detach().cpu().clone()
        return extra

    @staticmethod
    def _metadata(metadata):
        out = {}
        if not isinstance(metadata, dict):
            return out
        seen = set()
        for key, value in metadata.items():
            if id(value) in seen:
                continue
            seen.add(id(value))
            for field in ('block_table', 'query_start_loc', 'request_positions', 'live_counts', 'positions',
                          'req_id_per_token', 'slot_mapping', 'cache_lengths', 'swa_replay_start'):
                tensor = getattr(value, field, None)
                if tensor is not None and hasattr(tensor, 'detach'):
                    out[f'{key}.{field}'] = tensor.detach().cpu().clone()
            for field in ('num_actual_tokens', 'num_reqs', 'max_query_len', 'max_seq_len', 'is_decode'):
                if hasattr(value, field):
                    out[f'{key}.{field}'] = getattr(value, field)
        return out

    def after_prepare_inputs(self, input_batch, req_states):
        import torch
        if torch.cuda.is_current_stream_capturing() or self.step is None:
            return
        self._load_control()
        control = self.control
        if control is None or self.next_action >= len(control['actions']):
            return
        num_reqs = int(input_batch.num_reqs)
        if num_reqs != 1:
            return
        scheduled = int(input_batch.num_scheduled_tokens[0])
        if scheduled != control['scheduled_tokens']:
            return
        index = int(input_batch.idx_mapping_np[0])
        prompt_len = int(req_states.prompt_len.np[index])
        prompt = req_states.all_token_ids.gpu[index, :prompt_len].tolist()
        if not matches(control, num_reqs=num_reqs, scheduled=scheduled, prompt_sha=token_sha256(prompt)):
            return
        seq_len = int(input_batch.seq_lens[0])
        action = control['actions'][self.next_action]
        self.next_action += 1
        self.last_inventory = {}
        block_tables, kv_cache_config, metadata = self.step
        torch.cuda.current_stream().synchronize()
        if action['mode'] == 'observe':
            try:
                layers, omitted, counts = self.layers(kv_cache_config, block_tables, metadata)
                step_inputs = {'input_ids': input_batch.input_ids[:scheduled].detach().cpu().clone(),
                               'positions': input_batch.positions[:scheduled].detach().cpu().clone()}
                self._observe(action, layers, omitted, counts, metadata, seq_len, prompt_len, step_inputs)
            except Exception as error:
                if isinstance(error, torch.AcceleratorError) or 'CUDA error' in str(error):
                    raise
                # Observation is read-only. A failed diagnostic must not turn
                # a Python layout error into a serving outage or a PASS receipt.
                print(f"[DS41-STALE-PROBE] observe-failed run={action['run']} reason={error!r}", flush=True)
                try:
                    OUTPUT.mkdir(exist_ok=True)
                    (OUTPUT / f"{action['run']}-{self.node}.inventory.json").write_text(
                        json.dumps({'error': repr(error), 'inventory': getattr(self, 'last_inventory', {})}) + '\n')
                except OSError as receipt_error:
                    print(f'[DS41-STALE-PROBE] observe-failed receipt={receipt_error!r}', flush=True)
        else:
            layers, omitted, counts = self.layers(kv_cache_config, block_tables, metadata)
            self._write(action, layers, seq_len)

    def _observe(self, action, layers, omitted, counts, metadata, seq_len, prompt_len, step_inputs):
        import torch
        captured = {}
        for name, layer in layers.items():
            per_position, tail = request_slots(layer['block_row'], seq_len, page=layer['page'],
                                               ratio=layer['ratio'], circular=layer['circular'])
            view = payload_view(layer['cache'], layer['page'], layer['record_bytes'])
            record = {key: layer[key] for key in ('group', 'spec', 'page', 'ratio', 'circular',
                                                  'record_bytes', 'block_row', 'group_block_row')}
            record.update({'slots': per_position, 'tail_slots': tail,
                           'values': gather_records(view, per_position, layer['page']),
                           'tail': gather_records(view, tail, layer['page'])})
            captured[name] = record
        payload = {'run': action['run'], 'node': self.node, 'seq_len': seq_len, 'prompt_len': prompt_len,
                   'layers': captured, 'omitted': omitted, 'coverage_counts': counts,
                   'metadata': self._metadata(metadata), 'extra': self._extra_state(),
                   'step_inputs': step_inputs}
        OUTPUT.mkdir(exist_ok=True)
        path = OUTPUT / f"{action['run']}-{self.node}.pt"
        if path.exists():
            raise RuntimeError(f'stale-state snapshot exists: {path}')
        temporary = path.with_suffix('.pt.partial')
        torch.save(payload, temporary)
        temporary.rename(path)
        print(f'[DS41-STALE-PROBE] observe {path} layers={len(captured)} omitted={len(omitted)} '
              f'coverage={counts}', flush=True)

    def _write(self, action, layers, seq_len):
        import torch
        source_layers = {}
        if action['mode'] == 'transplant':
            source = torch.load(OUTPUT / f"{action['source']}-{self.node}.pt", weights_only=False)
            if source['seq_len'] != seq_len:
                raise RuntimeError('transplant source has a different sequence length')
            source_layers = source['layers']
        planned = [apply_write(action, name, layer, seq_len, source_layers, plan_only=True)
                   for name, layer in layers.items()]
        written = sum(count for count, _ in planned)
        if written == 0:
            raise RuntimeError(f"{action['run']}: {action['mode']} selected no backed slot")
        for _, write in planned:
            if write is not None:
                write()
        torch.cuda.current_stream().synchronize()
        print(f"[DS41-STALE-PROBE] {action['mode']} {action['run']} slots={written}", flush=True)


def install(model_state, vllm_config) -> None:
    """Wrap prepare_attn/prepare_inputs on one DeepseekV41ModelState instance."""
    probe = _Probe(model_state, vllm_config)
    original_attn, original_inputs = model_state.prepare_attn, model_state.prepare_inputs

    def prepare_attn(input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                     kv_cache_config, for_capture=False):
        metadata = original_attn(input_batch, cudagraph_mode, block_tables, slot_mappings,
                                 attn_groups, kv_cache_config, for_capture)
        probe.step = None if for_capture else (block_tables, kv_cache_config, metadata)
        return metadata

    def prepare_inputs(input_batch, req_states):
        inputs = original_inputs(input_batch, req_states)
        try:
            probe.after_prepare_inputs(input_batch, req_states)
        finally:
            probe.step = None
        return inputs

    model_state.prepare_attn = prepare_attn
    model_state.prepare_inputs = prepare_inputs
    model_state._claude_stale_probe = probe


# --------------------------------------------------------------------------
# Offline diff (needs torch; run inside the image)
# --------------------------------------------------------------------------

def diff(path_a, path_b) -> dict:
    import torch
    a, b = (torch.load(p, weights_only=False) for p in (path_a, path_b))
    report = {'seq_len': [a['seq_len'], b['seq_len']], 'omitted': [a['omitted'], b['omitted']],
              'layers': {}, 'metadata': [], 'extra': [], 'step_inputs': []}
    for name in sorted(set(a['layers']) | set(b['layers'])):
        la, lb = a['layers'].get(name), b['layers'].get(name)
        if la is None or lb is None:
            report['layers'][name] = {'missing_in': 'a' if la is None else 'b'}
            continue
        entry = {}
        if layout_key(la) != layout_key(lb):
            entry['layout_differs'] = True
        if la['block_row'] != lb['block_row']:
            entry['block_row_differs'] = True
        pa = [p for p, s in enumerate(la['slots']) if s is not None]
        pb = [p for p, s in enumerate(lb['slots']) if s is not None]
        if pa != pb:
            entry['backed_positions'] = [ranges(pa), ranges(pb)]
        ia = {p: k for k, p in enumerate(pa)}
        ib = {p: k for k, p in enumerate(pb)}
        differing = [p for p in sorted(set(pa) & set(pb)) if not torch.equal(la['values'][ia[p]], lb['values'][ib[p]])]
        if differing:
            entry['differing_positions'] = ranges(differing)
        if la['tail'] is not None and lb['tail'] is not None and la['tail'].shape == lb['tail'].shape:
            rows = [i for i in range(la['tail'].shape[0]) if not torch.equal(la['tail'][i], lb['tail'][i])]
            if rows:
                entry['differing_tail_offsets'] = ranges(rows)
        if entry:
            report['layers'][name] = entry
    for kind in ('metadata', 'extra', 'step_inputs'):
        for key in sorted(set(a[kind]) | set(b[kind])):
            va, vb = a[kind].get(key), b[kind].get(key)
            if hasattr(va, 'shape') and hasattr(vb, 'shape'):
                same = va.shape == vb.shape and va.dtype == vb.dtype and torch.equal(va, vb)
            else:
                same = va == vb
            if not same:
                report[kind].append(key)
    return report


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'diff':
        print(json.dumps(diff(sys.argv[2], sys.argv[3]), indent=1))
    else:
        sys.exit('usage: claude_stale_state_probe.py diff A.pt B.pt')
