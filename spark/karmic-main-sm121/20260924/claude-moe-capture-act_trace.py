"""DIAGNOSTIC ONLY: per-rank, per-module activation digests of eager prefill steps.

Installed as vllm/models/deepseek_v4_1/claude_act_trace.py in a diagnostic
derivative of the Engram repair candidate; DeepseekV4Model.__init__ calls
install(self) once. Nothing runs until the trigger file exists.

Purpose: localize a rare (about 1-3 percent), timing-correlated prefill
divergence without full tensor snapshots. For each traced step every rank
records, in call order, a 2-lane integer digest of each hooked tensor:
  lane 0 = sum of the tensor's raw 32-bit words (16/8-bit for odd sizes),
  lane 1 = sum of word * ((index mod 65521) + 1),
both in int64 with wraparound. Integer addition is associative, so the digest
of identical bytes is identical regardless of reduction order, and any
single-word change alters lane 0. claude_act_analyze.py then derives, per
(rank, step key), the modal digest vector and the first diverging module of
every non-modal step, aligned across ranks by step sequence.

Arming (write the SAME content on every node, after readiness; re-read when
the file's mtime changes):
  /cache/ds41-act-trace.json
  {"arm": "<8-64 of [a-z0-9-]>", "min_tokens": 100, "max_tokens": 1100,
   "post": [...optional...], "pre": [...optional...]}
Every record carries the arm token, a per-process session id, and the
target model's Engram epoch at the start of the step. The epoch advances
identically on every TP rank for every real or dummy step, so it is the
cross-rank step identity; the analyzer joins ranks on (arm, epoch) and
refuses mixed arms or sessions. A new arm token restarts `seq`.

MoE seams (v2, "moe" key): a pattern list selecting DeepseekV4MoE runners
(`layers.N.ffn.experts`, class MoERunner at vLLM 1794dcf1). For each selected
runner, while armed, two instance attributes shadow the only two callsites in
MoERunner.forward that bracket the rank-local partials and the collective:
  self._forward_entry(...)            -> records NAME#shared and NAME#routed,
                                         the custom op's (shared, fused) outputs;
  self._maybe_reduce_final_output(s)  -> records NAME#pre_reduce (s, the combined
                                         shared + routed input of the TP
                                         all-reduce) and NAME#post_reduce (its
                                         return value).
The wrappers call the original objects with the original arguments and return
their results unchanged; they only enqueue digests on the current (main)
stream. #routed is the raw kernel output: the runner's routed_scaling_factor
is 1.0 for DS4.1 (the factor is applied to the top-k weights), so nothing
scales it in place before the combine. Disarming restores the original
attributes (identity), so an unarmed runner is byte-for-byte the pinned code
path. Arming is refused if any post, pre or moe pattern matches nothing, or a
moe pattern selects a module without both callsites (the v1 kit silently
ignored a pattern naming a method, not a module).

MoE capture (v3, "moe" seams plus optional "capture" key). For every armed
runner the seams also record the routed call's inputs, without adding any
kernel inside the shared/routed overlap window: a forward hook on the
runner's gate and an instance wrapper of `runner.router.select_experts` only
keep Python references while the custom op runs; their digests are enqueued
after it returns. Per runner and step, in call order:
  NAME#input        hidden_states handed to the op (digest before the op)
  NAME#logits       gate output (router logits, fp32)
  NAME#topk_weights / NAME#topk_ids   the router's selection
  NAME#input_after  hidden_states again after the op (in-place write check)
  NAME#shared, NAME#routed, NAME#pre_reduce, NAME#post_reduce   as in v2.
"capture": {"rows": [254, 256], "moe": ["layers.1[0-3].ffn.experts"]}
arms a device-side latch per process: the first #routed digest for each
(rows, runner) becomes that pair's reference; the first later call whose
#routed digest differs from its reference copies the call's input, logits,
top-k weights and ids and routed output into preallocated device buffers
(torch.where with a device predicate; no host synchronization) and latches.
The reference also stores the step's `<input>` token-id digest, and a call
latches only when its step has that same digest, so a different prompt with
the same row count never latches. Each record carries a raw `<capture>` entry [latched, epoch]; once the writer
thread sees a latch, the next step start freezes the buffers and copies them
on the transport stream to pinned memory, and the writer saves
<out_dir>/<node>-<pid>-g<generation>-capture.pt. One latch per capture
generation (a new arm token or trigger rewrite starts a new generation).
Step records carry {generation, state}; `type: capture` status lines record
fetching, saved, error and discarded states, and a latch word that reaches the
writer after its generation was disarmed is written as latched-discarded. If the first
occurrence of a pair was itself the faulty one, the capture holds the first
modal call instead; the saved reference and trace digests tell which.

Default patterns are SPARSE checkpoints, to keep the added launches small
enough not to mask the race: the outputs of decoder layers 0, 1, 2, 8, 14,
19, 20 and 39 (all five carried tensors) and of the two Engram modules, plus
the model output. Refine around a first divergence with explicit patterns.
"post" patterns digest module outputs (a ":k" suffix selects tuple member k
only); "pre" patterns digest module inputs.

Call path at 1794dcf1 (source-verified): the DS4.1 target model is not
torch.compile'd (vLLM logs that the model does not support it), so prefill
steps above the 32-token FULL-graph sizes run eagerly and piecewise
compilation does not apply. DeepseekV4Model is called through Module.__call__
by the causal-LM wrapper; layers through layer(...); attention, FFN and
Engram through self.attn(...), self.ffn(...) and self.engram(...), so their
hooks run. attn.wo_a/wo_b are NOT invoked as modules: their weights are
packed into a fused WO projection, so hooks on them never fire, and no
module hook sees a rank-local tensor before a tensor-parallel reduction. Patterns match
names relative to the DeepseekV4Model segment by segment: each dotted
segment is an fnmatch pattern for exactly one name segment, so "layers.*"
selects the decoder layers and not their nested submodules. Remove the file to disarm:
module hooks are then unregistered, leaving only the model's two step hooks.

What is traced: only eager target-model forwards whose row count is within
[min_tokens, max_tokens] (at most 2048, bounding the int64 temporaries).
Steps under CUDA graph capture or torch.compile are skipped, and graph
replays never run Python, so captured decode graphs are unchanged. The draft
model is not traced.

No host synchronization on the forward path. Digests are written into a
preallocated device slot by stream-ordered kernels on the current stream;
at the end of the step a dedicated copy stream waits on an event and copies
the slot to pinned memory; a writer thread waits on that copy and appends one
JSON line per step to /cache/ds41-act-trace/<node>-<pid>.jsonl. If every slot
is still in flight the step is dropped (counted), never waited for.

Perturbation (document it with every result): each digested tensor adds about
six small kernels per 1M-word chunk (optional contiguous copy, int64
widening, multiply, two reductions, accumulate, slot write) on the current
stream, with int64 temporaries bounded to about 16 MiB per chunk plus one
8 MiB weight buffer per device, and Python hook overhead. With the sparse
defaults (45 digests: 8 layers x 5 carried tensors, 2 Engram outputs, the
model output and the 2 key digests) that is roughly 250-300 extra launches
for a 255-row prefill and roughly 500-700 at 1100 rows, dominated by the
chunked residual digests (rows x 8192 words each); broad patterns such as
every layer, attention and FFN reach 1500-2000. This changes
kernel timing and interleaving and can therefore change the rate of a
timing-dependent race; compare the divergence rate with the trigger present
and absent on the same boot before interpreting a low capture count. It does
not change any operand, output or stream dependency of the model.
"""
from __future__ import annotations

import fnmatch
import json
import os
import queue
import re
import threading
import time
import uuid

import torch

TRIGGER = '/cache/ds41-act-trace.json'
OUT_DIR = '/cache/ds41-act-trace'
MAX_TOKENS_CAP = 2048
CAPACITY = 2048          # digests per step
SLOTS = 8                # steps in flight between the forward and the writer
KEYS = {'arm', 'min_tokens', 'max_tokens', 'post', 'pre', 'moe', 'capture'}
ARM = re.compile(r'[a-z0-9][a-z0-9-]{7,63}\Z')
SPARSE_POST = ['layers.0', 'layers.1', 'layers.2', 'layers.8', 'layers.14', 'layers.19',
               'layers.20', 'layers.39', 'layers.1.engram', 'layers.14.engram']
PERIOD = 65521                 # weight period: lane 1 uses (index mod PERIOD) + 1
CHUNK = PERIOD * 16            # words per reduction chunk; a multiple of PERIOD, so
                               # every chunk uses the same weights (8 MiB int64)
_WEIGHTS: dict[str, torch.Tensor] = {}


def parse_config(text: str) -> dict:
    spec = json.loads(text)
    if not isinstance(spec, dict) or not set(spec) <= KEYS or not {'arm', 'min_tokens', 'max_tokens'} <= set(spec):
        raise ValueError('trigger needs arm, min_tokens and max_tokens; optional post, pre, moe, capture')
    if not isinstance(spec['arm'], str) or not ARM.match(spec['arm']):
        raise ValueError('arm must be 8-64 characters of [a-z0-9-]')
    low, high = spec['min_tokens'], spec['max_tokens']
    if type(low) is not int or type(high) is not int or not 1 <= low <= high <= MAX_TOKENS_CAP:
        raise ValueError(f'require 1 <= min_tokens <= max_tokens <= {MAX_TOKENS_CAP}')
    for key in ('post', 'pre', 'moe'):
        value = spec.setdefault(key, list(SPARSE_POST) if key == 'post' else [])
        if not isinstance(value, list) or not all(isinstance(p, str) and p for p in value):
            raise ValueError(key + ' must be a list of name patterns')
    capture = spec.setdefault('capture', None)
    if capture is not None:
        if not isinstance(capture, dict) or set(capture) != {'rows', 'moe'}:
            raise ValueError('capture needs exactly rows and moe')
        rows, moe = capture['rows'], capture['moe']
        if (not isinstance(rows, list) or not rows or len(set(rows)) != len(rows)
                or not all(type(r) is int and low <= r <= high for r in rows)):
            raise ValueError('capture rows must be distinct ints inside the row window')
        if not isinstance(moe, list) or not moe or not all(isinstance(p, str) and p for p in moe):
            raise ValueError('capture moe must be a list of runner patterns')
    return spec


def _weights(device) -> torch.Tensor:
    key = str(device)
    weights = _WEIGHTS.get(key)
    if weights is None:
        weights = torch.arange(CHUNK, dtype=torch.int64, device=device).remainder_(PERIOD).add_(1)
        _WEIGHTS[key] = weights
    return weights


def digest(tensor: torch.Tensor) -> torch.Tensor:
    """int64[2] order-independent digest of the tensor's bytes; device-side only."""
    data = tensor.detach()
    if not data.is_contiguous():
        data = data.contiguous()
    raw = data.reshape(-1).view(torch.uint8) if data.numel() else data.new_zeros(0, dtype=torch.uint8)
    if raw.numel() % 4 == 0:
        words = raw.view(torch.int32)
    elif raw.numel() % 2 == 0:
        words = raw.view(torch.int16)
    else:
        words = raw
    total = torch.zeros(2, dtype=torch.int64, device=words.device)
    weights = _weights(words.device)
    # Chunking bounds int64 temporaries to 16 bytes per word of one chunk;
    # the chunk count is known on the host, so no synchronization is needed.
    for start in range(0, words.numel(), CHUNK):
        wide = words[start:start + CHUNK].to(torch.int64)
        total += torch.stack((wide.sum(), (wide * weights[:wide.numel()]).sum()))
    return total


def split_member(pattern: str):
    """'layers.14:1' -> ('layers.14', 1); 'layers.14' -> ('layers.14', None)."""
    base, sep, member = pattern.rpartition(':')
    if sep and member.isdigit():
        return base, int(member)
    return pattern, None


def matches(name: str, pattern: str) -> bool:
    """Segment-wise match: each pattern segment (fnmatch syntax) matches exactly
    one dotted name segment, so 'layers.*' is the decoder layers themselves and
    never their nested submodules."""
    parts, wanted = name.split('.'), pattern.split('.')
    return len(parts) == len(wanted) and all(fnmatch.fnmatchcase(a, b) for a, b in zip(parts, wanted))


def _tensors(value, prefix=''):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensors(item, f'{prefix}:{index}')


class SyncTransport:
    """Host-memory transport for CPU tests: records are complete at submit."""

    def __init__(self, sink, slots=SLOTS, capacity=CAPACITY):
        self.sink = sink
        self.acc = torch.zeros(slots, capacity, 2, dtype=torch.int64)
        self.free = [True] * slots

    def begin(self, slot):
        pass

    def submit(self, slot, count, meta):
        meta['hashes'] = self.acc[slot, :count].tolist()
        meta['gpu_ms'] = None
        self.sink(meta)
        self.free[slot] = True

    def fetch(self, pairs, callback):
        for source, host in pairs:
            host.copy_(source)
        callback()


class CudaTransport:
    """Pinned-memory copy on a side stream; a writer thread finishes records."""

    def __init__(self, sink, device, slots=SLOTS, capacity=CAPACITY):
        self.sink = sink
        self.acc = torch.zeros(slots, capacity, 2, dtype=torch.int64, device=device)
        self.host = torch.zeros(slots, capacity, 2, dtype=torch.int64, pin_memory=True)
        self.free = [True] * slots
        self.stream = torch.cuda.Stream(device=device)
        self.start = [torch.cuda.Event(enable_timing=True) for _ in range(slots)]
        self.pending: queue.Queue = queue.Queue()
        threading.Thread(target=self._writer, name='ds41-act-trace', daemon=True).start()

    def begin(self, slot):
        self.start[slot].record()

    def submit(self, slot, count, meta):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.stream.wait_event(end)
        with torch.cuda.stream(self.stream):
            self.host[slot, :count].copy_(self.acc[slot, :count], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        self.pending.put((slot, count, meta, end, done))

    def fetch(self, pairs, callback):
        """Copy device tensors to preallocated pinned tensors after all work
        already enqueued on the current stream; the writer runs callback."""
        ready = torch.cuda.Event()
        ready.record()
        self.stream.wait_event(ready)
        with torch.cuda.stream(self.stream):
            for source, host in pairs:
                host.copy_(source, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        self.pending.put((None, callback, None, None, done))

    def _writer(self):
        while True:
            slot, count, meta, end, done = self.pending.get()
            if slot is None:
                try:
                    done.synchronize()
                    count()
                except Exception as error:
                    print(f'[DS41-ACT-TRACE] capture fetch error: {error!r}', flush=True)
                continue
            try:
                done.synchronize()
                meta['gpu_ms'] = self.start[slot].elapsed_time(end)
                meta['hashes'] = self.host[slot, :count].tolist()
                self.sink(meta)
            except Exception as error:  # never let the writer die silently
                print(f'[DS41-ACT-TRACE] writer error: {error!r}', flush=True)
            finally:
                self.free[slot] = True


SEAMS = ('#input', '#logits', '#topk_weights', '#topk_ids', '#input_after',
         '#shared', '#routed', '#pre_reduce', '#post_reduce')
CAPTURE_FIELDS = ('input', 'logits', 'topk_weights', 'topk_ids', 'routed')


class MoeSeams:
    """Armed-only wrappers of the two MoERunner.forward callsites of one runner.

    Refuses a module that is not an unwrapped runner: `_forward_entry` must be
    an instance attribute (MoERunner.__init__ stores the selected custom op
    there) and `_maybe_reduce_final_output` a class-level method not already
    shadowed on the instance.
    """

    def __init__(self, tracer, name, runner):
        entry = vars(runner).get('_forward_entry')
        reduce = getattr(type(runner), '_maybe_reduce_final_output', None)
        gate = getattr(runner, 'gate', None)
        router = getattr(runner, 'router', None)
        if (not callable(entry) or not callable(reduce) or getattr(entry, '_claude_seam', False)
                or '_maybe_reduce_final_output' in vars(runner)
                or not isinstance(gate, torch.nn.Module)
                or not callable(getattr(router, 'select_experts', None))
                or 'select_experts' in vars(router)):
            raise ValueError(f'{name} does not expose the MoE runner callsites')
        self.runner, self.entry, self.router = runner, entry, router
        bound = runner._maybe_reduce_final_output
        select = router.select_experts
        stash = {}

        # Inside the overlap window only references are kept: no kernel.
        def gate_hook(module, args, output):
            if tracer.active:
                stash['logits'] = output[0] if isinstance(output, tuple) else output

        def select_experts(*args, **kwargs):
            result = select(*args, **kwargs)
            if tracer.active:
                stash['route'] = result
            return result

        def forward_entry(*args, **kwargs):
            active = tracer.active
            hidden = args[0] if args else kwargs.get('hidden_states')
            if active:
                stash.clear()
                if isinstance(hidden, torch.Tensor):
                    tracer._record(name + '#input', hidden)
            result = entry(*args, **kwargs)
            if active and tracer.active:
                logits = stash.pop('logits', None)
                weights, ids = stash.pop('route', (None, None))
                stash.clear()
                for suffix, tensor in (('#logits', logits), ('#topk_weights', weights),
                                       ('#topk_ids', ids), ('#input_after', hidden)):
                    if isinstance(tensor, torch.Tensor):
                        tracer._record(name + suffix, tensor)
                shared, routed = result if isinstance(result, tuple) else (None, result)
                if isinstance(shared, torch.Tensor):
                    tracer._record(name + '#shared', shared)
                if isinstance(routed, torch.Tensor):
                    routed_digest = tracer._record(name + '#routed', routed)
                    if tracer.capture is not None:
                        tracer.capture.observe(name, {'input': hidden, 'logits': logits,
                                                      'topk_weights': weights, 'topk_ids': ids,
                                                      'routed': routed}, routed_digest)
            return result

        def reduce_final_output(states, *args, **kwargs):
            if tracer.active and isinstance(states, torch.Tensor):
                tracer._record(name + '#pre_reduce', states)
            result = bound(states, *args, **kwargs)
            if tracer.active and isinstance(result, torch.Tensor):
                tracer._record(name + '#post_reduce', result)
            return result

        forward_entry._claude_seam = reduce_final_output._claude_seam = True
        select_experts._claude_seam = True
        self.gate_handle = gate.register_forward_hook(gate_hook)
        router.select_experts = select_experts
        runner._forward_entry = forward_entry
        runner._maybe_reduce_final_output = reduce_final_output

    def remove(self):
        self.runner._forward_entry = self.entry
        vars(self.runner).pop('_maybe_reduce_final_output', None)
        vars(self.router).pop('select_experts', None)
        self.gate_handle.remove()


class Capture:
    """Device-latched copy of the first routed call that disagrees with its
    (rows, runner) reference digest. Forward-path work is device-only.

    Lifecycle (`state`): armed -> allocated -> latched (writer saw the latch
    word of this generation) -> fetching -> saved, or error, or discarded
    (disarmed before a save). Every state change after allocation is written
    as a `type: capture` status line to the receipts, and every step record
    carries {generation, state}, so a missing save is visible, never silent.
    """

    FIELDS = ('latched', 'epoch', 'rows', 'runner', 'seq', 'ref0', 'ref1', 'digest0', 'digest1')

    def __init__(self, tracer, rows, names, generation, arm):
        self.tracer, self.rows, self.names = tracer, sorted(rows), sorted(names)
        self.generation, self.arm, self.session = generation, arm, tracer.session
        self.state, self.error, self.path = 'armed', None, None
        self.allocated = self.ready = self.frozen = False
        self.latched_epoch = None

    def status(self):
        return {'generation': self.generation, 'state': self.state, 'error': self.error}

    def matches(self, meta):
        """True when a queued step record belongs to this very capture."""
        info = meta.get('capture') or {}
        return (meta.get('arm') == self.arm and meta.get('session') == self.session
                and info.get('generation') == self.generation and self.allocated)

    def fail(self, where, error):
        self.frozen = True
        self.state, self.error = 'error', f'{where}: {error!r}'
        self.tracer._status(self, {'where': where})
        print(f'[DS41-ACT-TRACE] capture generation {self.generation} failed in {where}: {error!r}',
              flush=True)

    def _allocate(self, tensors, rows):
        device = tensors['routed'].device
        top = max(self.rows)
        self.spec = {k: (t.dtype, tuple(t.shape[1:]), t.numel() // rows * t.element_size())
                     for k, t in tensors.items()}
        self.buf = {k: torch.zeros(top * per_row, dtype=torch.uint8, device=device)
                    for k, (_, _, per_row) in self.spec.items()}
        self.ref = torch.zeros(len(self.rows), len(self.names), 2, dtype=torch.int64, device=device)
        self.ref_input = torch.zeros(len(self.rows), len(self.names), 2, dtype=torch.int64, device=device)
        self.seen = torch.zeros(len(self.rows), len(self.names), dtype=torch.bool, device=device)
        # [latched, epoch, rows, runner index, seq, ref0, ref1, digest0, digest1]
        self.meta = torch.zeros(9, dtype=torch.int64, device=device)
        self.info = torch.zeros(5, dtype=torch.int64, device=device)
        self.allocated, self.state = True, 'allocated'

    def observe(self, name, tensors, routed_digest):
        if self.frozen or name not in self.names:
            return
        if any(not isinstance(t, torch.Tensor) for t in tensors.values()):
            return
        rows = tensors['routed'].shape[0]
        if rows not in self.rows:
            return
        try:                                   # a diagnostic must never fail the forward
            self._observe(name, tensors, routed_digest, rows)
        except Exception as error:
            self.fail('observe', error)

    def _observe(self, name, tensors, routed_digest, rows):
        if not self.allocated:
            self._allocate(tensors, rows)
        r, l = self.rows.index(rows), self.names.index(name)
        step_input = self.tracer.input_digest
        ref = torch.where(self.seen[r, l], self.ref[r, l], routed_digest)
        ref_input = torch.where(self.seen[r, l], self.ref_input[r, l], step_input)
        self.ref[r, l].copy_(ref)
        self.ref_input[r, l].copy_(ref_input)
        self.seen[r, l].fill_(True)
        # Same prompt (token-id digest) as the reference, different routed bytes.
        hit = ((routed_digest != ref).any() & (step_input == ref_input).all()
               & (self.meta[0] == 0))
        for key, tensor in tensors.items():
            source = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
            if source.numel() > self.buf[key].numel():
                raise ValueError(f'{key} needs {source.numel()} bytes, buffer holds {self.buf[key].numel()}')
            target = self.buf[key][:source.numel()]
            target.copy_(torch.where(hit, source, target))
        epoch = self.tracer.epoch if type(self.tracer.epoch) is int else -1
        for index, value in enumerate((1, epoch, rows, l, self.tracer.seq)):
            self.info[index].fill_(value)
        self.meta[:5].copy_(torch.where(hit, self.info, self.meta[:5]))
        self.meta[5:7].copy_(torch.where(hit, ref, self.meta[5:7]))
        self.meta[7:9].copy_(torch.where(hit, routed_digest, self.meta[7:9]))

    def freeze_and_fetch(self, transport, path, identity):
        """Forward thread, once: stop updating and copy everything to host."""
        self.frozen, self.state, self.path = True, 'fetching', path
        self.tracer._status(self, {})
        try:
            pin = torch.cuda.is_available()
            host = {k: torch.empty_like(b, device='cpu', pin_memory=pin) for k, b in self.buf.items()}
            extra = {k: torch.empty_like(getattr(self, k), device='cpu', pin_memory=pin)
                     for k in ('ref', 'ref_input', 'seen', 'meta')}
            pairs = ([(self.buf[k], host[k]) for k in host]
                     + [(getattr(self, k), extra[k]) for k in extra])
        except Exception as error:
            self.fail('fetch', error)
            return

        def save():
            try:
                meta = dict(zip(self.FIELDS, extra['meta'].tolist()))
                if not meta['latched']:
                    raise RuntimeError('fetched latch word is clear')
                rows = meta['rows']
                tensors = {k: host[k][:rows * per_row].view(dtype).view(rows, *tail)
                           for k, (dtype, tail, per_row) in self.spec.items()}
                record = dict(identity, generation=self.generation, meta=meta,
                              runner=self.names[meta['runner']], names=self.names, row_values=self.rows,
                              ref=extra['ref'], ref_input=extra['ref_input'], seen=extra['seen'],
                              tensors=tensors)
                temporary = path + '.tmp'
                torch.save(record, temporary)
                os.replace(temporary, path)
            except Exception as error:
                self.fail('save', error)
                return
            self.state = 'saved'
            self.tracer._status(self, {'path': path, 'epoch': meta['epoch'], 'rows': meta['rows'],
                                       'runner': record['runner']})
            print(f'[DS41-ACT-TRACE] capture saved {path}', flush=True)

        try:
            transport.fetch(pairs, save)
        except Exception as error:
            self.fail('fetch', error)


class Tracer:
    def __init__(self, model, *, transport=None, trigger=TRIGGER, out_dir=OUT_DIR, rank=None, node=None):
        self.model, self.trigger, self.out_dir = model, trigger, out_dir
        self.transport = transport
        self.rank, self.node = rank, node
        self.session = f'{os.getpid()}-{uuid.uuid4().hex[:12]}'
        self.config, self.mtime, self.handles = None, None, []
        self.seams = 0
        self.capture = None
        self.capture_generation = 0
        self.seq = self.dropped = 0
        self.active = False
        self.slot = self.count = 0
        self.names: list[str] = []
        self.overflow = False
        self.lock = threading.Lock()
        model.register_forward_pre_hook(self._begin, with_kwargs=True)
        model.register_forward_hook(self._end, with_kwargs=True)

    # ---- configuration (host only) -------------------------------------------------
    def _refresh(self):
        try:
            mtime = os.stat(self.trigger).st_mtime_ns
        except OSError:
            mtime = None
        if mtime == self.mtime:
            return
        self.mtime = mtime
        self._disarm()
        if mtime is None:
            self.config = None
            return
        try:
            with open(self.trigger, encoding='utf-8') as handle:
                config = parse_config(handle.read())
        except Exception as error:
            print(f'[DS41-ACT-TRACE] trigger ignored: {error!r}', flush=True)
            self.config = None
            return
        hits = {(key, p): 0 for key in ('post', 'pre', 'moe') for p in config[key]}
        seam_names = []
        try:
            for name, module in self.model.named_modules():
                if not name:
                    continue
                members = []
                for pattern in config['post']:
                    base, member = split_member(pattern)
                    if matches(name, base):
                        hits['post', pattern] += 1
                        members.append(member)
                if members:
                    keep = None if None in members else set(members)
                    self.handles.append(module.register_forward_hook(self._post(name, keep)))
                pre = [p for p in config['pre'] if matches(name, p)]
                for pattern in pre:
                    hits['pre', pattern] += 1
                if pre:
                    self.handles.append(module.register_forward_pre_hook(self._pre(name)))
                moe = [p for p in config['moe'] if matches(name, p)]
                for pattern in moe:
                    hits['moe', pattern] += 1
                if moe:
                    self.handles.append(MoeSeams(self, name, module))
                    self.seams += 1
                    seam_names.append(name)
            unmatched = sorted(f'{key}:{p}' for (key, p), n in hits.items() if n == 0)
            if unmatched:
                raise ValueError(f'patterns match no hook site: {unmatched}')
            if config['capture'] is not None:
                captured = {}
                for pattern in config['capture']['moe']:
                    chosen = [n for n in seam_names if matches(n, pattern)]
                    if not chosen:
                        raise ValueError(f'capture pattern selects no armed MoE seam: {pattern}')
                    captured.update(dict.fromkeys(chosen))
                self.capture_generation += 1
                self.capture = Capture(self, config['capture']['rows'], captured,
                                       self.capture_generation, config['arm'])
        except Exception as error:
            self._disarm()
            print(f'[DS41-ACT-TRACE] trigger refused: {error!r}', flush=True)
            self.config = None
            return
        if self.config is None or self.config['arm'] != config['arm']:
            self.seq = self.dropped = 0
        self.config = config
        print(f'[DS41-ACT-TRACE] armed {config["arm"]}: {len(self.handles) - self.seams} module hooks, '
              f'{self.seams} MoE seam runners', flush=True)

    def _disarm(self):
        for handle in reversed(self.handles):
            handle.remove()
        capture = self.capture
        if capture is not None and capture.state in ('allocated', 'latched'):
            capture.frozen, capture.state = True, 'discarded'
            self._status(capture, {'latched_epoch': capture.latched_epoch})
        self.handles, self.seams, self.capture = [], 0, None

    def _status(self, capture, extra):
        """Append a capture status line to this process's receipts (any thread)."""
        try:
            self._identity()
            self._append({'type': 'capture', 'arm': capture.arm, 'session': capture.session,
                          'rank': self.rank, 'node': self.node, 'pid': os.getpid(),
                          'wall': time.time(), **capture.status(), **extra})
        except Exception as error:
            print(f'[DS41-ACT-TRACE] capture status not written: {error!r}', flush=True)

    def _identity(self):
        if self.rank is None:
            try:
                from vllm.distributed import get_tensor_model_parallel_rank
                self.rank = get_tensor_model_parallel_rank()
            except Exception:
                self.rank = -1
        if self.node is None:
            self.node = os.environ.get('DS41_NODE', 'unknown')

    def _append(self, record):
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f'{self.node}-{os.getpid()}.jsonl')
        with self.lock, open(path, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')

    def _sink(self, meta):
        """Writer thread: persist a step record; mark only its own capture ready."""
        self._append(meta)
        if '<capture>' not in meta['names']:
            return
        latched, epoch = meta['hashes'][meta['names'].index('<capture>')]
        if not latched:
            return
        capture = self.capture
        if capture is not None and capture.matches(meta):
            if capture.state == 'allocated':
                capture.latched_epoch, capture.state = epoch, 'latched'
                capture.ready = True           # read by the forward thread at the next step
        else:
            info = meta.get('capture') or {}
            self._append({'type': 'capture', 'arm': meta.get('arm'), 'session': meta.get('session'),
                          'rank': meta.get('rank'), 'node': meta.get('node'), 'pid': os.getpid(),
                          'wall': time.time(), 'generation': info.get('generation'),
                          'state': 'latched-discarded', 'error': None, 'latched_epoch': epoch})

    # ---- forward path: device work only, no host synchronization -------------------
    def _begin(self, module, args, kwargs):
        self.active = False
        if torch.compiler.is_compiling() or (torch.cuda.is_available()
                                             and torch.cuda.is_current_stream_capturing()):
            return
        self._refresh()
        if self.config is None:
            return
        capture = self.capture
        if capture is not None and capture.ready and not capture.frozen and self.transport is not None:
            self._identity()
            capture.freeze_and_fetch(
                self.transport,
                os.path.join(self.out_dir, f'{self.node}-{os.getpid()}-g{capture.generation}-capture.pt'),
                {'arm': capture.arm, 'session': self.session, 'rank': self.rank, 'node': self.node})
        ids = kwargs.get('input_ids', args[0] if args else None)
        positions = kwargs.get('positions', args[1] if len(args) > 1 else None)
        embeds = kwargs.get('inputs_embeds')
        source = ids if isinstance(ids, torch.Tensor) else embeds
        if not isinstance(source, torch.Tensor):
            return
        rows = source.shape[0]
        if not self.config['min_tokens'] <= rows <= self.config['max_tokens']:
            return
        self.seq += 1
        if self.transport is None:
            self._identity()
            self.transport = CudaTransport(self._sink, source.device)
        free = self.transport.free
        slot = next((i for i, available in enumerate(free) if available), None)
        if slot is None:
            self.dropped += 1
            return
        free[slot] = False
        self.slot, self.count, self.names, self.overflow = slot, 0, [], False
        self.rows, self.wall = rows, time.time()
        epoch = getattr(self.model, '_engram_epoch', None)
        self.epoch = epoch if type(epoch) is int else None
        self.active = True
        self.transport.begin(slot)
        self.input_digest = self._record('<input>', source)
        if isinstance(positions, torch.Tensor):
            self._record('<positions>', positions)

    def _record(self, name, tensor):
        value = digest(tensor)
        self._store(name, value)
        return value

    def _store(self, name, value):
        if self.count >= self.transport.acc.shape[1]:
            self.overflow = True
            return
        self.transport.acc[self.slot, self.count].copy_(value)
        self.names.append(name)
        self.count += 1

    def _post(self, name, keep=None):
        def hook(module, args, output):
            if self.active:
                for index, (suffix, tensor) in enumerate(_tensors(output)):
                    if keep is None or index in keep:
                        self._record(name + suffix, tensor)
        return hook

    def _pre(self, name):
        def hook(module, args):
            if self.active:
                for suffix, tensor in _tensors(args):
                    self._record('>' + name + suffix, tensor)
        return hook

    def _end(self, module, args, kwargs, output):
        if not self.active:
            return
        for suffix, tensor in _tensors(output):
            self._record('<output>' + suffix, tensor)
        capture = self.capture
        if capture is not None and capture.allocated and not capture.frozen:
            self._store('<capture>', capture.meta[0:2])   # raw [latched, epoch], not a digest
        self.active = False
        self._identity()
        meta = {'type': 'step', 'arm': self.config['arm'], 'session': self.session,
                'epoch': self.epoch, 'rank': self.rank, 'node': self.node, 'pid': os.getpid(),
                'seq': self.seq, 'rows': self.rows, 'wall': self.wall, 'names': list(self.names),
                'dropped_before': self.dropped, 'overflow': self.overflow, 'seam_runners': self.seams,
                'capture': None if capture is None else capture.status()}
        self.transport.submit(self.slot, self.count, meta)


_INSTALLED: list = []


def install(model, **kwargs):
    """Attach the tracer's two step hooks; module hooks exist only while armed.

    One tracer per process: the epoch join assumes one traced model per rank.
    At 1794dcf1 only the target model builds a DeepseekV4Model (the DSpark
    draft does not); any second instance is refused and logged, never traced.
    """
    if _INSTALLED:
        print('[DS41-ACT-TRACE] second DeepseekV4Model in this process: not traced', flush=True)
        return None
    tracer = Tracer(model, **kwargs)
    model._claude_act_tracer = tracer
    _INSTALLED.append(tracer)
    return tracer
