#!/usr/bin/env python3
"""Real-input replay of the DS4.1 router projection (B12X bf16_gemv "prefill" backend).

DIAGNOSTIC ONLY. Runs the gate projection exactly as vLLM 1794dcf1
B12xLinearMethod does (bf16 source and weight [384, 5120], fp32 output,
`bf16_gemv.mm(x, weight, plan=plan, output_dtype=float32)`) on the real
captured MoE input and the real layer gate weight, many times, and compares
every output bitwise with the first. No model, no other weights.

Production plan resolution (b12x_layers.py 293-302): the exact (dtype, rows)
plan if rows is a planned count, else the max_tokens capacity plan. In the
retained caches 256 is planned and 254 is not, so:
  rows 256 -> the 256-row plan, rows 254 -> the 8192-row capacity plan.
The replay builds both, requires the selection cache to hold 256 and 8192
records and no 254 record (so the fallback is what production ran), and
records each selected config and its cached program identities.

Modes
  baseline (default): cache_only PreparationSession on a private copy of the
    boot's selection file (same identity/name/directory gates as the routed
    probe); every plan must report source "cached". The installed
    b12x/gemm/bf16_gemv/_prefill.py must hash to the a7d7d29b blob
    (d92cf099...); otherwise REFUSED.
  fenced (--fenced): for a SOURCE-CHANGED diagnostic derivative whose
    _prefill.py differs from the pinned blob by exactly one inserted line
    `cute.arch.fence_acq_rel_cta()` immediately before
    `load_pipeline.consumer_release(consumer_state)` (same indentation);
    anything else is REFUSED. Cached programs are NOT reused: the plan is
    pinned to GemvConfig(backend="prefill") (selection source "override") and
    compiled in process into --compile-cache-dir, which must be a fresh
    private directory (set before b12x is imported). The receipt records
    source_changed=true, the fenced source hash and selection source
    "override"; it never reports cached identity. Baseline and fenced receipts
    are comparable only by matched inputs, rows, arms and call counts.

Arms per row count (--arms):
  idle      gate alone on the main stream.
  overlap   a shared-expert-shaped bf16 matmul chain ([rows,5120]x[5120,1152],
            --side-iters) on a second stream, released by an event recorded on
            main just before the gate, then the gate on main. Overlap is
            MEASURED per repeat with timed events on both streams (gate
            interval inside a side interval, both relative to the common
            release event); the arm is valid only if at least --min-overlap of
            its repeats overlapped, and the report also compares the gate's
            duration against the idle arm (co-execution slows it).

Checks
  * the reference (first run) must equal the captured logits bitwise everywhere
    except the rows the forensics fit marks as corrupted (same kernel, same
    inputs); otherwise the replay is not running what production ran: REFUSED;
  * every repeat: the caller-owned output (bf16_gemv.mm(..., out=out), the
    same prepared state.run launcher production's allocating call uses) is
    NaN-filled before the timed window, so unwritten elements fail, and is compared
    bitwise with that row count's reference; mismatches counted, the first
    --keep failing outputs kept on CPU and classified with
    claude_gate_capture_forensics.analyze (kt+2 chunk signature etc.).
Memory is bounded: counters, a fixed reservoir of timings, --keep failures.

Exit 0: no mismatch; 1: mismatches; 2: refused; 3: clean but invalid overlap.

  /opt/venv/bin/python claude_replay_gate_prefill.py \\
      --tuning-receipt /cache/.../b12x/preparation/<digest>.json \\
      --capture dusty-524-g1-capture.pt \\
      --checkpoint-dir /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/fb2764a5cf321eaa5070ca8f9e892818f477c16d \\
      --weight-sha256 b98a45639ac40ddb... --repeats 100000 --out gate-replay.json
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
PINNED_PREFILL_SHA256 = 'd92cf099a097df66627a968e711a40366817b238f1efa7a68d43286e3708edc4'
RELEASE = 'load_pipeline.consumer_release(consumer_state)'
FENCE = 'cute.arch.fence_acq_rel_cta()'
IN_FEATURES, OUT_FEATURES, TILE = 5120, 384, 64


class Refused(SystemExit):
    def __init__(self, message):
        print(f'CLAUDE-GATE-REPLAY-REFUSED: {message}', flush=True)
        super().__init__(2)


def require(condition, message):
    if not condition:
        raise Refused(message)


# ---- pure helpers (CPU-tested) -------------------------------------------------------------

def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def query_dict(max_rows):
    """GemvQuery fields exactly as B12xLinearMethod builds them for the DS4.1 gate."""
    return {'source_dtype': 'bfloat16', 'weight_dtype': 'bfloat16', 'max_rows': int(max_rows),
            'in_features': IN_FEATURES, 'out_features': OUT_FEATURES, 'source_contiguous': True,
            'source_aligned': True, 'weight_contiguous': True, 'weight_aligned': True,
            'output_dtype': 'float32', 'output_contiguous': True, 'output_aligned': True, 'bias_dtype': None}


def choice_key(max_rows):
    """b12x.preparation.session._choice_key for gemm.bf16_gemv at a7d7d29b (no pin, no deps)."""
    return digest({'component': 'gemm.bf16_gemv', 'query_schema': 4, 'config_schema': 5,
                   'semantic_version': 1, 'candidate_contract_version': 3, 'query': query_dict(max_rows),
                   'invocation': {}, 'pin': None, 'dependencies': []})


def production_plan_rows(records, rows, capacity=8192):
    """Which plan production uses for `rows`: exact if cached, else capacity."""
    return rows if choice_key(rows) in records else capacity


def check_fence_source(text, pinned_text):
    """True only for the pinned text plus one fence line directly before the release."""
    lines, pinned = text.splitlines(keepends=True), pinned_text.splitlines(keepends=True)
    if len(lines) != len(pinned) + 1:
        return False
    for index, line in enumerate(lines):
        if line.strip() == FENCE:
            following = lines[index + 1] if index + 1 < len(lines) else ''
            indent = line[:len(line) - len(line.lstrip())]
            if (following.strip() == RELEASE and following.startswith(indent)
                    and following[len(indent):len(indent) + 1] != ' '
                    and lines[:index] + lines[index + 1:] == pinned):
                return True
    return False


def confined(tiles, rows):
    """Every differing element lies in one of `rows` (rows found corrupted by forensics)."""
    return all(set(tile['rows']) <= set(rows) for tile in tiles)


def overlapped(gate, side):
    """Intervals in ms from a common release event: gate fully inside a side interval."""
    return side[0] <= gate[0] and gate[1] <= side[1]


class Reservoir:
    def __init__(self, size=2048, seed=0):
        self.size, self.values, self.seen, self.rng = size, [], 0, random.Random(seed)

    def add(self, value):
        self.seen += 1
        if len(self.values) < self.size:
            self.values.append(value)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.size:
                self.values[j] = value

    def summary(self):
        if not self.values:
            return None
        ordered = sorted(self.values)
        pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]
        return {'count': self.seen, 'min': ordered[0], 'median': pick(0.5), 'p99': pick(0.99), 'max': ordered[-1]}


def tile_mismatches(reference, output):
    """64x64 tiles (row tile, column tile) where two outputs differ, with element counts."""
    import torch
    rows, cols = reference.shape
    different = ~((reference == output) | (torch.isnan(reference) & torch.isnan(output)))
    found = []
    for i in range((rows + TILE - 1) // TILE):
        for j in range((cols + TILE - 1) // TILE):
            block = different[i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE]
            count = int(block.sum())
            if count:
                found.append({'m_tile': i, 'n_tile': j, 'elements': count,
                              'rows': [i * TILE + r for r in block.any(1).nonzero().flatten().tolist()]})
    return found


def read_gate_weight(checkpoint_dir, layer):
    """Read layers.<layer>.ffn.gate.weight bytes from a local safetensors snapshot (no network)."""
    checkpoint_dir = Path(checkpoint_dir)
    index = json.loads((checkpoint_dir / 'model.safetensors.index.json').read_text())
    name = f'layers.{layer}.ffn.gate.weight'
    shard = checkpoint_dir / index['weight_map'][name]
    with open(shard, 'rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(size))
        info = header[name]
        require(info['dtype'] == 'BF16' and info['shape'] == [OUT_FEATURES, IN_FEATURES], f'{name} layout {info}')
        start, end = info['data_offsets']
        stream.seek(8 + size + start)
        data = stream.read(end - start)
    require(len(data) == end - start, 'short read of gate weight')
    return data, str(shard.name)


# ---- GPU replay --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--capture', type=Path, required=True)
    weight = p.add_mutually_exclusive_group(required=True)
    weight.add_argument('--checkpoint-dir', type=Path)
    weight.add_argument('--weight-bin', type=Path)
    p.add_argument('--weight-sha256', required=True)
    p.add_argument('--tuning-receipt', type=Path, help='baseline: the boot selection-cache file')
    p.add_argument('--fenced', action='store_true')
    p.add_argument('--compile-cache-dir', type=Path, help='fenced: fresh private B12X compile cache')
    p.add_argument('--rows', default='256,254')
    p.add_argument('--arms', default='idle,overlap')
    p.add_argument('--repeats', type=int, default=100000)
    p.add_argument('--side-iters', type=int, default=4)
    p.add_argument('--min-overlap', type=float, default=0.9)
    p.add_argument('--keep', type=int, default=8)
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    require(a.repeats >= 2, 'need at least two repeats')
    if a.fenced:
        require(a.compile_cache_dir is not None, '--fenced needs --compile-cache-dir')
        require(a.tuning_receipt is None, '--fenced never reads cached selections')
    else:
        require(a.tuning_receipt is not None, 'baseline needs --tuning-receipt')
        require(a.compile_cache_dir is None, 'baseline uses the boot compile cache')
    return a


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    a = parse_args(argv)
    if a.fenced:
        # Must precede the first b12x import: the compile cache location is read from it.
        require(not a.compile_cache_dir.exists() or not any(a.compile_cache_dir.iterdir()),
                'fenced compile cache directory must be new and empty')
        a.compile_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['B12X_COMPILE_CACHE_DIR'] = str(a.compile_cache_dir)
    import torch
    forensics = _load('claude_gate_forensics', HERE / 'claude_gate_capture_forensics.py')
    from b12x.gemm import bf16_gemv
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.preparation._cache import cache_identity, digest as b12x_digest
    from b12x._lib.compiler import _cute_compile_cache_dir

    prefill_path = Path(importlib.util.find_spec('b12x.gemm.bf16_gemv._prefill').origin)
    prefill_text = prefill_path.read_text()
    prefill_sha = hashlib.sha256(prefill_text.encode()).hexdigest()
    identity = {'prefill_path': str(prefill_path), 'prefill_sha256': prefill_sha, 'fenced': a.fenced,
                'source_changed': prefill_sha != PINNED_PREFILL_SHA256,
                'b12x_env': {k: v for k, v in sorted(os.environ.items()) if k.startswith('B12X_')}}
    if a.fenced:
        pinned = (HERE / 'claude-pinned-bf16_gemv-_prefill.py').read_text()
        require(hashlib.sha256(pinned.encode()).hexdigest() == PINNED_PREFILL_SHA256, 'pinned copy drifted')
        require(check_fence_source(prefill_text, pinned), 'installed _prefill.py is not the narrow fence derivative')
    else:
        require(prefill_sha == PINNED_PREFILL_SHA256, 'installed _prefill.py differs from a7d7d29b')

    device = torch.device('cuda', torch.cuda.current_device())
    saved = torch.load(a.capture, map_location='cpu', weights_only=True)
    layer = int(saved['runner'].split('.')[1])
    x_all = saved['tensors']['input']
    captured = saved['tensors']['logits']
    require(x_all.dtype == torch.bfloat16 and tuple(x_all.shape[1:]) == (IN_FEATURES,), 'capture input layout')
    data, shard = (read_gate_weight(a.checkpoint_dir, layer) if a.checkpoint_dir
                   else (a.weight_bin.read_bytes(), str(a.weight_bin)))
    require(hashlib.sha256(data).hexdigest() == a.weight_sha256, 'gate weight sha256 differs')
    weight_cpu = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).view(OUT_FEATURES, IN_FEATURES)
    weight = weight_cpu.to(device).contiguous()
    require(weight.data_ptr() % 16 == 0, 'weight not 16-byte aligned')
    rows_list = [int(r) for r in a.rows.split(',')]
    require(all(0 < r <= x_all.shape[0] for r in rows_list), 'rows exceed the captured input')
    identity.update(capture=str(a.capture), capture_epoch=saved['meta']['epoch'], capture_rank=saved['rank'],
                    capture_node=saved['node'], layer=layer, weight_shard=shard, weight_sha256=a.weight_sha256)

    # Plans exactly as production resolves them.
    if a.fenced:
        plan_rows = {r: (r if r == 256 else 8192) for r in rows_list}   # same resolution as the baseline caches
    else:
        payload = json.loads(a.tuning_receipt.read_text())
        require(set(payload) == {'identity', 'records'}, 'not a selection-cache file')
        require(a.tuning_receipt.stem == b12x_digest(payload['identity']), 'receipt name is not its identity digest')
        require(cache_identity(payload['identity']['namespace'], device.index) == payload['identity'],
                'live selection identity differs')
        require(a.tuning_receipt.resolve().parent == Path(_cute_compile_cache_dir()).resolve() / 'preparation',
                'receipt is not in this process compile cache')
        records = payload['records']
        plan_rows = {r: production_plan_rows(records, r) for r in rows_list}
        require(plan_rows.get(256, 256) == 256, 'rows 256 have no exact cached plan')
        require(plan_rows.get(254, 8192) == 8192, 'rows 254 have an exact plan; production would not fall back')
        identity['cached_records'] = {str(c): {'key': choice_key(c), 'config': records[choice_key(c)]['config'],
                                               'programs': records[choice_key(c)].get('programs')}
                                      for c in sorted(set(plan_rows.values()))}
    plans = {}
    for capacity in sorted(set(plan_rows.values())):
        query = bf16_gemv.GemvQuery(**{k: v for k, v in query_dict(capacity).items()})
        override = bf16_gemv.GemvConfig(backend='prefill') if a.fenced else None
        plans[capacity] = bf16_gemv.plan(query, override=override)

    def make_call(state):
        query = state.query
        source = torch.empty((query.max_rows, query.in_features), device=device, dtype=torch.bfloat16)
        out = torch.empty((query.max_rows, query.out_features), device=device, dtype=torch.float32)
        return PreparedCall(run=lambda: state.run(source, weight, out=out),
                            produce=lambda: source.normal_(std=0.25), owners=(weight,))

    requests = tuple(plan.request(name=f'claude-gate-replay.m{c}', prepare_call=make_call, benchmark_call=make_call)
                     for c, plan in plans.items())
    with tempfile.TemporaryDirectory(prefix='claude-gate-replay-') as private:
        if not a.fenced:
            shutil.copyfile(a.tuning_receipt, Path(private) / a.tuning_receipt.name)
            session_kwargs = dict(autotune=False, cache_only=True, cache_dir=private,
                                  namespace=payload['identity']['namespace'])
        else:
            session_kwargs = dict(autotune=False, cache_only=False, cache_dir=private)
        with PreparationSession(device=device, compile_workers=0, **session_kwargs) as session:
            try:
                session.prepare(requests)
            except LookupError as error:
                raise Refused(f'cached selection or compiled program missing: {error}')
            session.freeze()
            selections = {}
            for capacity, plan in plans.items():
                selection = plan.selection
                config = selection.config
                selections[str(capacity)] = {'source': selection.source,
                                             'config': {'backend': config.backend, 'rows_per_tile': config.rows_per_tile}}
                require(config.backend == 'prefill', f'plan m{capacity} backend {config.backend}')
                require(selection.source == ('override' if a.fenced else 'cached'),
                        f'plan m{capacity} selection source {selection.source}')
            identity['selections'] = selections
            print(json.dumps({'identity': identity}), flush=True)

            torch.manual_seed(20260925)
            side = torch.cuda.Stream(device=device)
            side_a = torch.randn((max(rows_list), IN_FEATURES), device=device, dtype=torch.bfloat16)
            side_b = torch.randn((IN_FEATURES, 1152), device=device, dtype=torch.bfloat16)
            side.wait_stream(torch.cuda.current_stream(device))       # operands initialized on main
            results, failed = [], False
            for rows in rows_list:
                plan = plans[plan_rows[rows]]
                x = x_all[:rows].to(device).contiguous()
                out = torch.empty((rows, OUT_FEATURES), device=device, dtype=torch.float32)
                out.fill_(float('nan'))
                bf16_gemv.mm(x, weight, plan=plan, out=out, output_dtype=torch.float32)
                torch.cuda.synchronize(device)
                reference = out.clone()
                require(not bool(torch.isnan(reference).any()), f'rows={rows}: reference has unwritten elements')
                reference_cpu = reference.cpu()
                row = {'rows': rows, 'plan_rows': plan_rows[rows], 'arms': {}}
                if rows == captured.shape[0]:
                    # Same kernel, same inputs: identical except the capture's corrupted rows.
                    bad = tile_mismatches(captured, reference_cpu)
                    corrupted = {f['row'] for f in forensics.analyze(x_all, captured, weight_cpu)['row_fits']}
                    row['capture_vs_reference_tiles'] = bad
                    row['capture_corrupted_rows'] = sorted(corrupted)
                    require(confined(bad, corrupted),
                            'reference differs from the captured logits outside the corrupted rows')
                idle_ms = Reservoir()
                for arm in a.arms.split(','):
                    require(arm in ('idle', 'overlap'), f'unknown arm {arm}')
                    mismatches, kept, overlaps, gate_ms = 0, [], 0, Reservoir()
                    release = torch.cuda.Event(enable_timing=True)
                    g0, g1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    started = time.monotonic()
                    done = 0
                    for repeat in range(a.repeats):
                        out.fill_(float('nan'))                      # before the timed window
                        release.record()
                        if arm == 'overlap':
                            side.wait_event(release)
                            with torch.cuda.stream(side):
                                s0.record(side)
                                for _ in range(a.side_iters):
                                    torch.matmul(side_a[:rows], side_b)
                                s1.record(side)
                        g0.record()
                        bf16_gemv.mm(x, weight, plan=plan, out=out, output_dtype=torch.float32)
                        g1.record()
                        torch.cuda.synchronize(device)
                        done += 1
                        gate = (release.elapsed_time(g0), release.elapsed_time(g1))
                        gate_ms.add(gate[1] - gate[0])
                        if arm == 'idle':
                            idle_ms.add(gate[1] - gate[0])
                        else:
                            if overlapped(gate, (release.elapsed_time(s0), release.elapsed_time(s1))):
                                overlaps += 1
                        if not torch.equal(out, reference):
                            mismatches += 1
                            if len(kept) < a.keep:
                                kept.append((repeat, out.cpu()))
                            if a.stop_after and mismatches >= a.stop_after:
                                break
                    arm_report = {'repeats': done, 'mismatches': mismatches, 'seconds': round(time.monotonic() - started, 1),
                                  'gate_ms': gate_ms.summary()}
                    if arm == 'overlap':
                        arm_report['overlapped_repeats'] = overlaps
                        arm_report['overlap_fraction'] = overlaps / max(1, done)
                        arm_report['valid'] = arm_report['overlap_fraction'] >= a.min_overlap
                    failures = []
                    for repeat, output in kept:
                        report = forensics.analyze(x.cpu(), output, weight_cpu)
                        failures.append({'repeat': repeat, 'tiles': tile_mismatches(reference_cpu, output),
                                         'all_rows_exact': report['all_rows_exact'],
                                         'fits': [{'row': f['row'], **f['best'][0]} for f in report['row_fits']]})
                    arm_report['failures'] = failures
                    row['arms'][arm] = arm_report
                    failed |= mismatches > 0
                    print(json.dumps({'rows': rows, 'arm': arm, **{k: v for k, v in arm_report.items()
                                                                   if k != 'failures'}}), flush=True)
                results.append(row)
    report = {'identity': identity, 'results': results,
              'verdict': 'mismatch' if failed else 'no-mismatch'}
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    invalid = [(r['rows'], arm) for r in results for arm, v in r['arms'].items() if v.get('valid') is False]
    if invalid:
        print(f'CLAUDE-GATE-REPLAY-WARNING overlap arms below --min-overlap: {invalid}', flush=True)
    print('CLAUDE-GATE-REPLAY-' + ('MISMATCH' if failed else 'CLEAN'), flush=True)
    return 1 if failed else (3 if invalid else 0)


if __name__ == '__main__':
    sys.exit(main())
