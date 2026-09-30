"""Isolated replay of captured Engram operands using the serving compiler identity.

Run only on an idle GPU. No model server, lookup, collective, or quantizer runs
inside the measured replay loop. This is a correctness probe, not a benchmark.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--trace', required=True)
p.add_argument('--manifest', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--report', required=True)
p.add_argument('--reference-sha256', required=True, help='Cross-rank unanimous original-output digest')
p.add_argument('--reference-key', choices=('projected_kv', 'dense_replay_synchronized'), default='projected_kv')
p.add_argument('--repeats', type=int, default=64)
p.add_argument('--delay-ms', type=float, default=0)
p.add_argument('--source-sha256', help='Explicit diagnostic source override, not a serving identity')
p.add_argument('--fresh-cache-dir', help='Fresh isolated compiler output; never deletes the serving cache')
p.add_argument('--column-slice', help='Diagnostic aligned output interval START:END')
p.add_argument('--poison-output', action='store_true', help='Initialize output to NaN to detect unwritten elements')
a = p.parse_args()
if a.repeats < 2 or Path(a.report).exists():
    raise RuntimeError('Require repeated trials and a fresh receipt')
manifest = json.loads(Path(a.manifest).read_text())
environment = dict(manifest['compile_environment'])
# Copy only the compiler's declared public inputs, never credentials or arbitrary
# process environment. Empty entries mean unset, as in the serving manifest.
allowed = {'CC', 'CUDACXX', 'CUDA_HOME', 'CUDA_PATH', 'CUDA_TOOLKIT_PATH',
           'CUTE_DSL_ARCH', 'CUTE_DSL_LIBS', 'CUTLASS_DSL_VERSION', 'CXX',
           'NVCC_APPEND_FLAGS', 'NVCC_PREPEND_FLAGS'}
for name in environment:
    if not name.startswith('B12X_') and name not in allowed:
        raise RuntimeError(f'Unexpected compiler environment input: {name}')
for name in list(os.environ):
    if name.startswith('B12X_') and name not in environment:
        del os.environ[name]
for name, value in environment.items():
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)
cache_root = Path(a.fresh_cache_dir) if a.fresh_cache_dir else Path(a.manifest).parent.parent
if a.fresh_cache_dir and cache_root.exists():
    raise RuntimeError('Fresh compiler cache already exists')
os.environ['B12X_COMPILE_CACHE_DIR'] = str(cache_root)
os.environ['B12X_PRINT_COMPILE_PROGRESS'] = '1'
object_path = Path(a.manifest).with_suffix('.o')
object_sha = hashlib.sha256(object_path.read_bytes()).hexdigest()
if object_sha != manifest['object_sha256']:
    raise RuntimeError('Serving compiled object differs from its manifest')

import torch
from safetensors import safe_open
from b12x.gemm import DenseGemmConfig, block_fp8_linear as bfl
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation.types import require_prepared
from b12x._lib.compile_plan import program_keys
from b12x._lib import dense_gemm as dense_source
from b12x._lib.compiler import compile_cache_info

actual_source_sha = hashlib.sha256(Path(dense_source.__file__).read_bytes()).hexdigest()
if a.source_sha256 and actual_source_sha != a.source_sha256:
    raise RuntimeError('Diagnostic source differs from its declared digest')

torch.set_num_threads(1)
trace = torch.load(a.trace, map_location='cpu', weights_only=True, mmap=True)
op, = (r for r in trace['operators'] if r['name'] == 'engram_projection')
reference_hash = hashlib.sha256(op['outputs'][a.reference_key].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
if reference_hash != a.reference_sha256:
    raise RuntimeError('Captured reference does not match cross-rank consensus')
m, k = op['inputs']['source'].shape
n = op['outputs']['projected_kv'].shape[1]
capacity = op['dense_rows']['expected_m']
if (k, n, capacity) != (6144, 25600, 256):
    raise RuntimeError('Unexpected captured geometry')
if op['dense_programs'][0]['key'] != manifest['cache_key']:
    raise RuntimeError('Trace and compiler manifest disagree')
with safe_open(a.checkpoint, framework='pt', device='cpu') as f:
    weight = f.get_tensor('layers.1.engram.wkv.weight')
    scales = f.get_tensor('layers.1.engram.wkv.scale')
if not torch.equal(weight[:1024].view(torch.uint8), op['outputs']['weight_prefix'].reshape(1024, k)):
    raise RuntimeError('Captured weight prefix differs from checkpoint')
column_interval = None
if a.column_slice:
    start, end = map(int, a.column_slice.split(':'))
    if not 0 <= start < end <= n or start % 128 or end % 128:
        raise RuntimeError('Output slice must consist of full 128-column tiles')
    column_interval = [start, end]
    weight = weight[start:end].contiguous()
    scales = scales[start // 32:end // 32].contiguous()
    n = end - start
weight, scales = weight.cuda(), scales.cuda()
packed = bfl.pack_weight(weight, scales, block_size=(32, 32))
if not a.column_slice and not torch.equal(packed.weight.scale_mma.view(torch.uint8).cpu(), op['outputs']['weight_scale_mma']):
    raise RuntimeError('Captured weight scales differ from checkpoint')
config = DenseGemmConfig(backend='cutedsl', tile_m=64, tile_n=128, tile_k=128,
                        load_path='tma', swap_ab=True, split_k_slices=1,
                        large_m_unroll=True, target_occupancy=None)
plan = bfl.plan(bfl.Caps(device=torch.device('cuda'), max_tokens=capacity,
                       in_features=k, out_features=n, block_size=(32, 32)), override=config)
spec, = plan.scratch_specs()
scratch = torch.empty(spec.shape, dtype=spec.dtype, device='cuda')
source = torch.zeros((capacity, k), dtype=torch.bfloat16, device='cuda')
source[:m].copy_(op['inputs']['source'])
output = torch.empty((capacity, n, 1), dtype=torch.bfloat16, device='cuda')


def prepare(state):
    binding = state.bind(scratch=scratch, source=source, packed_weight=packed, output=output)
    return PreparedCall(run=lambda: state.run_binding(binding))


with PreparationSession(device=torch.device('cuda'), autotune=False, compile_workers=20) as session:
    session.prepare((plan.request(name='engram-dense-replay', prepare_call=prepare),))
    session.freeze()
    binding = bfl.bind(plan, scratch=scratch, source=source[:m], packed_weight=packed, output=output[:m])
    state = require_prepared(plan, 'gemm.block_fp8_linear')
    keys = [{'dialect': x.dialect, 'key': x.key, 'name': x.name} for x in program_keys(state.dense.gemm)]
    if not (a.source_sha256 or a.column_slice) and keys != op['dense_programs']:
        raise RuntimeError(f'Isolated compiler identity differs: {keys} != {op["dense_programs"]}')
    if hashlib.sha256(object_path.read_bytes()).hexdigest() != object_sha:
        raise RuntimeError('Compiled object changed during preparation')
    executed_manifest_path = cache_root / keys[0]['key'][:2] / (keys[0]['key'] + '.json')
    executed_manifest = json.loads(executed_manifest_path.read_text())
    if not a.column_slice and executed_manifest['compile_spec_hash'] != manifest['compile_spec_hash']:
        raise RuntimeError('Diagnostic changed the compiled tactic or argument specialization')
    executed_sha = hashlib.sha256(executed_manifest_path.with_suffix('.o').read_bytes()).hexdigest()
    if executed_sha != executed_manifest['object_sha256']:
        raise RuntimeError('Executed object differs from its manifest')
    print('DIAGNOSTIC-PROGRAM' if a.source_sha256 or a.column_slice else 'IDENTICAL-SERVING-PROGRAM', json.dumps(keys), flush=True)
    print('COMPILER-CACHE', json.dumps(compile_cache_info()), flush=True)
    binding.x_q.values.view(torch.uint8).copy_(op['outputs']['quantized_values'])
    binding.x_q.scale_rows.view(torch.uint8).copy_(op['outputs']['quantized_scale_rows'])
    binding.x_q.scale_mma.view(torch.uint8).copy_(op['outputs']['quantized_scale_mma'])
    torch.cuda.synchronize()
    reference = op['outputs'][a.reference_key]
    if column_interval:
        reference = reference[:, start:end].contiguous()
    rows = []
    first = None
    for i in range(a.repeats):
        torch.cuda.synchronize()
        if a.delay_ms:
            time.sleep(a.delay_ms / 1000)
        out = torch.empty_like(binding.output)
        if a.poison_output:
            out.fill_(float('nan'))
        state.dense.run((binding.x_q.values.view(m, k, 1), binding.x_q.scale_mma),
                        (packed.weight.values.view(n, k, 1), packed.weight.scale_mma),
                        out=out, stream=None, split_k_workspace=binding.workspace)
        result = out.cpu()
        if first is None:
            first = result.clone()
        record = {'repeat': i, 'same_as_first': torch.equal(first, result),
                  'nan_elements': int(torch.isnan(result).sum()),
                  'same_as_serving': torch.equal(reference, result),
                  'sha256': hashlib.sha256(result.view(torch.uint8).numpy().tobytes()).hexdigest(),
                  'max_abs_vs_serving': float((result.float() - reference.float()).abs().max())}
        rows.append(record)
        if not record['same_as_serving']:
            indices = torch.nonzero(result[..., 0] != reference[..., 0])
            record['faulty_tiles'] = (indices // torch.tensor([64, 128])).unique(dim=0).tolist()
            record['faulty_rows'] = indices[:, 0].unique().tolist()
            record['faulty_columns'] = indices[:, 1].unique().tolist()
        if i % 32 == 0 or i == a.repeats - 1 or not record['same_as_serving']:
            print(json.dumps(record), flush=True)
        if not record['same_as_serving'] and not Path(a.report + '.first-mismatch.pt').exists():
            torch.save({'output': result, 'reference': reference}, a.report + '.first-mismatch.pt')
    for field, actual in [('quantized_values', binding.x_q.values),
                          ('quantized_scale_mma', binding.x_q.scale_mma)]:
        if not torch.equal(actual.view(torch.uint8).cpu(), op['outputs'][field]):
            raise RuntimeError('Replay operand changed: ' + field)
    report = {'programs': keys, 'trace': a.trace, 'manifest': a.manifest,
              'object_sha256': object_sha, 'reference_sha256': reference_hash, 'reference_key': a.reference_key,
              'executed_object_sha256': executed_sha, 'dense_source_sha256': actual_source_sha,
              'diagnostic_source_override': bool(a.source_sha256),
              'compiler_cache': compile_cache_info(), 'fresh_compile': bool(a.fresh_cache_dir),
              'addresses': {'values': binding.x_q.values.data_ptr(),
                            'scales': binding.x_q.scale_mma.data_ptr(),
                            'weight': packed.weight.values.data_ptr(),
                            'weight_scales': packed.weight.scale_mma.data_ptr()},
              'config': repr(config), 'column_interval': column_interval,
              'executed_compile_spec': executed_manifest['compile_spec_hash'],
              'delay_ms': a.delay_ms, 'poison_output': a.poison_output, 'results': rows}
    Path(a.report).write_text(json.dumps(report, indent=2) + '\n')
print('DENSE-REPLAY-COMPLETE', flush=True)
raise SystemExit(0 if all(r['same_as_first'] and r['same_as_serving'] for r in rows) else 1)
