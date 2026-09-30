"""Four-rank fixed-input PyNCCL probe, separate from the serving communicator.

Run only while the endpoint is idle. Uses its exact NCCL shared object and
transport environment; a result does not by itself attribute model variation.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_allocator import is_symmetric_memory_enabled

rank = int(os.environ['RANK'])
torch.cuda.set_device(0)
dist.init_process_group('gloo', rank=rank, world_size=4,
                        timeout=datetime.timedelta(seconds=180))
library = os.environ['VLLM_NCCL_SO_PATH']
print(json.dumps({'rank': rank, 'library': library,
                  'sha256': hashlib.sha256(Path(library).read_bytes()).hexdigest()}), flush=True)
comm = PyNcclCommunicator(dist.group.WORLD, device=torch.device('cuda:0'), library_path=library)
if comm.disabled:
    raise RuntimeError('PyNCCL communicator disabled')
if is_symmetric_memory_enabled():
    raise RuntimeError('Probe does not mirror symmetric-memory serving')
failed = False
for width in (5120, 6144):
    for rows in (128, 255, 256, 257, 300, 320, 384, 400, 416, 447, 448, 512):
        gen = torch.Generator(device='cuda').manual_seed(20260925 + rank)
        source = torch.randn((rows, width), generator=gen, device='cuda', dtype=torch.bfloat16)
        output = torch.empty_like(source)
        baseline = None
        changed = 0
        max_spread = 0.0
        for iteration in range(70):
            output.fill_(float('nan'))
            result = comm.all_reduce(source, output)
            torch.cuda.synchronize()
            if result is None or not torch.isfinite(output).all():
                raise RuntimeError('Missing or non-finite all-reduce output')
            if iteration < 6:
                continue
            if baseline is None:
                baseline = output.clone()
            elif not torch.equal(baseline, output):
                changed += 1
                max_spread = max(max_spread, (baseline.float() - output.float()).abs().max().item())
        digest = hashlib.sha256(baseline.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        rank_digests = [None] * 4
        dist.all_gather_object(rank_digests, digest)
        ranks_equal = len(set(rank_digests)) == 1
        failed |= changed > 0 or not ranks_equal
        print(json.dumps({'rank': rank, 'rows': rows, 'width': width,
                          'bytes': source.numel() * source.element_size(),
                          'measured_repeats': 64, 'different_from_first': changed,
                          'max_abs_spread': max_spread,
                          'rank_baselines_equal': ranks_equal}), flush=True)
        dist.barrier()
        del source, output, baseline, result
    cases = {}
    for rows in (256, 257, 384, 448):
        gen = torch.Generator(device='cuda').manual_seed(20260925 + rank)
        source = torch.randn((rows, width), generator=gen, device='cuda', dtype=torch.bfloat16)
        cases[rows] = (source, torch.empty_like(source))
    baselines, changed = {}, {rows: 0 for rows in cases}
    for iteration in range(64):
        for rows, (source, output) in cases.items():
            output.fill_(float('nan'))
            comm.all_reduce(source, output)
            torch.cuda.synchronize()
            if not torch.isfinite(output).all():
                raise RuntimeError('Non-finite interleaved output')
            if rows not in baselines:
                baselines[rows] = output.clone()
            else:
                changed[rows] += int(not torch.equal(output, baselines[rows]))
    failed |= any(changed.values())
    print(json.dumps({'rank': rank, 'width': width, 'interleaved_changes': changed,
                      'repeats_per_size': 64}), flush=True)
    del cases, baselines, source, output
dist.destroy_process_group()
if failed:
    raise SystemExit('NCCL-REPEATABILITY-FAIL')
print('NCCL-REPEATABILITY-PASS', flush=True)
