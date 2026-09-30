"""Generate independent isolated kernel-ordering controls from the frozen source."""
import difflib
import argparse
import re
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--only')
args = p.parse_args()
revision = 'a7d7d29b2ef8869086e0ceaa787321f17544e3c9'
path = 'b12x/_lib/dense_gemm.py'
source = subprocess.check_output(['git', '-C', '/home/jugs/git/b12x', 'show', f'{revision}:{path}'], text=True)
peek = 'mainloop_consumer_state, peek_ab_full_status'
unroll = 'unroll=4 if self.large_m_unroll else 2,'
if source.count(peek) != 2 or source.count(unroll) != 1:
    raise RuntimeError('Unexpected source sites')
tail_start = source.index('                # Hoist out last k_tile')
tail_end = source.index('                if cutlass.const_expr(self.swap_ab):', tail_start)
tail = source[tail_start:tail_end]
release = ('                    if k_block_idx == num_k_blocks - 1:\n'
           '                        mainloop_pipeline.consumer_release(mainloop_consumer_state)\n'
           '                        mainloop_consumer_state.advance()\n\n')
if tail.count(release) != 1:
    raise RuntimeError('Unexpected hoisted-tail release')
late_tail = tail.replace(release, '') + (
    '                mainloop_pipeline.consumer_release(mainloop_consumer_state)\n'
    '                mainloop_consumer_state.advance()\n\n')
variants = {'no-peek': source.replace(peek, 'mainloop_consumer_state'),
            'unroll2': source.replace(unroll, 'unroll=2,'),
            'late-release': source[:tail_start] + late_tail + source[tail_end:]}
initial_wait = ('                mainloop_pipeline.consumer_wait(\n'
                '                    mainloop_consumer_state, peek_ab_full_status\n'
                '                )\n')
if source.count(initial_wait) != 1:
    raise RuntimeError('Unexpected initial wait count')
variants['reinit'] = source.replace(initial_wait, initial_wait +
    '                accumulators.fill(0.0)\n'
    '                if cutlass.const_expr(self.block_fp8):\n'
    '                    stage_accumulators.fill(0.0)\n')
variants['prologue-barrier'] = source.replace(initial_wait, initial_wait +
    '                self.mma_sync_barrier.arrive_and_wait()\n')
variants['reinit-runtime'] = source.replace(initial_wait, initial_wait +
    '                accumulators.fill(alpha_value * 0.0)\n')
init = '                accumulators.fill(0.0)\n'
if source.count(init) != 1:
    raise RuntimeError('Unexpected accumulator initialization count')
variants['init-runtime-before'] = source.replace(init,
    '                accumulators.fill(alpha_value * 0.0)\n')
variants['init-negative-zero'] = source.replace(init,
    '                accumulators.fill(-0.0)\n')
main_gemm = ('                                    cute.gemm(\n'
             '                                        mma_atom,\n'
             '                                        accumulators[None, _mt, _nt],\n')
tail_gemm = ('                                cute.gemm(\n'
             '                                    mma_atom,\n'
             '                                    accumulators[None, _mt, _nt],\n')
if source.count(main_gemm) != 1 or source.count(tail_gemm) != 1:
    raise RuntimeError('Unexpected native MX GEMM sites')
variants['first-mma-overwrite'] = source.replace(main_gemm,
    '                                    mma_atom.set(WarpField.ACCUMULATE, (k_tile != 0) | (k_block_idx != 0))\n' + main_gemm).replace(tail_gemm,
    '                                mma_atom.set(WarpField.ACCUMULATE, (k_tile_iter_cnt > 1) | (k_block_idx != 0))\n' + tail_gemm)
grid_call = ('        tile_sched_params, grid = self._compute_grid(\n'
             '            c,\n            self.tile_shape_mnk,\n'
             '            max_active_clusters,\n')
if source.count(grid_call) != 1:
    raise RuntimeError('Unexpected grid setup')
variants['grid-all-tiles'] = source.replace(grid_call, grid_call.replace(
    '            max_active_clusters,', '            1048576,'))
variants['grid300'] = source.replace(grid_call, grid_call.replace(
    '            max_active_clusters,', '            300,'))
main_start = source.index('                for k_tile in range(')
advance_start = source.index('                        if k_block_idx == num_k_blocks - 1:\n'
                             '                            mainloop_pipeline.consumer_release', main_start)
advance_end = source.index('                        if cutlass.const_expr(self.a16):', advance_start)
advance_block = source[advance_start:advance_end]
remaining = source[:advance_start] + source[advance_end:]
before_next_load = ('                        if cutlass.const_expr(self.b_packed):\n'
                   '                            # Deferred expansion barrier')
if remaining.count(before_next_load) != 1:
    raise RuntimeError('Unexpected next-stage load marker')
variants['all-release-after-mma'] = remaining.replace(before_next_load,
    advance_block + before_next_load).replace(tail, late_tail)
variants['main-release-after-mma'] = remaining.replace(before_next_load,
    advance_block + before_next_load)
swap_epilogue = source.index('                if cutlass.const_expr(self.swap_ab):', tail_start)
advance_after_store = source.index('                    if cutlass.const_expr(self.single_work_tile_per_cta):', swap_epilogue)
variants['epilogue-barrier'] = (source[:advance_after_store] +
    '                    self.epilog_sync_barrier.arrive_and_wait()\n' + source[advance_after_store:])
variants['proxy-before-release'], count = re.subn(
    r'^( +)mainloop_pipeline.consumer_release\(mainloop_consumer_state\)$',
    r'\1cute.arch.fence_proxy("async.shared", space="cta")\n\1mainloop_pipeline.consumer_release(mainloop_consumer_state)',
    source, flags=re.MULTILINE)
if count != 2:
    raise RuntimeError('Unexpected release fence sites')
variants['group-before-release'], count = re.subn(
    r'^( +)mainloop_pipeline.consumer_release\(mainloop_consumer_state\)$',
    r'\1self.mma_sync_barrier.arrive_and_wait()\n\1mainloop_pipeline.consumer_release(mainloop_consumer_state)',
    source, flags=re.MULTILINE)
if count != 2:
    raise RuntimeError('Unexpected release group barrier sites')
release = 'mainloop_pipeline.consumer_release(mainloop_consumer_state)'
first_release = source.index(release)
last_release = source.rindex(release)
for name, position in [('group-main-release', first_release), ('group-tail-release', last_release)]:
    line_start = source.rfind('\n', 0, position) + 1
    indent = source[line_start:position]
    variants[name] = source[:line_start] + indent + 'self.mma_sync_barrier.arrive_and_wait()\n' + source[line_start:]
for name, replacement in [
    ('fence-before-release', r'\1cute.arch.fence_acq_rel_cta()\n\1mainloop_pipeline.consumer_release(mainloop_consumer_state)'),
    ('fence-after-release', r'\1mainloop_pipeline.consumer_release(mainloop_consumer_state)\n\1cute.arch.fence_acq_rel_cta()'),
]:
    variants[name], count = re.subn(
        r'^( +)mainloop_pipeline.consumer_release\(mainloop_consumer_state\)$',
        replacement, source, flags=re.MULTILINE)
    if count != 2:
        raise RuntimeError('Unexpected release fence sites')
group = '            pipeline.Agent.Thread, self.num_mma_warps\n'
if source.count(group) != 1:
    raise RuntimeError('Unexpected TMA consumer group')
all_lanes = source.replace(group, '            pipeline.Agent.Thread, self.num_mma_warps * self.num_threads_per_warp\n')
variants['all-lanes-release'], count = re.subn(
    r'^( +)mainloop_pipeline.consumer_release\(mainloop_consumer_state\)$',
    r'\1if cutlass.const_expr(self.load_path == "tma"):\n\1    mainloop_pipeline.sync_object_empty.arrive(mainloop_consumer_state.index, mainloop_pipeline.consumer_mask)\n\1else:\n\1    mainloop_pipeline.consumer_release(mainloop_consumer_state)',
    all_lanes, flags=re.MULTILINE)
if count != 2:
    raise RuntimeError('Unexpected all-lanes release sites')
inc = '            cute.arch.setmaxregister_increase(self.mma_register_requirement)'
dec = '            cute.arch.setmaxregister_decrease(self.load_register_requirement)'
if source.count(inc) != 1 or source.count(dec) != 1:
    raise RuntimeError('Unexpected register reallocation sites')
variants['no-reg-realloc'] = source.replace(inc, '            # Diagnostic: no register increase.').replace(dec, '            # Diagnostic: no register decrease.')
if args.only and args.only not in variants:
    raise RuntimeError('Unknown variant')
for name, updated in variants.items():
    if args.only and name != args.only:
        continue
    target = root / f'dense_gemm-{name}.py'
    if target.exists():
        raise RuntimeError('Diagnostic already generated: ' + name)
    compile(updated, str(target), 'exec')
    patch = ''.join(difflib.unified_diff(source.splitlines(keepends=True), updated.splitlines(keepends=True),
                                      fromfile='a/' + path, tofile='b/' + path))
    target.write_text(updated)
    (root / f'dense-{name}.patch').write_text(patch)
    identity = {'base_revision': revision, 'base_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'source_sha256': hashlib.sha256(updated.encode()).hexdigest(),
                'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
                'scope': 'isolated diagnostic only; not a qualified fix'}
    (root / f'dense-{name}.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(name, identity['source_sha256'], flush=True)
