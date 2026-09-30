#!/usr/bin/env python3
"""CPU-only protocol probe using the candidate's actual coordinator source.

The fake job exposes the exact cache-only progress state emitted by the pinned
B12X session before its distributed races. No GPU or live process is touched.
"""
import json
import subprocess
from types import SimpleNamespace

source = subprocess.check_output([
    'git', '-C', '/home/jugs/git/vllm', 'show',
    '8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368:vllm/v1/worker/b12x_startup.py',
], text=True)
module = {}
exec(compile(source, 'pinned/b12x_startup.py', 'exec'), module)
calls = []
progress = SimpleNamespace(pending_compilation=False, done=False,
    ready_collectives=(), ready_tuning=(), ready_cache=object())
job = SimpleNamespace(advance=lambda **kw: calls.append(kw) or progress)
session = SimpleNamespace(begin=lambda requests, autotune: job)
coordinator = module['B12xPreparationCoordinator'](
    session, [([object()], True)], global_rank=0, world_group=None,
    process_local_only=True)
# Avoid torch/distributed imports while retaining the actual multi-rank advance
# path. An exchange would fail the test, rather than silently return a fake result.
coordinator.process_local_only = False
coordinator.world_ranks = (0, 1)
exchanges = []
def unexpected_exchange():
    exchanges.append(True)
    raise RuntimeError('Coordinator observed cache-only progress')
coordinator._exchange = unexpected_exchange
outcomes = [coordinator.advance() for _ in range(100)]
reproduced = (len(calls) == 100 and not exchanges
    and all('cache' not in call for call in calls)
    and all(not item['done'] and item['error'] is None and item['round'] == 0
            for item in outcomes))
print(json.dumps({'cache_only_progress_stalls': reproduced,
    'advance_calls': len(calls), 'control_exchanges': len(exchanges),
    'final_round': outcomes[-1]['round'],
    'scope': 'real coordinator, synthetic B12X protocol progress'}))
if not reproduced:
    raise SystemExit('Expected pinned-source stall was not reproduced')
