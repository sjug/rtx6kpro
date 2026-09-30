import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from prepare_cache_backport import generate, port, PATHS

ROOT = Path(__file__).resolve().parent

class CacheBackportTests(unittest.TestCase):
    def test_generated_patch_and_lock(self):
        generated = generate()
        self.assertEqual(generated['patch'], (ROOT / 'cache-agreement.patch').read_text())
        self.assertEqual(generated['lock'], json.loads((ROOT / 'cache-agreement.lock.json').read_text()))

    def test_actual_coordinator_cache_progress_red_green(self):
        original, fixed, _, _ = port(PATHS[0])
        for source, expect_done in [(original, False), (fixed, True)]:
            with self.subTest(fixed=expect_done):
                module = {}
                exec(compile(source, 'pinned-coordinator', 'exec'), module)
                requirement = SimpleNamespace(ranks=(0, 1))
                calls = []
                def advance(**kw):
                    calls.append(kw)
                    supplied = kw.get('cache') is not None
                    return SimpleNamespace(pending_compilation=False, done=supplied,
                        ready_collectives=(), ready_tuning=(), ready_cache=None if supplied else requirement)
                job = SimpleNamespace(advance=advance, result=lambda: SimpleNamespace(close=lambda: None), close=lambda: None)
                session = SimpleNamespace(begin=lambda *a, **kw: job, close=lambda: None)
                c = module['B12xPreparationCoordinator'](session, [([object()], True)],
                    global_rank=0, world_group=None, process_local_only=True)
                c.process_local_only = False
                c.world_ranks = (0, 1)
                exchanges = []
                def exchange():
                    exchanges.append(True)
                    return dict(caches={(0, 1): (requirement, requirement)}, stop=False,
                        error=None, collective=None, tuning=(), done=c._local_done)
                c._exchange = exchange
                outcome = None
                for _ in range(4):
                    outcome = c.advance()
                self.assertEqual(outcome['done'], expect_done)
                self.assertEqual(bool(exchanges), expect_done)
                self.assertEqual(any(x.get('cache') is not None for x in calls), expect_done)

if __name__ == '__main__':
    unittest.main()
