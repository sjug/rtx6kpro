"""Stdlib tests for the DIAGNOSTIC one-shot Engram publication skip; no torch, GPU or nodes."""
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load('prep', HERE / 'claude_prepare_engram_fault_inject.py')
classify = load('classify', HERE / 'claude_classify_fault_inject.py')
TOKEN = 'kirby-rank3-20260925'


class Helper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.trigger = self.dir / 'ds41-engram-fault-inject.json'
        self.inject = load('inject', HERE / 'claude_engram_fault_inject.py')
        self.start = time.time() - 10

    def tearDown(self):
        self.tmp.cleanup()

    def arm(self, **fields):
        spec = dict(token=TOKEN, rank=3, node='kirby', min_tokens=1024, **fields)
        self.trigger.write_text(json.dumps(spec))
        now = time.time()
        os.utime(self.trigger, (now, now))

    def ask(self, epoch=41, tokens=2048, rank=3, node='kirby', imported_at=None):
        with mock.patch('sys.stderr'):
            return self.inject.should_skip(epoch, tokens, trigger=str(self.trigger),
                                           receipt_dir=str(self.dir), rank=rank, node=node,
                                           imported_at=self.start if imported_at is None else imported_at)

    def receipt(self):
        return self.dir / f'ds41-engram-fault-inject-{TOKEN}.consumed'

    def test_absent_trigger_never_fires(self):
        self.assertFalse(self.ask())
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_fires_once_with_durable_receipt(self):
        self.arm()
        self.assertTrue(self.ask(epoch=41))
        receipt = json.loads(self.receipt().read_text())
        self.assertEqual({k: receipt[k] for k in ('action', 'token', 'epoch', 'tokens', 'rank', 'node')},
                         {'action': 'skip-engram-publication', 'token': TOKEN, 'epoch': 41,
                          'tokens': 2048, 'rank': 3, 'node': 'kirby'})
        self.assertFalse(self.ask(epoch=42))                         # once per process
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()),
                         sorted([self.trigger.name, self.receipt().name]))   # no temporary left

    def test_token_is_consumed_across_restarts(self):
        self.arm()
        self.assertTrue(self.ask(epoch=41))
        self.inject = load('inject_restarted', HERE / 'claude_engram_fault_inject.py')
        self.assertFalse(self.ask(epoch=90))
        self.assertEqual(json.loads(self.receipt().read_text())['epoch'], 41)

    def test_stale_trigger_from_earlier_boot_is_ignored(self):
        self.arm()
        self.assertFalse(self.ask(imported_at=time.time() + 5))
        self.assertFalse(self.receipt().exists())

    def test_only_the_named_rank_and_node_fire(self):
        self.arm()
        for rank, node in ((0, 'dusty'), (3, 'dusty'), (2, 'kirby')):
            self.assertFalse(self.ask(rank=rank, node=node))
        self.assertFalse(self.receipt().exists())
        self.assertTrue(self.ask(rank=3, node='kirby'))

    def test_small_jobs_such_as_decode_never_fire(self):
        self.arm()
        for tokens in (1, 8, 170, 1023):
            self.assertFalse(self.ask(tokens=tokens))
        self.assertTrue(self.ask(tokens=1024))

    def test_invalid_triggers_never_fire(self):
        bad = [dict(min_tokens=170), dict(token='UPPER-case-x'), dict(token='short'),
               dict(rank=-1), dict(rank='3'), dict(node=''), dict(extra=1)]
        for fields in bad:
            spec = dict(token=TOKEN, rank=3, node='kirby', min_tokens=1024)
            spec.update(fields)
            self.trigger.write_text(json.dumps(spec))
            os.utime(self.trigger, (time.time(), time.time()))
            self.assertFalse(self.ask(), fields)
        self.trigger.write_text('{not json')
        self.assertFalse(self.ask())
        self.assertFalse(self.receipt().exists())

    def test_receipt_failure_means_no_skip(self):
        self.arm()
        receipts = self.dir / 'readonly'
        receipts.mkdir()
        receipts.chmod(0o500)
        try:
            with mock.patch('sys.stderr'):
                fired = self.inject.should_skip(41, 2048, trigger=str(self.trigger), receipt_dir=str(receipts),
                                                rank=3, node='kirby', imported_at=self.start)
            self.assertFalse(fired)
            self.assertEqual(list(receipts.iterdir()), [])
        finally:
            receipts.chmod(0o700)

    def test_short_writes_still_produce_a_complete_receipt(self):
        self.arm()
        real_write = os.write

        def dribble(fd, data):
            return real_write(fd, bytes(data[:5]))           # at most 5 bytes per call
        with mock.patch.object(self.inject.os, 'write', side_effect=dribble):
            self.assertTrue(self.ask(epoch=41))
        self.assertEqual(json.loads(self.receipt().read_text())['epoch'], 41)

    def test_write_without_progress_means_no_skip_and_no_receipt(self):
        self.arm()
        with mock.patch.object(self.inject.os, 'write', return_value=0):
            self.assertFalse(self.ask())
        self.assertFalse(self.receipt().exists())
        self.assertEqual([p.name for p in self.dir.iterdir()], [self.trigger.name])   # temp removed
        self.assertTrue(self.ask())                            # a later healthy write still fires

    def test_stat_errors_are_fail_safe(self):
        self.arm()
        for error in (PermissionError(13, 'denied'), OSError(5, 'io')):
            with mock.patch.object(self.inject.os, 'stat', side_effect=error):
                self.assertFalse(self.ask())
        self.assertFalse(self.receipt().exists())

    def test_preexisting_receipt_blocks_firing(self):
        self.arm()
        self.receipt().write_text('{}\n')
        self.assertFalse(self.ask())
        self.assertEqual(self.receipt().read_text(), '{}\n')

    def test_parse_floor_matches_rocenante_cap(self):
        self.assertGreater(self.inject.MIN_TOKENS_FLOOR * 6144 * 2, 2 * 1024 * 1024)
        self.assertLessEqual((self.inject.MIN_TOKENS_FLOOR - 1) * 6144 * 2, 2 * 1024 * 1024)


class ModelPatch(unittest.TestCase):
    def setUp(self):
        self.base, self.old, self.new = prep.compose()
        self.image = self.base['base_image_id']

    def method(self, source):
        return next(n for n in ast.walk(ast.parse(source))
                    if isinstance(n, ast.FunctionDef) and n.name == '_start_engram_job')

    def test_base_is_the_router_fence_candidate(self):
        repair = json.loads((HERE / 'engram-repair.lock.json').read_text())
        router_bytes = (HERE / 'router-release.lock.json').read_bytes()
        router = json.loads(router_bytes)
        receipt = json.loads((HERE / 'receipts/router-release-build-receipt.json').read_text())
        # model.py preimage: Engram repair output == router candidate's (unchanged) vLLM model.py
        self.assertEqual(hashlib.sha256(self.old.encode()).hexdigest(),
                         repair['targets'][prep.MODEL_TARGET]['output_sha256'])
        self.assertEqual(router['after']['vllm']['vllm/models/deepseek_v4_1/nvidia/model.py']['sha256'],
                         repair['targets'][prep.MODEL_TARGET]['output_sha256'])
        self.assertEqual(self.image, 'e06df11a8ca18fa514d9f28f67cc691aef296da2eeb22b113a734519853bccd7')
        self.assertEqual(self.image, receipt['image_id'])
        self.assertEqual(self.base['base_lock_sha256'], hashlib.sha256(router_bytes).hexdigest())
        self.assertEqual(self.base['base_lock_sha256'], receipt['lock_sha256'])
        self.assertEqual(self.base['base_parent_image_id'],
                         'ee03505df7a3b5b251d657ab201f62ac9d74eb2aee033cc5a094d7996caab4c5')
        self.assertEqual(self.base['base_router_target'],
                         {'path': 'b12x/gemm/bf16_gemv/_prefill.py',
                          'sha256': 'ca888af42b16eb27be445814154f98a3765cb5b054e7b0b77408f76939e801e8'})

    def test_base_identity_fails_closed(self):
        real = prep.root
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            receipt = json.loads((real / prep.ROUTER_RECEIPT).read_text())
            build = Path(receipt['directory']).name
            (tmp / 'receipts' / build).mkdir(parents=True)
            for name in (prep.ROUTER_LOCK, 'engram-repair.lock.json', prep.ROUTER_RECEIPT,
                         f'receipts/{build}/BUILD-OK', f'receipts/{build}/image-inspect.json'):
                shutil.copy(real / name, tmp / name)
            with mock.patch.object(prep, 'root', tmp):
                self.assertEqual(prep.base_identity()['base_image_id'], self.image)
                (tmp / f'receipts/{build}/BUILD-OK').write_text('0' * 64 + '\n')
                with self.assertRaisesRegex(RuntimeError, 'build gates'):
                    prep.base_identity()
                shutil.copy(real / f'receipts/{build}/BUILD-OK', tmp / f'receipts/{build}/BUILD-OK')
                inspect = json.loads((tmp / f'receipts/{build}/image-inspect.json').read_text())
                inspect[0]['Config']['Labels']['local-inference.ds41.diagnostic.kind'] = 'engram-prequeued-native-io-fix'
                (tmp / f'receipts/{build}/image-inspect.json').write_text(json.dumps(inspect))
                with self.assertRaisesRegex(RuntimeError, 'labels'):
                    prep.base_identity()
                shutil.copy(real / f'receipts/{build}/image-inspect.json', tmp / f'receipts/{build}/image-inspect.json')
                (tmp / prep.ROUTER_LOCK).write_bytes((real / prep.ROUTER_LOCK).read_bytes() + b' ')
                with self.assertRaisesRegex(RuntimeError, 'different router lock'):
                    prep.base_identity()

    def test_decision_precedes_cuda_work_and_only_gates_publication(self):
        body = self.method(self.new).body
        text = [ast.unparse(s) for s in body]
        decide = text.index('skip_publish = _fault_inject.should_skip(epoch, counts[0] if counts else 0)')
        stream = next(i for i, s in enumerate(body) if isinstance(s, ast.With))
        self.assertLess(decide, stream)
        self.assertGreater(decide, text.index('stream = self._engram_stream()'))
        inner = [ast.unparse(s) for s in body[stream].body]
        self.assertTrue(inner[1].startswith('self._engram_job = engram_native.enqueue_lookups('))
        self.assertTrue(inner[2].startswith('if not skip_publish:\n    _publish_engram_epoch[1,]('))
        self.assertEqual(len(inner), 3)

    def test_only_intended_changes(self):
        def normalized(text):
            kept = [l for l in text.splitlines() if '_fault_inject' not in l and 'skip_publish' not in l
                    and not l.strip().startswith('# DIAGNOSTIC ONLY') and 'is queued; True at most once' not in l]
            return ''.join(''.join(kept).split())
        self.assertEqual(normalized(self.new), normalized(self.old))
        added = [l.strip() for l in self.new.splitlines() if l not in self.old.splitlines()]
        self.assertEqual(added, [
            '# DIAGNOSTIC ONLY: one-shot publication skip for the cross-rank fail-closed test.',
            'import vllm.models.deepseek_v4_1.claude_engram_fault_inject as _fault_inject',
            '# DIAGNOSTIC ONLY: decided on the host before any CUDA work of this job',
            '# is queued; True at most once, after a durable receipt names `epoch`.',
            'skip_publish = _fault_inject.should_skip(epoch, counts[0] if counts else 0)',
            'if not skip_publish:',
            # the unchanged publication call, re-indented under the guard:
            '_publish_engram_epoch[(1,)](', 'self._engram_io_status.device_view,',
            'self._engram_epochs[0:1],', 'epoch,'])

    def test_outputs_and_lock_reproduce(self):
        lock = json.loads((HERE / 'claude-engram-fault-inject.lock.json').read_text())
        self.assertEqual(lock['base_image_id'], self.image)
        self.assertEqual({k: lock[k] for k in self.base if k != 'model_sha256'},
                         {k: v for k, v in self.base.items() if k != 'model_sha256'})
        self.assertIn('not numerics', lock['scope'])
        self.assertEqual(lock['status'], 'diagnostic-only-not-built-not-qualified')
        helper = (HERE / prep.HELPER_SOURCE).read_text()
        router = json.loads((HERE / 'router-release.lock.json').read_text())
        self.assertEqual(lock['trees'], {'vllm': prep.vllm_tree_after(self.new, helper),
                                         'b12x': router['trees']['b12x']})
        self.assertNotEqual(lock['trees']['vllm'], router['trees']['vllm'])     # honest: vLLM changed
        self.assertEqual((HERE / 'Dockerfile.claude-engram-fault-inject').read_text(), prep.render_dockerfile(lock))
        ignore = (HERE / 'claude-engram-fault-inject.ignore').read_text()
        self.assertEqual(ignore, '**\n' + ''.join('!' + n + '\n' for n in [*sorted(lock['inputs']),
                                                                           'claude-engram-fault-inject.lock.json']))
        target = lock['targets'][prep.MODEL_TARGET]
        self.assertEqual(target['output_sha256'], hashlib.sha256(self.new.encode()).hexdigest())
        self.assertEqual((HERE / prep.OUTPUT).read_text(), self.new)
        for name, digest in lock['inputs'].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)
        docker = (HERE / 'Dockerfile.claude-engram-fault-inject').read_text()
        copy = [l for l in docker.splitlines() if l.startswith('COPY ')][0].split()[1:-1]
        self.assertEqual(sorted(copy), sorted([*lock['inputs'], 'claude-engram-fault-inject.lock.json']))
        self.assertIn('FROM ' + self.image + '\n', docker)
        for label in (f'base-image="{self.image}"', f'base-lock.sha256="{lock["base_lock_sha256"]}"',
                      f'vllm.source-tree="{lock["trees"]["vllm"]}"', f'b12x.source-tree="{lock["trees"]["b12x"]}"',
                      'diagnostic.kind="engram-fault-inject"', 'status="diagnostic-only-not-qualified"'):
            self.assertIn(label, docker)
        self.assertNotIn('ee03505df7a3b5b251d657ab201f62ac9d74eb2aee033cc5a094d7996caab4c5', docker)

    def test_candidate_sources_untouched(self):
        repair = json.loads((HERE / 'engram-repair.lock.json').read_text())
        for name, digest in repair['inputs'].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)


class Installer(unittest.TestCase):
    def simulate(self, tamper=None):
        """Install into a simulated router-candidate filesystem; returns (first run, second run, model path)."""
        lock = json.loads((HERE / 'claude-engram-fault-inject.lock.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            kit, opt, router = Path(tmp) / 'kit', Path(tmp) / 'opt', Path(tmp) / 'router'
            kit.mkdir()
            router.mkdir()
            for name in [*lock['inputs'], 'claude-engram-fault-inject.lock.json']:
                shutil.copy(HERE / name, kit / name)
            model = opt / prep.MODEL_TARGET
            model.parent.mkdir(parents=True)
            shutil.copy(HERE / prep.MODEL_SOURCE, model)
            fenced = opt / 'b12x' / lock['base_router_target']['path']
            fenced.parent.mkdir(parents=True)
            shutil.copy(HERE / 'router-prefill-release-before.py', fenced)
            (opt / 'b12x/b12x/native.so').write_bytes(b'x')
            shutil.copy(HERE / 'router-release.lock.json', router / 'router-release.lock.json')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            preserved = {str(p): digest(p) for c in ('vllm', 'b12x') for p in (opt / c).rglob('*') if p.is_file()}
            (router / 'preserved-files.json').write_text(json.dumps(preserved, sort_keys=True) + '\n')
            if tamper:
                tamper(opt=opt, router=router, fenced=fenced)
            source = (HERE / 'claude_install_engram_fault_inject.py').read_text()
            script = Path(tmp) / 'install.py'
            script.write_text(source.replace("Path('/opt/ds41-engram-fault-inject')", f"Path({str(kit)!r})")
                              .replace("Path('/opt/jovian-judgement')", f"Path({str(opt)!r})")
                              .replace("Path('/opt/ds41-router-release')", f"Path({str(router)!r})"))
            run = lambda: subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
            first = run()
            second = run() if first.returncode == 0 else None
            return first, second, model.read_text()

    def test_simulated_install_is_confined(self):
        first, again, model = self.simulate()
        self.assertIn('ENGRAM-FAULT-INJECT-INSTALL-PASS', first.stdout, first.stderr)
        self.assertEqual(model, (HERE / prep.OUTPUT).read_text())
        self.assertNotEqual(again.returncode, 0)                          # helper now exists: refused
        self.assertIn('preexisting diagnostic file', again.stderr)

    def test_non_router_bases_are_refused(self):
        cases = {
            'router lock differs': lambda opt, router, fenced: (router / 'router-release.lock.json').write_text('{}'),
            'router source differs': lambda opt, router, fenced: shutil.copy(
                HERE / 'claude-pinned-bf16_gemv-_prefill.py', fenced),       # the unfenced parent source
            'inventory differs': lambda opt, router, fenced: (opt / 'vllm/extra.py').write_text('x'),
        }
        for message, tamper in cases.items():
            first, _, _ = self.simulate(tamper)
            self.assertNotEqual(first.returncode, 0, message)
            self.assertIn(message, first.stderr)


class Classifier(unittest.TestCase):
    RECEIPT = {'epoch': 57, 'tokens': 2048, 'rank': 3, 'node': 'kirby'}
    FAULT = ('RuntimeError: Model reported a fault in this step (fault epoch 57, step epoch 57): '
             'Engram rows were consumed before publication')
    REJECTED = {'status': 500, 'elapsed_s': 5.4, 'body': {'error': 'EngineDeadError'}}
    SERVED = {'status': 200, 'elapsed_s': 5.3, 'body': {'choices': [{'message': {'content': 'hi'}}]}}

    def verdict(self, receipt, log, response, tokens=2048):
        return classify.classify(receipt, log, response, tokens)[0]

    def test_rejection_must_be_proven(self):
        r, f = self.RECEIPT, self.FAULT
        served = lambda message: {'status': 200, 'elapsed_s': 5.2, 'body': {'choices': [{'message': message}]}}
        self.assertEqual(self.verdict(r, f, served({'content': '', 'reasoning_content': 'thinking'})), 'FAIL')
        self.assertEqual(self.verdict(r, f, served({'content': None, 'tool_calls': [{'id': 'x'}]})), 'FAIL')
        self.assertEqual(self.verdict(r, f, {'status': 200, 'body': {'choices': [{'text': 'x'}]}}), 'FAIL')
        for body in ({}, {'choices': []}, {'choices': [{'message': {'content': ''}}]}, 'data: [DONE]', None):
            self.assertEqual(self.verdict(r, f, {'status': 200, 'elapsed_s': 5.1, 'body': body}),
                             'INCONCLUSIVE', body)
        self.assertEqual(self.verdict(r, f, {'status': 503, 'body': 'Service Unavailable'}), 'PASS')
        self.assertEqual(self.verdict(r, f, {'status': None, 'error': 'connection reset by peer'}), 'PASS')
        self.assertEqual(self.verdict(r, f, {'status': None, 'error': ''}), 'INCONCLUSIVE')
        self.assertEqual(self.verdict(r, f, {'status': 302, 'body': ''}), 'INCONCLUSIVE')

    def test_first_model_fault_must_be_the_injected_one(self):
        r, f = self.RECEIPT, self.FAULT
        earlier_other = f.replace('57', '12') + '\n'
        earlier_step = f.replace('this step', 'an earlier step').replace('epoch 57)', 'epoch 60)') + '\n'
        self.assertEqual(self.verdict(r, earlier_other + f, self.REJECTED), 'CONFOUNDED')
        self.assertEqual(self.verdict(r, earlier_step + f, self.REJECTED), 'CONFOUNDED')
        self.assertEqual(self.verdict(r, f + '\n' + earlier_step, self.REJECTED), 'PASS')   # later echoes

    def test_verdicts(self):
        r, f = self.RECEIPT, self.FAULT
        self.assertEqual(self.verdict(r, 'x\n' + f + '\n', self.REJECTED), 'PASS')
        self.assertEqual(self.verdict(r, f + '\nNCCL error: remote process exited', self.REJECTED), 'PASS')
        self.assertEqual(self.verdict(r, f, self.SERVED), 'FAIL')
        self.assertEqual(self.verdict(r, 'RoCE collective on rank 0 timed out waiting for rank 3\n' + f,
                                      self.REJECTED), 'CONFOUNDED')
        self.assertEqual(self.verdict(r, f.replace('57', '58'), self.REJECTED), 'CONFOUNDED')
        self.assertEqual(self.verdict(r, f.replace('this step', 'an earlier step'), self.REJECTED), 'CONFOUNDED')
        self.assertEqual(self.verdict(r, 'TimeoutError: disk batch read has not completed', self.REJECTED),
                         'CONFOUNDED')
        self.assertEqual(self.verdict(None, '', self.SERVED), 'NOT-FIRED')
        self.assertEqual(self.verdict(None, f, self.REJECTED), 'NOT-FIRED')
        self.assertEqual(self.verdict(r, f, self.REJECTED, tokens=385), 'MISTARGETED')
        self.assertEqual(self.verdict(r, f, self.REJECTED, tokens=4096), 'PASS')   # chunked prefill
        self.assertEqual(self.verdict(r, f, self.REJECTED, tokens=2000), 'PASS')   # padded job
        self.assertEqual(self.verdict(r, 'engine stopped', self.REJECTED), 'INCONCLUSIVE')
        self.assertEqual(self.verdict(r, 'DS41-IO-URING-BYTE-PARITY-PASS io_uring ready\n' + f,
                                      self.REJECTED), 'PASS')        # benign startup text is not a confounder

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'receipt').write_text(json.dumps(self.RECEIPT))
            (tmp / 'log').write_text(self.FAULT)
            (tmp / 'response').write_text(json.dumps(self.REJECTED))
            args = [sys.executable, str(HERE / 'claude_classify_fault_inject.py'), '--receipt',
                    str(tmp / 'receipt'), '--rank0-log', str(tmp / 'log'), '--response', str(tmp / 'response'),
                    '--prompt-tokens', '2048']
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(json.loads(result.stdout)['verdict'], 'PASS')
            (tmp / 'response').write_text(json.dumps(self.SERVED))
            self.assertEqual(subprocess.run(args, capture_output=True).returncode, 1)


if __name__ == '__main__':
    unittest.main()
