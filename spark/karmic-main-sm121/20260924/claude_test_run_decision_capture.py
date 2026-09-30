"""CPU tests for claude_run_decision_capture.py and claude_needle_spans.py (no nodes, no torch).
  .venv-snapshot-cpu/bin/python -m unittest claude_test_run_decision_capture
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile

import claude_run_decision_capture as drv
import claude_needle_spans as ns
from prepare_decision_reference import signature

HERE = Path(__file__).resolve().parent
IMAGE = 'a3178f679c6287b728d867054103ed9e4a1f9815c03928db2472e367a08a77ef'
LOCK = 'lock' * 16
BUILD = {'image_id': IMAGE, 'lock_sha256': LOCK}


def container(node='toby', **changes):
    env = {'DS41_DECISION_ROW_BLOCKS': '80927', 'DS41_NODE': node, 'DS41_KIT_SHA256': 'kit',
           'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '1', 'B12X_DENSE_SPLITK_TURBO': '0',
           'B12X_COMPILE_CACHE_DIR': '/cache/jit/ds41-router-d8af8395f4328ee6df88/b12x'}
    env.update(changes.pop('env', {}))
    info = {'Id': 'abcdef0123456789', 'Image': 'sha256:' + IMAGE, 'State': {'Running': True},
            'Config': {'Labels': {'local-inference.ds41.kit.sha256': 'kit'},
                       'Env': [f'{k}={v}' for k, v in env.items()]}}
    info.update(changes)
    return info


def image(**labels):
    base = {'local-inference.ds41.diagnostic.kind': 'decision-row-capture',
            'local-inference.ds41.diagnostic.lock.sha256': LOCK,
            'vllm.source-tree': 'v-diag', 'b12x.source-tree': 'b-tree'}
    base.update(labels)
    return {'Id': 'sha256:' + IMAGE, 'Config': {'Labels': base}}


class Pure(unittest.TestCase):
    def test_same_boot_controls_must_match_before_arming(self):
        identities = {n: {'kit_sha256': 'kit', 'source_trees': {'vllm': 'v', 'b12x': 'b'}} for n in drv.NODES}
        regimes = {n: {'plan': 'same'} for n in drv.NODES}
        rows = [{'response_signature': 's'}, {'response_signature': 's'}]
        control = drv.matched_control(rows, BUILD, identities, regimes)
        self.assertEqual(control['response_signature'], 's')
        self.assertEqual(control['image_id'], IMAGE)
        for bad in (rows[:1], [rows[0], {'response_signature': 'different'}]):
            with self.assertRaisesRegex(RuntimeError, 'must match before arming'):
                drv.matched_control(bad, BUILD, identities, regimes)
        self.assertIn('--control', drv.audit_argv([], spans=False, control=True))
        self.assertNotIn('--control', drv.audit_argv([], spans=False))

    def test_aborted_capture_does_not_wait_for_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(drv, 'ssh', return_value='CLAUDE-DECISION-ROW aborted token=decision-row-test IndexError: row'):
                with self.assertRaisesRegex(RuntimeError, 'capture helper aborted'):
                    drv.wait_captures('decision-row-test', 900, Path(directory))
            self.assertIn('IndexError', (Path(directory) / 'dusty-helper.log').read_text())

    def test_token_grammar_matches_the_helper(self):
        self.assertEqual(drv.token_for('20260926T031500Z'), 'decision-row-20260926t031500z')
        self.assertTrue(drv.TOKEN.match(drv.token_for('20260926T031500Z')))
        with self.assertRaises(ValueError):
            drv.token_for('bad_stamp!')
        self.assertEqual(drv.trigger_payload('decision-row-x'),
                         {'token': 'decision-row-x', 'prompt_tokens': 524288, 'chunk_rows': 8192})   # the hook's historical default

    def test_container_checks_pass_and_fail_closed(self):
        problems, identity = drv.check_container('toby', container(), image(), BUILD, LOCK)
        self.assertEqual(problems, [])
        self.assertEqual(identity, {'rank': 1, 'kit_sha256': 'kit',
                                    'cache_dir': '/cache/jit/ds41-router-d8af8395f4328ee6df88/b12x',
                                    'source_trees': {'vllm': 'v-diag', 'b12x': 'b-tree'}})
        cases = {
            'not running': (container(State={'Running': False}), image()),
            'not the capture build': (container(Image='sha256:' + 'e' * 64), image()),
            'KV pin': (container(env={'DS41_DECISION_ROW_BLOCKS': '81592'}), image()),
            'DS41_NODE': (container(node='rusty'), image()),
            'repaired runtime': (container(env={'B12X_DENSE_SPLITK_TURBO': '1'}), image()),
            'prefetch override': (container(env={'VLLM_DS41_L2_PREFETCH': '0'}), image()),
            'kit digest': (container(env={'DS41_KIT_SHA256': 'other'}), image()),
            'decision-row capture': (container(), image(**{'local-inference.ds41.diagnostic.kind': 'engram-fault-inject'})),
            'lock differs': (container(), image(**{'local-inference.ds41.diagnostic.lock.sha256': 'x' * 64})),
        }
        for needle, (c, i) in cases.items():
            problems, _ = drv.check_container('toby', c, i, BUILD, LOCK)
            self.assertTrue(any(needle in p for p in problems), (needle, problems))
        missing = container()
        missing['Config']['Env'] = [e for e in missing['Config']['Env'] if not e.startswith('DS41_DECISION_ROW_BLOCKS')]
        self.assertTrue(any('KV pin' in p for p in drv.check_container('toby', missing, image(), BUILD, LOCK)[0]))

    def test_blocks_line_and_paths(self):
        self.assertTrue(drv.blocks_pinned('x\n(Worker_TP0 pid=414) INFO ' + drv.BLOCKS_LINE + '\n'))
        self.assertFalse(drv.blocks_pinned(drv.BLOCKS_LINE.replace('80927', '81592')))
        self.assertEqual(drv.selection_host_path('/cache/jit/ns/b12x'),
                         '/home/jugs/.cache/vllm-jj-ds41-tp4/jit/ns/b12x/' + drv.SELECTION)
        self.assertEqual(drv.host_path('/cache/claude-decision-row/toby-rank1-t.pt'),
                         '/home/jugs/.cache/vllm-jj-ds41-tp4/claude-decision-row/toby-rank1-t.pt')
        with self.assertRaises(ValueError):
            drv.host_path('/tmp/x.pt')

    def test_request_is_the_frozen_gate_request_with_a_fresh_salt(self):
        spec = {'messages': [{'role': 'user', 'content': 'p'}], 'chat_template_kwargs': {'thinking': False}}
        body = drv.request_body(spec, 'salt')
        self.assertEqual(body, {'model': 'DeepSeek-V4.1-Flash', 'messages': spec['messages'],
                                'chat_template_kwargs': {'thinking': False}, 'temperature': 0, 'max_tokens': 64,
                                'logprobs': True, 'top_logprobs': 20, 'cache_salt': 'salt'})
        gate = (HERE / 'qualify_original_needle.py').read_text()
        for needle in ("'max_tokens': 64", "'top_logprobs': 20", "'temperature': 0", "'logprobs': True"):
            self.assertIn(needle, gate)

    def test_response_row_uses_the_reference_signature_helper(self):
        result = {'usage': {'prompt_tokens': 524288},
                  'choices': [{'message': {'content': '739184, 482617'}, 'finish_reason': 'stop',
                               'logprobs': {'content': [{'token': '7', 'logprob': -0.1}]}}]}
        spec = {'expected': '739184, 482617'}
        reference = {'response_signature': signature(result)}
        row = drv.response_row(result, spec, reference)
        self.assertTrue(row['correct'] and row['signature_equals_b2'])
        wrong = copy.deepcopy(result)
        wrong['choices'][0]['message']['content'] = '510c94b1, 739184'
        row = drv.response_row(wrong, spec, reference)
        self.assertFalse(row['correct'] or row['signature_equals_b2'])
        self.assertEqual(row['response_signature'], signature(wrong))

    def test_consumed_receipt_must_name_this_token_and_rank(self):
        text = json.dumps({'file': '/cache/claude-decision-row/toby-rank1-decision-row-a.pt', 'sha256': 'f' * 64, 'problems': []})
        record = drv.parse_consumed(text, 'decision-row-a', 1)
        self.assertEqual(record['host_file'], '/home/jugs/.cache/vllm-jj-ds41-tp4/claude-decision-row/toby-rank1-decision-row-a.pt')
        with self.assertRaises(ValueError):
            drv.parse_consumed(text, 'decision-row-a', 2)
        with self.assertRaises(ValueError):
            drv.parse_consumed(text, 'decision-row-b', 1)

    def test_observed_identity_requires_rank_agreement(self):
        ident = {n: {'rank': i, 'kit_sha256': 'kit', 'cache_dir': '/cache/x',
                     'source_trees': {'vllm': 'v', 'b12x': 'b'}} for i, n in enumerate(drv.NODES)}
        regime = {n: {'swa.extend': {'v41_compute_mode': 'bf16'}} for n in drv.NODES}
        observed = drv.observed_identity(BUILD, ident, regime, 'sig')
        self.assertEqual(observed, {'image_id': IMAGE, 'kit_sha256': 'kit', 'source_trees': {'vllm': 'v', 'b12x': 'b'},
                                    'regime': {'swa.extend': {'v41_compute_mode': 'bf16'}}, 'response_signature': 'sig'})
        self.assertEqual(sorted(observed), sorted(drv.expected_identity(
            {'image_id': 1, 'kit_sha256': 2, 'source_trees': 3, 'regime': 4, 'response_signature': 5, 'extra': 6})))
        split = copy.deepcopy(regime)
        split['kirby']['swa.extend']['v41_compute_mode'] = 'fp8'
        with self.assertRaises(RuntimeError):
            drv.observed_identity(BUILD, ident, split, 'sig')
        other = copy.deepcopy(ident)
        other['rusty']['kit_sha256'] = 'drift'
        with self.assertRaises(RuntimeError):
            drv.observed_identity(BUILD, other, regime, 'sig')

    def test_gather_uses_fabric_addresses_only_and_verifies_hashes(self):
        captures = {n: {'host_file': f'/home/jugs/.cache/vllm-jj-ds41-tp4/claude-decision-row/{n}-rank{i}-t.pt',
                        'sha256': hashlib.sha256(n.encode()).hexdigest(), 'problems': []}
                    for i, n in enumerate(drv.NODES)}
        script = drv.gather_script('/remote/receipts/x', captures)
        self.assertIn('set -euo pipefail', script)
        for node in ('toby', 'rusty', 'kirby'):
            address = drv.CONTRACT_NODES[node][1]
            self.assertTrue(address.startswith('10.11.11.'))
            self.assertIn(f'ip -j route get {address}', script)
            self.assertIn(f'{address}:/home/jugs/.cache/vllm-jj-ds41-tp4/claude-decision-row/{node}-', script)
            self.assertNotIn(f' {node}:', script)                         # never the management hostname
        self.assertIn("r.get('dev')=='enp1s0f0np0' and r.get('prefsrc')=='10.11.11.7'", script)
        self.assertIn('cp --reflink=auto /home/jugs/.cache/vllm-jj-ds41-tp4/claude-decision-row/dusty-rank0-t.pt', script)
        self.assertEqual(script.count('sha256sum -c -'), 4)
        self.assertEqual(script.count('echo GATHERED'), 4)
        self.assertNotIn('rm ', script)
        self.assertNotIn('--remove-source-files', script)

    def test_cpu_containers_expose_no_gpu_and_no_network(self):
        command = drv.cpu_container(IMAGE, '/remote/x', ['/gate/a.py'], memory='24g')
        self.assertNotIn('--device', command)
        self.assertNotIn('nvidia.com/gpu=all', ' '.join(command))
        self.assertIn('--network=none', command)
        self.assertIn('--memory=24g', command)
        self.assertIn('--rm', command)
        self.assertEqual(command[-3:], [IMAGE, '-u', '/gate/a.py'])
        unbounded = drv.cpu_container(IMAGE, '/remote/x', ['/gate/a.py'], memory='none',
                                      hf_mount='/home/jugs/.cache/huggingface')
        self.assertFalse(any(c.startswith('--memory') for c in unbounded))
        self.assertIn('/home/jugs/.cache/huggingface:/root/.cache/huggingface:ro', unbounded)
        argv = drv.audit_argv(['d0.pt', 't1.pt', 'r2.pt', 'k3.pt'], spans=True)
        self.assertEqual(argv[1:5], ['/gate/captures/d0.pt', '/gate/captures/t1.pt', '/gate/captures/r2.pt', '/gate/captures/k3.pt'])
        self.assertIn('--spans', argv)
        self.assertNotIn('--spans', drv.audit_argv(['a'], spans=False))

    def test_local_inputs_are_the_frozen_ones(self):
        self.assertEqual(hashlib.sha256(drv.INPUT.read_bytes()).hexdigest(), drv.INPUT_SHA)
        reference = json.loads(drv.REFERENCE.read_text())
        self.assertEqual((reference['image_id'], reference['proposed_capture_blocks']), (drv.B2_IMAGE, 80927))
        self.assertEqual(json.loads(drv.B2_SELECTION_MANIFEST.read_text())['cache_path'].rsplit('/b12x/', 1)[1], drv.SELECTION)
        self.assertTrue(drv.ENVELOPE.is_file())
        for name in drv.AUDIT_FILES:
            self.assertTrue((HERE / name).is_file(), name)


class Source(unittest.TestCase):
    text = (HERE / 'claude_run_decision_capture.py').read_text()

    def test_driver_never_changes_boot_or_node_state(self):
        for forbidden in ('podman stop', 'podman rm', 'podman build', 'podman load', 'podman exec', 'podman run -d',
                          '--device', 'nvidia.com/gpu', 'rm -rf', 'launch_contract.py', 'start_moe_repaired'):
            self.assertNotIn(forbidden, self.text, forbidden)
        writes = [n for n in ast.walk(ast.parse(self.text)) if isinstance(n, ast.Call)
                  and getattr(n.func, 'id', None) == 'ssh' and n.args and isinstance(n.args[1], ast.JoinedStr)]
        self.assertTrue(writes)                                              # remote commands are explicit strings

    def test_capture_follows_two_matched_cold_controls_and_fails_closed(self):
        tree = ast.parse(self.text)
        capture = ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_capture'))
        self.assertEqual(capture.count('send('), 2)
        self.assertIn('range(2)', capture)
        self.assertLess(capture.index('matched_control('), capture.index('arm(token, armed, '))
        self.assertIn('uuid.uuid4().hex', capture)
        send = ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'send'))
        self.assertIn("raise RuntimeError('Historical replay was not cold", send)
        self.assertIn('idle_snapshot(base_url)', send)
        self.assertIn('/v1/chat/completions', send)
        self.assertNotIn('/v1/models', self.text)
        audit = ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'phase_audit'))
        self.assertIn("podman ps -q", audit)
        self.assertIn('while_serving', audit)


class Matched(unittest.TestCase):
    """--mode matched: 81389 blocks, pinned 045954Z regime, explicit chunk grid equal to the boot threshold."""

    def test_profiles(self):
        hist = drv.profile_for('historical')
        self.assertEqual((hist['blocks'], hist['chunk_rows'], hist['threshold'], hist['tag']), (80927, 8192, None, ''))
        for chunk in (8192, 4096):
            prof = drv.profile_for('matched', chunk)
            self.assertEqual((prof['blocks'], prof['chunk_rows'], prof['threshold'], prof['tag']),
                             (81389, chunk, chunk, f'matched{chunk}-'))
        for bad in (('matched', None), ('matched', 7936), ('historical', 4096), ('other', 8192)):
            with self.assertRaises(ValueError):
                drv.profile_for(*bad)
        self.assertEqual(drv.blocks_line(81389), drv.BLOCKS_LINE.replace('80927', '81389'))
        self.assertTrue(drv.blocks_pinned('x ' + drv.blocks_line(81389), 81389))
        self.assertFalse(drv.blocks_pinned('x ' + drv.blocks_line(81389)))

    def test_tokens_and_trigger_carry_the_grid(self):
        self.assertEqual(drv.token_for('20260926T031500Z', 4096), 'decision-row-m4096-20260926t031500z')
        self.assertTrue(drv.TOKEN.match(drv.token_for('20260926T031500Z', 8192)))
        self.assertEqual(drv.trigger_payload('t', 4096), {'token': 't', 'prompt_tokens': 524288, 'chunk_rows': 4096})
        self.assertEqual(drv.trigger_payload('t'), {'token': 't', 'prompt_tokens': 524288, 'chunk_rows': 8192})
        with self.assertRaises(ValueError):
            drv.trigger_payload('t', 7936)

    def test_container_gate_per_mode(self):
        matched = drv.profile_for('matched', 4096)
        good = container(env={'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '4096'})
        self.assertEqual(drv.check_container('toby', good, image(), BUILD, LOCK, matched)[0], [])
        cases = {
            'KV pin': container(env={'DS41_DECISION_ROW_BLOCKS': '80927', 'DS41_PREFILL_THRESHOLD': '4096'}),
            'matched grid': container(env={'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192'}),
            'matched grid ': container(env={'DS41_DECISION_ROW_BLOCKS': '81389'}),
            'router chunking control': container(env={'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '4096',
                                                      'DS41_CHUNKING_BLOCKS': '81389'}),
        }
        for needle, c in cases.items():
            problems, _ = drv.check_container('toby', c, image(), BUILD, LOCK, matched)
            self.assertTrue(any(needle.strip() in p for p in problems), (needle, problems))
        # the historical gate refuses a threshold and still wants 80927
        problems, _ = drv.check_container('toby', container(env={'DS41_PREFILL_THRESHOLD': '8192'}), image(), BUILD, LOCK)
        self.assertTrue(any('threshold present on a historical' in p for p in problems))
        self.assertEqual(drv.check_container('toby', container(), image(), BUILD, LOCK)[0], [])

    def test_boot_profile_and_api_args_per_mode(self):
        def rendered(node='toby', blocks='81389', threshold='4096', extra=None):
            argv = ['/opt/venv/bin/vllm', 'serve', 'm', '--max-num-batched-tokens', '8192', '--max-num-seqs', '4',
                    '--enable-chunked-prefill', '--num-gpu-blocks-override', blocks]
            env = {'DS41_DECISION_ROW_BLOCKS': blocks}
            if threshold:
                argv += ['--long-prefill-token-threshold', threshold]
                env['DS41_PREFILL_THRESHOLD'] = threshold
            env.update(extra or {})
            return {'node': node, 'rank': 1, 'model': argv, 'env': env}
        matched = drv.profile_for('matched', 4096)
        self.assertEqual(drv.check_boot_profile(rendered(), 'toby', matched), [])
        self.assertTrue(drv.check_boot_profile(rendered(threshold='8192'), 'toby', matched))
        self.assertTrue(drv.check_boot_profile(rendered(blocks='80927'), 'toby', matched))
        self.assertTrue(drv.check_boot_profile(rendered(node='rusty'), 'toby', matched))
        self.assertTrue(drv.check_boot_profile(rendered(extra={'DS41_CHUNKING_BLOCKS': '81389'}), 'toby', matched))
        hist = drv.profile_for('historical')
        self.assertEqual(drv.check_boot_profile(rendered(blocks='80927', threshold=None), 'toby', hist), [])
        self.assertTrue(drv.check_boot_profile(rendered(blocks='80927', threshold='8192'), 'toby', hist))
        args = {'num_gpu_blocks_override': 81389, 'max_num_batched_tokens': 8192, 'long_prefill_token_threshold': 4096,
                'max_num_seqs': 4, 'enable_chunked_prefill': True}
        self.assertEqual(drv.check_api_args(args, matched), [])
        self.assertTrue(drv.check_api_args(dict(args, long_prefill_token_threshold=8192), matched))
        self.assertTrue(drv.check_api_args(dict(args, max_num_batched_tokens=4096), matched))
        self.assertEqual(drv.check_api_args({k: v for k, v in dict(args, num_gpu_blocks_override=80927).items()
                                             if k != 'long_prefill_token_threshold'}, hist), [])

    def test_pinned_regime_is_verified_and_differences_named(self):
        reference = json.loads(drv.REFERENCE.read_text())
        regime, differences = drv.pinned_regime(drv.PINNED_REGIME.read_bytes(), reference)
        self.assertEqual(set(regime), set(reference['regime']))
        self.assertEqual(list(differences), ['ratio2.extend'])
        self.assertEqual(differences['ratio2.extend']['pinned']['v41_compute_mode'], 'fp8')
        self.assertEqual(differences['ratio2.extend']['b2']['v41_compute_mode'], 'bf16')
        with self.assertRaises(RuntimeError):
            drv.pinned_regime(drv.PINNED_REGIME.read_bytes() + b' ', reference)
        record = drv.regime_reference_record(drv.profile_for('matched', 8192), regime, differences)
        self.assertEqual((record['mode'], record['sha256'], record['blocks'], record['chunk_rows']),
                         ('matched-pinned-regime', drv.PINNED_REGIME_SHA256, 81389, 8192))
        self.assertIn('not claimed', record['scope'])

    def test_identity_and_audit_argv_carry_the_grid(self):
        ident = {n: {'rank': i, 'kit_sha256': 'kit', 'cache_dir': '/cache/x',
                     'source_trees': {'vllm': 'v', 'b12x': 'b'}} for i, n in enumerate(drv.NODES)}
        regime = {n: {'swa.extend': {'v41_compute_mode': 'bf16'}} for n in drv.NODES}
        self.assertEqual(drv.observed_identity(BUILD, ident, regime, 'sig', 4096)['chunk_rows'], 4096)
        self.assertNotIn('chunk_rows', drv.observed_identity(BUILD, ident, regime, 'sig'))
        rows = [{'response_signature': 's'}, {'response_signature': 's'}]
        self.assertEqual(drv.matched_control(rows, BUILD, ident, regime, 4096)['chunk_rows'], 4096)
        argv = drv.audit_argv(['a'], spans=False, control=True, regime_reference=True)
        self.assertEqual(argv[-2:], ['--regime-reference', '/gate/regime-reference.json'])
        self.assertNotIn('--regime-reference', drv.audit_argv(['a'], spans=False, control=True))

    def test_cli_modes(self):
        with patch.object(drv, 'phase_capture') as capture:
            drv.main(['--phase', 'preflight', '--mode', 'matched', '--chunk-rows', '4096'])
            self.assertEqual(capture.call_args[0][0].chunk_rows, 4096)
        for argv in (['--mode', 'matched'], ['--mode', 'historical', '--chunk-rows', '4096'], ['--chunk-rows', '7936']):
            with self.assertRaises(SystemExit):
                drv.main(['--phase', 'preflight', *argv])


class FakeTokenizer:
    """Models the deepseek_v41 wrapper: apply_chat_template renders the V4.1 prompt itself (tokenize
    defaults to True) and encoding with add_special_tokens=False yields the same ids. Tokens are the
    space-separated pieces; a newline is its own token unless merge_newline is set."""

    def __init__(self, merge_newline=False):
        self.merge_newline = merge_newline

    def _pieces(self, text):
        pieces, position = [], 0
        for chunk in text.split(' '):
            if not self.merge_newline and '\n' in chunk:
                head, tail = chunk.split('\n', 1)
                for piece in (head, '\n', tail):
                    if piece:
                        pieces.append((position, position + len(piece)))
                    position += len(piece)
            elif chunk:
                pieces.append((position, position + len(chunk)))
                position += len(chunk)
            position += 1
        return pieces

    def _render(self, messages):
        assert [m['role'] for m in messages] == ['user']
        return '<s> <user> ' + messages[0]['content'] + ' <asst> </think>'

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        assert kwargs == {'thinking': False, 'reasoning_effort': 'high'}, kwargs
        text = self._render(messages)
        return text if not tokenize else self(text, add_special_tokens=False, return_offsets_mapping=True)['input_ids']

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens and return_offsets_mapping
        offsets = self._pieces(text)
        return {'input_ids': [text[s:e] for s, e in offsets], 'offset_mapping': offsets}


def fake_spans(tokenizer, spec):
    kwargs = ns.chat_template_kwargs(spec)
    served_ids, text = ns.render(tokenizer, spec['messages'], kwargs)
    offset_ids, offsets = ns.encode_with_offsets(tokenizer, text)
    prefix_ids, _ = ns.render(tokenizer, ns.prefix_messages(spec), kwargs)
    return text, offsets, dict(served_ids=served_ids, offset_ids=offset_ids, prefix_ids=prefix_ids)


CONTENT = ('Archive identity 510c94b1bb4f42d2998093bd6b9c3d99. code: 739184. filler filler\n'
           'The late retrieval code is 482617.\nfiller')
# <s> <user> Archive identity <id>. code: 739184. filler filler \n The late retrieval code is 482617. \n filler <asst> </think>
SPEC = {'messages': [{'role': 'user', 'content': CONTENT}], 'chat_template_kwargs': {'thinking': False},
        'prompt_tokens': 20, 'late_marker_prefix_tokens_including_template': 11}


class Spans(unittest.TestCase):
    def test_launch_defaults_match_the_frozen_serve_arguments(self):
        model = json.loads((HERE / 'upstream-launch.json').read_text())['reference']['model']
        self.assertIn('deepseek_v41', model[model.index('--tokenizer-mode') + 1])
        defaults = {}
        for arg in model:
            if arg.startswith('--default-chat-template-kwargs.'):
                key, value = arg.removeprefix('--default-chat-template-kwargs.').split('=', 1)
                defaults[key] = {'true': True, 'false': False}.get(value, value)
        self.assertEqual(defaults, ns.DEFAULT_CHAT_TEMPLATE_KWARGS)
        self.assertEqual(ns.chat_template_kwargs(SPEC), {'thinking': False, 'reasoning_effort': 'high'})
        self.assertEqual(ns.chat_template_kwargs({'chat_template_kwargs': {'thinking': None}}),
                         ns.DEFAULT_CHAT_TEMPLATE_KWARGS)

    def test_spans_map_markers_to_contiguous_tokens(self):
        text, offsets, ids = fake_spans(FakeTokenizer(), SPEC)
        result = ns.spans(text, offsets, SPEC, **ids)
        self.assertEqual(result, {'identity': [4, 5], 'early_code': [6, 7], 'late_marker': [10, 15], 'late_code': [15, 16]})
        self.assertEqual(len(ids['prefix_ids']), 11)          # 9 tokens before the newline + 2 header tokens
        self.assertEqual(ns.common_prefix(ids['prefix_ids'], ids['served_ids']), 9)
        with self.assertRaises(ValueError):
            ns.spans(text, offsets, dict(SPEC, prompt_tokens=21), **ids)
        with self.assertRaises(ValueError):
            ns.spans(text, offsets, dict(SPEC, late_marker_prefix_tokens_including_template=10), **ids)
        with self.assertRaises(ValueError):
            ns.spans(text, offsets, SPEC, **dict(ids, served_ids=ids['served_ids'][:-1] + ['x']))
        with self.assertRaises(ValueError):
            ns.spans(text + ' 482617', offsets + [(len(text) + 1, len(text) + 7)], dict(SPEC, prompt_tokens=21),
                     **dict(ids, served_ids=ids['served_ids'] + ['482617'], offset_ids=ids['offset_ids'] + ['482617']))

    def test_newline_absorbed_into_the_marker_token_is_tolerated(self):
        spec = dict(SPEC, prompt_tokens=16, late_marker_prefix_tokens_including_template=11)
        text, offsets, ids = fake_spans(FakeTokenizer(merge_newline=True), spec)
        result = ns.spans(text, offsets, spec, **ids)
        self.assertEqual(result['late_marker'], [8, 13])       # the token 'filler\nThe' starts the span
        self.assertEqual(ns.common_prefix(ids['prefix_ids'], ids['served_ids']), 8)
        # A prefix that agrees two tokens short of the marker is not the recorded measurement.
        with self.assertRaises(ValueError):
            ns.spans(text, offsets, spec, **dict(ids, prefix_ids=ids['prefix_ids'][:6] + ['a', 'b', 'c', 'd', 'e']))

    def test_prefix_messages_reproduce_the_historical_before_late_string(self):
        self.assertEqual(ns.prefix_messages(SPEC), [{'role': 'user', 'content': CONTENT.split('\nThe late')[0]}])
        with self.assertRaises(ValueError):
            ns.prefix_messages({'messages': [{'role': 'user', 'content': 'x The late retrieval code is 1'}]})
        with self.assertRaises(ValueError):
            ns.prefix_messages({'messages': [{'role': 'system', 'content': 's'}, SPEC['messages'][0]]})

    def test_historical_input_matches_the_helper_assumptions(self):
        spec = json.loads(drv.INPUT.read_text())
        content = spec['messages'][0]['content']
        for needle in ns.MARKERS.values():
            self.assertEqual(content.count(needle), 1, needle)
        self.assertEqual(ns.prefix_messages(spec)[0]['content'], content.split('\nThe late retrieval code is')[0])
        self.assertEqual(ns.chat_template_kwargs(spec), {'thinking': False, 'reasoning_effort': 'high'})

    def test_helper_source_has_no_fallback_template(self):
        source = ns.__file__ and Path(ns.__file__).read_text()
        self.assertNotIn('chat_template=', source)
        self.assertNotIn('AutoTokenizer', source)
        self.assertIn('from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer', source)


if __name__ == '__main__':
    unittest.main()
