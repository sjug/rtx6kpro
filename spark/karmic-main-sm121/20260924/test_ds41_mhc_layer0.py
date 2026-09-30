"""Local tests for the standalone layer-0 mHC comparison (no GPU, no nodes).

The fp64 reference is checked against B12X's own pinned references (a7d7d29b
b12x/testing/mhc.py pre_reference and tests/norm/test_mhc_lagged.py _lagged_reference, read with
git show from ~/git/b12x), and the DS4.1 layer-0 broadcast form against four identical streams.

    .venv-snapshot-cpu/bin/python -m unittest test_ds41_mhc_layer0
"""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import torch
import torch.nn.functional as F

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
import claude_compare_selections  # noqa: E402,F401  (before the replay strips the kit dir from sys.path)
import ds41_mhc_layer0_manifest as manifest_tool  # noqa: E402
import ds41_mhc_layer0_replay as replay  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402

B12X = Path.home() / 'git/b12x'
PARAMS = {'rms_eps': 1e-20, 'hc_eps': 1e-6, 'sinkhorn_iters': 20}


def pinned(path):
    return subprocess.run(['git', '-C', str(B12X), 'show', f'a7d7d29b:{path}'], capture_output=True, text=True,
                          check=True).stdout


def pinned_references():
    namespace = {}
    exec(compile(pinned('b12x/testing/mhc.py'), 'b12x/testing/mhc.py', 'exec'), namespace)
    source = pinned('tests/norm/test_mhc_lagged.py')
    start = source.index('def _lagged_reference')
    end = source.index('\n@', start)
    lagged = {'torch': torch, 'F': F, '_mhc_pre_reference': namespace['pre_reference']}
    exec(compile(source[start:end], 'test_mhc_lagged.py', 'exec'), lagged)
    return namespace, lagged['_lagged_reference']


def inputs(tokens=5, hidden=256, seed=1):
    g = torch.Generator().manual_seed(seed)
    residual = (torch.randn((tokens, 4, hidden), generator=g) / 3).to(torch.bfloat16)
    fn = torch.randn((24, 4 * hidden), generator=g) / 64
    scale = torch.randn((3,), generator=g) / 3
    base = torch.randn((24,), generator=g) / 5
    weight = torch.linspace(0.5, 1.5, hidden).to(torch.bfloat16)
    incoming = torch.softmax(torch.randn((tokens, 4), generator=g), -1)
    return residual, fn, scale, base, weight, incoming


class Reference(unittest.TestCase):
    def test_matches_pinned_b12x_reference(self):
        _, lagged = pinned_references()
        residual, fn, scale, base, weight, incoming = inputs()
        mine = replay.lagged_reference(residual, fn, scale, base, incoming, weight, **PARAMS)
        post, comb, y, predicted = lagged(residual, fn, scale, base, incoming, weight)  # float32 as pinned
        for name, want in (('post', post), ('comb', comb), ('pre_out', predicted)):
            torch.testing.assert_close(mine[name].float(), want, rtol=2e-5, atol=4e-5, msg=name)
        # y: both round the collapsed stream to bf16 first; the final bf16 may differ by one ulp.
        torch.testing.assert_close(mine['y'].to(torch.bfloat16).float(), y.float(), rtol=2e-5, atol=0.008)
        # A deliberate change must be caught (non-separable: Sinkhorn ignores row/column shifts).
        wrong = replay.lagged_reference(residual, fn, scale, base + torch.randn((24,), generator=torch.Generator().manual_seed(7)) / 20, incoming, weight, **PARAMS)
        for name, want in (('post', post), ('comb', comb), ('pre_out', predicted)):
            with self.assertRaises(AssertionError, msg=name):
                torch.testing.assert_close(wrong[name].float(), want, rtol=2e-5, atol=4e-5)

    def test_layer0_broadcast_equals_four_identical_streams(self):
        residual, fn, scale, base, weight, _ = inputs()
        x = residual[:, 0]
        e0 = torch.zeros((x.shape[0], 4))
        e0[:, 0] = 1
        served, full = replay.references(x, replay.broadcast_fn(fn, x.shape[-1]), fn, scale, base, e0, weight, PARAMS)
        extra, _ = replay.references(x, replay.broadcast_fn(fn, x.shape[-1]), fn, scale, base,
                                     e0, weight, dict(PARAMS, engram_layer_ids=[1, 14]))
        for name in replay.OUTPUTS:
            self.assertTrue(torch.equal(served[name], extra[name]))
        for name in replay.OUTPUTS:
            torch.testing.assert_close(served[name], full[name], rtol=1e-6, atol=1e-9, msg=name)
        torch.testing.assert_close(served['y'].to(torch.bfloat16),
                                   (x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-20)
                                    * weight.double()).to(torch.bfloat16), rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, 'incoming mix'):
            replay.references(x, replay.broadcast_fn(fn, x.shape[-1]), fn, scale, base, e0.roll(1, 1), weight, PARAMS)

    def test_broadcast_matches_serving_formula(self):
        _, fn, *_ = inputs()
        self.assertTrue(torch.equal(replay.broadcast_fn(fn, 256), fn.detach().view(-1, 4, 256).sum(dim=1)))


class Propagation(unittest.TestCase):
    def test_matches_pinned_post_and_collapse(self):
        namespace, lagged = pinned_references()
        residual, fn, scale, base, weight, _ = inputs(tokens=6)
        emb = residual[:, 0]
        e0 = torch.zeros((6, 4))
        e0[:, 0] = 1
        served, _ = replay.references(emb, replay.broadcast_fn(fn, 256), fn, scale, base, e0, weight, PARAMS)
        x = (torch.randn((6, 256), generator=torch.Generator().manual_seed(3)) / 2).to(torch.bfloat16)
        ffn_norm = torch.linspace(1.5, 0.5, 256).to(torch.bfloat16)
        post, comb, pre_out = (served[n].float() for n in ('post', 'comb', 'pre_out'))
        got = replay.propagate(post, comb, pre_out, x, emb, ffn_norm, 1e-20)
        streams = emb[:, None, :].expand(-1, 4, -1).contiguous()
        want_residual = namespace['post_reference'](x, streams, post, comb)
        self.assertTrue(torch.equal(got['residual'], want_residual))
        collapsed = (pre_out.unsqueeze(-1) * want_residual.float()).sum(1).bfloat16().float()
        want_y = (collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + 1e-20) * ffn_norm.float())
        self.assertTrue(torch.equal(got['ffn_in'], want_y.bfloat16()))

    def test_counts(self):
        residual, fn, scale, base, weight, _ = inputs(tokens=200)
        emb = residual[:, 0]
        e0 = torch.zeros((200, 4))
        e0[:, 0] = 1
        served, _ = replay.references(emb, replay.broadcast_fn(fn, 256), fn, scale, base, e0, weight, PARAMS)
        side = {n: served[n].float() for n in ('post', 'comb', 'pre_out')}
        x = torch.randn((128, 256), generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)
        ffn_norm = torch.ones(256, dtype=torch.bfloat16)
        same = replay.propagation({'passing': side, 'release': dict(side)}, x, emb[-128:], ffn_norm, 1e-20)
        self.assertEqual((same['residual']['changed_elements'], same['ffn_in']['changed_elements']), (0, 0))
        moved = dict(side, pre_out=side['pre_out'].clone())
        moved['pre_out'][-1, 1] += 1e-3  # last captured row only; the tail slice must be aligned
        diff = replay.propagation({'passing': side, 'release': moved}, x, emb[-128:], ffn_norm, 1e-20)
        self.assertEqual(diff['residual']['changed_elements'], 0)
        self.assertEqual(diff['ffn_in']['rows_changed'], 1)


class Inputs(unittest.TestCase):
    def test_locate(self):
        names = ['embed.weight', 'layers.0.hc_attn_fn', 'layers.0.hc_attn_scale', 'layers.0.hc_attn_base',
                 'layers.0.attn_norm.weight', 'layers.10.hc_attn_fn', 'layers.1.hc_attn_fn', 'mtp.0.hc_attn_fn',
                 'layers.0.engram.embed.weight']
        self.assertEqual(replay.locate(names, 'layer_fn', 0), 'layers.0.hc_attn_fn')
        self.assertEqual(replay.locate(names, 'embed'), 'embed.weight')
        with self.assertRaisesRegex(RuntimeError, 'matched 0'):
            replay.locate(names, 'layer_fn', 3)
        with self.assertRaisesRegex(RuntimeError, 'matched 2'):
            replay.locate(names + ['model.layers.0.hc_attn_fn'], 'layer_fn', 0)

    def test_config_and_tokens(self):
        cfg = {'hidden_size': 5120, 'hc_mult': 4, 'rms_norm_eps': 1e-20, 'hc_eps': 1e-6, 'hc_sinkhorn_iters': 20,
               'engram_layer_ids': [2, 15]}
        self.assertEqual(replay.check_config(cfg, {}), dict(PARAMS, engram_layer_ids=[2, 15]))
        nested = {'model_type': 'deepseek_v41',
                  'text_config': dict(cfg, model_type='deepseek_v41_text')}
        self.assertEqual(replay.check_config(nested, {}), replay.check_config(cfg, {}))
        with self.assertRaisesRegex(RuntimeError, 'text model type'):
            replay.check_config(dict(nested, text_config=dict(cfg, model_type='other')), {})
        for bad, message in ((dict(cfg, engram_layer_ids=[0, 7]), 'engram'), (dict(cfg, hc_mult=2), 'hc_mult')):
            with self.assertRaisesRegex(RuntimeError, message):
                replay.check_config(bad, {})
        m = {'prompt_tokens': 10, 'chunk_rows': 4}
        self.assertEqual(replay.final_chunk(list(range(10)), m), [6, 7, 8, 9])
        with self.assertRaises(RuntimeError):
            replay.final_chunk(list(range(9)), m)
        good = {'tokens': list(range(524288)), 'count': 524288}
        self.assertEqual(len(manifest_tool.check_tokens(good)), 524288)
        for bad in ({'tokens': list(range(524287)), 'count': 524287}, dict(good, count=1),
                    {'tokens': [-1] + list(range(524287)), 'count': 524288}):
            with self.assertRaises(RuntimeError):
                manifest_tool.check_tokens(bad)

    def test_tokenize_body_is_the_gate_request(self):
        spec = json.loads(manifest_tool.NEEDLE.read_text())
        body = manifest_tool.tokenize_body()
        self.assertEqual(body['messages'], spec['messages'])
        self.assertEqual(body['chat_template_kwargs'], spec['chat_template_kwargs'])
        self.assertTrue(body['add_generation_prompt'])

    def test_manifest_from_receipts(self):
        built = manifest_tool.build()
        from claude_compare_selections import mhc_key
        self.assertEqual(built['keys'], {'pre': mhc_key('pre', 8192), 'pre_expanded': mhc_key('pre', 8192, True)})
        self.assertEqual(built['keys']['pre'], '2394c70075c7fe2b87a1bd135e1ba48faaf75dfbb048b7f683b8df73d6de75c3')
        self.assertEqual((built['configs']['passing']['pre']['partials_per_cta'],
                          built['configs']['release']['pre']['partials_per_cta']), (25, 13))
        self.assertEqual((built['configs']['passing']['pre_expanded']['projection_k_splits'],
                          built['configs']['release']['pre_expanded']['projection_k_splits']), (8, 1))
        self.assertEqual(built['capture']['sha256'], '8e5c8a4203e59ada7048ddfad5c845634d63a21294bcda8078b1f1e4d7408032')
        self.assertEqual(built['capture']['positions'], [524160, 524287])
        tol = built['tolerances']
        self.assertEqual(set(tol['reference']), set(replay.OUTPUTS))
        self.assertEqual(tol['reference']['post'], {'atol': 1e-6, 'rtol': 1e-6})
        self.assertEqual(tol['config_parity']['bf16'], {'atol': 0.0, 'rtol': 0.0})
        self.assertIn("'ffn_norm'", Path(replay.__file__).read_text())

    def test_tolerances_are_b12x_asserts(self):
        source = pinned('tests/norm/test_mhc_lagged.py')
        for text in ('rtol=1e-6, atol=1e-6', 'rtol=2e-6, atol=2e-6', 'rtol=0, atol=0'):
            self.assertIn(text, source)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class Source(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        files = {'b12x/b12x/norm/mhc/_impl.py': b'impl', 'vllm/vllm/models/deepseek_v4_1/b12x_layers.py': b'layers',
                 'b12x/b12x/other.py': b'other', 'vllm/vllm/_C.so': b'elf'}
        for rel, data in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(data)
        after = {'b12x': {'b12x/norm/mhc/_impl.py': {'sha256': sha256(b'impl')}, 'b12x/other.py': {'sha256': sha256(b'other')}},
                 'vllm': {'vllm/models/deepseek_v4_1/b12x_layers.py': {'sha256': sha256(b'layers')}}}
        lock = json.dumps({'trees': {'b12x': 't1', 'vllm': 't2'}, 'after': after}).encode()
        (root / 'lock.json').write_bytes(lock)
        source = {'image_lock': str(root / 'lock.json'), 'lock_sha256': sha256(lock), 'trees': {'b12x': 't1', 'vllm': 't2'},
                  'root': str(root), 'pythonpath': f'{root}/vllm:{root}/b12x',
                  'required': {'b12x': {'b12x/norm/mhc/_impl.py': sha256(b'impl')},
                               'vllm': {'vllm/models/deepseek_v4_1/b12x_layers.py': sha256(b'layers')}}}
        return source, after

    @staticmethod
    def read(path):
        try:
            return Path(path).read_bytes()
        except OSError:
            return None

    def test_source_problems(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = self.fixture(tmp)
            env = {'PYTHONPATH': source['pythonpath']}
            good_path = ['', f'{tmp}/vllm', f'{tmp}/b12x', '/usr/lib/python3.12', '/opt/venv/lib/python3.12/site-packages']
            self.assertEqual(replay.source_problems(source, env, good_path, self.read), [])
            cases = [
                ({'PYTHONPATH': '/gate'}, good_path, 'PYTHONPATH'),
                ({'PYTHONPATH': f'/gate:{source["pythonpath"]}'}, good_path, 'PYTHONPATH'),
                (env, ['', f'{tmp}/b12x', f'{tmp}/vllm'], 'serving order'),
                (env, ['/opt/venv/lib/python3.12/site-packages', f'{tmp}/vllm', f'{tmp}/b12x'], 'installed-package'),
                (env, [str(replay.HERE)] + good_path, 'kit directory'),
            ]
            for environ, path, message in cases:
                self.assertTrue(any(message in p for p in replay.source_problems(source, environ, path, self.read)), message)
            (Path(tmp) / 'b12x/b12x/norm/mhc/_impl.py').write_bytes(b'patched')
            self.assertTrue(any('_impl.py' in p for p in replay.source_problems(source, env, good_path, self.read)))
            (Path(tmp) / 'lock.json').write_bytes(b'{}')
            self.assertTrue(any('release lock' in p for p in replay.source_problems(source, env, good_path, self.read)))

    def test_registry_pseudofiles_are_not_source_files(self):
        import torch
        from types import SimpleNamespace
        fake = SimpleNamespace(__file__='/gate/b12x/__init__.py')
        modules = {'torch.ops': torch.ops, 'torch.classes': torch.classes,
                   'b12x.shadow': fake, 'sys': sys}
        files = replay.module_files(modules)
        self.assertNotIn('torch.ops', files)
        self.assertNotIn('torch.classes', files)
        self.assertEqual(files['b12x.shadow'], '/gate/b12x/__init__.py')
        self.assertEqual(replay.module_files({'torch.ops': fake})['torch.ops'], '/gate/b12x/__init__.py')

    def test_loaded_problems(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, after = self.fixture(tmp)
            ok = {'b12x.norm.mhc._impl': f'{tmp}/b12x/b12x/norm/mhc/_impl.py', 'vllm._C': f'{tmp}/vllm/vllm/_C.so',
                  'torch': '/opt/venv/lib/python3.12/site-packages/torch/__init__.py', 'sys': None,
                  'ds41_mhc_layer0_replay': str(replay.HERE / 'ds41_mhc_layer0_replay.py'),
                  'ds41_mhc_layer0_selection_helper': str(replay.HERE / 'claude_compare_selections.py')}
            problems, unlisted = replay.loaded_problems(source, ok, after, self.read)
            self.assertEqual((problems, unlisted), ([], ['vllm/vllm/_C.so']))
            for extra, message in (
                    ({'b12x.shadow': '/gate/b12x/__init__.py'}, 'outside the source root'),
                    ({'b12x': '/opt/venv/lib/python3.12/site-packages/b12x/__init__.py'}, 'outside the source root'),
                    ({'runtime': str(replay.HERE / 'runtime.py')}, 'from the kit'),
                    ({'b12x.new': f'{tmp}/b12x/b12x/unlocked.py'}, 'not the release lock source')):
                found, _ = replay.loaded_problems(source, dict(ok, **extra), after, self.read)
                self.assertTrue(any(message in p for p in found), message)
            (Path(tmp) / 'b12x/b12x/other.py').write_bytes(b'changed')
            found, _ = replay.loaded_problems(source, dict(ok, **{'b12x.other': f'{tmp}/b12x/b12x/other.py'}), after, self.read)
            self.assertTrue(any('b12x/b12x/other.py' in p for p in found))

    def test_import_path_hygiene_and_helper(self):
        self.assertNotIn(str(replay.HERE), [str(Path(p or '.').resolve()) for p in sys.path if p])
        module = replay.helper(manifest_tool.sha((KIT / 'claude_compare_selections.py').read_bytes()))
        self.assertEqual(module.__name__, 'ds41_mhc_layer0_selection_helper')
        self.assertEqual(module.mhc_key('pre', 8192), claude_compare_selections.mhc_key('pre', 8192))
        with self.assertRaisesRegex(RuntimeError, 'helper differs'):
            replay.helper('0' * 64)

    def test_documented_command(self):
        doc = replay.__doc__
        self.assertIn('-e PYTHONPATH=' + manifest_tool.PYTHONPATH + ' ' + chr(92), doc)
        self.assertNotIn('PYTHONPATH=/gate', doc)
        self.assertIn('-P /gate/ds41_mhc_layer0_replay.py', doc)
        self.assertNotRegex(doc, r'-e (B12X|XDG|VLLM|TRITON|TORCHINDUCTOR|CUTE|SPARKINFER)_?[A-Z_]*=')
        self.assertIn(chr(92) + chr(10), doc)  # raw docstring keeps shell continuations
        launch = (KIT / 'launch_contract.py').read_text()
        self.assertIn(f"PYTHONPATH='{manifest_tool.PYTHONPATH}'", launch)

    def test_manifest_source_identity(self):
        built = manifest_tool.source_identity()
        release = json.loads((KIT / manifest_tool.RELEASE_PIN_RECORD).read_text())['candidate']
        self.assertEqual(release['diagnostic']['kind'], 'precision-release-candidate')
        self.assertEqual(built['lock_sha256'], release['diagnostic']['lock_sha256'])
        self.assertEqual(built['trees'], {'b12x': '03a4e0b363d17c291516678689ae72d3e266749e',
                                          'vllm': '0a446838b6cc8d5ad3de0d4e2020e117813d1d57'})
        self.assertIn('b12x/norm/mhc/_impl.py', built['required']['b12x'])
        self.assertIn('vllm/models/deepseek_v4_1/nvidia/model.py', built['required']['vllm'])
        self.assertEqual(json.loads((KIT / 'ds41-mhc-layer0-inputs.json').read_text())['source'], built)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('ds41-precision-release.lock.json', 'ratio1-restore-20260927T015500Z-amendment.json',
                         'claude_compare_selections.py'):
                (Path(tmp) / name).write_bytes((KIT / name).read_bytes())
            (Path(tmp) / 'engram-progress-model.py').write_bytes(b'not the release model')
            with self.assertRaisesRegex(RuntimeError, 'is not engram-progress-model.py'):
                manifest_tool.source_identity(Path(tmp))


class Contract(unittest.TestCase):
    def test_builder_contract_and_mutations(self):
        from claude_compare_selections import MHC_QUERY, mhc_invocation
        for expanded, key in ((False, '2394c70075c7fe2b87a1bd135e1ba48faaf75dfbb048b7f683b8df73d6de75c3'),
                              (True, '1b6c36a14eaadabaa5b558f21ee4755eb16d6f5c5bc7c1f8ff4daae6b64f10a6')):
            invocation = mhc_invocation('pre', expanded)
            query = dict(MHC_QUERY, max_tokens=8192, **invocation)
            self.assertEqual(replay.contract_check(query, invocation, expanded, key), key)
            with self.assertRaisesRegex(RuntimeError, 'query differs'):
                replay.contract_check(dict(query, split_k=64), invocation, expanded, key)
            with self.assertRaisesRegex(RuntimeError, 'invocation differs'):
                replay.contract_check(query, dict(invocation, rms_eps=1e-6), expanded, key)
            with self.assertRaisesRegex(RuntimeError, 'no longer rebuilds'):
                replay.contract_check(query, invocation, expanded, '0' * 64)


class GateAndVerdict(unittest.TestCase):
    def capture(self, rows):
        return {'layers': {0: {'positions': torch.arange(524160, 524288), 'hidden_in': rows.clone()}}}

    def test_gate(self):
        y = torch.randn((8192, 16)).to(torch.bfloat16)
        self.assertTrue(replay.gate(y, self.capture(y[-128:]), (524160, 524288))['passed'])
        bad = y[-128:].clone()
        bad[5, 3] += 1
        result = replay.gate(y, self.capture(bad), (524160, 524288))
        self.assertEqual((result['passed'], result['changed_elements']), (False, 1))
        shifted = self.capture(y[-128:])
        shifted['layers'][0]['positions'] = torch.arange(524159, 524287)
        self.assertFalse(replay.gate(y, shifted, (524160, 524288))['passed'])

    def result(self, *, equal=False, errors=(1e-6, 1e-6), outside=0, gate=True, parity=0, repeat=(True, True)):
        err = lambda e: {n: {'max_abs': e, 'outside_tolerance': outside} for n in replay.OUTPUTS}
        return {'gate': {'passed': gate}, 'repeat_bit_equal': {'passing': repeat[0], 'release': repeat[1]},
                'config_diff': {n: {'bit_equal': equal, 'outside_parity': parity} for n in replay.OUTPUTS},
                'errors': {'passing': err(errors[0]), 'release': err(errors[1])}}

    def test_verdicts(self):
        self.assertEqual(replay.verdict(self.result(gate=False)), 'not-faithful')
        self.assertEqual(replay.verdict(self.result(equal=True)), 'configs-bit-identical')
        self.assertEqual(replay.verdict(self.result()), 'reduction-order-level')
        self.assertEqual(replay.verdict(self.result(errors=(1e-6, 5e-6))), 'asymmetric-error-review')
        self.assertEqual(replay.verdict(self.result(outside=3)), 'outside-operator-tolerance')
        self.assertEqual(replay.verdict(self.result(parity=2)), 'outside-config-parity')
        self.assertEqual(replay.verdict(self.result(parity=2, outside=1)), 'outside-operator-tolerance')
        no_gate = self.result()
        no_gate['gate'] = None
        self.assertEqual(replay.verdict(no_gate), 'reduction-order-level')
        self.assertEqual(replay.verdict(self.result(repeat=(True, False))), 'nonrepeatable')

    def test_failed_gate_outranks_nonrepeatable(self):
        # Regression: a nonrepeatable run with a failed gate was labeled nonrepeatable, and main only
        # tested verdict != not-faithful before interpreting.
        both = self.result(gate=False, repeat=(False, True))
        self.assertEqual(replay.verdict(both), 'not-faithful')
        self.assertFalse(replay.may_continue(both))
        for bad_gate in (False, None, 'true', 1):
            case = self.result(repeat=(False, True))
            case['gate'] = {'passed': bad_gate}
            self.assertFalse(replay.may_continue(case), bad_gate)
            case['verdict'] = 'nonrepeatable'
            self.assertFalse(replay.may_continue(case))
        missing = self.result()
        missing['gate'] = None
        self.assertFalse(replay.may_continue(missing))  # section A must have run its gate
        self.assertFalse(replay.may_continue(self.result(repeat=(True, False))))
        self.assertTrue(replay.may_continue(self.result()))
        empty = self.result()
        empty['repeat_bit_equal'] = {}
        self.assertFalse(replay.may_continue(empty))

    def test_main_continues_only_through_may_continue(self):
        main = Path(replay.__file__).read_text().split('def main():', 1)[1]
        self.assertNotIn("verdict'] != 'not-faithful'", main)
        self.assertEqual(main.count('may_continue('), 1)
        self.assertIn('if proceed:', main)
        self.assertIn("if proceed and params['engram_layer_ids']:", main)

    def test_error_stats_and_diff(self):
        want = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        got = torch.tensor([1.0, 2.0001, 3.0], dtype=torch.float32)
        stats = replay.error_stats(got, want, {'atol': 4e-5, 'rtol': 2e-5})
        self.assertEqual(stats['outside_tolerance'], 1)
        a = {n: torch.zeros(3) for n in replay.OUTPUTS}
        b = {n: torch.zeros(3) for n in replay.OUTPUTS}
        b['comb'][1] = 1e-7
        diff = replay.compare_outputs(a, b, manifest_tool.TOLERANCES['config_parity'])
        self.assertTrue(diff['post']['bit_equal'])
        self.assertEqual((diff['comb']['bit_equal'], diff['comb']['changed_elements']), (False, 1))
        self.assertEqual(diff['comb']['outside_parity'], 0)
        b['comb'][1] = 1e-5
        self.assertEqual(replay.compare_outputs(a, b, manifest_tool.TOLERANCES['config_parity'])['comb']['outside_parity'], 1)
        ya, yb = torch.ones(4, dtype=torch.bfloat16), torch.ones(4, dtype=torch.bfloat16)
        yb[2] = torch.tensor(1.0078125, dtype=torch.bfloat16)  # one bf16 ulp: parity requires bit equality
        a['y'], b['y'] = ya, yb
        self.assertEqual(replay.compare_outputs(a, b, manifest_tool.TOLERANCES['config_parity'])['y']['outside_parity'], 1)


if __name__ == '__main__':
    unittest.main()
