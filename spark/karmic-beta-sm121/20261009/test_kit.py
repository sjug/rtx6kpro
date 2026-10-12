#!/usr/bin/env python3
"""Local, stdlib-only checks of the Karmic beta refresh kit (no GPU, no image)."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOUNDATION = ROOT.parents[1] / 'karmic-main-sm121/20260922'
sys.path.insert(0, str(FOUNDATION))
from contracts import git_tree, manifest

spec = importlib.util.spec_from_file_location('beta_prepare', ROOT / 'prepare.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class KitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = json.loads((ROOT / 'runtime.lock.json').read_text())
        cls.pin = json.loads((ROOT / 'build.lock.json').read_text())
        with tarfile.open(ROOT / 'refresh.tar') as bundle:
            cls.members = {m.name: bundle.extractfile(m).read() for m in bundle.getmembers()}

    def test_preflight_passes(self):
        out = subprocess.run([sys.executable, str(ROOT / 'preflight.py')], check=True,
                             capture_output=True, text=True).stdout
        self.assertIn('KARMIC-BETA-INPUTS-PASS', out)

    def test_prepare_is_deterministic(self):
        before = {n: digest(ROOT / n) for n in ('runtime.lock.json', 'refresh.tar', 'build.lock.json')}
        subprocess.run([sys.executable, str(ROOT / 'prepare.py')], check=True, capture_output=True)
        after = {n: digest(ROOT / n) for n in before}
        self.assertEqual(before, after)

    def test_install_replay_reaches_locked_trees(self):
        # Pure replay of install.py semantics over the independently replayed base.
        bases = {'vllm': prepare.base_vllm(self.lock['sources']['vllm']['base_commit']),
                 'b12x': prepare.previous.source('b12x', self.lock['sources']['b12x']['base_commit'])}
        # The refreshed B12X package must be the FlashInfer-embedded subtree, nothing else.
        inputs = json.loads((ROOT / 'inputs.lock.json').read_text())
        embedded = prepare.b12x_from_flashinfer(inputs)
        root = prepare.root_package(inputs)
        row = self.lock['sources']['b12x']
        self.assertEqual(manifest(root), row['files'])
        self.assertEqual({k for k in embedded if embedded[k] != root[k]}, {'b12x/__init__.py'})
        self.assertEqual(row['root_initializer']['sha256'], digest(ROOT / 'b12x_root_init.py'))
        self.assertEqual(hashlib.sha256(root['b12x/__init__.py'][1]).hexdigest(), row['root_initializer']['sha256'])
        self.assertIn(b'submodule_search_locations=[str(_root)]', root['b12x/__init__.py'][1])
        self.assertEqual(sorted(row['native_changed']['paths']), sorted(prepare.B12X_NATIVE_CHANGED))
        self.assertEqual(row['carrier']['commit'], prepare.FLASHINFER)
        self.assertEqual(self.lock['proxy_abi'], prepare.PROXY_ABI)
        self.assertNotIn('b12x/preparation/_memory.cpp', row['files'])
        self.assertTrue(all(not name.startswith('b12x/_compat/') for name in row['files']))
        for component, files in bases.items():
            row = self.lock['sources'][component]
            self.assertEqual(manifest(files), row['base_files'])
            result = dict(files)
            for name in row['base_files'].keys() - row['files'].keys():
                del result[name]
            for name, entry in row['files'].items():
                if row['base_files'].get(name) != entry:
                    data = self.members[component + '/' + name]
                    self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'])
                    result[name] = (entry['mode'], data)
            self.assertEqual(manifest(result), row['files'])
            self.assertEqual(git_tree(result), row['refreshed_tree'])

    def test_inventory_matches_changes(self):
        expected = {c + '/' + p for c in ('vllm', 'b12x')
                    for p, e in self.lock['sources'][c]['files'].items()
                    if self.lock['sources'][c]['base_files'].get(p) != e}
        self.assertEqual(set(self.members), expected)
        for name in self.members:
            if name.startswith('b12x/') and name[len('b12x/'):] in prepare.B12X_NATIVE_CHANGED:
                continue  # declared B12X C sources (proxy ABI 5, loader reformat), compiled in-image or at first use
            self.assertIn(Path(name).suffix, prepare.SHIPPED_SUFFIXES, name)
            self.assertFalse(name.startswith('vllm/rust/'), name)
        for path in prepare.B12X_NATIVE_CHANGED:
            self.assertIn('b12x/' + path, self.members)
        proxy = self.members['b12x/b12x/comm/roce/_roce_proxy.c']
        self.assertIn(b'#define ROCE_ABI_VERSION 5', proxy)

    def test_rust_parser_stays_at_base(self):
        row = self.lock['sources']['vllm']
        for path in prepare.RUST_DEFERRED:
            self.assertEqual(row['files'].get(path), row['base_files'].get(path), path)
        self.assertEqual(sorted(row['deferred_native']['paths']), sorted(prepare.RUST_DEFERRED))

    def test_dependency_metadata_inputs_are_declared_and_shipped_as_text(self):
        row = self.lock['sources']['vllm']
        self.assertEqual(sorted(row['deferred_metadata']['paths']), sorted(prepare.METADATA_INPUTS))
        for path in prepare.METADATA_INPUTS:
            self.assertNotEqual(row['files'][path], row['base_files'][path], path)
            self.assertIn('vllm/' + path, self.members)
        requirements = self.members['vllm/requirements/cuda.txt'].decode()
        self.assertIn('flashinfer-python==0.7.1', requirements)
        self.assertNotIn('flashinfer-python==0.6.18', requirements)
        self.assertIn('"b12x": ["flashinfer-python[b12x]==0.7.1"]', self.members['vllm/setup.py'].decode())

    def test_sm121_overlay_is_cmake_only(self):
        row = self.lock['sources']['vllm']
        # CMakeLists.txt keeps the base's overlaid bytes (unchanged, so not in the refresh); replay it.
        self.assertEqual(row['files']['CMakeLists.txt'], row['base_files']['CMakeLists.txt'])
        overlaid = prepare.beta_overlay(prepare.previous.source('vllm', row['commit']))
        cmake = overlaid['CMakeLists.txt'][1].decode()
        self.assertEqual(cmake.count(';11.0;12.0;12.1")'), 2)
        self.assertEqual(hashlib.sha256(overlaid['CMakeLists.txt'][1]).hexdigest(), row['files']['CMakeLists.txt']['sha256'])
        head = self.members['vllm/vllm/models/glm5next/nvidia/mtp_draft_head.py'].decode()
        self.assertNotIn('!= (12, 0)', head)
        self.assertIn('SM120-family', head)
        self.assertIn('CMakeLists.txt', row['sm121_overlay'])

    def test_foundation_gate_copies_differ_only_by_proxy_abi(self):
        for kit, foundation in (('verify_identity.py', 'verify.py'), ('verify_runtime.py', 'inherited/verify_runtime.py'),
                                ('glm/launchers/serve-glm53-flash-karmic-spark.sh', 'inherited/launchers/serve-glm53-flash-karmic-spark.sh')):
            with self.subTest(kit=kit):
                ours = [line for line in (ROOT / kit).read_text().splitlines()
                        if not line.startswith(('# Kit copy of the foundation', '# refreshed B12X RoCE proxy',
                                                '# are imported from /opt/karmic-main-build'))
                        and line not in ('import sys', "sys.path.insert(0, '/opt/karmic-main-build')")]
                theirs = (FOUNDATION / foundation).read_text().splitlines()
                self.assertEqual(len(ours), len(theirs))
                changed = [(a, b) for a, b in zip(ours, theirs) if a != b]
                self.assertEqual(len(changed), 1, changed)
                self.assertEqual(changed[0][0].replace('== 5', '== 4'), changed[0][1])
                self.assertIn('roce_abi_version() == 5', changed[0][0])

    def test_runtime_wheel_lock_matches_frozen_inputs(self):
        inputs = json.loads((ROOT / 'inputs.lock.json').read_text())['wheels']
        lines = [line.split() for line in (ROOT / 'runtime-arm64.lock').read_text().splitlines() if line.strip()]
        self.assertEqual({line[0].split('==')[0] for line in lines}, {'nccl4py', 'nccl-extensions', 'nvidia-cudnn-frontend'})
        for spec, hashed in lines:
            name, version = spec.split('==')
            self.assertEqual(inputs[name]['version'], version)
            self.assertEqual(hashed, '--hash=sha256:' + inputs[name]['sha256'])
            self.assertEqual(digest(ROOT / 'inputs' / inputs[name]['file']), inputs[name]['sha256'])
        self.assertEqual(self.pin['runtime_lock_sha256_deps'], digest(ROOT / 'runtime-arm64.lock'))

    def test_image_launcher_and_build_context_are_consistent(self):
        launcher = digest(ROOT / 'glm/launchers/serve-glm53-flash-karmic-spark.sh')
        dockerfile = (ROOT / 'Dockerfile').read_text()
        self.assertIn('ENV GLM_LAUNCHER_SHA256=' + launcher, dockerfile)
        self.assertIn('local-inference.glm.launcher.sha256="' + launcher + '"', dockerfile)
        for runner in ('glm/run-glm-tp4-node.sh',):
            self.assertIn('."local-inference.glm.launcher.sha256" == "' + launcher + '"', (ROOT / runner).read_text())
        for runner in ('run-qwen.sh', 'glm/run-glm-tp4-node.sh'):
            text = (ROOT / runner).read_text()
            self.assertIn('."vllm.source-tree" == "' + self.lock['sources']['vllm']['refreshed_tree'] + '"', text)
            self.assertIn('."b12x.source-tree" == "' + self.lock['sources']['b12x']['refreshed_tree'] + '"', text)
        # Every COPY source of the Dockerfile must survive the .containerignore allowlist.
        import fnmatch
        rules = [line for line in (ROOT / '.containerignore').read_text().splitlines() if line.strip()]
        def admitted(path):
            keep = True
            for rule in rules:
                negate = rule.startswith('!')
                pattern = rule[1:] if negate else rule
                parts = path.split('/')
                hit = fnmatch.fnmatchcase(path, pattern) or any(
                    fnmatch.fnmatchcase('/'.join(parts[:i]), pattern) for i in range(1, len(parts)))
                if pattern.endswith('/**') and path.startswith(pattern[:-3] + '/'):
                    hit = True
                if hit:
                    keep = negate
            return keep
        import re
        for line in dockerfile.splitlines():
            if line.startswith('COPY ') and '--from' not in line:
                for source in line.split()[1:-1]:
                    self.assertTrue(admitted(source.rstrip('/')), f'COPY source excluded by .containerignore: {source}')
                    self.assertTrue((ROOT / source).exists(), f'COPY source missing: {source}')

    def test_initializer_contract_is_shared_by_every_consumer(self):
        # b12x_root_init.py executes _api.py as the package: b12x.__file__ is the root's _api.py.
        for name in ('upgrade_dependencies.py', 'gate_compiler.py', 'gate_flashinfer.py'):
            text = (ROOT / name).read_text()
            self.assertIn("b12x/_api.py", text, name)
            self.assertNotIn("b12x/__init__.py').resolve()", text, name)
            self.assertNotIn("b12x/b12x/__init__.py')", text, name)
        dockerfile = (ROOT / 'Dockerfile').read_text()
        self.assertIn('local-inference.roce.proxy-abi="5"', dockerfile)
        self.assertIn('b12x.roce-proxy-abi="5"', dockerfile)

    def test_flashinfer_wheel_identification_accepts_the_built_names(self):
        # Names the 2026-10-09 15:24Z build produced (first attempt failed on a wrong aarch64 count).
        names = ['flashinfer_jit_cache-0.7.1+lil.cu134.sm121.gb4c35e6ec1d7-cp39-abi3-manylinux_2_28_aarch64.whl',
                 'flashinfer_jit_cache_sm121a-0.7.1+lil.cu134.sm121.gb4c35e6ec1d7-cp39-abi3-linux_aarch64.whl',
                 'flashinfer_python-0.7.1+lil.cu134.sm121.gb4c35e6ec1d7-py3-none-any.whl']
        for script in ('verify_flashinfer_component.py', 'upgrade_dependencies.py'):
            text = (ROOT / script).read_text()
            self.assertNotIn('== 1', text.split('aarch64.whl')[0][-80:] if 'aarch64.whl' in text else '')
            self.assertIn('flashinfer_jit_cache_sm121a-', text, script)
        wheels = [Path(n) for n in names]
        (python_wheel,) = [p for p in wheels if p.name.startswith('flashinfer_python-')]
        (shim_wheel,) = [p for p in wheels if p.name.startswith('flashinfer_jit_cache-')]
        (provider_wheel,) = [p for p in wheels if p.name.startswith('flashinfer_jit_cache_sm121a-')]
        self.assertTrue(python_wheel.name.endswith('-py3-none-any.whl'))
        self.assertIn('aarch64', shim_wheel.name)
        self.assertIn('aarch64', provider_wheel.name)

    def test_publication_has_no_b12x_component(self):
        publication = json.loads((ROOT / 'publication.json').read_text())
        self.assertNotIn('b12x', publication['components'])
        self.assertEqual(publication['components']['flashinfer']['source_commit'], prepare.FLASHINFER)
        self.assertEqual(publication['components']['vllm']['source_commit'], prepare.VLLM)
        self.assertEqual(publication['image'], 'ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261008.2')

    def test_release_tooling_stays_at_base(self):
        row = self.lock['sources']['vllm']
        for path in prepare.TOOLING_DEFERRED:
            self.assertEqual(row['files'].get(path), row['base_files'].get(path), path)
            self.assertNotIn('vllm/' + path, self.members)
        self.assertEqual(sorted(row['deferred_tooling']['paths']), sorted(prepare.TOOLING_DEFERRED))

    def test_key_beta_fixes_are_shipped(self):
        # vllm #976: NVFP4 MoE input scales start at zero, never uninitialized memory.
        modelopt = self.members['vllm/vllm/model_executor/layers/quantization/modelopt.py'].decode()
        self.assertIn('w13_input_scale = PerTensorScaleParameter(\n            data=torch.zeros(', modelopt)
        # b12x #469: W4A16 skips empty 16-row blocks by default (bit-identity gated in-image).
        w4a16 = self.members['b12x/b12x/moe/_shared/kernels/w4a16/kernel.py'].decode()
        self.assertIn('B12X_W4A16_SKIP_EMPTY_M_BLOCKS', w4a16)
        # vllm #943: the ring spec no longer disables worker slot mapping.
        ring = self.members['vllm/vllm/v1/kv_cache_interface.py'].decode()
        block = ring[ring.index('class CircularBufferSpec'):]
        block = block[:block.index('\nclass ', 1)]
        if 'def uses_slot_mapping' in block:
            body = block.split('def uses_slot_mapping', 1)[1].split('\n    @', 1)[0]
            self.assertNotIn('return False', body)
        # vllm #1016: NVFP4-CSF expert projections load straight into packed layer weights.
        self.assertIn('nvfp4_csf', self.members['vllm/vllm/model_executor/layers/quantization/nvfp4_csf.py'].decode())
        # hybrid A4 prefill is the per-row switch (docker-146), never the token threshold.
        self.assertNotIn('B12X_W4A16_A4_PREFILL_MIN_TOKENS', self.members['vllm/vllm/envs.py'].decode())

    def test_foundation_and_publication_are_separate(self):
        base = json.loads((FOUNDATION / 'qsa865/runtime.lock.json').read_text())
        for key in prepare.INHERITED:
            self.assertNotIn(key, self.lock, key)
            self.assertEqual(self.lock['foundation'][key], base[key], key)
        self.assertEqual(self.lock['foundation']['image_id'], prepare.BASE)
        self.assertEqual(self.lock['foundation']['source_lock_sha256'], self.pin['base_lock_sha256'])
        pub = self.lock['publication']
        self.assertEqual((pub['tag'], pub['image'], pub['recipe_commit']),
                         (prepare.PUBLICATION, prepare.PUBLISHED_IMAGE, prepare.RECIPE))
        self.assertEqual((pub['vllm_commit'], pub['b12x_commit'], pub['flashinfer_commit']),
                         (prepare.VLLM, prepare.FLASHINFER, prepare.FLASHINFER))
        self.assertEqual((self.pin['publication'], self.pin['recipe_commit']), (pub['tag'], pub['recipe_commit']))

    def test_overlay_and_pins(self):
        cmake = self.lock['sources']['vllm']['files']['CMakeLists.txt']
        self.assertEqual(cmake, self.lock['sources']['vllm']['base_files']['CMakeLists.txt'])
        self.assertEqual(self.pin['vllm_commit'], prepare.VLLM)
        self.assertEqual(self.pin['b12x_commit'], prepare.FLASHINFER)
        self.assertEqual(self.pin['flashinfer_commit'], prepare.FLASHINFER)
        self.assertEqual(self.pin['base_image_id'], prepare.BASE)

    def test_source_archives_match_remote_git_blobs(self):
        import unpack_sources
        inputs = json.loads((ROOT / 'inputs.lock.json').read_text())
        for name, row in inputs['sources'].items():
            with self.subTest(component=name):
                files = unpack_sources.inspect_archive(ROOT / 'inputs' / row['file'], row)
                self.assertTrue(files)

    def test_wrong_source_digest_is_rejected(self):
        import unpack_sources
        row = dict(json.loads((ROOT / 'inputs.lock.json').read_text())['sources']['spdlog'])
        row['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'Archive digest mismatch'):
            unpack_sources.inspect_archive(ROOT / 'inputs' / row['file'], row)

    def test_compiler_metadata_overlay_preserves_versions_and_unrelated_requirements(self):
        import csv
        import io
        import upgrade_dependencies
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            relative = Path('vllm-0.11.dist-info/METADATA')
            path = root / relative
            path.parent.mkdir()
            original = 'Name: vllm\nVersion: 0.11\nRequires-Dist: torch>=2.13\nRequires-Dist: nvidia-cutlass-dsl==4.6.2\n'
            path.write_text(original)
            (path.parent / 'RECORD').write_text(str(relative) + ',,\n')

            class Distribution:
                files = [relative]

                def locate_file(self, name):
                    return root / name

            with patch.object(upgrade_dependencies.metadata, 'distributions', return_value=[Distribution()]), \
                 patch.object(upgrade_dependencies.metadata, 'distribution', return_value=Distribution()):
                changed = upgrade_dependencies.update_compiler_metadata()
            self.assertIn('vllm', changed)
            self.assertEqual(path.read_text(), original.replace('4.6.2', '4.7.1'))
            row = next(csv.reader(io.StringIO((path.parent / 'RECORD').read_text())))
            self.assertTrue(row[1].startswith('sha256='))
            self.assertEqual(int(row[2]), len(path.read_bytes()))

    def test_compiler_overlay_updates_source_egg_and_wheel_metadata(self):
        import importlib.metadata as metadata
        import sys
        import upgrade_dependencies
        from verify_compiler import validate_compiler_metadata
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            source, installed = root / 'source', root / 'installed'
            source.mkdir()
            installed.mkdir()
            for name in ('vllm', 'b12x'):
                egg = source / (name + '.egg-info')
                wheel = installed / (name + '-1.3.0.dist-info')
                egg.mkdir()
                wheel.mkdir()
                original = ('Metadata-Version: 2.1\nName: ' + name + '\nVersion: 1.3.0\n'
                            'Requires-Dist: torch>=2.12\nRequires-Dist: nvidia-cutlass-dsl==4.6.2\n'
                            'Requires-Dist: nvidia-cutlass-dsl-libs-cu13==4.6.2\n')
                requirements = ('torch>=2.12\nnvidia-cutlass-dsl==4.6.2\n'
                                'nvidia-cutlass-dsl-libs-cu13==4.6.2\n\n[dev]\npytest\n'
                                '\n[dev:python_version < "3.11"]\ntomli>=2.0\n')
                (egg / 'PKG-INFO').write_text(original)
                (egg / 'requires.txt').write_text(requirements)
                (egg / 'SOURCES.txt').write_text(name + '.egg-info/PKG-INFO\n' + name + '.egg-info/requires.txt\n')
                (wheel / 'METADATA').write_text(original)
                (wheel / 'RECORD').write_text(name + '-1.3.0.dist-info/METADATA,,\n')
            with patch.object(sys, 'path', [str(source), str(installed), *sys.path]):
                for name in ('vllm', 'b12x'):
                    self.assertEqual(len(list(metadata.distributions(name=name))), 2)
                self.assertEqual(metadata.distribution('b12x')._path, source / 'b12x.egg-info')
                changes = upgrade_dependencies.update_compiler_metadata()
                audit = validate_compiler_metadata()
                for name in ('vllm', 'b12x'):
                    egg = source / (name + '.egg-info')
                    wheel = installed / (name + '-1.3.0.dist-info')
                    expected = original.replace('Name: b12x', 'Name: ' + name).replace('4.6.2', '4.7.1')
                    self.assertEqual((egg / 'PKG-INFO').read_text(), expected)
                    self.assertEqual((wheel / 'METADATA').read_text(), expected)
                    self.assertEqual((egg / 'requires.txt').read_text(), requirements.replace('4.6.2', '4.7.1'))
                    copies = list(metadata.distributions(name=name))
                    self.assertEqual(len(copies), 2)
                    for dist in copies:
                        self.assertEqual(dist.version, '1.3.0')
                        compiler = [r for r in dist.requires if r.startswith('nvidia-cutlass-dsl')]
                        self.assertTrue(compiler)
                        self.assertTrue(all('4.7.1' in r and '4.6.2' not in r for r in compiler))
                    self.assertEqual(audit[name]['selected'], str(egg / 'PKG-INFO'))
                    self.assertEqual(len(changes[name]), 3)
                with self.assertRaisesRegex(RuntimeError, 'Uncovered selected'):
                    validate_compiler_metadata(set())
                (installed / 'b12x-1.3.0.dist-info/METADATA').write_text(original)
                with self.assertRaisesRegex(RuntimeError, 'Wrong compiler requirement'):
                    validate_compiler_metadata()

    def test_candidate_preserves_foundation_gates_and_expands_warmup(self):
        import ast
        import gate_inherited
        tree = ast.parse((FOUNDATION / 'inherited/tests/run_r38_regressions.py').read_text())
        cases = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                      and any(isinstance(t, ast.Name) and t.id == 'CASES' for t in n.targets)))
        adapted = gate_inherited.adapted_cases(cases)
        self.assertEqual(adapted.pop('warmup'), (gate_inherited.WARMUP, 20, None))
        self.assertEqual(adapted, {k: v for k, v in cases.items() if k != 'warmup'})
        with self.assertRaisesRegex(RuntimeError, 'contract changed'):
            gate_inherited.adapted_cases({**cases, 'warmup': (gate_inherited.WARMUP, 7, None)})
        expected = (FOUNDATION / 'gate.sh').read_text().replace(
            'cd "$(dirname "${BASH_SOURCE[0]}")"',
            'KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\ncd "$KIT/../../karmic-main-sm121/20260922"').replace(
            '-v "$PWD:/gate:ro" "$image" python',
            '-v "$PWD:/gate:ro" -v "$KIT:/kit:ro" "$image" python').replace(
            '/gate/inherited/tests/run_r38_regressions.py', '/kit/gate_inherited.py').replace(
            'CC=/bin/false run identity /gate/verify.py\nCC=/bin/false run native /gate/inherited/verify_runtime.py',
            '# ABI-5 adapted kit copies of the foundation identity and native gates (B12X from FlashInfer).\n'
            'CC=/bin/false run identity /kit/verify_identity.py\nCC=/bin/false run native /kit/verify_runtime.py')
        self.assertEqual((ROOT / 'gate_foundation.sh').read_text(), expected)

    def test_beta_gate_paths_exist_in_pinned_source(self):
        import gate_beta
        import re
        for path in gate_beta.FILES:
            self.assertIn(path.split('::', 1)[0], self.lock['sources']['vllm']['files'])
        # Named nodes must be functions of the pinned file (2026-10-09 build 2 failed collection on a stale name).
        for path in gate_beta.FILES:
            if '::' not in path:
                continue
            file, name = path.split('::', 1)
            source = prepare.previous.git('vllm', 'show', f"{self.lock['sources']['vllm']['commit']}:{file}").decode()
            self.assertRegex(source, rf'(?m)^(async )?def {re.escape(name)}\(', path)

    def test_candidates_preserve_serving_arguments(self):
        # Exercise shell dry-runs: only image/name/source identity may change.
        import shlex
        env = dict(__import__('os').environ, DRY_RUN='1', ROLE='head')
        for baseline in json.loads((ROOT / 'runner-baseline.json').read_text())['commands']:
            name, role = baseline['runner'], baseline['role']
            with self.subTest(runner=name):
                runner_env = dict(env, ROLE=role)
                if name.startswith('glm/'):
                    runner_env.update(NODE_RANK='0' if role == 'head' else '1',
                                      HOST_IP='10.11.11.1' if role == 'head' else '10.11.11.2')
                if name.startswith('ds4-vision/'):
                    runner_env['DS4_KIT_ROOT'] = '/home/jugs/git/ds4-vision-r38/karmic-beta-sm121/20261009'
                new = subprocess.check_output(['bash', str(ROOT / name)], env=runner_env, text=True)
                argv = [token.replace('20261009', '20260929') for token in shlex.split(new)]
                self.assertEqual(argv, baseline['argv'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
