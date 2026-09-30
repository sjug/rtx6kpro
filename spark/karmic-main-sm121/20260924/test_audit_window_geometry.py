"""Local-only tests for the geometry audit orchestrator: planning, structural gates, digest, fake execution."""
import copy
import json
from pathlib import Path
import shlex
import tempfile
import unittest

import audit_window_geometry as audit
from audit_window_geometry import NODES, REMOTE

GEOMETRY = 'tree-simple-1ch'


def make_receipt(root, name, chunk_rows, geometry, *, verdict='within-conformance-envelope', problems=None):
    path = root / 'receipts' / name
    path.mkdir(parents=True)
    tag = f'm{chunk_rows}-{name[-7:].lower()}'
    summary = {'chunk_rows': chunk_rows, 'image_id': 'img' * 21 + 'x', 'blocks': 81389, 'mode': 'matched',
               'regime_source': 'pinned-045954Z', 'signature_equals_control': True, 'regime_equals_reference': True,
               'capture_problems': {n: [] for n in NODES}, 'nccl_geometry': geometry,
               'captures': {n: f'{n}-rank{i}-decision-row-{tag}.pt' for i, n in enumerate(NODES)},
               'remote_dir': REMOTE + '/receipts/' + name}
    (path / 'summary.json').write_text(json.dumps(summary))
    windows = {n: {'container_file': f'/cache/claude-decision-row/{n}-rank{i}-decision-row-{tag}-window.pt',
                   'problems': (problems or {}).get(n, [])} for i, n in enumerate(NODES)}
    (path / 'window-captures.json').write_text(json.dumps(windows))
    (path / 'audit.json').write_text(json.dumps({'verdict': verdict}))
    return path


def within(ok=True):
    return {'packed': {'rows_bit_equal': 128 if ok else 127, 'rows_not_in_gathered_records': []},
            'slots': {'write_slots_equal_last_row_window': True, 'write_offsets_equal_position_offsets': True,
                      'write_slots_ascending': True}}


def self_report(ok=True):
    return {'decision_consistency_left': {'0': {'all_equal': True}, '1': {'all_equal': ok}},
            'within_left': {'0': within(), '1': within(ok)}}


def window_report(first=None, changed=0, rel=0.0, kit_equal=False, decision=True):
    report = {'window': {'first_differing_boundary': first, 'kit_sha256': {'equal': kit_equal},
                         'layers': {'0': {'boundaries': {'wo_reduced': {'changed_row_count': changed, 'relative_l2_to_left': rel,
                                                                        'equal_values': changed == 0}}}}}}
    if decision:
        report['decision_row'] = {'first_unequal_layer': None if changed == 0 else 0, 'equal_observations': changed == 0}
    return report


class Receipts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.g8 = make_receipt(self.root, 'decision-row-matched8192-geometry-capture-20260926T190000Z', 8192, GEOMETRY)
        self.g4 = make_receipt(self.root, 'decision-row-matched4096-geometry-capture-20260926T193000Z', 4096, GEOMETRY)
        self.b8 = make_receipt(self.root, 'decision-row-matched8192-capture-20260926T182527Z', 8192, None)
        self.b4 = make_receipt(self.root, 'decision-row-matched4096-capture-20260926T185004Z', 4096, None)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self):
        return (audit.load_receipt(self.g8, geometry=True, root=self.root), audit.load_receipt(self.g4, geometry=True, root=self.root),
                audit.load_receipt(self.b8, geometry=False, root=self.root), audit.load_receipt(self.b4, geometry=False, root=self.root))

    def test_load_receipt_identity_rules(self):
        self.load()
        with self.assertRaisesRegex(ValueError, 'nccl_geometry'):
            audit.load_receipt(self.b8, geometry=True, root=self.root)
        with self.assertRaisesRegex(ValueError, 'nccl_geometry'):
            audit.load_receipt(self.g8, geometry=False, root=self.root)
        with self.assertRaisesRegex(ValueError, 'outside this kit'):
            audit.load_receipt(self.root, geometry=True, root=self.root)
        bad = make_receipt(self.root, 'decision-row-matched8192-geometry-capture-20260926T191111Z', 8192, GEOMETRY,
                           problems={'toby': ['aborted']})
        with self.assertRaisesRegex(ValueError, 'Window capture failures'):
            audit.load_receipt(bad, geometry=True, root=self.root)
        bad = make_receipt(self.root, 'decision-row-matched8192-geometry-capture-20260926T192222Z', 8192, GEOMETRY,
                           verdict='deviation-found')
        with self.assertRaisesRegex(ValueError, 'verdict'):
            audit.load_receipt(bad, geometry=True, root=self.root)

    def test_plan_is_complete_read_only_and_refuses_mixed_arms(self):
        g8, g4, b8, b4 = self.load()
        image, commands = audit.plan(g8, g4, b8, b4, memory='24g')
        self.assertEqual(image, g8['summary']['image_id'])
        stems = sorted(stem for _, stem in commands)
        self.assertEqual(len(commands), 22)
        self.assertEqual(stems.count('window-ranks'), 2)
        self.assertEqual(sum(s.startswith('window-geometry-grids-') for s in stems), 4)
        for (name, stem), command in commands.items():
            self.assertIn('--network=none', command)
            self.assertIn('--memory=24g', command)
            mounts = [command[i + 1] for i, a in enumerate(command) if a == '-v']
            self.assertTrue(mounts and all(m.endswith(':ro') for m in mounts), (stem, mounts))
            self.assertNotIn('--device', command)
            if stem.startswith('window-geometry-grids-'):
                self.assertEqual(name, g4['path'].name)
                node = stem.removeprefix('window-geometry-grids-')
                rank = NODES.index(node)
                self.assertIn(f'/left/captures/{node}-rank{rank}-decision-row-m8192-190000z-window.pt', command)
                self.assertIn(f'/gate/captures/{node}-rank{rank}-decision-row-m4096-193000z-window.pt', command)
            if stem.startswith('window-self-'):
                self.assertEqual(command.count(command[command.index('compare') + 1]), 2)
            if stem.startswith('window-geometry-pair-'):
                self.assertIn('/base/captures/', ' '.join(command))
        with self.assertRaises(ValueError):
            audit.plan(g8, g4, b8, dict(b4, summary=dict(b4['summary'], nccl_geometry=GEOMETRY)))
        with self.assertRaises(ValueError):
            audit.plan(g8, dict(g4, summary=dict(g4['summary'], image_id='other')), b8, b4)

    def test_validate_within_requires_structural_checks_only(self):
        reports = {n: self_report() for n in NODES}
        audit.validate_within(reports, {'replicated_all_equal': True})
        with self.assertRaisesRegex(ValueError, 'Replicated'):
            audit.validate_within(reports, {'replicated_all_equal': False})
        broken = copy.deepcopy(reports)
        broken['rusty'] = self_report(ok=False)
        with self.assertRaisesRegex(ValueError, 'Sibling'):
            audit.validate_within(broken, {'replicated_all_equal': True})
        broken = copy.deepcopy(reports)
        broken['kirby']['decision_consistency_left']['1']['all_equal'] = True
        broken['kirby']['within_left']['1']['packed']['rows_bit_equal'] = 100
        with self.assertRaisesRegex(ValueError, 'Packed'):
            audit.validate_within(broken, {'replicated_all_equal': True})
        broken = copy.deepcopy(reports)
        broken['toby']['within_left']['0']['slots']['write_slots_ascending'] = False
        with self.assertRaisesRegex(ValueError, 'slot'):
            audit.validate_within(broken, {'replicated_all_equal': True})
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            audit.validate_within({n: self_report() for n in NODES[:3]}, {'replicated_all_equal': True})

    def test_summary_reports_without_gating(self):
        pairs8 = {n: window_report({'layer': 0, 'boundary': 'wo_reduced'}, 96, 0.003) for n in NODES}
        pairs4 = {n: window_report({'layer': 0, 'boundary': 'wo_reduced'}, 128, 0.003) for n in NODES}
        grids = {n: window_report(None, 0, 0.0, decision=False) for n in NODES}
        summary = audit.summarize(pairs8, pairs4, grids)
        self.assertTrue(summary['layer0_wo_reduced_equal_across_geometry_grids_all_ranks'])
        self.assertEqual(summary['geometry_vs_baseline']['8192']['dusty']['layer0_wo_reduced_changed_rows'], 96)
        self.assertEqual(summary['geometry_vs_baseline']['4096']['toby']['decision_row_first_unequal_layer'], 0)
        self.assertNotIn('decision_row_first_unequal_layer', summary['geometry_8192_vs_4096']['dusty'])
        grids['rusty'] = window_report({'layer': 0, 'boundary': 'wo_reduced'}, 32, 0.001, decision=False)
        self.assertFalse(audit.summarize(pairs8, pairs4, grids)['layer0_wo_reduced_equal_across_geometry_grids_all_ranks'])

    def fake_run(self, argv, stdout=None, stderr=None, check=False):
        text = ' '.join(shlex.split(argv[-1])) if argv[0] == 'ssh' else ' '.join(argv)
        if argv[0] == 'scp':
            self.staged.append(argv[1:])
            return type('R', (), {'returncode': 0})()
        if ' ranks ' in text:
            payload = {'replicated_all_equal': True}
        elif ' compare ' in text:
            payload = self_report()
        elif ' pair ' in text:
            payload = window_report({'layer': 0, 'boundary': 'wo_reduced'}, 96, 0.003)
        else:
            payload = window_report(None, 0, 0.0, decision=False)
        stdout.write(json.dumps(payload))
        return type('R', (), {'returncode': 1 if self.fail_stem and self.fail_stem in text else 0})()

    def test_execute_writes_outputs_and_summary_without_touching_baselines(self):
        self.staged, self.fail_stem = [], None
        g8, g4, b8, b4 = self.load()
        idle_calls = []
        summary = audit.execute(g8, g4, b8, b4, idle=lambda: idle_calls.append(1), run=self.fake_run,
                                stage_files=lambda receipt, files: self.staged.append((receipt['path'].name, sorted(files))))
        self.assertEqual(idle_calls, [1])
        self.assertEqual([s[0] for s in self.staged], [g8['path'].name, g4['path'].name])
        self.assertEqual(self.staged[0][1], sorted(audit.STAGED))
        for receipt in (g8, g4):
            for n in NODES:
                self.assertTrue((receipt['path'] / f'window-self-{n}.json').is_file())
                self.assertTrue((receipt['path'] / f'window-geometry-pair-{n}.json').is_file())
            self.assertTrue((receipt['path'] / 'window-ranks.json').is_file())
            invocation = json.loads((receipt['path'] / 'window-geometry-invocation.json').read_text())
            self.assertEqual(sorted(invocation['files']), sorted(audit.STAGED))
        self.assertTrue((g4['path'] / 'window-geometry-summary.json').is_file())
        self.assertTrue(summary['layer0_wo_reduced_equal_across_geometry_grids_all_ranks'])
        for baseline in (b8, b4):
            self.assertEqual(sorted(p.name for p in baseline['path'].iterdir()),
                             ['audit.json', 'summary.json', 'window-captures.json'])

    def test_execute_stops_on_container_failure(self):
        self.staged, self.fail_stem = [], 'window-geometry-pair-rusty'
        g8, g4, b8, b4 = self.load()
        self.fail_stem = 'rusty-rank2-decision-row-m8192-190000z-window.pt --baseline-window'
        with self.assertRaisesRegex(RuntimeError, 'failed; preserve'):
            audit.execute(g8, g4, b8, b4, idle=None, run=self.fake_run, stage_files=lambda r, f: None)
        self.assertFalse((g4['path'] / 'window-geometry-summary.json').exists())


if __name__ == '__main__':
    unittest.main()
