import copy
import hashlib
import json
import unittest

from qualify_engram_repair import check_transition, validate_identity, check_residual_stress, validate_pin, validate_router_build


class RepairPinVerdict(unittest.TestCase):
    def build_data(self):
        raw = json.dumps({'trees': {'vllm': 'v', 'b12x': 'b'}}).encode()
        digest = hashlib.sha256(raw).hexdigest()
        pin = {'image_id': 'image', 'diagnostic': {'lock_sha256': digest, 'vllm_tree': 'v', 'b12x_tree': 'b'}}
        return pin, {'image_id': 'image', 'lock_sha256': digest}, raw

    def test_build_bound(self):
        validate_router_build(*self.build_data())

    def test_build_drift_rejected(self):
        for field in ('image', 'lock', 'tree'):
            pin, build, raw = self.build_data()
            if field == 'image':
                pin['image_id'] = 'other'
            elif field == 'lock':
                build['lock_sha256'] = 'other'
            else:
                pin['diagnostic']['b12x_tree'] = 'other'
            with self.assertRaises(RuntimeError):
                validate_router_build(pin, build, raw)

    def test_router_requires_explicit_selection(self):
        pin = {'diagnostic': {'kind': 'router-stage-release-candidate'}}
        validate_pin(pin, 'router-stage-release-candidate')
        with self.assertRaises(RuntimeError):
            validate_pin(pin, 'engram-prequeued-native-io-fix')

    def test_overlay_rejected(self):
        pin = {'diagnostic': {'kind': 'router-stage-release-candidate'}, 'diagnostic_overlay': {}}
        with self.assertRaises(RuntimeError):
            validate_pin(pin, 'router-stage-release-candidate')


class ResidualStressVerdict(unittest.TestCase):
    def data(self):
        lengths = [127,128,143,247,248,249,250,251,252,253,254,255,256,257,258,271,383,384,511,512,767,1023]
        summary = [{'length': n, 'trials': 50, 'unique_signatures': 1, 'nonmodal_trials': 0} for n in lengths]
        return summary, [{'unexplained_ttft_ge_5s': False} for _ in range(1100)]

    def test_complete(self):
        check_residual_stress(*self.data())

    def test_divergence(self):
        summary, records = self.data()
        summary[0]['unique_signatures'] = 2
        summary[0]['nonmodal_trials'] = 1
        with self.assertRaises(RuntimeError):
            check_residual_stress(summary, records)

    def test_incomplete_or_stall(self):
        for problem in ('records', 'summary', 'stall'):
            summary, records = self.data()
            if problem == 'records':
                records.pop()
            elif problem == 'summary':
                summary.pop()
            else:
                records[0]['unexplained_ttft_ge_5s'] = True
            with self.assertRaises(RuntimeError):
                check_residual_stress(summary, records)


class IdentityVerdict(unittest.TestCase):
    def info(self):
        return {'State': {'Running': True, 'StartedAt': 'boot1'}, 'Image': 'sha256:image',
                'Id': 'container', 'RestartCount': 0, 'Config': {'Env': ['MODE=1'],
                'Cmd': ['serve'], 'Labels': {'local-inference.ds41.kit.sha256': 'kit'}}}

    def test_valid_identity(self):
        self.assertEqual(validate_identity(self.info(), 'image', 'kit', {'MODE': '1'})['Id'], 'container')

    def test_fixed_capacity_control_cannot_qualify_serving(self):
        info = self.info()
        info['Config']['Env'].append('DS41_ROUTER_CONTROL_BLOCKS=79000')
        with self.assertRaises(RuntimeError):
            validate_identity(info, 'image', 'kit', {'MODE': '1'})

    def test_drift_rejected(self):
        for path, value in ((('State', 'Running'), False), (('Image',), 'other'),
                            (('Config', 'Env'), ['MODE=0']),
                            (('Config', 'Labels'), {})):
            info = self.info()
            target = info
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(RuntimeError):
                validate_identity(info, 'image', 'kit', {'MODE': '1'})

    def test_restart_and_extra_environment_change_snapshot(self):
        before = validate_identity(self.info(), 'image', 'kit', {'MODE': '1'})
        for field, value in (('StartedAt', 'boot2'), ('extra_env', 'UNEXPECTED=1')):
            info = self.info()
            if field == 'StartedAt':
                info['State'][field] = value
            else:
                info['Config']['Env'].append(value)
            self.assertNotEqual(before, validate_identity(info, 'image', 'kit', {'MODE': '1'}))


class TransitionVerdict(unittest.TestCase):
    def rows(self):
        return [{'cycle': cycle, 'prior': 385, 'label': label, 'returncode': 0,
                 'answers': [answer] * count, 'logprob_sha256': [label] * count}
                for cycle in range(3)
                for label, count, answer in (('prime', 2, '826493'), ('target', 3, '739184'))]

    def test_complete_pass(self):
        check_transition(self.rows())

    def test_incomplete_fails(self):
        for rows in ([], self.rows()[:-1]):
            with self.assertRaises(RuntimeError):
                check_transition(rows)

    def test_cross_cycle_drift_fails(self):
        rows = self.rows()
        rows[3]['logprob_sha256'] = ['different'] * 3
        with self.assertRaises(RuntimeError):
            check_transition(rows)

    def test_local_failure_is_not_a_pass(self):
        for field, value in (('returncode', 1), ('answers', ['wrong'] * 3),
                             ('logprob_sha256', ['a', 'b', 'a']), ('cycle', 10)):
            rows = copy.deepcopy(self.rows())
            rows[1][field] = value
            with self.assertRaises(RuntimeError):
                check_transition(rows)


if __name__ == '__main__':
    unittest.main()
