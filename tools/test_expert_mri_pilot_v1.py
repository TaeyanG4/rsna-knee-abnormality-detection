"""Meaningful synthetic selection/blinding checks; no medical data printed."""
import json
import unittest
from unittest.mock import patch
import numpy as np
from tools import expert_mri_pilot_v1 as m


class PilotTests(unittest.TestCase):
    def fixture(self):
        train = [f't{i}' for i in range(400)]
        val = [f'v{i}' for i in range(260)]
        ids = train + val
        groups = {u: u for u in ids}
        # Repeated study IDs in a component must never become separate examples.
        for i in range(0, 60, 2):
            groups[train[i+1]] = groups[train[i]]
        a = np.random.default_rng(17).random((len(ids), 12)).astype(np.float32)
        b = .8*a + .2*np.random.default_rng(22).random(a.shape).astype(np.float32)
        labels = np.where(a > .5, 0., 1.).astype(np.float32)
        r = dict(train=train, val=val, gold=['excluded_gold'])
        return ids, dict(R50=a, S=b), labels, r, groups

    def test_disjoint_full_selection_and_repeated_groups(self):
        ids, f, y, r, g = self.fixture()
        entries, test, fallback, pool = m.select(ids, f, y, r, g)
        selected = [e[0] for e in entries]+test
        self.assertEqual((len(entries), len(test), len({g[u] for u in selected})), (96, 200, 296))
        self.assertEqual({role: sum(x[1] == role for x in entries)
                          for role in ('calibration', 'training', 'audit_dev')},
                         dict(calibration=16, training=48, audit_dev=32))
        self.assertTrue(all(u in r['train'] for u, role, _ in entries if role != 'audit_dev'))
        self.assertTrue(all(u in r['val'] for u, role, _ in entries if role == 'audit_dev'))
        self.assertFalse(set(selected)&set(r['gold']))
        self.assertFalse({g[u] for u, role, _ in entries if role == 'calibration'}&{g[u] for u in pool})

    def test_deterministic_input_iteration_independence(self):
        args = self.fixture()
        first = m.select(*args)
        args[3]['train'].reverse(); args[3]['val'].reverse()
        self.assertEqual(first, m.select(*args))

    def test_random_component_uniform_representative_once(self):
        ids = ['a1', 'a2', 'a3', 'b', 'c']
        groups = {u: ('a' if u.startswith('a') else u) for u in ids}
        out = m.random_groups(ids, groups, 3, 'x', set())
        self.assertEqual({groups[u] for u in out}, {'a', 'b', 'c'})
        self.assertEqual(len(out), 3)

    def test_rank_ties_and_constant_target(self):
        p = m.percentiles(np.array([[0., 1.], [0., 1.], [1., 1.]]))
        np.testing.assert_allclose(p[:, 0], [.5, .5, 1.])
        np.testing.assert_allclose(p[:, 1], [2/3]*3)

    def test_contradiction_signs_and_intermediate_label_exclusion(self):
        a = np.full((4, 12), .5); b = a.copy(); y = a.copy()
        a[:, 0] = b[:, 0] = [.9, .1, .9, .1]
        y[:, 0] = [0, 1, .5, .5]
        q = m.suspicion_queues(['hi', 'lo', 'midhi', 'midlo'], a, b, y, 'contradiction')
        self.assertEqual([u for u, _ in q[0]], ['hi'])
        self.assertEqual([u for u, _ in q[1]], ['lo'])
        self.assertTrue(all(not qq for qq in q[2:]))

    def test_round_robin_skips_same_group_across_targets(self):
        q = [[('a', 't1'), ('b', 't1')], [('a2', 't2'), ('c', 't2')]]
        out = m.round_robin(q, dict(a=1, a2=1, b=2, c=3), set(), 3)
        self.assertEqual([u for u, _ in out], ['a', 'c', 'b'])

    def test_empty_strata_fall_back_without_threshold_relaxation(self):
        ids, f, y, r, g = self.fixture(); y[:] = .5
        entries, _, fallback, _ = m.select(ids, f, y, r, g)
        self.assertEqual(fallback['contradiction'], 15)
        self.assertEqual(fallback['rare'], 6)
        self.assertEqual(sum('random_fallback' in e[2] for e in entries), 21)

    def test_insufficient_unique_components_fail(self):
        with self.assertRaisesRegex(ValueError, 'INSUFFICIENT_COMPONENTS'):
            m.random_groups(['a', 'b'], dict(a=1, b=1), 2, 'test', set())

    def test_input_hash_change_rejected(self):
        with patch.object(m, 'sha', return_value='tampered'):
            with self.assertRaisesRegex(ValueError, 'INPUT_HASH_MISMATCH'):
                m.checked('artifacts/synthetic.json', 'expected', {})

    def test_blank_is_not_negative_and_no_private_fields(self):
        rows = m.blank_forms(['MRI-test'])
        self.assertEqual(len(rows), 12)
        self.assertEqual([r['target'] for r in rows], m.TARGETS)
        self.assertTrue(all(r[k] == '' for r in rows for k in m.FORM_FIELDS[2:]))
        text = json.dumps(rows)
        for secret in ('StudyInstanceUID', 'component', 'role', 'reason', 'prediction', 'label'):
            self.assertNotIn(secret, text)

    def test_nonfinite_rank_rejected(self):
        with self.assertRaisesRegex(ValueError, 'RANK_INPUT'):
            m.percentiles(np.array([[float('nan')]]))


if __name__ == '__main__':
    unittest.main()
