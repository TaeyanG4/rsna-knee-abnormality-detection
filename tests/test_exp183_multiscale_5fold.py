"""CPU-only synthetic checks for EXP183's pooled analysis and frozen schedule."""
import unittest

import numpy as np

from tools import exp115_vlm_eval as evaluator
from tools import exp183_multiscale_5fold as experiment


class FiveFoldScreenTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(183)
        self.folds = {}
        labels = []
        for fold in range(5):
            y = np.tile(np.arange(24)[:, None] % 2, (1, 12)).astype(float)
            labels.append(y)
            mixed = .1 + .8 * y
            neutral = np.full_like(y, .5)
            self.folds[fold] = {'mixed': mixed, 'control': neutral,
                                'rad42': rng.uniform(.25, .75, y.shape),
                                'rad43': rng.uniform(.25, .75, y.shape),
                                'dino': rng.uniform(.25, .75, y.shape)}
        self.y = np.concatenate(labels)
        self.refs = {name: (self.y, np.ones_like(self.y, dtype=bool), 2 if name == 'consensus' else 1)
                     for name in ('jev', 'consensus', 'dread', 'aquila')}
        groups = [np.array([2*i, 2*i+1]) for i in range(60)]
        self.draws = evaluator.bootstrap_index(groups, 30, 20261003)

    def test_pooled_screen_uses_all_five_matched_folds(self):
        result = experiment.score_arrays(self.folds, self.refs, self.draws, evaluator)
        self.assertEqual(result['studies'], 120)
        self.assertEqual(result['folds'], [0, 1, 2, 3, 4])
        self.assertEqual(result['bootstrap']['draws'], 30)
        self.assertEqual(result['registered_gate']['status'], 'ADVANCE')
        self.assertGreater(result['direct']['primary_mean_ci95'][0], 0)
        self.assertGreater(result['bundle_replacement']['primary_mean_delta'], 0)
        self.assertEqual(set(result['per_fold_direct']), set('01234'))

    def test_missing_fold_or_bundle_input_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'five folds'):
            experiment.score_arrays({k: v for k, v in self.folds.items() if k != 4},
                                    self.refs, self.draws, evaluator)
        broken = dict(self.folds)
        broken[1] = {k: v for k, v in broken[1].items() if k != 'dino'}
        with self.assertRaisesRegex(ValueError, 'fixed family'):
            experiment.score_arrays(broken, self.refs, self.draws, evaluator)

    def test_no_effect_on_one_fold_is_visible(self):
        folds = {f: dict(row) for f, row in self.folds.items()}
        folds[2]['mixed'] = folds[2]['control'].copy()
        result = experiment.score_arrays(folds, self.refs, self.draws, evaluator)
        self.assertAlmostEqual(result['per_fold_direct']['2']['jev'], 0)
        self.assertGreater(result['per_fold_direct']['1']['jev'], 0)

    def test_frozen_schedule_and_derived_runner(self):
        self.assertEqual([q['tag'] for q in experiment.queue()],
                         [experiment.tag(f) for f in range(1, 5)])
        source = experiment.runner_source()
        self.assertIn('from tools import exp183_multiscale_5fold as guard', source)
        self.assertNotIn('from tools import exp179_multiscale as guard', source)
        with self.assertRaises(ValueError):
            experiment.tag(0)


if __name__ == '__main__':
    unittest.main()
