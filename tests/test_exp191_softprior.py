# Compatibility tests v2.0.0: retired launch guards, archived scientific parity, collected readouts.
"""CPU checks for the registered EXP191 labels and readout guards."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from tools import exp191_softprior as exp


class SoftPriorTests(unittest.TestCase):
    def test_historical_recipe_keeps_control_weights_with_distinct_train_labels(self):
        overlay = exp.ROOT / '.local/exp191-overlay-v101'
        m = exp.validate(overlay)
        for arm in ('P', 'PA'):
            for seed in exp.SEEDS:
                q = exp.read(overlay / m['runs'][exp.tag(arm, seed)]['protocol'])
                control = exp.read(overlay / f'protocols/S{seed}.json')
                self.assertEqual(q['folds']['0']['pos_weight'], control['folds']['0']['pos_weight'])
                self.assertTrue(q['exp191']['frozen_pos_weight'])
                self.assertEqual(q['paths']['labels'], exp.LABELS[arm])
                self.assertEqual(q['paths']['eval_labels'], exp.L2)
                self.assertEqual(q['exp191']['eval_label_semantic_sha256'], control['label_set']['semantic_sha256'])
        for entry, args in [('runner_source', ()), ('prepare', (None,)), ('admit', ()), ('run', ('PA', 42))]:
            with self.assertRaisesRegex(ValueError, 'retired'):
                getattr(exp, entry)(*args)

    def test_verified_controls_reuse_existing_seeds(self):
        for seed in exp.SEEDS:
            control = exp.control_reuse(seed)
            self.assertEqual(control['prediction_sha256'],
                             exp.sha(exp.ROOT / control['path'] / 'oof_k36_flip_tta.csv'))

    def test_real_labels_touch_only_eligible_cells(self):
        data, counts, nrows = exp.build_label_arrays()
        self.assertEqual(nrows, 4347)
        self.assertEqual(counts['Synovitis']['eligible'], 3646)
        self.assertEqual(counts['Synovitis']['changed_P'], 3646)
        self.assertEqual(counts['Synovitis']['changed_PA'], 3618)
        self.assertEqual(counts['PF OA']['eligible'], 749)
        self.assertEqual(counts['PF OA']['changed_P'], 741)
        self.assertEqual(counts['PF OA']['changed_PA'], 717)
        original = exp._rows(exp.ROOT / exp.L2)[1]
        marker = exp._lookup(exp.ROOT / exp.MARKER)
        relabel = exp._lookup(exp.ROOT / exp.RELABEL)
        aquila = exp._lookup(exp.ROOT / exp.AQUILA)
        for arm in ('P', 'PA'):
            self.assertEqual(hashlib.sha256(data[arm]).hexdigest(), exp.digest_bytes(data[arm]))
            decoded = list(csv.reader(data[arm].decode().splitlines()))
            self.assertEqual(decoded[0], [exp.fill.UID, *exp.TARGETS])
            self.assertEqual(len(decoded) - 1, nrows)
            for old, new in zip(original, decoded[1:]):
                uid = old[0]
                self.assertEqual(new[0], uid)
                for col, target in enumerate(exp.TARGETS, start=1):
                    eligible = (target in exp.TARGETS_CHANGED and
                                marker[uid][target] == 0.5 and
                                relabel[uid][target] < 0.59 and
                                uid in aquila)
                    if eligible:
                        expected = (0.75 * exp.BASE_RATES[target] +
                                    0.25 * aquila[uid][target]) if arm == 'P' else aquila[uid][target]
                        self.assertEqual(float(new[col]), exp.f32(expected))
                    else:
                        self.assertEqual(new[col], old[col])

    def test_panel_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel = root / 'panel.csv'
            panel.write_text('StudyInstanceUID,PatientID,Synovitis,PF OA\nA,pat1,1,0\nB,pat2,0,1\n',
                             encoding='utf-8')
            lock = root / 'lock.json'
            lock.write_text(json.dumps({
                'panel_sha256': exp.sha(panel),
                'locked_before_predictions': True,
                'independent_blinded_image_labels': True,
                'definition_matched': {'Synovitis': True, 'PF OA': True}}), encoding='utf-8')
            self.assertTrue(exp._panel_lock(panel, lock))
            self.assertFalse(exp._panel_lock(panel, None))
            panel.write_text(panel.read_text(encoding='utf-8') + 'C,pat3,1,1\n', encoding='utf-8')
            self.assertFalse(exp._panel_lock(panel, lock))

    @unittest.skipIf(np is None, 'NumPy is absent from the CPU packaging venv')
    def test_sparse_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            panel = Path(directory) / 'gold.csv'
            panel.write_text('StudyInstanceUID,Synovitis,PF OA\nA,1,0\nB,0,1\n',
                             encoding='utf-8')
            y, _ = exp._binary_reference(panel, ['A', 'B'])
            self.assertTrue(np.isnan(y[:, exp.TARGETS.index('ACL')]).all())

    @unittest.skipIf(np is None, 'NumPy is absent from the CPU packaging venv')
    def test_gold58_cannot_use_fold0_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            gold = Path(directory) / 'gold58.csv'
            gold.write_text('StudyInstanceUID,Synovitis\n' +
                            ''.join(f'G{i},1\n' for i in range(58)), encoding='utf-8')
            pending = exp._gold58_pending(gold, ['V888-study'])
            self.assertEqual(pending['status'], 'PENDING_SEPARATE_INFERENCE')
            self.assertEqual(set(pending['delta']), set(exp.TARGETS))
            self.assertTrue(all(value is None for value in pending['delta'].values()))
            with self.assertRaisesRegex(Exception, 'disjoint'):
                exp._gold58_pending(gold, ['G0'])

    def test_auc_and_complete_macro_guards(self):
        complete = {target: 0.01 for target in exp.TARGETS}
        self.assertAlmostEqual(exp._macro(complete), 0.01)
        complete['Synovitis'] = None
        self.assertIsNone(exp._macro(complete))

    @unittest.skipIf(np is None, 'NumPy is absent from the CPU packaging venv')
    def test_auc(self):
        self.assertEqual(exp._auc_binary(np.array([0, 1]), np.array([0.5, 0.5])), 0.5)
        self.assertEqual(exp._auc_binary(np.array([0, 1]), np.array([0.1, 0.9])), 1.0)
        self.assertIsNone(exp._auc_binary(np.array([1, 1]), np.array([0.1, 0.9])))


if __name__ == '__main__':
    unittest.main()
