"""Small structural and numerical gates for the v4 visible parity audit."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np

from tools import exp169_v4_visible_compare as audit
from tools import exp169_v4_visible_refs as refs


class VisibleCompareTests(unittest.TestCase):
    def test_s43_cpu_bind_sees_v4_checkpoint_name(self):
        module_name = 'exp152_own0949_release_s43_bind_test'
        helper = refs.STAGE / 's43/assets' / refs.local.EXP152_FILE
        module = refs.local._load_module(module_name, helper)
        self.addCleanup(sys.modules.pop, module_name, None)
        refs._install_s43_cpu_bind(module)
        module.OWN_NAME = refs.v4.S43_NAME
        self.assertIs(module.bind.__globals__, module.__dict__)
        self.assertEqual(module.bind.__globals__['OWN_NAME'], refs.v4.S43_NAME)

    def test_npz_rejects_id_and_target_order(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'raw.npz'
            ids = ['one', 'two', 'three']
            np.savez(path, ids=np.asarray(ids[::-1]), targets=np.asarray(audit.TARGETS),
                     own=np.full((3, 12), .5, np.float32))
            with self.assertRaisesRegex(ValueError, 'IDs/order'):
                audit.npz(path, {'ids', 'targets', 'own'}, ids, True)
            np.savez(path, ids=np.asarray(ids), targets=np.asarray(audit.TARGETS[::-1]),
                     own=np.full((3, 12), .5, np.float32))
            with self.assertRaisesRegex(ValueError, 'targets/order'):
                audit.npz(path, {'ids', 'targets', 'own'}, ids, True)

    def test_absolute_tolerance_failure(self):
        reference = np.zeros((3, 12), np.float32)
        visible = reference.copy()
        visible[0, 0] = .01
        result = audit.metric(reference, visible,
                              {'max_abs': .002, 'mean_abs': .0005, 'min_spearman': .99})
        self.assertFalse(result['pass'])
        self.assertGreater(result['max_abs'], .002)

    def test_own_anchor_csv_rank_preserves_serving_tie_order(self):
        raw = np.full((3, 12), .5, np.float32)
        raw[:, 0] = [.2, .2, .9]
        expected = raw.argsort(0).argsort(0).astype(np.float64) / 2
        self.assertTrue(np.array_equal(audit.own_anchor_csv_values(raw), expected))
        self.assertFalse(np.array_equal(audit.own_anchor_csv_values(raw),
                                        np.asarray(audit.v4.v2.rank(raw), np.float32)))

    def test_missing_independent_reference(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                audit.local_reference('S43', Path(td), ['one', 'two', 'three'], 'sha')

    def test_anchor_requires_complete_receipt_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / 'receipt.json').write_text(json.dumps({
                'status': 'COMPLETE_INFERENCE', 'files': {'raw.npz': 'ignored'}}),
                encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'anchor receipt incomplete'):
                audit.anchor(folder, {'own'}, ['one', 'two', 'three'], folder / 'sample.csv')

    def test_final_csv_text_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'submission.csv'
            ids = ['one', 'two', 'three']
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow([audit.UID, *audit.TARGETS])
                for uid in ids:
                    writer.writerow([uid, *(['0.5000000000'] * 12)])
            expected = [['0.5'] * 12 for _ in ids]
            self.assertTrue(np.array_equal(audit.csv_values(path, ids),
                                           np.full((3, 12), .5)))
            self.assertNotEqual(audit.csv_cells(path, ids), expected)

    def test_missing_bundle_writes_hold(self):
        with tempfile.TemporaryDirectory() as td:
            result = audit.compare('A4', stage=Path(td))
            self.assertEqual(result['status'], 'HOLD')
            self.assertTrue(result['missing'])
            self.assertIn('parity-plan.json', result['missing'][0])

    def test_hold_preserves_completed_metrics_and_timing(self):
        with tempfile.TemporaryDirectory() as td:
            def interrupted(candidate, stage, refs, root, report):
                report['comparisons']['S42'] = {'pass': True, 'max_abs': .0001}
                report['comparisons']['DINO'] = {'pass': False, 'max_abs': .01}
                report['timing']['wall_clock_seconds'] = 100
                raise FileNotFoundError('missing-reference.npz')

            with mock.patch.object(audit, '_compare', side_effect=interrupted):
                result = audit.compare('B4', stage=Path(td))
            self.assertEqual(result['status'], 'HOLD')
            self.assertEqual(result['comparisons']['S42']['max_abs'], .0001)
            self.assertEqual(result['timing']['wall_clock_seconds'], 100)
            self.assertEqual(result['failed_comparisons'], ['DINO'])
            self.assertIn('missing-reference.npz', result['missing'])

    def test_false_check_prevents_pass(self):
        result = audit.new_report('A4', Path('stage'))
        result['comparisons']['S42'] = {'pass': True}
        result['checks']['receipt'] = False
        result = audit.finish_report(result)
        self.assertEqual(result['status'], 'HOLD')
        self.assertEqual(result['failed_checks'], ['receipt'])

    def test_file_provenance_hash(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'raw.npz'
            path.write_bytes(b'raw')
            result = audit.new_report('A4', Path(td))
            audit.record_file(result, 'reference', path)
            self.assertEqual(result['files']['reference']['sha256'], audit.sha(path))
            audit.record_file(result, 'missing', Path(td) / 'absent.npz')
            self.assertEqual(result['status'], 'HOLD')
            self.assertEqual(len(result['missing']), 1)

    def test_candidate_specific_s42_input_folder(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            ids = ['one', 'two', 'three']
            records = {uid: {'series': uid} for uid in reversed(ids)}
            for candidate, folder in [('A4', 'exp169-own42'), ('B4', 'exp169-trio')]:
                visible = base / ('visible-' + candidate.lower())
                (visible / folder).mkdir(parents=True)
                (visible / refs.v4.RECEIPT).write_text(json.dumps({'candidate': candidate}), encoding='utf-8')
                (visible / folder / 'inputs.json').write_text(json.dumps(records), encoding='utf-8')
                self.assertEqual(audit.s42_folder(candidate), folder)
                with patch.object(refs.local, 'visible_inputs', return_value=(ids, {'studies': 3})), \
                     patch.object(refs, '_established_inputs') as established:
                    got_ids, _, got_records = refs._inputs(base, base, base, visible)
                self.assertEqual(got_ids, ids)
                self.assertEqual(got_records, records)
                established.assert_called_once_with(ids, {'studies': 3}, records)

    def test_reference_inputs_require_prior_local_records_and_source_digest(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            anchor = base / 'b172-own-anchor'
            anchor.mkdir()
            ids = ['one', 'two', 'three']
            records = {uid: {'volume': uid} for uid in ids}
            source = {'studies': 3, 'dicom_tree': {'sha256': 'prior-tree'}}
            inputs = anchor / 'inputs.json'
            inputs.write_text(json.dumps(records), encoding='utf-8')
            receipt = anchor.parent / 'b172-own-anchor.local-receipt.json'
            receipt.write_text(json.dumps({
                'status': 'COMPLETE_LOCAL_OWN_ANCHOR_REFERENCE',
                'ids': ids, 'inputs': source,
                'files': {'inputs.json': refs.local.sha(inputs)},
            }), encoding='utf-8')
            refs._established_inputs(ids, source, records, anchor)
            with self.assertRaisesRegex(ValueError, 'source CSV/DICOM tree'):
                refs._established_inputs(ids, {'studies': 3, 'dicom_tree': {'sha256': 'other-tree'}}, records, anchor)
            with self.assertRaisesRegex(ValueError, 'Visible decoded inputs'):
                refs._established_inputs(ids, source, {'one': {'volume': 'copied-visible'}}, anchor)
            inputs.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'independent input receipt'):
                refs._established_inputs(ids, source, records, anchor)

    def test_reference_publication_is_complete_or_absent(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            raw = base / 'source.npz'
            ids = ['one', 'two', 'three']
            np.savez(raw, ids=np.asarray(ids), targets=np.asarray(audit.TARGETS),
                     own=np.full((3, 12), .5, np.float32))
            (base / 'inputs.json').write_text(json.dumps({'a': 1}), encoding='utf-8')
            visible_input = base / 'visible-inputs.json'
            visible_input.write_text(json.dumps({'a': 1}), encoding='utf-8')
            evidence = {'visible_inputs_path': visible_input}
            bad = base / 'bad'
            with self.assertRaisesRegex(ValueError, 'decoded inputs'):
                refs._publish(bad, raw, ids, {'a': 2}, 'S43', 'cpu-fp32', evidence)
            self.assertFalse(bad.exists())
            out = base / 'good'
            receipt = refs._publish(out, raw, ids, {'a': 1}, 'S43', 'cpu-fp32', evidence)
            self.assertEqual(receipt['status'], 'COMPLETE_LOCAL_REFERENCE')
            self.assertEqual(set(p.name for p in out.iterdir()), {'raw.npz', 'local-receipt.json'})
            self.assertEqual(receipt['raw_sha256'], audit.sha(out / 'raw.npz'))


if __name__ == '__main__':
    unittest.main()
