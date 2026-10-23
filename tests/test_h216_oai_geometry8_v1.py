import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from pydicom.dataset import Dataset
from tools import h216_oai_geometry8_v1 as m


class GeometryContract(unittest.TestCase):
    def fixture(self, side='R', tag=None):
        ds = Dataset(); ds.Rows = 6; ds.Columns = 8
        ds.ImageOrientationPatient = [1, 0, 0, 0, 0, -1]
        ds.ImagePositionPatient = [20, 30, 40]; ds.PixelSpacing = [2, 3]
        if tag: ds.Laterality = tag
        aff = m.bank.pixel_affine(ds, 'Coronal', side)
        r = dict(affines_col_row_origin_lps_mm=[aff.tolist()], quality=dict(aligned=True),
                 raw_explicit_sides=[] if tag is None else [tag])
        s = dict(files=['one.dcm'], native_shape=[1, 6, 8], plane='Coronal', side=side)
        return ds, r, s

    def test_correct_right_mapping(self):
        ds, r, s = self.fixture()
        self.assertLess(m.validate_record(r, s, [ds]), 1e-10)

    def test_omitted_right_flip_is_detected(self):
        ds, r, s = self.fixture()
        r['affines_col_row_origin_lps_mm'] = [m.bank.pixel_affine(ds, 'Coronal', None).tolist()]
        with self.assertRaisesRegex(ValueError, 'NATIVE_AFFINE_MISMATCH'):
            m.validate_record(r, s, [ds])

    def test_corrupt_spacing_is_detected(self):
        ds, r, s = self.fixture(); ds.PixelSpacing = [3, 2]
        with self.assertRaisesRegex(ValueError, 'NATIVE_AFFINE_MISMATCH'):
            m.validate_record(r, s, [ds])

    def test_count_and_nan_fail(self):
        ds, r, s = self.fixture()
        with self.assertRaisesRegex(ValueError, 'AFFINE_FRAME_COUNT'):
            m.validate_record(r, s, [])
        r['affines_col_row_origin_lps_mm'][0][0][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'NONFINITE_AFFINE'):
            m.validate_record(r, s, [ds])

    def test_raw_side_cannot_be_invented(self):
        ds, r, s = self.fixture(); r['raw_explicit_sides'] = ['R']
        with self.assertRaisesRegex(ValueError, 'RAW_LATERALITY_REWRITTEN'):
            m.validate_record(r, s, [ds])
        side, e = m.bank.resolve_anatomy_laterality([dict(raw_explicit_sides=[])])
        self.assertIsNone(side); self.assertEqual(e['status'], 'unknown_no_explicit_tags')

    def test_conflicting_raw_tag(self):
        ds, r, s = self.fixture(tag='L')
        with self.assertRaisesRegex(ValueError, 'DICOM_AUDIT_SIDE_CONFLICT'):
            m.validate_record(r, s, [ds])

    def test_incorrect_native_shape(self):
        ds, r, s = self.fixture(); s['native_shape'] = [1, 8, 6]
        with self.assertRaisesRegex(ValueError, 'NATIVE_SHAPE_GEOMETRY'):
            m.validate_record(r, s, [ds])

    def test_extracted_byte_drift_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); name = 'archive/member'
            path = root/(hashlib.sha256(name.encode()).hexdigest()+'.dcm'); path.write_bytes(b'original')
            digest = hashlib.sha256(json.dumps({name:m.sha(path)}, sort_keys=True).encode()).hexdigest()
            expected = dict(frames=1, frame_bytes_sha256=digest)
            self.assertEqual(len(m.frame_binding(root, [name], expected)), 1)
            with self.assertRaisesRegex(ValueError, 'DUPLICATE_OR_MISSING_FRAMES'):
                m.frame_binding(root, [name, name], expected)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'EXTRACTION_BYTES_DRIFT'):
                m.frame_binding(root, [name], expected)


if __name__ == '__main__':
    unittest.main()
