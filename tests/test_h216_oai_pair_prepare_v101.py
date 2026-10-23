import unittest
import numpy as np
import torch
from tools import h216_oai_pair_prepare_v101 as m


class OaiInputMechanics(unittest.TestCase):
    def test_wrong_returned_logical_index_rejected(self):
        x = torch.zeros(2, 3, 2, 2, dtype=torch.uint8); meta = torch.zeros(2, 6)
        with self.assertRaisesRegex(ValueError, 'LOGICAL_INDEX'):
            m.compare_item((x, meta, 4), (x.numpy(), meta.numpy()), 9)

    def test_changed_pixel_or_metadata_rejected(self):
        x = torch.zeros(2, 3, 2, 2, dtype=torch.uint8); meta = torch.zeros(2, 6)
        altered = x.numpy().copy(); altered[0, 0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'ACTUAL_WINDOW_PARITY'):
            m.compare_item((x, meta, 9), (altered, meta.numpy()), 9)
        with self.assertRaisesRegex(ValueError, 'ACTUAL_WINDOW_PARITY'):
            m.compare_item((x, meta, 9), (x.numpy(), meta.numpy()+1), 9)

    def test_reference_uses_logical_seed_and_actual_rows(self):
        observed = []
        class CT:
            @staticmethod
            def window_rows(bs, get, mode, slices, stride, cover, physical, rng, jitter, sampling):
                observed.append(int(rng.integers(100000)))
                self.assertEqual((mode, slices, stride, cover, physical, jitter, sampling), ('train',3,3,3,3.5,.25,'cover'))
                return [(0, 1, 0, 1, 2), (0, 2, 1, 2, 3)]
        px = np.arange(16, dtype=np.uint8).reshape(4,2,2)
        meta = np.arange(48, dtype=np.float32).reshape(4,12)
        x, a = m.exact_windows(CT,[dict(views=[[0,0]])],px,meta,73,epoch=2)
        self.assertEqual(observed, [int(np.random.default_rng(22601+2*1000003+73).integers(100000))])
        np.testing.assert_array_equal(x[0], px[[0,1,2]])
        np.testing.assert_array_equal(a, meta[[1,2]][:,[1,3,4,5,6,7]])


if __name__ == '__main__':
    unittest.main()
