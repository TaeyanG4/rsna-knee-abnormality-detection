import copy
import unittest
from tools import h216_oai_native8_v1 as m


class NativeChecks(unittest.TestCase):
    def setup_fixture(self):
        row=dict(StudyInstanceUID='OAI_1234567_L_V00',SeriesInstanceUID='s',Anatomical_Plane='Coronal',
                 slot=4,Fat_Suppression=0,Fluid_Sensitive=0,frame_count=2)
        study=dict(errors=[],series=[dict(SeriesInstanceUID='s',errors=[],plane='Coronal',slot=4,
                   fat_suppression=0,fluid_sensitive=0,positions=[0,3],files=['a','b'],side='L')])
        return study,[row]

    def test_valid_fixture(self):
        s,r=self.setup_fixture();self.assertEqual(set(m.validate_native(s,r)),{'s'})

    def test_wrong_slot_or_metadata(self):
        for k,v in [('slot',8),('fat_suppression',-1),('fluid_sensitive',1)]:
            s,r=self.setup_fixture();s['series'][0][k]=v
            with self.assertRaisesRegex(ValueError,'NATIVE_METADATA_DRIFT'):m.validate_native(s,r)

    def test_frame_loss(self):
        s,r=self.setup_fixture();s['series'][0]['files']=['a']
        with self.assertRaisesRegex(ValueError,'NATIVE_FRAME_COUNT'):m.validate_native(s,r)

    def test_wrong_side(self):
        s,r=self.setup_fixture();s['series'][0]['side']='R'
        with self.assertRaisesRegex(ValueError,'NATIVE_SIDE'):m.validate_native(s,r)

    def test_error_or_missing_duplicate_series(self):
        s,r=self.setup_fixture();s['errors']=['synthetic']
        with self.assertRaises(ValueError):m.validate_native(s,r)
        s,r=self.setup_fixture();s['series'].append(copy.deepcopy(s['series'][0]))
        with self.assertRaisesRegex(ValueError,'SERIES_IDENTITY'):m.validate_native(s,r)


if __name__=='__main__':unittest.main()
