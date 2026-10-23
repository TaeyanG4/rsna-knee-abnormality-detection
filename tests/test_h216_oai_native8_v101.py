import unittest
from dataclasses import dataclass
from tools import h216_oai_native8_v101 as m


@dataclass
class Side:
    side: str|None
    source: str
    tag_values: tuple
    sagittal_ipp_x_median: float|None


class SideAdapter(unittest.TestCase):
    def original(self,study,tags,x):
        return Side(next(iter(tags),None),'dicom' if tags else 'unknown_no_flip',tuple(tags),None)

    def test_absent_tag_absent_sagittal_uses_bound_side(self):
        f=m.side_resolver(self.original,{'u':'R'})
        self.assertIsNone(self.original('u',[],[]).side)
        r=f('u',[],[]);self.assertEqual(r.side,'R')
        self.assertEqual(r.source,'authenticated_OAI_side_join')

    def test_conflicting_tags_fail(self):
        f=m.side_resolver(self.original,{'u':'R'})
        for tags in [('L',),('L','R')]:
            with self.assertRaisesRegex(ValueError,'DICOM_AUDIT_SIDE_CONFLICT'):f('u',tags,[])

    def test_unknown_identity_fail(self):
        with self.assertRaisesRegex(ValueError,'UNBOUND_OAI_SIDE'):
            m.side_resolver(self.original,{'u':'R'})('other',[],[])

    def test_correct_tag_preserved(self):
        r=m.side_resolver(self.original,{'u':'L'})('u',['L'],[1])
        self.assertEqual((r.side,r.tag_values),('L',('L',)))

    def test_real_geometry_reproduces_missing_side_and_enables_right_flip(self):
        import importlib.util
        import sys
        spec=importlib.util.spec_from_file_location('_h216_frozen_geometry',m.CODE/'training/dicom_geometry.py')
        geom=importlib.util.module_from_spec(spec);sys.modules[spec.name]=geom;spec.loader.exec_module(geom)
        old=geom.decide_study_side('u',[],[])
        self.assertIsNone(old.side);self.assertFalse(geom.flip_for_plane('Coronal',old.side))
        actual=m.side_resolver(geom.decide_study_side,{'u':'R'})('u',[],[])
        self.assertTrue(geom.flip_for_plane('Coronal',actual.side))
        self.assertTrue(geom.flip_for_plane('Axial',actual.side))
        self.assertFalse(geom.flip_for_plane('Sagittal',actual.side))


if __name__=='__main__':unittest.main()
