"""Meaningful metadata/task leakage and drift checks; no patient data or GPU."""
import tempfile
import unittest
from pathlib import Path

from tools import h216_oai_inputs_v1 as m


def reading(value, project='22', uid='OAI_1234567_R_V00'):
    s={'MCMFMC':value}
    return m.extlabels.Reading(uid,s,s,project,'1','baseline.txt')


class Inputs(unittest.TestCase):
    def group(self, source):
        p,f,c,method=m.EXPECTED[source]
        return dict(source_sequence=source,plane=p,proposed_fluid=f,contrast_family=c,
                    fat_suppression_method=method,contrast_verified=True,contrast_issues=[],
                    contrast_certainty='HEADER_SUPPORTED_SOURCE_PROTOCOL',study_key='synthetic',
                    series_key='series',group_key='group',frame_count=12)

    def test_four_routes_keep_fluid_separate(self):
        for source,slot in [('SAG_IW_TSE',1),('COR_IW_TSE',4),('COR_MPR',3),('AX_MPR',5)]:
            r=m.bridge(self.group(source),m.slots())
            self.assertEqual(r['slot'],slot)
        r=m.bridge(self.group('AX_MPR'),m.slots())
        self.assertEqual((r['Fluid_Sensitive'],r['Fat_Suppression']),(-1,1))

    def test_wrong_plane_rejected(self):
        g=self.group('SAG_IW_TSE');g['plane']='Coronal'
        with self.assertRaisesRegex(ValueError,'CONTRAST_ROUTE_CONFLICT'):m.bridge(g,m.slots())

    def test_fs_not_copied_from_fluid(self):
        g=self.group('AX_MPR');g['fat_suppression_method']='NONE'
        with self.assertRaises(ValueError):m.bridge(g,m.slots())

    def test_unverified_contrast_rejected(self):
        g=self.group('COR_MPR');g['contrast_verified']=False
        with self.assertRaisesRegex(ValueError,'UNVERIFIED_CONTRAST'):m.bridge(g,m.slots())

    def test_unknown_method_stays_unknown_not_admitted(self):
        self.assertEqual(m.fs_flag('UNKNOWN'),-1)
        g=self.group('COR_MPR');g['fat_suppression_method']='UNKNOWN'
        with self.assertRaises(ValueError):m.bridge(g,m.slots())

    def test_compound_cartilage_split(self):
        vals=m.native_values({'MCMFMC':'3.1: category'})
        self.assertEqual(vals[:2],[3,1]);self.assertTrue(all(x is None for x in vals[2:]))

    def test_invalid_or_followup_cartilage_masked(self):
        for value in ['1.2', '.5', '-.5', '3.33', '99', 'NaN', '.M', '.U', '']:
            self.assertEqual(m.native_values({'MCMFMC':value})[:2],[None,None])

    def test_zero_is_observed_missing_is_not_negative(self):
        vals=m.native_values({'MCMFMC':'0','MEFFWK':'0: None','MSYIC':'.M'})
        self.assertEqual(vals[:2],[0,0]);self.assertEqual(vals[-2:],[0,None])

    def test_severity_invalid_masked(self):
        for v in [4,9,99,1.5,-1,float('inf')]:
            self.assertEqual(m.native_values({'MEFFWK':v})[-2],None)

    def test_participant_excludes_both_sides_and_visits(self):
        excluded=m.exclusion_union(['OAI_1234567_R_V00'],['OAI_2345678_L_V00'],[])
        self.assertIn(m.oai_exclusions.participant('OAI_1234567_L_V03'),excluded)
        self.assertEqual(len(excluded),2)

    def test_bad_exclusion_identity_fails(self):
        with self.assertRaises(ValueError):m.exclusion_union(['bad'],[],[])

    def test_same_rank_conflict_excludes(self):
        picked,conflicts=m.choose([reading(1.0),reading(2.0)])
        self.assertFalse(picked);self.assertEqual(len(conflicts),1)

    def test_duplicate_identical_reading_not_double_counted(self):
        picked,conflicts=m.choose([reading(1.0),reading(1.0)])
        self.assertEqual(len(picked),1);self.assertFalse(conflicts)

    def test_precedence_not_largest_grade(self):
        picked,conflicts=m.choose([reading(0,project='22'),reading(3.3,project='30')])
        self.assertEqual(next(iter(picked.values())).scores['MCMFMC'],0)
        self.assertFalse(conflicts)

    def test_unknown_project_fails(self):
        with self.assertRaisesRegex(ValueError,'UNREGISTERED_READING_PROJECT'):m.choose([reading(1,'unknown')])

    def test_source_drift_and_output_no_overwrite(self):
        # Fixed project scratch, no system temp or original files.
        base=m.ROOT/'.local/scratch/slot-b/h216-test-fixtures';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            p=Path(folder)/'fixture.json';m.write(p,{'x':1});bindings={str(p):m.sha(p)}
            m.check_bindings(bindings)
            with self.assertRaises(FileExistsError):m.write(p,{'x':2})
            p.write_text('changed',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'SOURCE_DRIFT'):m.check_bindings(bindings)


if __name__=='__main__':unittest.main()
