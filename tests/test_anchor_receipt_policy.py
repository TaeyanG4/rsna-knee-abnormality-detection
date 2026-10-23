"""CPU-only tests for the distinction between output validity and metadata."""
import copy
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('receipt_policy',ROOT/'tools/anchor_receipt_policy.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

def receipt(members=p.MEMBERS,reduction='rank_of_member_probability_mean'):
    return {'contract':p.CONTRACT,'members':list(members),'study_count':3,'private_alpha':.4,'public_raptor_alpha':.6,'inner_rerank':False,'finding_specific_weights':False,'within_coat':{m:1/len(members) for m in members},'family_reduction':reduction}

class ReceiptPolicyTests(unittest.TestCase):
    def check(self,r,a=None,ok=True):return p.classify(r,a,csv_valid=ok,study_count=3)
    def test_full_receipt_without_live_namespace_is_not_failure(self):
        x=self.check(receipt());self.assertEqual(x['coat_member_state'],'full_reported');self.assertTrue(x['current_anchor_use_permitted']);self.assertTrue(x['full_coat_contract_reported'])
    def test_member_order_is_not_recipe_change(self):
        self.assertTrue(self.check(receipt(p.MEMBERS[::-1]))['full_coat_contract_reported'])
    def test_missing_all_metadata_preserves_usable_csv_as_unknown(self):
        x=self.check(None);self.assertEqual(x['coat_member_state'],'unknown');self.assertTrue(x['current_anchor_use_permitted']);self.assertFalse(x['full_coat_contract_reported'])
    def test_missing_receipt_and_explicit_zero_has_producer_explanation(self):
        x=self.check(None,{'coat_family_members':0,'events':[{'kind':'coat_family_unavailable'}]});self.assertEqual(x['coat_member_state'],'none_reported');self.assertTrue(x['current_anchor_use_permitted'])
    def test_partial_has_no_full_score_identity_claim(self):
        x=self.check(receipt(p.MEMBERS[:2]));self.assertEqual(x['coat_member_state'],'partial_reported');self.assertFalse(x['full_coat_contract_reported']);self.assertFalse(x['scored_anchor_prediction_or_score_recovered'])
    def test_probability_fallback_is_recipe_degradation_even_with_four_members(self):
        x=self.check(receipt(reduction='rank_sum_fallback_probabilities_unavailable'));self.assertEqual(x['coat_member_state'],'full_reported');self.assertEqual(x['coat_reduction_state'],'changed_reported');self.assertFalse(x['full_coat_contract_reported'])
    def test_single_member_preserved_and_marked_partial(self):
        x=self.check(receipt(p.MEMBERS[:1],'single_member_rank'));self.assertEqual(x['coat_member_state'],'partial_reported');self.assertEqual(x['coat_reduction_state'],'changed_reported')
    def test_disagreeing_counts_are_metadata_conflict_not_csv_failure(self):
        x=self.check(receipt(),{'coat_family_members':3});self.assertEqual(x['coat_member_state'],'conflicting_evidence');self.assertTrue(x['current_anchor_use_permitted'])
    def test_wrong_cohort_count_cannot_certify_full(self):
        r=receipt();r['study_count']=4;self.assertFalse(self.check(r)['full_coat_contract_reported'])
    def test_duplicate_unknown_member_cannot_certify_full(self):
        for members in [list(p.MEMBERS)+['other'],[p.MEMBERS[0]]*4]:
            r=receipt();r['members']=members;self.assertFalse(self.check(r)['full_coat_contract_reported'])
    def test_failure_event_conflicts_with_claimed_full_recipe(self):
        x=self.check(receipt(),{'events':[{'kind':'coat_member_probability_unavailable'}]});self.assertEqual(x['coat_member_state'],'conflicting_evidence')
    def test_invalid_csv_never_becomes_usable_from_complete_metadata(self):
        x=self.check(receipt(),ok=False);self.assertFalse(x['current_anchor_use_permitted']);self.assertEqual(x['output_action'],'reject_for_blending_do_not_overwrite')
    def test_contract_fields_missing_or_changed_are_not_full_recipe(self):
        r=receipt();del r['inner_rerank'];self.assertFalse(self.check(r)['full_coat_contract_reported'])
        r=receipt();r['private_alpha']=.5;self.assertFalse(self.check(r)['full_coat_contract_reported'])
    def test_csv_schema_cohort_and_nonfinite_checks(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);sample=d/'sample.csv';pred=d/'pred.csv'
            def write(path,ids,number):
                with path.open('w',newline='',encoding='utf-8') as f:
                    w=csv.writer(f);w.writerow(['StudyInstanceUID',*p.TARGETS]);w.writerows([[u,*([number]*12)] for u in ids])
            write(sample,['a','b'],0);write(pred,['a','b'],.5);self.assertTrue(p.validate_csv(pred,sample)[0])
            write(pred,['b','a'],.5);self.assertFalse(p.validate_csv(pred,sample)[0])
            write(pred,['a','b'],'nan');self.assertFalse(p.validate_csv(pred,sample)[0])
            write(pred,['a','b'],.5)
            lines=pred.read_text().splitlines();lines[1]+=',unexpected';pred.write_text('\n'.join(lines))
            self.assertFalse(p.validate_csv(pred,sample)[0])
            self.assertFalse(p.validate_csv(pred,d/'missing_sample.csv')[0])

if __name__=='__main__':unittest.main()
