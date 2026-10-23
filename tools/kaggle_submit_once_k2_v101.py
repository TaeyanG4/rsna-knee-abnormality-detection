"""K2 v1.0.1 narrow admission wrapper around unchanged locked submit-once v139.

Same ledger, lock, durable IN_FLIGHT, send, list reconciliation and retry policy.
Unknown families use unchanged v139 admission. No live action on import.
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import strict_variant_gate as base
from tools import strict_variant_gate_k2_v101 as gate
from tools import l1_rco_k2_worker_v101 as w
from tools import l1_rco_k2_v101 as b

VERSION='1.0.1'
PARENT_SHA='62df1fb0bc062c3a7f9a763e37c9a06e4261656ecab720c9c9c1c8d2387c3a86'


def context(root=ROOT):
    source=ROOT/'tools/kaggle_submit_once_v139.py'
    base.need(w.sha(source)==PARENT_SHA,'immutable v139 submitter pin')
    spec=importlib.util.spec_from_file_location('_k2_private_submit_v139',source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    # Root override is useful for isolated tests only; production main uses ROOT.
    m.ROOT=Path(root).resolve()
    prior_gate,prior_qualify,prior_submit=m._strict_gate,m.qualify,m.submit_once
    def strict(kernel,binding,hashes):
        if kernel!=w.SLUG:return prior_gate(kernel,binding,hashes)
        path=m._local(Path('.local/state/gates')/(kernel.split('/')[1]+'.json'))
        inputs=base.Inputs(m.ROOT)
        observed=gate.verify_receipt(path,inputs,'FINAL_READY')
        base.need(binding.get('gate_sha256')==inputs.digest(path),'K2 gate_sha256 mandatory even non-FAST')
        inputs.recheck()
        hashes.update({str(m.ROOT/name):pin for name,pin in inputs.hashes().items()})
        return observed
    def qualify(kernel,version,receipt,review):
        if kernel!=w.SLUG:return prior_qualify(kernel,version,receipt,review)
        m._need(receipt and review,'K2 receipt/review required')
        hashes={};path=m._local(receipt);record=m._json_evidence(path,hashes)
        binding_path=path.parent/'submission-binding.json'
        binding=m._json_evidence(binding_path,hashes) if binding_path.exists() else record
        m._need(binding.get('phase')=='FINAL_READY' and binding.get('qualified_for_submission') is True,
                'K2 FINAL_READY binding required')
        observed=strict(kernel,binding,hashes)
        m._need(str(observed['kernel_version'])==str(version),'K2 exact remote version')
        m._need(m._local(review)==m._local(observed['final_review']),'K2 exact independent final review')
        from tools import l1_rco_k2_admission_v101 as admission
        inputs=base.Inputs(m.ROOT);bp=gate.checked_pin(observed['pin'],inputs)
        checked=admission.replay_final(observed,inputs,bp)
        m._need(m._local(checked['visible_receipt']['path'])==path and
            checked['visible_receipt']['sha256']==w.sha(path),'submission receipt differs from qualified visible output')
        result=prior_qualify(kernel,version,receipt,review)
        result['input_sha256'].update(hashes)
        result['input_sha256'].update({str(m.ROOT/name):pin for name,pin in inputs.hashes().items()})
        result.update(phase='FINAL_READY',qualified_for_submission=True,gate_version=gate.VERSION)
        m._recheck(result)
        return result
    def submit_once(sub_id,kernel,*args,**kwargs):
        m._need(kernel!=w.SLUG or sub_id==b.CANDIDATE,'exact K2 candidate ID required')
        return prior_submit(sub_id,kernel,*args,**kwargs)
    m._strict_gate=strict;m.qualify=qualify;m.submit_once=submit_once
    return m


def main():
    return context().main()


if __name__=='__main__':sys.exit(main())
