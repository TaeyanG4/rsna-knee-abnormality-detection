"""New bridge failure/receipt/thread checks; no real inputs/models or old suite rerun."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('thread_bridge', ROOT/'tools/b_rn3f_thread_input_v2.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

def write(path, value): path.write_text(json.dumps(value), encoding='utf-8')

@pytest.fixture
def case(tmp_path, monkeypatch):
    h = b.load_math(ROOT/'tools/b_rn3f_parity.py')
    ids = [str(i) for i in range(100)]
    assets = tmp_path/'assets'; assets.mkdir()
    write(assets/'b-rn3f-assets.json', {})
    (assets/'b_rn3f_serving.py').write_text('# synthetic identity only')
    write(assets/'engineering100.json', {'ids': ids})
    h.SERVING_SHA = b.sha(assets/'b-rn3f-assets.json')
    h.PRODUCER_SHA = b.sha(assets/'b_rn3f_serving.py')
    h.UID_SHA = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    work = tmp_path/'work'; (work/'result').mkdir(parents=True)
    def row(uid):
        return {'uid': uid, 'volume_sha256': '1'*64, 'mask_sha256': '2'*64,
                'selected_centers': [-1]*24, 'empty_windows': 24, 'normalized_sha256': '3'*64,
                'shape': [24,3,224,224], 'dtype': 'float32', 'pass': True,
                **{k:{'max_abs':0.,'p99_abs':0.,'mean_abs':0.,'pass':True,'limits':h.LIMITS[k]}
                   for k in ['window','normalized']}}
    rows = [row(uid) for uid in ids]
    worker = {'schema':'B_RN3F_SERVING_V1','status':'COMPLETE','manifest_sha256':h.SERVING_SHA,
              'studies':100,'ordered_uids_sha256':h.UID_SHA,'members':h.MEMBERS,'targets':h.TARGETS,
              'runtime':{'target':'local-reference','cohort':'engineering100','input_evidence':copy.deepcopy(rows),
                         'allocation':{'selected_threads':12,'effective_allocation':16,'torch_interop_threads':1},
                         'runtime_policy':{'policy':'strict','backends':{'deterministic_algorithms':True}}},
              'inference':{'status':'COMPLETE','studies':100,'member_count':6,'input_hash_scheme':'normalized-float32-shape-v1',
                           'fp32_fallbacks':0,'neutral_fills':0,'input_tensor_hashes':[{'uid':uid,'sha256':'3'*64} for uid in ids]}}
    def publish():
        write(work/'result/receipt.json', worker)
        write(work/'run-receipt.json', {'schema':'B_RN3F_SERVING_V1','status':'COMPLETE_ENGINEERING_OUTPUT_ONLY',
                                      'elapsed_seconds':1.,'worker_receipt_sha256':b.sha(work/'result/receipt.json')})
    publish()
    args = SimpleNamespace(out=tmp_path/'input.json', acceptance_receipt=tmp_path/'acceptance.json',
                           parity_helper='mocked', protocol_sha256='math', thread_protocol_sha256='thread',
                           assets=assets, work=work, target='local-reference')
    report = {'schema':h.SCHEMA,'protocol_sha256':'math','helper_sha256':b.PARITY_SHA,
              'serving_manifest_sha256':h.SERVING_SHA,'target':args.target,'rows':rows,
              'synthetic':[row(name) for name in h.SYNTHETIC_NAMES],'pass':True,'cpu_threads':12}
    monkeypatch.setattr(b,'load_math',lambda path:h)
    monkeypatch.setattr(b,'validate_contract',lambda a,p:None)
    monkeypatch.setattr(b,'reconstruct',lambda *unused:report)
    return args,h,worker,ids,report,publish

@pytest.mark.parametrize('bad',['parent_failed','parent_hash','worker_failed','threads0','threads_bool','threads_over',
                                'interop','incomplete_hashes','reordered_inputs','nonstrict'])
def test_completion_thread_and_identity_fail_closed(case,bad):
    args,h,w,ids,report,publish=case
    if bad=='worker_failed': w['status']='FAILED'
    if bad=='threads0': w['runtime']['allocation']['selected_threads']=0
    if bad=='threads_bool': w['runtime']['allocation']['selected_threads']=True
    if bad=='threads_over': w['runtime']['allocation']['selected_threads']=17
    if bad=='interop': w['runtime']['allocation']['torch_interop_threads']=2
    if bad=='incomplete_hashes': w['inference']['input_tensor_hashes'].pop()
    if bad=='reordered_inputs': w['runtime']['input_evidence'].reverse()
    if bad=='nonstrict': w['runtime']['runtime_policy']['backends']['deterministic_algorithms']=False
    publish()
    parent=b.read(args.work/'run-receipt.json')
    if bad=='parent_failed': parent['status']='FAILED'
    if bad=='parent_hash': parent['worker_receipt_sha256']='bad'
    write(args.work/'run-receipt.json',parent)
    assert b.execute(args)==1
    assert b.read(args.acceptance_receipt)['status']=='FAILED'
    assert not args.out.exists()

@pytest.mark.parametrize('bad',['pass_false','pass_string','hash','mask','centers','math_limit'])
def test_explicit_math_and_exact_producer_binding_required(case,bad):
    args,h,w,ids,report,publish=case
    if bad=='pass_false': report['pass']=False
    if bad=='pass_string': report['pass']='true'
    if bad=='hash': report['rows'][0]['normalized_sha256']='4'*64
    if bad=='mask': report['rows'][0]['mask_sha256']='4'*64
    if bad=='centers': report['rows'][0]['selected_centers'][0]=1
    if bad=='math_limit': report['rows'][0]['normalized']['max_abs']=4e-6
    assert b.execute(args)==1
    assert b.read(args.acceptance_receipt)['status']=='FAILED'
    assert args.out.exists() # preserve failed numerical/binding evidence

def test_exact_valid_report_success_and_no_overwrite(case):
    args,h,w,ids,report,publish=case
    assert b.execute(args)==0
    rec=b.read(args.acceptance_receipt)
    assert rec['status']==b.PASS and rec['exact_bound_studies']==100 and rec['actual_threads']==12
    assert rec['input_report_sha256']==b.sha(args.out)
    with pytest.raises(ValueError,match='new output'): b.execute(args)

def test_set_threads_checks_observed_threads_without_cuda():
    class Torch:
        cuda=SimpleNamespace(is_initialized=lambda:False)
        def set_num_threads(self,n): self.n=n
        def set_num_interop_threads(self,n): self.inter=n
        def get_num_threads(self): return self.n
        def get_num_interop_threads(self): return self.inter
    torch=Torch(); b.set_threads(torch,12); assert (torch.n,torch.inter)==(12,1)
    torch.get_num_threads=lambda:1
    with pytest.raises(ValueError,match='actual CPU thread mismatch'): b.set_threads(torch,12)

def test_real_contract_freeze_and_tamper_rejection(tmp_path):
    h=b.load_math(ROOT/'tools/b_rn3f_parity.py')
    protocol=ROOT/'docs/handoffs/evidence/2026-10/2026-10-04-b-rn3f-parity-v1/protocol.json'
    args=SimpleNamespace(protocol=protocol,protocol_sha256=b.sha(protocol),out=tmp_path/'thread.json')
    b.freeze(args,h)
    args.thread_protocol=args.out; args.thread_protocol_sha256=b.sha(args.out)
    b.validate_contract(args,h)
    value=b.read(args.out); value['thread_policy']['same_host_normalized_hash']='approximate'
    write(args.out,value); args.thread_protocol_sha256=b.sha(args.out)
    with pytest.raises(ValueError,match='thread contract changed'): b.validate_contract(args,h)

def test_real_cli_invalid_math_returns_one_and_failure_receipt(tmp_path):
    import subprocess,sys
    fake=tmp_path/'fake.py'; fake.write_text('raise Exception("must not import")')
    receipt=tmp_path/'receipt.json'
    result=subprocess.run([sys.executable,str(ROOT/'tools/b_rn3f_thread_input_v2.py'),'input',
        '--parity-helper',str(fake),'--protocol','unused','--protocol-sha256','unused',
        '--thread-protocol','unused','--thread-protocol-sha256','unused','--assets','unused','--work','unused',
        '--target','local-reference','--out',str(tmp_path/'input.json'),'--acceptance-receipt',str(receipt)],
        capture_output=True,text=True,timeout=10)
    assert result.returncode==1 and b.read(receipt)['status']=='FAILED'
