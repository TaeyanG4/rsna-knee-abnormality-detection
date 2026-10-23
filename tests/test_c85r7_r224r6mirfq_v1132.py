"""H179 exact branch composition, evidence authentication and numerical gates."""
import copy
import csv
import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from tools import c85r7_r224r6mirfq_v1132 as b
from tools import c85r7_r224r6mir_cpu as cpu


@pytest.fixture(scope='module')
def snapshots():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    return b.snapshots()


def test_exact_composition_and_bootstrap(snapshots):
    s = snapshots
    nb, meta, files, blob, proof = b.payload(s)
    p, r, m = (s[k] for k in ('parent','r6','mirror'))
    assert len(files) == 31 and set(files) == set(p['files'])
    assert {k for k in files if files[k] != p['files'][k]} == {b.SOURCE,b.WRAPPER,b.r6.PORT,b.r6.CHAIN}
    assert {k for k in files if files[k] != r['files'][k]} == {b.SOURCE,b.WRAPPER}
    assert files[b.SOURCE] == m['files'][b.SOURCE]
    assert files[b.HELPER] == p['files'][b.HELPER]
    assert proof['both_merge_orders_equal']
    assert nb['cells'][:-1] == p['nb']['cells'][:-1]
    raw = ''.join(nb['cells'][-1]['source']).encode()
    assignments = b.old.unpack(b.fast.restored(nb))[2]
    for key in ('BLOB','PINS'):
        raw = b.old.replace(raw,key+' = '+repr(assignments[key]),key+' = '+repr(p['assignments'][key]))
    raw = b.old.replace(raw,repr(b.old.digest(blob)),repr(b.old.digest(p['blob'])))
    assert raw == ''.join(p['nb']['cells'][-1]['source']).encode()
    assert meta == dict(r['meta'],id=b.SLUG,title=b.SLUG.split('/')[1])


@pytest.mark.parametrize('branch,member',[('r6',b.r6.PORT),('r6',b.HELPER),('mirror',b.SOURCE),('mirror',b.WRAPPER),('parent',b.HELPER)])
def test_source_drift_rejected(snapshots,branch,member):
    s=copy.deepcopy(snapshots); s[branch]['files'][member]+=b'\n# drift\n'
    with pytest.raises(ValueError): b.merge(s)


def test_registration_and_stage(snapshots):
    r=b.registration(snapshots)
    assert r['keep_score_min']==.956 and r['clean_oof'] is False and r['exposure']=='EXPOSED'
    assert r['runtime']['HB170_N32_mirror_increment_seconds'] is None
    assert r['runtime']['HB174_N32_R6_increment_seconds'] is None
    assert r['runtime']['parent_hidden_upper_hours']==6.73 and r['runtime']['limit_seconds']==32400
    assert not r['qualified_for_submission']
    assert b.validate()['status']=='PASS_CPU_VALIDATE'


@pytest.mark.parametrize('kind',['verdict','spec','reviewer'])
def test_prereview_fail_closed(snapshots,tmp_path,monkeypatch,kind):
    for name in ('SPEC.md','PRE-REVIEW.md'):
        (tmp_path/name).write_bytes((b.WORK/name).read_bytes())
    if kind=='spec': (tmp_path/'SPEC.md').write_text('altered spec')
    else:
        p=tmp_path/'PRE-REVIEW.md'; text=p.read_text()
        text=text.replace('PASS-WITH-NOTES','FAIL') if kind=='verdict' else text.replace('distinct from builder','same builder')
        p.write_text(text)
    monkeypatch.setattr(b,'WORK',tmp_path)
    with pytest.raises(ValueError,match='independent prereview'): b.registration(snapshots)


@pytest.mark.parametrize('score,mi,ri,expect',[(.956,0,0,True),(.955999,0,0,False),(.957,100,8072,True),(.957,100,8072.001,False)])
def test_decision_boundaries(score,mi,ri,expect):
    assert b.decision(score,mi,ri)==expect


@pytest.mark.parametrize('index,value',[(0,None),(0,True),(0,float('nan')),(0,1.1),(1,None),(1,-1),(2,float('inf')),(2,-1)])
def test_decision_invalid(index,value):
    values=[.956,100,100];values[index]=value
    with pytest.raises(ValueError): b.decision(*values)


@pytest.mark.parametrize('bad',['none','duplicate','missing','permuted','nan','inf','negative','above_one'])
def test_predictions_strict(tmp_path,bad):
    labels=[f't{i}' for i in range(12)];ids=['s1','s2']
    header=['StudyInstanceUID',*labels];rows=[['s2',*[.2]*12],['s1',*[.1]*12]]
    if bad=='duplicate': rows[0][0]='s1'
    if bad=='missing': rows.pop()
    if bad=='permuted': header[1],header[2]=header[2],header[1]
    if bad in ('nan','inf','negative','above_one'): rows[0][1]={'nan':'nan','inf':'inf','negative':-.1,'above_one':1.1}[bad]
    path=tmp_path/'p.csv'
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)
    if bad=='none': np.testing.assert_array_equal(cpu.predictions(path,ids,labels),[[.1]*12,[.2]*12])
    else:
        with pytest.raises(ValueError): cpu.predictions(path,ids,labels)


def load_test(name):
    spec=importlib.util.spec_from_file_location('hb179_'+name,Path(__file__).with_name(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('mode',['healthy','failures','all_own_fail','all_r384_fail','all_r7_fail'])
def test_candidate_independent_failure_oracle(snapshots,mode):
    legacy=load_test('test_c85r7_r224q_v1123')
    helper=SimpleNamespace(**b.r6.parent_builder.math_functions(b.merge(snapshots)[0]))
    legacy.h=helper
    legacy.test_independent_pandas_oracle(mode)
    legacy.test_sentinel_and_exact_three_percent_gate()
    for bad in ('nan','inf','negative','shape','duplicate'): legacy.test_invalid_predictions_reject(bad)


def test_actual_isolated_embedded_collect_with_synthetic_identity(tmp_path):
    legacy=load_test('test_c85r7_stackq_v1125')
    source=inspect.getsource(legacy.test_isolated_embedded_import_and_real_collect)
    source=source.replace("CUDA_VISIBLE_DEVICES='-1'","CUDA_VISIBLE_DEVICES=''").replace('1.1.25-stackq',b.VERSION)
    source=source.replace("b.SCRATCH/'candidate-runtime'","b.WORK/'runtime'")
    # The recorded three-study S6 stand-in is a formula fixture, not R6 evidence.
    # First require the R6 pin to reject its legacy identity; then explicitly
    # tag the isolated test fixture and test the unchanged collection path.
    source=source.replace("ids,r,ours,_=c.authenticated_pr7(staged)","""try:
    c.authenticated_pr7(staged)
except ValueError as exc:
    assert 'manifest pin' in str(exc)
else:
    raise AssertionError('R6 accepted the unpinned stand-in')
for relative in ('r7-identity.json','s_r7/provenance.json'):
    path=staged/'pr7-run'/relative
    value=json.loads(path.read_text())
    value.update(manifest_sha256=c.r7.MANIFEST_SHA256,stand_in='SYNTHETIC COLLECTION FIXTURE ONLY')
    path.write_text(json.dumps(value))
ids,r,ours,_=c.authenticated_pr7(staged)""")
    ns=dict(legacy.__dict__,b=b)
    exec(compile(source,'<hb179-isolated-collect>','exec'),ns)
    ns['test_isolated_embedded_import_and_real_collect'](tmp_path)


@pytest.mark.parametrize('kind',['stale_kernel','stale_status','evidence_drift','valid_hold'])
def test_admission_evidence_and_hold(tmp_path,monkeypatch,kind):
    monkeypatch.setattr(b,'ROOT',tmp_path);monkeypatch.setattr(b,'WORK',tmp_path)
    monkeypatch.setattr(b,'validate',lambda:dict(kernel_sha256='current'))
    path=tmp_path/'output.json';path.write_bytes(b'original')
    rec=dict(status='PASS_REAL_TWO_STUDY_COMPOSITION',kernel_sha256='current',evidence_sha256={'output.json':b.old.digest(path.read_bytes())})
    if kind=='stale_kernel': rec['kernel_sha256']='stale'
    if kind=='stale_status': rec['status']='RUNNING'
    if kind=='evidence_drift': path.write_bytes(b'changed')
    (tmp_path/'cpu-replay.json').write_text(json.dumps(rec))
    if kind=='valid_hold':
        result=b.admit();assert result['status'].startswith('HOLD') and not result['gate_bound']
    else:
        with pytest.raises(ValueError): b.admit()
