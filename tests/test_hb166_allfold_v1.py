"""All-fold v1.2.2: recipe identity, population, decision and admission failures."""
import ast
import copy
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pytest
from tools import hb166_allfold_build_v1 as b
from tools import hb166_allfold_spec_v1 as s
from tools import hb166_fold3_readout_v1 as r
from tools import multi_lens_readout as base_lens


def test_complete_config_diff():
    allowed={'seed','fold','train','val','version','inputs','optimizer_steps','control_config',
             'control_config_sha256','control_member','control_fits','parent_config','parent_config_sha256',
             'gate','bias_statement','estimand','recipe'}
    for seed in b.SEEDS:
        c=b.validate(b.config_path(seed)); p=b.read(b.ROOT/c['parent_config'])
        assert {k for k in c.keys()|p.keys() if c.get(k)!=p.get(k)} <= allowed
        cr,pr=copy.deepcopy(c['recipe']),copy.deepcopy(p['recipe'])
        for x in (cr,pr):
            x.pop('roots')
            for k in ('loader_batches','steps','max_skips'): x['updates'].pop(k)
        assert cr==pr
        assert (c['workers'],c['epochs'],c['batch'],c['microbatch'],c['accumulation'])==(2,20,8,2,4)
        assert c['optimizer_steps']==len(c['train'])//8*20
        assert c['selection']=='fixed_epoch20' and c['window_budget']==p['window_budget']
        assert b.validate(b.config_path(seed,True))['engineering_only'] is True


def test_five_fold_population():
    sets=[]
    for fold in range(5):
        if fold in (0,1,4): a,v=b.roster(fold)
        else:
            w=b.parent.b155.WORK if fold==2 else b.PARENT_WORK
            c=b.read(w/'configs'/('control-s42.json' if fold==2 else 'multiview-s44.json'))
            a,v=c['train'],c['val']
        assert len(v)==[888,864,864,864,863][fold]
        assert len(a)+len(v)==4343 and not set(a)&set(v)
        sets.append(set(v))
    assert len(set.union(*sets))==sum(map(len,sets))==4343


def test_registration_repeat_no_reference_rebuild(monkeypatch):
    monkeypatch.setattr(s,'make_spec',lambda:pytest.fail('must not reopen local OOF references remotely'))
    assert b.register()['status']=='REGISTERED'
    assert b.registration_binding()


@pytest.mark.parametrize('kind', ['macro','ci','target','stream','nan','length'])
def test_replacement_gate_failures(kind):
    a,c,t,z=[.003]*4,[.001]*4,[.003]*12,[.001]*2
    assert s.replacement_gate(a,c,t,z)=='SUPPORT'
    if kind=='macro': a=[.001]*4
    if kind=='ci': c=[.001,0,0,0]
    if kind=='target': t[0]=-.006
    if kind=='stream': z[0]=0
    if kind=='nan': t[0]=float('nan')
    if kind=='length': t.pop()
    assert s.replacement_gate(a,c,t,z)==('HOLD' if kind in ('nan','length') else 'NO_SUPPORT')


def support_result():
    # Completed immutable scientific result; aggregate JSON only, never label rows.
    return b.read(b.PARENT_WORK/'readout.json')


def test_measurement_accepts_authentic_no_pass_unchanged():
    x=support_result(); original=copy.deepcopy(x)
    assert x['registered_gate']['status']=='NO_PASS'
    assert x['combined_fold2_fold3']['status']=='NO_COMBINED_SUPPORT'
    s.require_measurement_history(x)
    assert x==original


@pytest.mark.parametrize('field', ['identity','engineering','pass','combined','target','seed','ci','macro','checks'])
def test_history_fails_closed(field):
    x=support_result(); g=x['registered_gate']
    if field=='identity': x['registration']={}
    if field=='engineering': x['engineering_only']=True
    if field=='pass': g['PASS']=True
    if field=='combined': x['combined_fold2_fold3']['equal_fold_macro']=1
    if field=='target': g['target_four_ruler_mean'][r.lens.TARGETS[0]]=-.9
    if field=='seed': g['seed_four_ruler_macro']['44']=0
    if field=='ci': g['ruler_ci95_lower']={k:0 for k in r.lens.RULERS}
    if field=='macro': g['four_ruler_macro']=float('nan')
    if field=='checks': g['checks']['target_floor']=True
    with pytest.raises(ValueError): s.require_measurement_history(x)


@pytest.mark.parametrize('failure',[None,'review_fail','stale','self','no_b','wrong_registration','old_purpose','wrong_hypotheses','missing_purpose','missing_result'])
def test_dependency_evidence(tmp_path,monkeypatch,failure):
    monkeypatch.setattr(b,'WORK',tmp_path)
    def write(n,x): (tmp_path/n).write_text(json.dumps(x))
    write('registration.json',{'test':1})
    (tmp_path/'fold3-readout.json').write_bytes((b.PARENT_WORK/'readout.json').read_bytes())
    review=dict(schema='hb166-fold3-result-review/v1',verdict='PASS',independent=True,reviewer='independent-test',
                readout_sha256=b.sha(tmp_path/'fold3-readout.json'),registration_sha256=b.PARENT_REG_SHA)
    if failure=='review_fail': review['verdict']='FAIL'
    if failure=='stale': review['readout_sha256']='0'*64
    if failure=='self': review['reviewer']='hb166-allfold-builder'
    write('fold3-result-review.json',review)
    decision=dict(schema='hb166-allfold-b-admission/v1',coordinator='B',decision='ADMIT_FRESH_FOLD_MEASUREMENT',purpose='measurement',hypotheses_sha256=s.hypotheses_sha256(),
                  recorded_at='test',registration_sha256=b.sha(tmp_path/'registration.json'),
                  fold3_readout_sha256=b.sha(tmp_path/'fold3-readout.json'),
                  fold3_result_review_sha256=b.sha(tmp_path/'fold3-result-review.json'))
    if failure=='no_b': decision['coordinator']='builder'
    if failure=='wrong_registration': decision['registration_sha256']='0'*64
    if failure=='old_purpose': decision['decision']='ADMIT_ALLFOLD_AFTER_CONFIRMATION'
    if failure=='wrong_hypotheses': decision['hypotheses_sha256']='0'*64
    if failure=='missing_purpose': decision.pop('purpose')
    if failure=='missing_result': write('fold3-readout.json',{})
    write('b-admission.json',decision)
    if failure:
        with pytest.raises(ValueError): b.dependency_bindings()
    else: assert len(b.dependency_bindings())==3


def test_no_direct_cuda_bypass(tmp_path,monkeypatch):
    from tools import hb166_allfold_fit_v1 as f
    def block(): raise ValueError('test admission HOLD')
    monkeypatch.setattr(b,'require_gpu',block)
    for smoke in (True,False):
        with pytest.raises(ValueError,match='test admission HOLD'):
            f.fit(b.config_path(46),tmp_path/'never',device='cuda',smoke=smoke)
    with pytest.raises(ValueError,match='test admission HOLD'): b.sequence('smoke')
    with pytest.raises(ValueError,match='test admission HOLD'): b.snapshot_runtime(tmp_path/'never','cuda')
    assert not (tmp_path/'never').exists()


@pytest.mark.parametrize('stale',[False,True])
def test_pre_review_binding(tmp_path,monkeypatch,stale):
    monkeypatch.setattr(b,'WORK',tmp_path)
    monkeypatch.setattr(b,'dependency_bindings',lambda:{'dependency':'1'})
    monkeypatch.setattr(b,'pre_bindings',lambda:{'cpu':'1'})
    (tmp_path/'pre-review.json').write_text(json.dumps(dict(schema='hb166-allfold-pre-review/v1',
        verdict='PASS',independent=True,reviewer='independent-test',bindings={'cpu':'0' if stale else '1'})))
    if stale:
        with pytest.raises(ValueError): b.require_gpu()
    else: assert len(b.require_gpu())==2


@pytest.mark.parametrize('stale',[False,True])
def test_fit_review_binding(tmp_path,monkeypatch,stale):
    monkeypatch.setattr(b,'require_gpu',lambda:{})
    monkeypatch.setattr(b,'review_bindings',lambda:{'gpu':'1'})
    p=tmp_path/'review.json'; p.write_text(json.dumps(dict(schema='hb166-allfold-fit-review/v1',
        verdict='PASS',independent=True,reviewer='independent-test',bindings={'gpu':'0' if stale else '1'})))
    assert b.admit(p)['status']==('HOLD' if stale else 'ADMITTED')
    with pytest.raises(ValueError): b.require_fit(b.config_path(46,True),p)


def test_full_training_and_watchdog_ast_identity():
    for kind in ('fit','watchdog'):
        p=(b.ROOT/f'tools/hb166_fold3_{kind}_v1.py').read_text()
        n=(b.ROOT/f'tools/hb166_allfold_{kind}_v1.py').read_text()
        n=n.replace('hb166_allfold_','hb166_fold3_').replace('allfold-v122','pilot-fold3-v110')
        n=n.replace('1.2.2','1.1.0').replace('EXPOSED_ALLFOLD_REPLACEMENT','EXPOSED_PAIRED_FOLD3')
        n=n.replace('    if device == "cuda":\n        b.require_gpu()\n','')
        a,c=ast.parse(p),ast.parse(n)
        for tree in (a,c):
            if isinstance(tree.body[0],ast.Expr): tree.body.pop(0)
        assert ast.dump(a)==ast.dump(c),kind


def test_transfer_is_only_new_and_paths_safe():
    from tools import hb166_allfold_transfer_v1 as t
    names=t.new_files()
    assert all(p.startswith(b.rel(b.WORK)+'/') or p in t.NEW_CODE for p in names)
    assert not any(p.endswith(('.npy','.pt','.csv')) and '/fold' not in p for p in names)
    with pytest.raises(ValueError): t.check_result_names({'../outside':{}})
    with pytest.raises(ValueError): t.ssh_base('-bad',51626)
    assert 'StrictHostKeyChecking=yes' in t.ssh_base(t.DEFAULT_HOST,51626)


def test_control_design_and_panel_spec():
    c=b.read(b.WORK/'control-inventory.json'); spec=b.read(b.WORK/'readout-spec.json')
    assert c['new_control_fits']==0 and c['optional_causal_design']['extra_control_fits']==6
    assert [f for f,v in c['folds'].items() if v['status']=='EXISTING_MATCHED_RECIPE']==['2','3']
    assert spec['L7']['gold58']['rows']==58 and spec['L7']['oai557']['rows']==557
    assert len(spec['reference_members'])==35 and spec['projection']['fit_hours']==6*4700/3600


def measurement_inputs():
    statistics={k:[.002,.001,.003] for k in ('H1_L6','H2_Contusion','H2_L6')}
    harm={f'{route}/{ruler}/{target}':[0,0,0] for route in ('S','L6')
          for ruler in ('jev','consensus','dread','aquila','primary')
          for target in (*s.HARM_TARGETS,'weak_four')}
    return statistics,harm


def test_two_fresh_fold_hypotheses_and_target_identity():
    spec=b.read(b.WORK/'readout-spec.json')
    assert spec['aggregation']==s.RULE
    assert s.RULE['folds']==[0,1,4] and set(s.RULE['hypotheses'])=={'H1','H2'}
    assert s.HARM_TARGETS==['Fracture',*[r.lens.TARGETS[i] for i in base_lens.WEAK]]
    assert r.lens.TARGETS[10]=='Contusion'
    assert spec['hypotheses_sha256']==s.hypotheses_sha256()
    assert b.read(b.WORK/'registration.json')['admission_purpose']=='measurement'
    out=s.measurement_gate(*measurement_inputs())
    assert out==dict(H1='PASS',H2='PASS',H2_harm='NOT_HARM',release='OUT_OF_SCOPE')


@pytest.mark.parametrize('key',['H1_L6','H2_Contusion','H2_L6'])
@pytest.mark.parametrize('value',[[0,-.01,.01],[.002,0,.003],[.002,-.001,.003],
                                 [float('nan'),.001,.003],[.002,.003,.001]])
def test_measurement_strict_positive_ci_and_nan(key,value):
    statistics,harm=measurement_inputs(); statistics[key]=value
    out=s.measurement_gate(statistics,harm)
    assert out['H1' if key=='H1_L6' else 'H2']!='PASS'
    assert out['H2' if key=='H1_L6' else 'H1']=='PASS'
    assert out['release']=='OUT_OF_SCOPE'


@pytest.mark.parametrize('target',s.HARM_TARGETS+['weak_four'])
@pytest.mark.parametrize('route',['S','L6'])
def test_h2_harm_blocks_only_h2(target,route):
    statistics,harm=measurement_inputs()
    harm[f'{route}/jev/{target}']=[-.002,-.003,-.001]
    assert s.measurement_gate(statistics,harm)==dict(H1='PASS',H2='NO_PASS',H2_harm='HARM',release='OUT_OF_SCOPE')


@pytest.mark.parametrize('case',['missing','nan','reversed'])
def test_h2_undefined_harm_holds(case):
    statistics,harm=measurement_inputs(); key=next(iter(harm))
    if case=='missing': harm.pop(key)
    if case=='nan': harm[key]=[0,float('nan'),0]
    if case=='reversed': harm[key]=[0,1,-1]
    assert s.measurement_gate(statistics,harm)['H2']=='HOLD'


def test_exactly_two_hypotheses_no_extra_statistic():
    statistics,harm=measurement_inputs(); statistics['selected_target']=[1,1,1]
    with pytest.raises(ValueError): s.measurement_gate(statistics,harm)


@pytest.mark.parametrize('case',['valid','peak-mismatch','over-12gib','no-headroom','negative-used','no-time'])
def test_measured_memory_binding(tmp_path,monkeypatch,case):
    monkeypatch.setattr(b,'ROOT',tmp_path); monkeypatch.setattr(b,'WORK',tmp_path)
    monkeypatch.setattr(b,'registration_binding',lambda:{})
    monkeypatch.setattr(b,'prepare',lambda engineering=False:{'status':'PREPARED_ENGINEERING' if engineering else 'PREPARED'})
    monkeypatch.setattr(b,'smoke_paths',lambda:[])
    monkeypatch.setattr(b,'require_runtime',lambda runtime:None)
    monkeypatch.setattr(b,'require_gpu',lambda:{'dependency':'1'})
    peak=(13 if case=='over-12gib' else 8)*(1<<30)
    for name in ('prepared.json','cpu-prepared.json'): (tmp_path/name).write_text('{}')
    for seed in b.SEEDS:
        d=tmp_path/'cuda'/b.member(seed); d.mkdir(parents=True)
        (d/'runtime.json').write_text('{}'); (d/'summary.json').write_text(json.dumps({'peak_reserved_bytes':peak}))
    m=dict(schema='hb166-allfold-memory/v1',device=b.GPU,measured_at='test',peak_reserved_bytes=peak,
           used_before_cuda_bytes=1<<30,total_bytes=11*(1<<30))
    if case=='peak-mismatch': m['peak_reserved_bytes']-=1
    if case=='no-headroom': m['total_bytes']-=1
    if case=='negative-used': m['used_before_cuda_bytes']=-1
    if case=='no-time': m['measured_at']=''
    (tmp_path/'memory-admission.json').write_text(json.dumps(m))
    if case=='valid': assert b.review_bindings()['dependency']=='1'
    else:
        with pytest.raises(ValueError,match='memory|headroom'): b.review_bindings()
