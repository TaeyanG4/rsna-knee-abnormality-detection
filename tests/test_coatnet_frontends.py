"""v1.0.0: explicit frontend CPU cards, immutable history and screen parity."""
from pathlib import Path
import json
import os
import runpy
import subprocess
import sys

import pytest
from tools import coatnet_config as config
from training.arms import coatnet_history as history, coatnet_screen as screen

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = {
    'control-f0-s42': ['tools.exp181_coatnet_f0_bridge','lifecycle'],
    'aug1-f0-s42': ['tools.exp181_coatnet_f0_bridge','lifecycle','--preset-arm','AUG1'],
    'coverage-f0-s42': ['tools.exp188_coatnet_coverage_f0','lifecycle'],
    'crop-f0-s42': ['tools.exp192_coatnet_crop_f0','lifecycle'],
    'control-full-s43': ['tools.exp182_own0949_s43','lifecycle'],
}

@pytest.mark.parametrize('preset',config.PRESETS)
def test_real_frontend_cpu_card(preset,monkeypatch):
    location=os.environ.get('COATNET_FRONTEND_SMOKE_OUT')
    if not location: pytest.skip('Use run-step8-tests.py for durable real frontend CPU cards')
    monkeypatch.setenv('COATNET_LIFECYCLE_SMOKE_OUT',location)
    old_run=subprocess.run
    def routed(argv,**kw):
        if len(argv)>3 and argv[3]=='training.runners.coatnet_lifecycle':
            args=list(argv[4:])
            if args[0]=='register':
                for flag in ('--variant','--population','--seed'):
                    at=args.index(flag); del args[at:at+2]
            entry=ENTRIES[preset]
            # The historical crop smoke alias uses the same immutable spec/stage.
            if preset=='crop-f0-s42' and args[0]=='smoke':
                entry=['tools.exp192_cpu_smoke']; args=args[1:]
            argv=argv[:3]+entry+args
        return old_run(argv,**kw)
    monkeypatch.setattr(subprocess,'run',routed)
    suite=runpy.run_path(str(ROOT/'tests/test_coatnet_lifecycle.py'))
    suite['test_real_lifecycle_cpu_card'](*config.PRESETS[preset])
    card=Path(location)/f'{config.PRESETS[preset][0]}-{config.PRESETS[preset][1]}-s{config.PRESETS[preset][2]}'
    # Original helper records abstract commands; additionally record actual argv.
    commands=json.loads((card/'command-results.json').read_text())
    for row in commands:
        args=row['argv'][4:]
        if args[0]=='register':
            for flag in ('--variant','--population','--seed'):
                at=args.index(flag); del args[at:at+2]
        entry=ENTRIES[preset]
        if preset=='crop-f0-s42' and args[0]=='smoke': entry=['tools.exp192_cpu_smoke'];args=args[1:]
        row['argv']=row['argv'][:3]+entry+args
    screen.write_new(card/'frontend-command-results.json',commands,ROOT)

@pytest.mark.parametrize('preset',config.PRESETS)
@pytest.mark.parametrize('override',['--variant=crop','--population','--seed=42'])
def test_preset_cannot_be_overridden(preset,override):
    with pytest.raises(ValueError,match='cannot be overridden'):
        config.dispatch(preset,['register',override])

@pytest.mark.parametrize('module',['tools.exp188_coatnet_coverage_f0','tools.exp192_coatnet_crop_f0'])
@pytest.mark.parametrize('command',['register','validate','train','pilot','predict-oof','queue-json'])
def test_old_producers_retired(module,command,tmp_path):
    result=subprocess.run([sys.executable,'-B','-m',module,command],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==2
    assert 'invalid choice' in result.stderr
    assert list(tmp_path.iterdir())==[]

@pytest.mark.parametrize('case',['positive','zero','macro','other','dread','undefined'])
def test_historical_score_equations_and_harm_guards(monkeypatch,tmp_path,case):
    from tools import exp181_coatnet_f0_bridge as bridge, exp179_screen as helper
    ids=[f'SYNTHETIC_{i}' for i in range(888)]
    rec=dict(p=dict(val_ids=ids),run=tmp_path)
    delta={t:0. for t in bridge.TARGETS}
    for t in screen.WEAK: delta[t]=0. if case=='zero' else .004
    if case=='other': delta['ACL']=-.011
    if case=='undefined': delta[screen.WEAK[0]]=None
    primary=dict(primary_mean_delta=-.0011 if case=='macro' else .001,
        references=dict(jev=dict(per_target_delta=delta),consensus=dict(per_target_delta=delta),
                        dread=dict(delta=-.0031 if case=='dread' else 0.)))
    paired=dict(primary_k94_mirror_tta=primary,diagnostic_k94_plain={},registered_gate={},ruler_advisory={})
    monkeypatch.setattr(helper,'load_context',lambda _: (ids,{},[],{}))
    monkeypatch.setattr(bridge,'load_oof',lambda *_:None)
    monkeypatch.setattr(bridge,'score',lambda *_:paired)
    if case=='undefined':
        with pytest.raises(ValueError,match='undefined primary'): screen.score_records(rec,rec)
    else:
        result=screen.score_records(rec,rec)
        assert result['historical_weak_gate']==('REVIEW' if case=='positive' else 'HOLD')
        assert result['weak_target_mean_delta']==(0. if case=='zero' else .004)
        if case in ('macro','other','dread'):
            assert result['harm_guard']['checks'][dict(macro='macro12',other='other_targets',dread='dread')[case]] is False

@pytest.mark.parametrize('name,candidate,module',[
    ('EXP181_COATNET_F0_BRIDGE','AUG1','tools.exp181_coatnet_f0_bridge'),
    ('EXP188_COATNET_COVERAGE_F0','COVERAGE','tools.exp188_coatnet_coverage_f0'),
    ('EXP192B_COATNET_CROP_F0','CROP','tools.exp192_coatnet_crop_f0')])
def test_real_historical_collection_and_screen(name,candidate,module,monkeypatch):
    location=os.environ.get('COATNET_HISTORY_SMOKE_OUT')
    if not location: pytest.skip('Use run-step8-tests.py for durable historical paired cards')
    card=Path(location)/candidate;card.mkdir(parents=True,exist_ok=False)
    commands=[]
    def run(args,code,status):
        argv=[sys.executable,'-B','-m',module,*map(str,args)]
        p=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        log=card/f'command-{len(commands):02d}.log';log.write_text(p.stdout+p.stderr,encoding='utf-8')
        commands.append(dict(argv=argv,returncode=p.returncode,log=log.name,sha256=history.sha256(log)))
        assert p.returncode==code,p.stderr[-2000:]+p.stdout[-2000:]
        assert json.loads(p.stdout)['status']==status
    extra=['--preset-arm','AUG1'] if candidate=='AUG1' else []
    collection=card/'collection.json';out=card/'screen.json'
    run(['collect',*extra,'--out',collection,'--verification-note','CPU engineering readout'],0,'COLLECTED')
    run(['screen',*extra,'--out',out,'--collection',collection,'--verification-note','CPU engineering readout'],0,'B_DECISION_PENDING')
    result=json.loads(out.read_text())
    assert result['launch_authorized'] is False and result['holistic']['launch_authorized'] is False
    assert result['holistic']['holistic_decision']=='B_DECISION_PENDING'
    oldpath=ROOT/f'artifacts/experiments/{name}/screen/result-{candidate}-vs-CONTROL.json'
    if oldpath.exists():
        old=json.loads(oldpath.read_text())
        for key in ('primary_k94_mirror_tta','diagnostic_k94_plain','weak_target_mean_delta','per_target_mean_delta','harm_guard'):
            if key in old: assert result[key]==old[key],key
    before=out.read_bytes()
    run(['screen',*extra,'--out',out,'--collection',collection,'--verification-note','duplicate'],2,'REJECTED')
    assert out.read_bytes()==before
    tampered=json.loads(collection.read_text());tampered['candidate']='WRONG'
    screen.write_new(card/'tampered.json',tampered,ROOT)
    run(['screen',*extra,'--out',card/'rejected.json','--collection',card/'tampered.json','--verification-note','tamper'],2,'REJECTED')
    assert not (card/'rejected.json').exists()
    screen.write_new(card/'command-results.json',commands,ROOT)
