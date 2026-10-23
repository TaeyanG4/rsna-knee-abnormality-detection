"""Collected paired screen v1.0.0: integrity and real historical CPU migration.

Historical CSV migration verifies engineering paths only; it creates no new
training evidence. Existing artifacts are read-only and no label rows are logged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from training.arms import lifecycle as life
from training.arms import collected_screen as cs
from test_arm_artifacts import registered

ROOT = Path(__file__).resolve().parents[1]


def put(path, value):
    life.create_new(path, life.canonical_bytes(value))
    return {'path': str(path), 'sha256': life.file_sha256(path)}


@pytest.fixture
def pair(registered, monkeypatch):
    spec, root = registered
    monkeypatch.setattr(cs, 'SOURCES', ())
    monkeypatch.setattr(cs, 'EVALUATION', ())
    doc = spec.document
    ids = ['a', 'b']
    roster = put(root/'ordered.json', dict(train=['c'], val=ids))
    doc['inputs']['rosters'] = {'ordered.json': roster['sha256']}
    doc['config']['targets'] = list(cs.multi.TARGETS)
    doc['config']['roots'] = dict(ordered='ordered.json', n_val=2,
                                val_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
    doc['outputs'] = ['predictions.csv']
    spec = life.ArmSpec.from_document(doc)
    sp = put(root/'spec.json', doc)
    outputs = root/'run'; outputs.mkdir()
    raw = life.prediction_csv_bytes(ids, np.full((2, 12), .5), cs.multi.TARGETS,
                                    id_column=cs.multi.UID)
    life.create_new(outputs/'predictions.csv', raw)
    receipt = life.artifact_receipt(spec, outputs, verification_sha256='1'*64,
                                    exit_code=0, elapsed_seconds=10**9, metrics={'rows':2})
    rh = life.content_sha256(receipt)
    life.export_artifacts(spec, outputs, receipt, root/'archive.tar', expected_receipt_sha256=rh)
    life.collect_artifacts(spec, root/'archive.tar', root/'collected', expected_receipt_sha256=rh)
    control = root/'control/predictions.csv'; life.create_new(control, raw)
    entry = dict(spec=sp, collection={'path':str(root/'collected/collection.json'),
                                    'sha256':life.file_sha256(root/'collected/collection.json')},
                 prediction='predictions.csv')
    request = dict(schema=cs.REQUEST, readout=dict(name='fixture',family='convnext-mil'),
                   collections={'42':dict(arm=[entry],control=[dict(path=str(control),sha256=life.file_sha256(control))])})
    return root, request


def test_collected_rederivation_and_full_lenses_remain_pending(pair, monkeypatch):
    root, request = pair
    full = dict(verdict={'verdict':'ADVANCE'}, lenses={f'L{i}': {'PASS':True} for i in range(1,8)},
                harm_guard={'PASS':False}, seed_details={'42':'retained'})
    monkeypatch.setattr(cs.multi, 'run_spec', lambda *a, **k: full)
    gate = put(root/'gate.json', {'status':'OLD_HOLD'})
    request['registered_gate'] = gate
    spec = cs.register(root, request, root/'screen-spec.json')
    assert cs.validate(root, spec) == spec
    out = cs.screen(root, spec, root/'result.json')
    assert out['multi_lens'] == full
    assert out['registered_gate']['result'] == {'status':'OLD_HOLD'}
    assert out['holistic_decision'] == 'B_DECISION_PENDING' and not out['launch_authorized']
    with pytest.raises(ValueError):
        cs.screen(root, spec, root/'result.json')


@pytest.mark.parametrize('mutation', ['prediction', 'receipt', 'manifest', 'spec', 'roster', 'control'])
def test_collected_inputs_fail_closed_after_registration(pair, mutation):
    root, request = pair
    spec = cs.draft(root, request)
    paths = dict(prediction='collected/outputs/predictions.csv', receipt='collected/receipt.json',
                 manifest='collected/collection.json', spec='spec.json', roster='ordered.json',
                 control='control/predictions.csv')
    with (root/paths[mutation]).open('ab') as f:
        f.write(b' ')
    with pytest.raises(ValueError):
        cs.validate(root, spec)


@pytest.mark.parametrize('mutation', ['seed', 'family', 'duplicate', 'missing_hash', 'wrong_control', 'spec_edit'])
def test_collected_request_identity_rejection(pair, mutation):
    root, req = pair
    if mutation == 'seed': req['collections']['43'] = req['collections'].pop('42')
    elif mutation == 'family': req['readout']['family'] = 'coatnet'
    elif mutation == 'duplicate': req['collections']['42']['arm'] *= 2
    elif mutation == 'missing_hash': del req['collections']['42']['arm'][0]['spec']['sha256']
    elif mutation == 'wrong_control':
        shutil.copyfile(root/'control/predictions.csv', root/'wrong.csv')
        req['collections']['42']['control'][0]['path'] = str(root/'wrong.csv')
    else:
        spec = cs.draft(root, req); spec['collections']['42'][0]['fold'] = 4
        with pytest.raises(ValueError): cs.validate(root, spec)
        return
    with pytest.raises(ValueError): cs.draft(root, req)


def test_mid_read_mutation_prevents_publication(pair, monkeypatch):
    root, req = pair
    spec = cs.draft(root, req)
    def mutate(*a, **k):
        with (root/'control/predictions.csv').open('ab') as f: f.write(b' ')
        return {'lenses':{}}
    monkeypatch.setattr(cs.multi, 'run_spec', mutate)
    with pytest.raises(ValueError): cs.screen(root, spec, root/'result.json')
    assert not (root/'result.json').exists()


@pytest.mark.parametrize('name', ['../outside.csv', '.local/relabel/panel.csv', 'H:/kaggle/api-keys/key.csv'])
def test_restricted_paths_rejected_before_read(pair, name):
    root, _ = pair
    with pytest.raises(ValueError): cs.pin(root, name, {}, suffix='.csv')


def test_real_expression_bundle_and_gold58_cpu_readout(tmp_path):
    """Exercise optional expressions/image inputs on designated numeric artifacts."""
    prior = cs.read(ROOT/'artifacts/experiments/EXP179_MULTISCALE/screen/result.json')
    fixed = {k: dict(path=v, sha256=prior['input_sha256'][k])
             for k,v in prior['input_files'].items()}
    blend = lambda a,b: dict(blend=[dict(files=a,weight=.5),dict(files=b,weight=.5)])
    arm = str(ROOT/'artifacts/experiments/EXP185_FINDING_MIL/outputs/S42/results/EXP185_FINDING_MIL_F0_S42/oof_k36_flip_tta.csv')
    control = str(ROOT/'artifacts/experiments/EXP179_MULTISCALE/outputs/M42/results/EXP179_M42_F0/oof_k36_flip_tta.csv')
    gold = ROOT/'artifacts/experiments/EXP149_LABEL_SOURCES/screen/gold58'
    spec = dict(name='engineering-expression-card', family='convnext-mil',
        seeds={'42':dict(arm=dict(files=arm,rank_folds_like=[arm]),
                         control=dict(files=control,rank_folds_like=[control]))},
        partners=dict(r50=blend(fixed['R42'],fixed['R43']),dino=fixed['D172'],convnext=fixed['C42']),
        image_reads=dict(gold58=dict(arm=blend(str(gold/'EXP149_L2_S42_F0.csv'),str(gold/'EXP149_L2_S42_F1.csv')),
            control=blend(str(gold/'EXP127_AUG1_S42_F0.csv'),str(gold/'EXP134_AUG1_S42_F1.csv')),
            labels='storage/competition/train.csv',components='gold58_singletons',eligible=False)))
    result = cs.readout(spec, root=ROOT)
    assert result['readout_spec']['family'] == 'convnext'
    direct = cs.multi.clean(cs.multi.run_spec(result['readout_spec'],root=ROOT))
    assert result['multi_lens'] == direct
    assert not direct['lenses']['L7']['entries']['gold58']['eligible']
    put(tmp_path/'optional-readout.json',result)
    spec['image_reads']['gold58']['eligible'] = True
    with pytest.raises(ValueError,match='diagnostic'):
        cs.bind_readout(spec,ROOT)


def test_real_historical_artifact_cpu_collection_and_holistic_card(tmp_path):
    base = Path(os.environ.get('COLLECTED_SCREEN_SMOKE_OUT', tmp_path/'real'))
    base.mkdir(parents=True, exist_ok=False)
    # Inputs are genuine full 888-study EXP185 predictions. New specs explicitly
    # identify a retrospective artifact migration, never a newly trained model.
    commands = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    def command(module, *args, expected=0):
        argv = [sys.executable, '-B', '-m', module, *map(str,args)]
        cp = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
        commands.append(dict(argv=argv, exit_code=cp.returncode, expected=expected,
                             stdout=cp.stdout, stderr=cp.stderr))
        (base/'command-results.json').write_text(json.dumps(commands,indent=2), encoding='utf-8')
        assert cp.returncode == expected, f'CPU command {len(commands)} exit mismatch; see command-results.json'
    req = dict(schema=cs.REQUEST, readout=dict(name='historical-EXP185-artifact-migration',
                family='convnext-mil', label_change=False, old_verdict='retained in registered_gate'), collections={})
    for seed in (42,43):
        run = base/str(seed); run.mkdir()
        doc = cs.read(ROOT/f'training/configs/mil-s{seed}-f0-v3.json')
        doc.update(arm=f'HISTORICAL_EXP185_MIGRATION_S{seed}', attempt='engineering-only-v1')
        pred = 'oof_k36_flip_tta.csv'
        doc['outputs'] = [pred]
        put(run/'draft.json', doc)
        spec_path = run/'spec.json'
        command('tools.arm_lifecycle', 'register', '--draft', run/'draft.json','--root',ROOT,'--out',spec_path)
        command('tools.arm_lifecycle', 'validate','--spec',spec_path,'--root',ROOT)
        outputs = run/'outputs'; outputs.mkdir()
        source = ROOT/f'artifacts/experiments/EXP185_FINDING_MIL/outputs/S{seed}/results/EXP185_FINDING_MIL_F0_S{seed}'/pred
        shutil.copyfile(source, outputs/pred)
        note = put(run/'migration.json',dict(status='ENGINEERING_HISTORICAL_MIGRATION_ONLY',
                                            source_sha256=life.file_sha256(source), launch_authorized=False))
        put(run/'metrics.json', {'historical_rows':888})
        command('tools.arm_lifecycle','receipt','--spec',spec_path,'--outputs',outputs,'--out',run/'receipt.json',
                '--verification-sha256',note['sha256'],'--exit-code','0','--elapsed-seconds','1000000000','--metrics',run/'metrics.json')
        receipt_sha = life.content_sha256(cs.read(run/'receipt.json'))
        command('tools.arm_lifecycle','export','--spec',spec_path,'--outputs',outputs,'--receipt',run/'receipt.json',
                '--receipt-sha256',receipt_sha,'--out',run/'archive.tar')
        command('tools.arm_lifecycle','collect','--spec',spec_path,'--archive',run/'archive.tar',
                '--receipt-sha256',receipt_sha,'--out',run/'collected')
        control = ROOT/f'artifacts/experiments/EXP179_MULTISCALE/outputs/M{seed}/results/EXP179_M{seed}_F0'/pred
        binding = lambda p: dict(path=str(p),sha256=life.file_sha256(p))
        req['collections'][str(seed)] = dict(arm=[dict(spec=binding(spec_path),
                    collection=binding(run/'collected/collection.json'),prediction=pred)],control=[binding(control)])
    gate = ROOT/'artifacts/experiments/EXP185_FINDING_MIL/screen/result.json'
    req['registered_gate'] = binding(gate)
    # Actual collection cards must live within the project for portable binding.
    if not base.resolve().is_relative_to(ROOT):
        # Regular local pytest temp is project-owned under the mandated basetemp.
        pytest.fail('Use --basetemp under the project or COLLECTED_SCREEN_SMOKE_OUT')
    put(base/'request.json',req)
    command('tools.mil_screen','register-collected','--root',ROOT,'--request',base/'request.json','--out',base/'screen-spec.json')
    command('tools.mil_screen','validate','--root',ROOT,'--spec',base/'screen-spec.json')
    command('tools.mil_screen','screen','--root',ROOT,'--spec',base/'screen-spec.json','--out',base/'result.json')
    command('tools.mil_screen','screen','--root',ROOT,'--spec',base/'screen-spec.json','--out',base/'result.json',expected=2)
    result = cs.read(base/'result.json')
    assert set(result['multi_lens']['lenses']) == {f'L{i}' for i in range(1,8)}
    assert result['holistic_decision'] == 'B_DECISION_PENDING' and not result['launch_authorized']
    direct = cs.multi.clean(cs.multi.run_spec(cs.read(base/'screen-spec.json')['readout'],root=ROOT))
    assert result['multi_lens'] == direct
    assert result['registered_gate']['result'] == cs.read(gate)
