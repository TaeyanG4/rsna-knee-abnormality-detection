"""v1.0.0: collection pipeline and engineering-only receipt attestations."""
import copy
from pathlib import Path

import pytest

from tools import cpu_pipeline as cp
from tools import job_runtime as rt
from training.arms import lifecycle as life
from training.arms import collected_screen as cs
from test_arm_artifacts import registered
from test_collected_screen import pair


def pin(file):
    return dict(path=str(file),sha256=life.file_sha256(file))


def fixture_tests(root,spec):
    # Explicitly synthetic unit receipt. Real command evidence is built by the
    # parent gate and consumed by the separate real-data CLI card.
    cp.put(root/'unit-receipt-marker.json',dict(synthetic=True))
    life.create_new(root/'test.log',b'unit fixture only\n')
    life.create_new(root/'test.xml',b'<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="unit"/></testsuite></testsuites>')
    sources=spec.document['inputs']['sources']
    receipt=dict(schema='executed-cpu-tests/v1',command=['synthetic-unit-fixture'],exit_code=0,
                 counts=dict(tests=1,failures=0,errors=0,skipped=0),source_before=sources,source_after=sources,
                 artifacts={n:pin(root/f'test.{s}') for n,s in [('log','log'),('junit','xml')]})
    cp.put(root/'tests.json',receipt)
    return root/'tests.json'


def test_attestation_is_hold_with_copied_evidence_and_admission(registered):
    spec,root=registered
    tests=fixture_tests(root,spec)
    note=cp.attest(root,spec,tests,root/'evidence-out')
    assert note['verdict']=='HOLD' and note['smokes']==dict(cpu=None,target=None)
    life.prepare(spec,root,root/'stage')
    admission=life.admission_dry_run(spec,root/'stage',note,root/'evidence-out')
    assert admission['status']=='HOLD' and not admission['launch_authorized']
    with pytest.raises(ValueError): cp.attest(root,spec,tests,root/'evidence-out')


@pytest.mark.parametrize('mutation',['exit','counts','source','drift','coverage','log','junit','junit_failure','command'])
def test_bad_test_evidence_cannot_generate_attestation(registered,mutation):
    spec,root=registered; tests=fixture_tests(root,spec); receipt=rt.read_json(tests)
    if mutation=='exit': receipt['exit_code']=1
    elif mutation=='counts': receipt['counts']['skipped']=1
    elif mutation=='source': (root/'runner.py').write_text('changed')
    elif mutation=='drift': receipt['source_after']={'runner.py':'f'*64}
    elif mutation=='coverage': receipt['source_before']=receipt['source_after']={'data.bin':life.file_sha256(root/'data.bin')}
    elif mutation=='log': (root/'test.log').write_text('changed')
    elif mutation=='junit': receipt['counts']['tests']=2
    elif mutation=='junit_failure':
        (root/'test.xml').write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase><failure/></testcase></testsuite></testsuites>')
        receipt['artifacts']['junit']=pin(root/'test.xml')
    else: receipt['command']=[]
    rt.atomic_json(tests,receipt)
    with pytest.raises(ValueError): cp.attest(root,spec,tests,root/'evidence-out')
    assert not (root/'evidence-out').exists()


@pytest.fixture
def pipeline(pair,monkeypatch):
    root,request=pair
    monkeypatch.setattr(cs.multi,'run_spec',lambda *a,**k:dict(lenses={f'L{i}':{} for i in range(1,8)}))
    entry=request['collections']['42']['arm'][0]
    spec=life.ArmSpec(Path(entry['spec']['path']).read_bytes())
    tests=fixture_tests(root,spec)
    entry['collection']={'path':'@arm'}
    plan=dict(schema='cpu-pipeline/v1',collections=[dict(name='arm',spec=entry['spec'],archive=pin(root/'archive.tar'),
              receipt_sha256=rt.read_json(root/'collected/collection.json')['receipt_sha256'])],
              screen_request=request,next=dict(spec=entry['spec'],tests=pin(tests)))
    return root,plan


def test_pipeline_idempotent_only_exact_input_and_output(pipeline):
    root,plan=pipeline
    result=cp.run(root,plan,root/'pipeline')
    before=cp.inventory(root/'pipeline')
    assert result['holistic_decision']=='B_DECISION_PENDING' and not result['launch_authorized']
    assert rt.read_json(root/'pipeline/admission.json')['status']=='HOLD'
    assert cp.run(root,plan,root/'pipeline')==result
    assert before==cp.inventory(root/'pipeline')
    screen=rt.read_json(root/'pipeline/screen.json')
    assert len(screen['multi_lens']['lenses'])==7


@pytest.mark.parametrize('change',['archive','plan','prediction','stage','evidence','extra_output','screen'])
def test_pipeline_rejects_changed_inputs_or_results(pipeline,change):
    root,plan=pipeline; cp.run(root,plan,root/'pipeline')
    paths=dict(archive='archive.tar',prediction='pipeline/collections/arm/outputs/predictions.csv',
               stage='pipeline/stage/runner.py',evidence='pipeline/attestation/artifacts/log.log',
               extra_output='pipeline/extra',screen='pipeline/screen.json')
    if change=='plan': plan['screen_request']['readout']['name']='new contrast'
    else:
        with (root/paths[change]).open('ab') as stream: stream.write(b'changed')
    with pytest.raises((ValueError,FileNotFoundError)): cp.run(root,plan,root/'pipeline')


def test_pipeline_collection_failure_never_publishes_next_stage(pipeline):
    root,plan=pipeline; plan['collections'][0]['receipt_sha256']='f'*64
    with pytest.raises(ValueError): cp.run(root,plan,root/'pipeline')
    assert not (root/'pipeline/stage').exists() and not (root/'pipeline/pipeline.json').exists()
    with pytest.raises((ValueError,FileNotFoundError)): cp.run(root,plan,root/'pipeline')


@pytest.mark.parametrize('name',['../escape','.local/relabel/labels.json','.codex/config.toml','api-keys/key'])
def test_pipeline_restricted_paths_before_read(tmp_path,name):
    with pytest.raises(ValueError): cp.path(tmp_path,name)


def test_screen_only_pipeline_has_no_stage(pipeline):
    root,plan=pipeline; plan['next']=None
    result=cp.run(root,plan,root/'screen-only')
    assert result['status']=='CPU_COMPLETE' and not (root/'screen-only/stage').exists()
