"""CPU-only candidate binding and admission tests; fixtures are not measurements."""
import copy
import json

import numpy as np
import pytest

from tools import k1n_qualify_v1 as q
from tools import k1_admit_v1 as a
from tools import k1n_stage_v1 as b
from test_k1_admit_v1 import case, put, change, transport
from test_k1n_members_v1 import filled, runtime
from test_k1n_v7 import records, replay


def attest(directory, kernel):
    path = directory/'k1sr-receipt.json'
    put(directory/(path.name+'.download.json'), dict(status='DOWNLOADED',
        kernel=kernel, version=1, file=path.name, local_name=path.name,
        sha256=a.sha(path.read_bytes())))


@pytest.fixture
def ratio_case(case, tmp_path, monkeypatch, request):
    options = dict(getattr(request, 'param', {}))
    template = options.pop('template', 'c7-bal-dropo-p25-v9.template.json')
    c = filled(b.ROOT/'configs/k1n'/template)
    c.update(release_version='1.1.77-cpu-ratio', kernel_id='test-owner/cpu-ratio-v1')
    c.update(options)
    c = b.canonical(c)
    original = b.stage
    def destination(value, diagnostic=False):
        if value['release_version'] == c['release_version']:
            return tmp_path/('v'+c['release_version']+('-n32' if diagnostic else ''))
        return original(value, diagnostic)
    monkeypatch.setattr(b, 'stage', destination)
    b.run('prepare', c)
    case.update(stage=b.stage(c), kernel=c['kernel_id'], n32_dir=tmp_path/'n32')
    visible = case['visible_dir']
    k = json.loads((visible/'k1sr-receipt.json').read_bytes())
    k.update(b.config(c))
    k['status'] = 'COMPLETE_SHARED_K1N'
    put(visible/'k1sr-receipt.json', k)
    def parent(p):
        p.update(version=c['release_version'], drop_o=c['drop_o'],
            intended_members=['R','S_R7','R224'], executed_members=['R','S_R7','R224'],
            skipped_members=['EXP143v2','EXP155','Q_A5q'])
        if c['drop_o']:
            del p['fallbacks']['a172_prediction']
        else:
            p['skipped_members'] = []
            for key in ('intended_members', 'executed_members'):
                p[key] += ['EXP143v2','EXP155','Q_A5q']
    change(visible/'c85r7-receipt.json', parent)
    transport(visible, kernel=c['kernel_id'])
    directory = case['n32_dir']
    directory.mkdir()
    # Produce a real v9 receipt using the baked finish implementation and
    # synthetic successful model predictions/timings. No model execution.
    r = runtime(c)
    values = np.full((32,12), .4, dtype=np.float32)
    (directory/'diagnostic-c85r7.csv').write_bytes(
        replay.csv_bytes(list(map(str,range(32))), values))
    rows = records(r, [values for _ in c['models']])
    for row in rows:
        row.update(model_identities=c['models'], k1_eval_stride=c['k1_eval_stride'],
            k1_owner_seconds=800/32, extra_render_seconds=0., failed_capture_seconds=0.)
    worker = directory/'run/pr7/pr7-run/s_r7/controller/worker0'
    worker.mkdir(parents=True)
    (worker/'k1-studies.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    r.finish(directory, directory/'run', b.config(c, True), 1500.)
    attest(directory, c['kernel_id']+'-n32')
    return case


@pytest.mark.parametrize('ratio_case', [dict(template=t, p_weight_mode='flat') for t in (
    'b5-flat-r6s2-v9.template.json', 'folds5-r6s2-av2-v9.template.json',
    'c5-dropo-p25-v9.template.json', 'c7-bal-dropo-p25s3-v9.template.json')], indirect=True)
def test_all_flat_recipes_prepare_validate_admit(ratio_case, capsys):
    case = dict(ratio_case)
    case.pop('n32_dir')
    a.prepare(**case)
    assert 'ADMITTED_OFFLINE' in capsys.readouterr().out
    c = json.loads((case['stage']/'config.json').read_bytes())
    assert b.run('validate', c)['status'] == 'PASS_CPU_VALIDATE'


def test_flat_templates_are_registered_twins():
    for stem in ('c5-dropo-p25', 'c7-bal-dropo-p25'):
        parent = json.loads((b.ROOT/f'configs/k1n/{stem}-v9.template.json').read_bytes())
        flat = json.loads((b.ROOT/f'configs/k1n/{stem}-flatp.template.json').read_bytes())
        assert {k for k in parent.keys() | flat.keys() if parent.get(k) != flat.get(k)} == {
            'release_version', 'kernel_id', 'p_weight_mode'}
        with pytest.raises(ValueError, match='real SHA256'):
            b.run('prepare', flat)
        assert flat['p_weight_mode'] == 'flat'
        if stem.startswith('c7'):
            assert flat['projection_basis'] == 'n32_ratio'


@pytest.mark.parametrize('ratio_case', [{}, {'p_weight_mode':'flat'},
    {'k1_eval_stride':3}, {'p_weight_mode':'flat', 'k1_eval_stride':3}], indirect=True)
def test_ratio_qualification_and_real_offline_admission(ratio_case, capsys):
    case = ratio_case
    with pytest.raises(ValueError, match='missing input'):
        a.prepare(**case)
    q.qualify(case['stage'], case['n32_dir'])
    record = q.qualify(case['stage'], case['n32_dir'])  # immutable/idempotent
    assert record['qualified_for_submission'] is True
    assert record['projection']['limit_hours'] == 9.0
    receipt = json.loads((case['n32_dir']/'k1sr-receipt.json').read_bytes())
    assert receipt['runtime_projection']['qualification_limit_hours'] == 9.0
    assert receipt['runtime_projection']['refuse_above_hours'] == 9.0
    assert receipt['runtime_projection']['warn_above_hours'] == 8.3
    assert record['projection']['hours'] == pytest.approx(6.2*2300/1785)
    output = a.prepare(**case)
    assert 'ADMITTED_OFFLINE' in capsys.readouterr().out
    assert json.loads((output/'ordinary-visible-registration.json').read_bytes())[
        'projection']['hours'] == record['projection']['hours']
    assert not json.loads((case['stage']/'registration.json').read_bytes())['qualified_for_submission']
    assert not (case['stage']/q.RECORD).exists()
    c = json.loads((case['stage']/'config.json').read_bytes())
    assert b.run('validate', c)['status'] == 'PASS_CPU_VALIDATE'


def test_ratio_cannot_omit_n32(ratio_case):
    case = dict(ratio_case)
    case.pop('n32_dir')
    with pytest.raises(ValueError, match='qualification record required'):
        a.prepare(**case)


@pytest.mark.parametrize('mutation', [
    lambda n:n.update(k1_studies=31),
    lambda n:n.update(k1_studies=True),
    lambda n:n.update(k1_executed=False),
    lambda n:n.update(failed_study_count=1),
    lambda n:n.update(failed_study_count=False),
    lambda n:n.update(failed_study_fraction=.01),
    lambda n:n.update(fail_loud_triggered=True),
    lambda n:n.update(status='FAILED_K1_FAIL_LOUD'),
    lambda n:n.update(kernel_id='test-owner/other-n32'),
    lambda n:n.update(release_version='1.1.67-other-n32'),
    lambda n:n.update(config_sha256='0'*64),
    lambda n:n.update(k1_eval_stride=3),
    lambda n:n.update(member_weights=[1/7]*7),
    lambda n:n.update(model_sha256={}),
    lambda n:n.update(diagnostic=False),
    lambda n:n.update(never_submit=False),
    lambda n:n.update(parent_seconds=0),
    lambda n:n.update(parent_seconds=True),
    lambda n:n.update(parent_seconds=2000),  # >9 h with K1=800
    lambda n:n.update(k1_seconds=1),  # timing inconsistency
    lambda n:n['per_study'].pop(),
    lambda n:n['per_study'][0].update(uid='1'),
    lambda n:n['per_study'][0].update(ok=False),
    lambda n:n['per_study'][0].update(systemic_failure=True),
    lambda n:n['per_study'][0].update(config_sha256='bad'),
    lambda n:n['per_study'][0].update(k1_eval_stride=3),
    lambda n:n['per_study'][0].update(model_identities=[]),
    lambda n:n['per_study'][0].update(predictions={}),
    lambda n:n['per_study'][0]['predictions'].update(k1_v2=[1.1]*12),
    lambda n:n['per_study'][0].update(k1_owner_seconds=0),
])
def test_unqualified_receipts_refused(ratio_case, mutation):
    case = ratio_case
    change(case['n32_dir']/'k1sr-receipt.json', mutation)
    attest(case['n32_dir'], case['kernel']+'-n32')
    with pytest.raises((ValueError, KeyError)):
        q.qualify(case['stage'], case['n32_dir'])
    assert not (case['n32_dir']/q.RECORD).exists()


@pytest.mark.parametrize('which', ['record', 'receipt', 'download', 'stage', 'n32stage', 'config', 'notebook'])
def test_admission_rechecks_qualification(ratio_case, which):
    case = ratio_case
    q.qualify(case['stage'], case['n32_dir'])
    if which == 'record':
        change(case['n32_dir']/q.RECORD, lambda n:n['projection'].update(hours=1))
    elif which == 'receipt':
        change(case['n32_dir']/'k1sr-receipt.json', lambda n:n.update(parent_seconds=1501))
        attest(case['n32_dir'], case['kernel']+'-n32')
    elif which == 'download':
        change(case['n32_dir']/'k1sr-receipt.json.download.json', lambda n:n.update(version=2))
    elif which in ('stage','n32stage'):
        stage = case['stage'] if which == 'stage' else case['stage'].with_name(case['stage'].name+'-n32')
        change(stage/'registration.json', lambda n:n.update(builder_version='tampered'))
    elif which == 'config':
        change(case['stage']/'config.json', lambda n:n.pop('projection_basis'))
    else:
        with (case['stage']/'kernel/release.ipynb').open('ab') as f:
            f.write(b' ')
    with pytest.raises(ValueError):
        a.prepare(**case)


@pytest.mark.parametrize('value', [None, True, -1, 0, float('inf'), float('nan'), '1500'])
def test_ratio_invalid_timing(value):
    with pytest.raises(ValueError):
        q.measured_hours(value, 419)
    with pytest.raises(ValueError):
        q.measured_hours(1366, value)


def test_ratio_threshold_and_anchor():
    assert q.measured_hours(1366,419)['hours'] == 6.2
    boundary = 1785*9.0/6.2
    assert q.measured_hours(boundary-800,800)['hours'] <= 9.0
    assert q.measured_hours(boundary-800+.001,800)['hours'] > 9.0
    assert q.measured_hours(1785*8.8/6.2-800,800)['limit_hours'] == 9.0


@pytest.mark.parametrize('ratio_case', [{'p_weight_mode':'flat'}], indirect=True)
@pytest.mark.parametrize('hours', [8.8, 9.0, 9.0001])
def test_flat_ratio_measured_boundary(ratio_case, hours):
    case = ratio_case
    change(case['n32_dir']/'k1sr-receipt.json',
           lambda n:n.update(parent_seconds=1785*hours/6.2-800))
    attest(case['n32_dir'], case['kernel']+'-n32')
    if hours > 9:
        with pytest.raises(ValueError, match='exceeds 9'):
            q.qualify(case['stage'], case['n32_dir'])
    else:
        record = q.qualify(case['stage'], case['n32_dir'])
        assert record['projection']['hours'] == pytest.approx(hours)


@pytest.mark.parametrize('ratio_case', [{'p_weight_mode':'flat'}], indirect=True)
@pytest.mark.parametrize('location', ['visible', 'stage', 'n32'])
@pytest.mark.parametrize('key,value', [('p_weight_mode',None),
    ('p_weight_mode','public'), ('p_raptor_weights',[.2]*12)])
def test_flat_identity_cannot_drift(ratio_case, location, key, value):
    case = ratio_case
    q.qualify(case['stage'], case['n32_dir'])
    directory = case['stage'] if location == 'stage' else case[location+'_dir']
    path = directory/('registration.json' if location == 'stage' else 'k1sr-receipt.json')
    change(path, lambda n:n.pop(key) if value is None else n.update({key:value}))
    if location == 'visible':
        transport(directory, kernel=case['kernel'])
    elif location == 'n32':
        attest(directory, case['kernel']+'-n32')
    with pytest.raises(ValueError):
        a.prepare(**case)


@pytest.mark.parametrize('basis', [None, False, 'formula', 'N32_ratio', {}, 1])
def test_basis_is_exact_opt_in(basis):
    c = filled(b.ROOT/'configs/k1n/c7-bal-dropo-p25-v9.template.json')
    with pytest.raises(ValueError, match='projection_basis'):
        b.artifacts(dict(c, projection_basis=basis))


def test_absent_basis_preserves_formula_refusal():
    c = filled(b.ROOT/'configs/k1n/c7-bal-dropo-p25-v9.template.json')
    c.pop('projection_basis')
    with pytest.raises(ValueError, match='exceeds 8.8'):
        b.artifacts(c)


@pytest.mark.parametrize('field,value', [('alpha',.2), ('alpha_vector',[.25]*12),
    ('r6_eval_stride',1), ('k1_guard_hours',8.6), ('k1_max_failed_fraction',.1)])
def test_p25s3_exact_recipe(field, value):
    c = filled(b.ROOT/'configs/k1n/c7-bal-dropo-p25s3-v9.template.json')
    with pytest.raises(ValueError):
        b.artifacts(dict(c, **{field:value}))


def test_p25s3_baked_stride_and_formula():
    c = filled(b.ROOT/'configs/k1n/c7-bal-dropo-p25s3-v9.template.json')
    r = runtime(c)
    cfg = b.config(c)
    assert r.K1_EVAL_STRIDE == cfg['k1_eval_stride'] == cfg['contract']['eval_stride'] == 3
    assert b.members.projection(c)['hours'] == 7.459
    assert cfg['drop_o'] and cfg['k1_weight'] == .25
    assert b.members.projection(dict(c, models=c['models'][:3]))['hours'] == 5.579
