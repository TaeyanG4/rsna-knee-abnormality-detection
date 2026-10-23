import pytest
from tools import pl_production_release_v1 as m


@pytest.fixture
def sealed(monkeypatch):
    members = [e for e in m.pl.member_roster() if e['kind'] == 'MIL']
    reg = dict(members=members, arm='global')
    monkeypatch.setattr(m.release, 'validate', lambda stage: {'status': 'PASS_PREPARED'})
    monkeypatch.setattr(m.pl, 'registration', lambda stage: reg)
    return reg


def test_seed42_cannot_satisfy_frozen_ten_checkpoint_release(sealed):
    result = m.preflight('stage', arm='global')
    assert result['status'] == 'HOLD'
    assert result['missing_members'] == [f'PL-S-s43-f{fold}' for fold in range(5)]
    assert result['requested_members'] == [f'PL-S-s42-f{fold}' for fold in range(5)]
    assert not result['extra_members']


def test_exact_roster_preflight(sealed):
    assert m.preflight('stage', arm='global', seeds=(42, 43))['status'] == 'READY_FOR_EXPORT'


def test_missing_folds_refuse_before_binder(tmp_path, monkeypatch, sealed):
    inputs = tmp_path/'inputs.json'
    m.pl.dump(inputs, dict(arm='global', members=sealed['members'][:5]))
    monkeypatch.setattr(m.release, 'bind', lambda *a: pytest.fail('binder must not run'))
    with pytest.raises(ValueError, match='checkpoint roster'):
        m.bind_completed(tmp_path, inputs, tmp_path/'bound.json')
    assert not (tmp_path/'bound.json').exists()


def test_binder_is_existing_a7(tmp_path, monkeypatch, sealed):
    inputs = tmp_path/'inputs.json'
    m.pl.dump(inputs, dict(arm='global', members=sealed['members']))
    calls = []
    monkeypatch.setattr(m.release, 'bind', lambda *a: calls.append(a) or {'status': 'BOUND'})
    output = tmp_path/'bound.json'
    assert m.bind_completed(tmp_path, inputs, output)['status'] == 'BOUND'
    assert calls == [(tmp_path, inputs, output.resolve())]


@pytest.mark.parametrize('name', ['registration.json', 'registration.sha256', 'variant-a.json', 'variant-b.json'])
def test_frozen_files_cannot_be_binding_outputs(tmp_path, monkeypatch, sealed, name):
    inputs = tmp_path/'inputs.json'
    m.pl.dump(inputs, dict(arm='global', members=sealed['members']))
    monkeypatch.setattr(m.release, 'bind', lambda *a: pytest.fail('binder must not run'))
    with pytest.raises(ValueError, match='frozen stage'):
        m.bind_completed(tmp_path, inputs, tmp_path/name)


def test_cannot_replace_input_template(tmp_path, sealed):
    inputs = tmp_path/'inputs.json'
    m.pl.dump(inputs, dict(arm='global', members=sealed['members']))
    with pytest.raises(ValueError, match='differ from inputs'):
        m.bind_completed(tmp_path, inputs, inputs)


def test_child_stage_has_only_seed42_and_preserves_parent(tmp_path):
    parent_hash=m.pl.sha(m.release.OUT/'registration.json')
    child=tmp_path/'production-v1'
    assert m.register(child)['members']==5
    assert m.validate(child)['status']=='PASS_PREPARED_CHILD'
    for arm in ('global','fill'):
        assert m.preflight(child,arm=arm)['status']=='READY_FOR_EXPORT'
        reg=m.pl.registration(child/'student'/arm)
        assert reg['members']==m.MIL
        assert [x['fold'] for x in reg['members']]==list(range(5))
    assert m.pl.sha(m.release.OUT/'registration.json')==parent_hash


def test_child_parent_drift_blocks(tmp_path):
    child=tmp_path/'production-v1'
    m.register(child)
    reg=m.pl.read(child/'registration.json')
    reg['parent_registration_sha256']='0'*64
    m.pl.dump(child/'registration.json',reg)
    (child/'registration.sha256').write_text(m.pl.digest(reg)+'\n')
    with pytest.raises(ValueError,match='parent release changed'):
        m.validate(child)


def test_qualification_refuses_absent_inputs():
    from tools import pl_production_qualify_v1 as q
    result=q.plan(m.OUT,'visible')
    assert result['status']=='HOLD'
    assert len(result['reasons'])==3


def test_qualification_unpacks_authentic_v122_only(tmp_path):
    from tools import pl_production_qualify_v1 as q
    module,runtime,bootstrap=q.load_runtime(m.OUT,tmp_path)
    assert module.VERSION=='1.2.2'
    assert module.RELEASE=='C85R7+codecs+insurance'
    assert 'pl_production_qualify_v1' in (runtime/(module.MODULE+'.py')).read_text()
    assert (bootstrap/'sitecustomize.py').exists()


def test_serving_adapters_isolated_from_original():
    from tools import pl_production_serve_v1 as serve
    assert len(serve.adapter.MIL)==5
    assert len(m.release.MIL)==10
    assert serve.adapter.registration is m.registration
    assert serve.adapter.pl is m.production_a7.adapter
    assert m.production_a7.adapter is not m.pl


def test_variant_b_n32_refused():
    from tools import pl_production_qualify_v1 as q
    with pytest.raises(ValueError, match='variant a only'):
        q.plan(m.OUT,'n32',variant='b')


def test_scoped_a7_checks_genuine_new_teacher_and_receipts(tmp_path,monkeypatch):
    from tools import pl_production_a7_v1 as a7
    from tools import vnext_r7_pl_go_v111 as go
    reg=dict(arm='global',members=m.MIL)
    def asset(name,value):
        path=tmp_path/name;m.pl.dump(path,value)
        return dict(path=str(path),sha256=m.pl.sha(path))
    teacher=asset('teacher.json',dict(producer='H-B-145',producer_version='1.1.3',
        producer_exposure='EXPOSED',status='PASS',role='heldout',output_kind='pre_stacker',
        model_fold='int(sha1(uid)[:8],16)%4',complete=True,failures=0,clean_oof=False,
        exposure='EXPOSED_EXPLORATORY',producer_identity='id',csv_sha256='csv'))
    seen=[]
    monkeypatch.setattr(a7.teacher,'validate',lambda path:seen.append(m.pl.read(path)['producer_version']))
    monkeypatch.setattr(go,'validate',lambda *a:'genuine-go')
    monkeypatch.setattr(go,'admission',lambda *a:'genuine-go')
    entries=[]
    for member in m.MIL:
        checkpoint=asset(member['id']+'.pt',{'fixture':'no tensors'})
        receipt=asset(member['id']+'.json',dict(status='COMPLETE',checkpoint_sha256=checkpoint['sha256'],
            arm='global',seed=42,fold=member['fold'],teacher_sha256=teacher['sha256'],
            registration_sha256=m.pl.digest(reg),clean_oof=False,exposure='EXPOSED_EXPLORATORY',
            target_sha256='target',train_ids_sha256='ids',l2_pos_weight_sha256='weights',
            spec_sha256='spec',go={}))
        entries.append(dict(member,**checkpoint,training_receipt=receipt))
    bindings=dict(registration_sha256=m.pl.digest(reg),arm='global',teacher=teacher,go={},members=entries)
    assert a7.validate_bindings(reg,bindings)==entries
    assert seen==['1.1.3']
    with pytest.raises(ValueError,match='heldout teacher seal'):
        m.pl.validate_bindings(reg,bindings)
    entries[0]['fold']=4
    with pytest.raises(ValueError,match='member recipe changed'):
        a7.validate_bindings(reg,bindings)


def test_scoped_a7_invalid_teacher_refused_first(monkeypatch):
    from tools import pl_production_a7_v1 as a7
    monkeypatch.setattr(a7.adapter,'pin',lambda entry:'teacher')
    def invalid(path): raise ValueError('production heldout seal required')
    monkeypatch.setattr(a7.teacher,'validate',invalid)
    monkeypatch.setattr(a7,'inherited_validate',lambda *a:pytest.fail('A7 must not bypass seal'))
    with pytest.raises(ValueError,match='production heldout seal'):
        a7.validate_bindings({},dict(teacher={}))


def test_older_baseline_refused_before_prediction_read(tmp_path,monkeypatch):
    from tools import pl_production_serve_v1 as s
    monkeypatch.setattr(s,'registration',lambda stage:{})
    monkeypatch.setattr(s.pl,'registration',lambda stage:{})
    monkeypatch.setattr(s.pl,'validate_bindings',lambda *a:[])
    binding=tmp_path/'binding.json';s.pl.dump(binding,dict(arm='global'))
    receipt=tmp_path/'old.json';s.pl.dump(receipt,dict(version='1.2.1',status='COMPLETE_C85R7'))
    with pytest.raises(ValueError,match='authenticated v122'):
        s.compose(tmp_path,binding,'a',tmp_path/'absent-a',tmp_path/'absent-b',[],
                  'run',tmp_path/'output',base_receipt=dict(path=str(receipt),sha256=s.pl.sha(receipt)))
